//===- SwarmABI.cpp - Swarm hardware interface ----------------------------===//
//
//                       The SCC Parallelizing Compiler
//
//          Copyright (c) 2020 Massachusetts Institute of Technology
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Provides the interface to generate calls into hardware or simulators
// for features implemented by Swarm extensions to a conventional ISA.
// This interface does the low-level dirty work of passes such as LowerToSwarm.
//
//===----------------------------------------------------------------------===//

#include "Utils/SwarmABI.h"

#include "Impl.h"

#include "Utils/CFGRegions.h"
#include "Utils/Flags.h"
#include "Utils/Misc.h"
#include "Utils/Tasks.h"

#include "llvm/Analysis/SwarmAA.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Transforms/Swarm.h"
#include "llvm/Transforms/Tapir/CilkABI.h"  // for cilk::populateDetachedCFG
#include "llvm/Transforms/Tapir/Outline.h"  // for findInputsOutputs
#include "llvm/Transforms/Utils/TapirUtils.h"  // for MoveStaticAllocasInBlock

#include <sstream>
#include <map>

using namespace llvm;

#define DEBUG_TYPE "swarmabi"

static cl::opt<bool> DisableHints("swarm-disableallhints", cl::init(false),
    cl::desc("Transform all non-SAMEHINT tasks to NOHINT"));

enum class SwarmABIs {
  ordspecsim_v1,
  serial,
};
static cl::opt<SwarmABIs> SwarmABI("swarm-abi", cl::values(
    clEnumValN(SwarmABIs::ordspecsim_v1, "ordspecsim_v1",
               "Old ordspecsim magic ops and enqueue flags"),
    clEnumValN(SwarmABIs::serial, "serial",
               "Queue tasks in libsccrt (eschew Swarm simulator/hardware)")
    ), cl::init(SwarmABIs::ordspecsim_v1));


static ImplBase& Impl() {
  static ImplBase *ImplPtr = nullptr;
  if (!ImplPtr) {
    switch (SwarmABI) {
    case SwarmABIs::ordspecsim_v1:
      ImplPtr = new oss_v1();
      break;
    case SwarmABIs::serial:
      ImplPtr = new sccrt_serial();
      break;
    }
  }
  return *ImplPtr;
}


static std::multimap<unsigned, Attribute> getParamsWithBadAttribute(
        const Function &F) {
  std::multimap<unsigned, Attribute> BadParamAttrs;
  const AttributeList AL = F.getAttributes();
  for (unsigned i = 0; i < F.arg_size() ; i++) {
    for (const Attribute &Attr : AL.getParamAttributes(i)) {
      if (Attr.isStringAttribute()) {
        if (Attr.getKindAsString() != SwarmFlag::Continuation) {
          BadParamAttrs.insert({i, Attr});
        }
      } else {
        switch (Attr.getKindAsEnum()) {
        case Attribute::Alignment:
        case Attribute::Dereferenceable:
        case Attribute::DereferenceableOrNull:
        case Attribute::NoAlias:
        case Attribute::NoCapture:
        case Attribute::NonNull:
        case Attribute::ReadNone:
        case Attribute::ReadOnly:
        case Attribute::WriteOnly:
          // The above are safe as they only convey guarantees about pointers,
          // and do not affect ABI
          continue;
        case Attribute::SExt:
        case Attribute::ZExt:
          continue;
        default:
          BadParamAttrs.insert({i, Attr});
        }
      }
    }
  }
  return BadParamAttrs;
}


static void remarkParamsWithBadAttributes(
        const Function &F,
        const SetVector<Value*>& Inputs,
        const std::multimap<unsigned, Attribute>& BadParamAttrs) {
  for (const auto &pair: BadParamAttrs) {
    unsigned i = pair.first;
    const Argument &Arg = F.arg_begin()[i];
    const Value *V = Inputs[i];
    const Attribute &Attr = pair.second;

    std::string Msg;
    raw_string_ostream OS(Msg);
    OS << "Detached CFG wants to have the argument " << Arg
       << "\n with value " << *V
       << "\n with attribute " << Attr.getAsString() << "\n";
    OS.flush();
    F.getContext()
     .diagnose(DiagnosticInfoUnsupported(F, Msg, F.getSubprogram()));
  }
}


template<typename ValueCollection>
static StructType *createRegArgsStructTy(const ValueCollection& Values) {
  assert(Values.size());
  SmallVector<Type *, 8> Types;
  for (Value *V : Values)
    Types.push_back(V->getType());
  return StructType::create(Types, StringRef(), /*isPacked*/ true);
}


/// Copy the body of the detached code into a function
/// suitable for enqueuing to hardware.
/// Also set Args to contain the actual argument values
/// that should be passed from the parent task.
/// If the detached code cannot reach a matching reattach, do nothing.
/// \return the newly created task function,
///   or null if DI has no matching reattach.
static Function *extractDetachedCFG(SDetachInst &DI, DominatorTree &DT,
                                    const TargetTransformInfo &TTI,
                                    SmallVectorImpl<Value *> &Args) {
  BasicBlock *DetBB = DI.getParent();
  BasicBlock *Spawned  = DI.getDetached();
  Function *F = DetBB->getParent();

  // Shrink inputs by sinking cheap instructions into the task. We do this
  // before populateDetachedCFG as doing it after leads to erased values (by
  // eraseDetach()) being still used.
  //TODO(victory): Change shrinkInputs() API to look more like our outline()
  // utility, allowing us to reduce the boilerplate code here.
  {
    SmallVector<BasicBlock *, 8> Blocks;
    BasicBlock *Spawned = DI.getDetached();
    Blocks.push_back(Spawned);
    DT.getDescendants(DI.getDetached(), Blocks);
    shrinkInputs(Blocks, {DI.getTimestamp()}, TTI);
  }

#ifndef NDEBUG
  SmallVector<const SReattachInst *, 4> MatchingReattaches;
  getMatchingReattaches(&DI, DT, MatchingReattaches);
  SmallPtrSet<const BasicBlock *, 4> VerifyReattachBBs;
  for (const SReattachInst *RI : MatchingReattaches)
    VerifyReattachBBs.insert(RI->getParent());
#endif

  // Collect the BBs in the detached CFG.
  SmallPtrSet<BasicBlock *, 32> SpawnedBBs;
  SmallVector<BasicBlock *, 32> ReattachBBs;
  SmallPtrSet<BasicBlock *, 4> ExitBBs;
  //N.B. this call will replace reattaches with branches, leaving things in
  // a temporarily invalid state where the detach does not have matching
  // reattaches. Calls to assertVerifyFunction() will fail until the entire
  // detached region is removed.
  // In a future version of Tapir, the populateDetachedCFG() utility is no
  // longer responsible for performing this replacement.
  bool success = cilk::populateDetachedCFG(cast<DetachInst>(DI), DT,
                                           SpawnedBBs, ReattachBBs, ExitBBs,
                                           true /*replace*/);
  (void)success;
  assert(success && "Unable to collect detached basic blocks to extract");
  assert(all_of(ReattachBBs, [&VerifyReattachBBs](const BasicBlock *BB) {
                 return VerifyReattachBBs.count(BB); })
         && all_of(VerifyReattachBBs, [&ReattachBBs](const BasicBlock *BB) {
                 return find(ReattachBBs, BB) != ReattachBBs.end(); })
         && "getMatchingReattaches() out of sync with populateDetachedCFG()");

  if (ReattachBBs.empty()) {
    DEBUG(dbgs() << "Detach cannot reach any matching reattaches.\n");
    return nullptr;
  }

  // Check the spawned blocks' predecessors.
  assert(Spawned->getUniquePredecessor() &&
         "Entry block of detached CFG has multiple predecessors.");
  assert(Spawned->getUniquePredecessor() == DetBB &&
         "Broken CFG.");
  //FIXME(victory): Iterating over an unordered set yields non-determinism.
  for (BasicBlock *BB: SpawnedBBs) {
    if (BB == Spawned)
      continue;
    if (ExitBBs.count(BB))
      continue;
    for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
      if (!SpawnedBBs.count(*PI)) {
        dbgs() << "\n\n";
        DI.getDebugLoc().print(dbgs());
        dbgs() << "\nFor detach: " << DI << "\n"
               << "Block inside detached CFG:" << *BB << '\n'
               << "Has predecessor outside detached CFG:" << **PI << '\n';
        DEBUG(F->viewCFG());
        llvm_unreachable("Block inside of detached CFG reached from outside detached CFG");
      }
    }
    assert(DT.dominates(Spawned,BB));
  }

  // Hoist extractvalue instructions to avoid unncessarily passing aggregates
  //FIXME(victory): Iterating over an unordered set yields non-determinism.
  for (BasicBlock *BB : SpawnedBBs) {
    for (BasicBlock::iterator II = BB->begin(), E = BB->end(); II != E;) {
      Instruction& I = *II++;
      if (auto EVI = dyn_cast<ExtractValueInst>(&I)) {
        if (auto Aggregate = dyn_cast<Instruction>(EVI->getAggregateOperand())) {
          if (!SpawnedBBs.count(Aggregate->getParent())) {
            DEBUG(dbgs() << "Hoisting instruction before detach: " << I << '\n');
            I.moveBefore(&DI);
          }
        }
      }
    }
  }

  // Get the inputs that will be the arguments for the detached CFG.
  //FIXME(victory): Avoid Tapir's findInputsOutputs(), which populates Inputs
  // in a non-deterministic order by iterating through the unordered set Blocks.
  SetVector<Value *> Inputs, Outputs;
  findInputsOutputs(SpawnedBBs, Inputs, Outputs, &ExitBBs);
  assert(Outputs.empty() &&
         "All results from detached CFG should be passed by memory already.");
  DEBUG({
    for (Value *Input : Inputs)
      dbgs() << "Detached CFG input: " << *Input << '\n';
  });

  // In the current ABI, task functions must take the timestamp as their first
  // parameter, even if they don't need it.
  // There are two cases here:
  // 1.) DI.getTimestamp() is a parameter value or a value computed locally by
  //     some instruction, in which case we want to actually put that value at
  //     the start of Params, and make sure not to redundantly pass it as an
  //     additional argument.
  // 2.) DI.getTimestamp() is a constant, in which case there's no need to
  //     use any actual parameter value, and it will not show up in Inputs,
  //     so all we need to do is have an unused dummy timestamp parameter.
  Value *TimestampParam;
  Instruction *TimestampDummy = nullptr;
  if (isa<Constant>(DI.getTimestamp())) {
    TimestampDummy = createDummyValue(DI.getTimestamp()->getType(), "ts", &DI);
    TimestampParam = TimestampDummy;
  } else {
    TimestampParam = DI.getTimestamp();
    assert(isa<Argument>(TimestampParam) || isa<Instruction>(TimestampParam));
    Inputs.remove(TimestampParam);
  }
  SetVector<Value *> Params;
  Params.insert(TimestampParam);

  const DataLayout &DL = F->getParent()->getDataLayout();

  const unsigned NumInputs = Inputs.size();
  const unsigned InputBytes =
      NumInputs ? DL.getTypeAllocSize(createRegArgsStructTy(Inputs)) : 0;
  DEBUG(dbgs() << NumInputs << " inputs add up to "
               << InputBytes << " bytes.\n");

  // Prep to pack the inputs.
  Args.clear();
  ValueToValueMapTy OldValueMap;
  auto SubstituteInput = [&](Value *Input, Value *NewInput) {
    auto UI = Input->use_begin(), E = Input->use_end();
    while (UI != E) {
      Use &U = *(UI++); // Carefully crafted to avoid iterator invalidation
      auto *Usr = cast<Instruction>(U.getUser());
      if (SpawnedBBs.count(Usr->getParent()))
        U.set(NewInput);
    }
    OldValueMap[NewInput] = Input;
  };

  Value* MemArgsPtr = nullptr;
  if (!Inputs.empty()) {
    DEBUG(dbgs() << "Packing task live-ins\n");

    // Sort inputs by decreasing length to minimize unaligned accesses
    auto isLargerThan = [&DL](Value* v, Value* other) {
      return DL.getTypeSizeInBits(v->getType()) >
             DL.getTypeSizeInBits(other->getType());
    };
    auto IVec = Inputs.takeVector();
    std::stable_sort(IVec.begin(), IVec.end(), isLargerThan);
    for (Value* Input : IVec) Inputs.insert(Input);

    // Split inputs into passed-through-registers and passed-through-memory
    using ValueVector = SmallVector<Value *, 8>;
    ValueVector RegInputs(Inputs.begin(), Inputs.end());
    ValueVector MemInputs;
    StructType* RegArgsStructTy = createRegArgsStructTy(RegInputs);
    unsigned RegArgsBytes = DL.getTypeStoreSize(RegArgsStructTy);
    if (RegArgsBytes <= SwarmRegistersTransferred * 8) {
      DEBUG(dbgs() << "Passing all task live-ins through registers.\n");
    } else {
      DEBUG(dbgs() << "Passing some task live-ins through memory.\n");

      if (getDetachKind(&DI) == DetachKind::BalancedSpawner && !DisableEnvSharing) {
        F->getContext().diagnose(DiagnosticInfoUnsupported(*F,
                "Spawners are expected to use a shared closure if necessary",
                DI.getDebugLoc()));
        llvm_unreachable("Failure of shared closures to eliminate heap allocation");
      }
      if (getDetachKind(&DI) == DetachKind::BalancedIter && !DisableEnvSharing) {
        F->getContext().diagnose(DiagnosticInfoUnsupported(*F,
                "Iteration tasks should use a shared closure if necessary",
                DI.getDebugLoc()));
        llvm_unreachable("Failure of shared closures to eliminate heap allocation");
      }

      do {
        MemInputs.push_back(RegInputs.back());
        RegInputs.pop_back();
        RegArgsStructTy = createRegArgsStructTy(RegInputs);
        RegArgsBytes = DL.getTypeStoreSize(RegArgsStructTy);
        // Leave last reg arg for MemArgs pointer
      } while (RegArgsBytes > (SwarmRegistersTransferred - 1) * 8);
      std::reverse(MemInputs.begin(), MemInputs.end());
    }

    // Handle pass-through-register inputs
    IRBuilder<> OuterBuilder(&DI);
    IRBuilder<> InnerBuilder(Spawned->getFirstNonPHI());
    InnerBuilder.SetCurrentDebugLocation(DI.getDebugLoc());

    // Pad struct to a whole number of 64-bit/8-byte words.
    unsigned NumRegs = (RegArgsBytes + 7) / 8;
    unsigned PaddingBytes = NumRegs * 8 - RegArgsBytes;
    SmallVector<Type *, 8> RegInputTypes;
    for (Value *V : RegInputs)
      RegInputTypes.push_back(V->getType());
    RegInputTypes.push_back(
        ArrayType::get(Type::getInt8Ty(F->getContext()), PaddingBytes));
    RegArgsStructTy =
        StructType::create(RegInputTypes, StringRef(), /*isPacked*/ true);
    assert(DL.getTypeStoreSize(RegArgsStructTy) == NumRegs * 8);
    assert(DL.getTypeAllocSize(RegArgsStructTy) == NumRegs * 8);
    ConstantInt *AllocaSize =
        ConstantInt::get(Type::getInt64Ty(F->getContext()), NumRegs * 8);

    // 1. Store inputs to stack-allocated struct (will be optimized out)
    SDetachInst *EnclosingTask = getEnclosingTask(&DI, DT);
    BasicBlock *AllocaBlock =
        EnclosingTask ? EnclosingTask->getDetached() : &F->getEntryBlock();
    auto *OuterArgsPtr =
        IRBuilder<>(AllocaBlock->getFirstNonPHI())
            .CreateAlloca(RegArgsStructTy, nullptr, "outer_args_ptr");
    OuterBuilder.CreateLifetimeStart(OuterArgsPtr, AllocaSize);
    unsigned i = 0;
    for (Value *Input : RegInputs) {
      Value *Pointer =
          OuterBuilder.CreateConstGEP2_32(RegArgsStructTy, OuterArgsPtr, 0, i);
      OuterBuilder.CreateStore(Input, Pointer);
      ++i;
    }

    // 2. Cast args struct into/out of registers (emulating an union)
    ArrayType *RegsArrayTy =
        ArrayType::get(Type::getInt64Ty(F->getContext()), NumRegs);

    Value *OuterRegsPtr = OuterBuilder.CreateBitCast(
        OuterArgsPtr, PointerType::getUnqual(RegsArrayTy), "outer_regs_ptr");
    Value *InnerRegsPtr =
        InnerBuilder.CreateAlloca(RegsArrayTy, nullptr, "inner_regs_ptr");
    InnerBuilder.CreateLifetimeStart(InnerRegsPtr, AllocaSize);

    LoadInst *OuterRegs = OuterBuilder.CreateLoad(OuterRegsPtr, "outer_regs");
    for (i = 0; i < NumRegs; i++) {
      Value *RegArg = OuterBuilder.CreateExtractValue(OuterRegs, i);
      Params.insert(RegArg);
      Args.push_back(RegArg);

      Value *Pointer =
          InnerBuilder.CreateConstGEP2_32(RegsArrayTy, InnerRegsPtr, 0, i);
      InnerBuilder.CreateStore(RegArg, Pointer);
    }
    OuterBuilder.CreateLifetimeEnd(OuterArgsPtr, AllocaSize);

    // 3. Load inputs from stack-allocated struct (also optimized out)
    Value *InnerArgsPtr = InnerBuilder.CreateBitCast(
        InnerRegsPtr, PointerType::getUnqual(RegArgsStructTy),
        "inner_args_ptr");
    LoadInst *InnerArgs = InnerBuilder.CreateLoad(InnerArgsPtr, "inner_args");
    i = 0;
    for (Value *Input : RegInputs) {
      Value *UnpackedParam = InnerBuilder.CreateExtractValue(
          InnerArgs, i, Input->getName() + ".unpacked");
      SubstituteInput(Input, UnpackedParam);
      ++i;
    }
    InnerBuilder.CreateLifetimeEnd(InnerRegsPtr, AllocaSize);

    // Handle pass-through-memory
    if (!MemInputs.empty()) {
      MemArgsPtr = createClosure(MemInputs, &DI, "mem_args");
      LoadInst *ParamsToUnpack =
          InnerBuilder.CreateLoad(MemArgsPtr, "params_to_unpack");
      addSwarmMemArgsMetadata(ParamsToUnpack);
      uint64_t MallocAlignment = SwarmCacheLineSize;
      ParamsToUnpack->setAlignment(MallocAlignment);

      unsigned i = 0;
      for (Value *Input : MemInputs) {
        Value *UnpackedParam = InnerBuilder.CreateExtractValue(
            ParamsToUnpack, i, Input->getName() + ".unpacked");
        SubstituteInput(Input, UnpackedParam);
        ++i;
      }

      Params.insert(MemArgsPtr);
      Args.push_back(MemArgsPtr);
    }
  }

  // Conservatively remove attributes that may become invalid after outlining
  for (Argument &Arg : F->args()) {
    // We may now be capturing pointers even if we weren't before
    Arg.removeAttr(Attribute::NoCapture);
  }

  // Clone the detached CFG into a helper function.
  ValueToValueMapTy VMap;
  Function *Extracted = outline(Params, SpawnedBBs, Spawned, ".d2c", VMap);

  if (TimestampDummy) TimestampDummy->eraseFromParent();

  // Provide a useful failure message for strange parameter attributes:
  auto BadParamAttrs = getParamsWithBadAttribute(*Extracted);
  remarkParamsWithBadAttributes(*Extracted, Params, BadParamAttrs);

  assert(BadParamAttrs.empty() && "Outlined function has bad attribute");
  assert(Extracted->getReturnType()->isVoidTy()
         && "Outlined function has non-void return");
  assert(!Extracted->getAttributes().getRetAttributes().getNumAttributes()
         && "Outlined function has a return attribute");

  Extracted->addFnAttr("TaskFunc");
  Extracted->removeFnAttr(SwarmAttr::AssertSwarmified);

  Extracted->setName(
      Extracted->getName() + Twine(static_cast<int>(getDetachKind(&DI))) + "." +
      Twine(NumInputs) + "in" + Twine(InputBytes) + "bytes_" +
      (DI.hasHint() ? "hint_" :
       DI.hasSameHintCondition() ? "conditionalsamehint_" :
       DI.isSameHint() ? "samehint_" : "") +
      (MemArgsPtr ? "mem" : "reg") + "_runner");

  if (MemArgsPtr) {
    cast<Argument>(VMap[MemArgsPtr])->addAttr(Attribute::NoAlias);
    // Delay call to free() to the end of Extracted to reduce register pressure.
    Instruction *Free =
        CallInst::CreateFree(VMap[MemArgsPtr], getUniqueReturnInst(*Extracted));
    addSwarmMemArgsForceAliasMetadata(cast<CallInst>(Free));
  }

  // Move allocas in the newly cloned detached CFG to the entry block of the
  // helper.
  {
    // Collect reattach instructions.
    SmallVector<Instruction *, 4> ReattachPoints;
    //FIXME(victory): This loop does nothing, because the call to
    // populateDetachedCFG() already replaced reattaches with branches.
    // See: https://github.com/wsmoses/Tapir-LLVM/issues/65
    // Perhaps this isn't important enough to warrant an urgent fix,
    // but if Tapir ever implements a fix upstream, we should copy it here.
    //for (pred_iterator PI = pred_begin(Continue), PE = pred_end(Continue);
    //     PI != PE; ++PI) {
    //  BasicBlock *Pred = *PI;
    //  if (!isa<SReattachInst>(Pred->getTerminator())) continue;
    //  if (SpawnedBBs.count(Pred))
    //    ReattachPoints.push_back(cast<BasicBlock>(VMap[Pred])->getTerminator());
    //}

    // Move allocas in cloned detached block to entry of helper function.
    MoveStaticAllocasInBlock(&Extracted->getEntryBlock(),
                             cast<BasicBlock>(VMap[Spawned]),
                             ReattachPoints);
  }

  // ExitBBs might not have been be dominated by the detach,
  // so they may still be reachable in the original function.
  // Restore their direct usage of the original input values.
  //FIXME(victory): Iterating over an unordered set yields non-determinism.
  for (BasicBlock *ExitBB : ExitBBs)
    for (Instruction &I : *ExitBB)
      RemapInstruction(&I, OldValueMap,
                       RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);

  // Warn if passing pointers to the stack.
  for (Value* Arg : Args) {
    if (!Arg->getType()->isPointerTy())
      continue;
    if (auto Alloca = dyn_cast<AllocaInst>(GetUnderlyingObject(Arg, DL))) {
      assert(Alloca->getFunction() == F);
      std::string Msg;
      raw_string_ostream OS(Msg);
      OS << "Pointer to local stack passed to child task: "
         << *(Arg->getType()) << " %" << Arg->getName()
         << " points to underlying stack allocation " << *Alloca;
      F->getContext().diagnose(DiagnosticInfoOptimizationFailure(
              *F, DI.getDebugLoc(), OS.str()));
      assert(!F->hasFnAttribute(SwarmFlag::Parallelized) &&
             "Autoparallelized function passing stack pointers to child task");
    }
  }

  swarm_abi::optimizeTaskFunction(Extracted);

  DEBUG(assertVerifyFunction(*Extracted, "Extracted task function"));

  return Extracted;
}


void llvm::swarm_abi::optimizeTaskFunction(Function *TaskFn) {
  Impl().optimizeTaskFunction(TaskFn);
}


CallInst *llvm::swarm_abi::createGetTimestampInst(Instruction *InsertBefore,
                                                  bool isSuper) {
  CallInst *CI = Impl().createGetTimestampInst(InsertBefore, isSuper);

  // By treating these as an unknown memory read, we allow them to be moved
  // in limited ways, while ensuring they are not reordered with respect to
  // detaches, reattaches, deepens, or undeepens.
  CI->setOnlyReadsMemory();
  assert(CI->mayReadFromMemory());
  assert(!CI->doesNotAccessMemory() && !CI->onlyAccessesArgMemory());

  assert(isGetTimestampInst(CI) &&
         "isGetTimestampInst() is out of sync with createGetTimestampInst()!");

  return CI;
}

CallInst *llvm::swarm_abi::createDeepenInst(Instruction *InsertBefore) {
  CallInst *CI = Impl().createDeepenInst(InsertBefore);

  // By treating this as an unknown memory write, we ensure this is not
  // reordered with respect to detaches, reattaches, deepens, undeepens,
  // or GetTimestamps.
  assert(CI->mayWriteToMemory());
  assert(!CI->doesNotAccessMemory() && !CI->onlyAccessesArgMemory());
  assert(!CI->onlyReadsMemory());

  assert(isDeepenInst(CI) &&
         "isDeepenInst() is out of sync with createDeepenInst()!");

  return CI;
}

CallInst *llvm::swarm_abi::createUndeepenInst(Instruction *InsertBefore) {
  CallInst *CI = Impl().createUndeepenInst(InsertBefore);

  // By treating this as an unknown memory write, we ensure this is not
  // reordered with respect to detaches, reattaches, deepens, undeepens,
  // or GetTimestamps.
  assert(CI->mayWriteToMemory());
  assert(!CI->doesNotAccessMemory() && !CI->onlyAccessesArgMemory());
  assert(!CI->onlyReadsMemory());

  assert(isUndeepenInst(CI) &&
         "isUndeepenInst() is out of sync with createUndeepenInst()!");
  return CI;
}

CallInst *llvm::swarm_abi::createHeartbeatInst(Instruction *InsertBefore) {
  CallInst *CI = Impl().createHeartbeatInst(InsertBefore);
  assert(isHeartbeatInst(CI) &&
         "isHeartbeatInst() is out of sync with createHeartbeatInst()!");
  return CI;
}


static CallInst *createSwarmEnqueue(Function *TaskFn,
                                    ArrayRef<Value *> Args,
                                    SDetachInst *DI) {
  CallInst *Call = Impl().createSwarmEnqueue(TaskFn, Args, DI);

  // By treating this as an unknown memory write, we ensure this is not
  // reordered with respect to detaches, reattaches, deepens, undeepens,
  // or GetTimestamps.
  assert(Call->mayWriteToMemory());
  assert(!Call->doesNotAccessMemory() && !Call->onlyAccessesArgMemory());
  assert(!Call->onlyReadsMemory());

  DEBUG(dbgs() << "Created enqueue instruction: " << *Call << '\n');
  assert(swarm_abi::isSwarmEnqueueInst(Call));
  assert(swarm_abi::getEnqueueFunction(Call) == TaskFn);

  return Call;
}


Function *llvm::swarm_abi::lowerDetach(SDetachInst &DI, DominatorTree &DT,
                                       const TargetTransformInfo &TTI,
                                       LoopInfo *LI) {
  BasicBlock* DetBB = DI.getParent();

  //DEBUG(dbgs() << "\nDetach Block:" << *DetBB);

  if (DisableHints && DI.hasHint()) {
    DEBUG(dbgs() << "Discarding hint " << *DI.getHint()
                 << " because of -swarm-disableallhints.\n");
    DI.setHint(nullptr);
  }

  SmallVector<Value *, 8> Args;
  Function *Extracted = extractDetachedCFG(DI, DT, TTI, Args);

  if (!Extracted) {
    DEBUG(dbgs() << "Replacing detach with branch to detached region.\n");
    // This a a dangerous action made on the assumption that the program will
    // crash anyway if this code runs.
    // Print out a message so we know if this transformed code runs.
    createPrintString("\nReached spawn point that will crash.\n",
                      "swarmabi_detach_without_reattach",
                      &DI);

    // Now replace the detach.
    BasicBlock *Continue = DI.getContinue();
    ReplaceInstWithInst(&DI, BranchInst::Create(DI.getDetached()));

    if (DT.dominates(DetBB, Continue)) {
      DEBUG(dbgs() << "Deleting now-unreachable continuation.\n");
      eraseDominatorSubtree(Continue, DT, LI);
      // If the continuation can reach some non-dominated blocks,
      // the erasure may change the dominators for those blocks.
      // TODO(victory): Do an incremental update of the dominator tree instead
      // of throwing it out and recalculating it.
      DT.recalculate(*DetBB->getParent());
    } else {
      DEBUG(dbgs() << "Continuation is still reachable by some other path.\n");
      DT.deleteEdge(DetBB, Continue);
    }
    return nullptr;
  }

  DEBUG(dbgs() << "Extracted Function " << Extracted->getName() << '\n');
  //DEBUG(dbgs() << " with entry block:\n " << Extracted->getEntryBlock() << '\n');

  createSwarmEnqueue(Extracted, Args, &DI);

  // FIXME(mcj) This assert doesn't catch empty tasks where Extracted simply
  // calls an empty function, because inlining hasn't yet been run on Extracted.
  if (none_of(instructions(Extracted),
              [] (const Instruction &I) {
                return I.mayHaveSideEffects() && !isDequeueInst(&I);
              })) {
    // Unfortunately the Swarm passes are still given read-only loops, e.g.
    // resulting from assertions that are optimized away
    // https://github.mit.edu/swarm/benchmarks/blob/4ca734f43f98f5e030984f24a72a4804e636220a/speccpu2006/450.soplex/factor.cc#L1727-L1735
    Function *F = DI.getParent()->getParent();
    F->getContext().diagnose(DiagnosticInfoOptimizationFailure(
        *F, DI.getDebugLoc(), "Extracted task function has no side effects"));
  }

  eraseDetach(&DI, DT, LI);

  //DEBUG(dbgs() << "\nModified Detach Block:" << *DetBB);

  DEBUG(assertVerifyFunction(*DetBB->getParent(), "After lowering detach", &DT));

  return Extracted;
}


bool llvm::swarm_abi::isSwarmEnqueueInst(const Instruction *I) {
  if (auto *CI = dyn_cast<CallInst>(I))
    return Impl().isSwarmEnqueueInst(CI);
  return false;
}

const Function *llvm::swarm_abi::getEnqueueFunction(const Instruction *I) {
  if (!isSwarmEnqueueInst(I)) return nullptr;
  const CallInst *EnqueueI = cast<CallInst>(I);
  const Constant *FunctionPtr = Impl().getEnqueueFunction(EnqueueI);
  if (auto *TaskFn = dyn_cast<Function>(FunctionPtr)) {
    return TaskFn;
  } else {
    auto *FunctionPtrExpr = cast<ConstantExpr>(FunctionPtr);
    assert(FunctionPtrExpr->isCast());
    return cast<Function>(FunctionPtrExpr->getOperand(0));
  }
}

bool llvm::swarm_abi::isGetTimestampInst(const Instruction *I) {
  if (auto CI = dyn_cast<CallInst>(I))
    return Impl().isGetTimestampInst(CI);
  return false;
}

bool llvm::swarm_abi::isDeepenInst(const Instruction *I) {
  if (auto CI = dyn_cast<CallInst>(I))
    return Impl().isDeepenInst(CI);
  return false;
}

bool llvm::swarm_abi::isUndeepenInst(const Instruction *I) {
  if (auto CI = dyn_cast<CallInst>(I))
    return Impl().isUndeepenInst(CI);
  return false;
}

bool llvm::swarm_abi::isDequeueInst(const Instruction *I) {
  if (auto CI = dyn_cast<CallInst>(I))
    return Impl().isDequeueInst(CI);
  return false;
}

bool llvm::swarm_abi::isHeartbeatInst(const Instruction *I) {
  if (auto CI = dyn_cast<CallInst>(I))
    return Impl().isHeartbeatInst(CI);
  return false;
}
