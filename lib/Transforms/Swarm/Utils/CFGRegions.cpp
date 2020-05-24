//===-- CFGRegions.cpp - Manipulation of CFG subgraphs ---------*- C++ -*--===//
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
// Utility methods for traversing and finding subgraph structure within a CFG,
// for forming such subgraphs into single-entry, single-valid-exit regions,
// for manipulating the communication of live-in values into such regions, and
// for restructuring such regions by separating them into new functions (a.k.a.
// "outlining", roughly the reverse of function inlining).
//
//===----------------------------------------------------------------------===//


#include "CFGRegions.h"

#include "Flags.h"
#include "Misc.h"
#include "Tasks.h"

#include "llvm/Analysis/CostModel.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationDiagnosticInfo.h"
#include "llvm/Analysis/SwarmAA.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Swarm.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "regions"

cl::opt<bool> llvm::DisableEnvSharing(
        "swarm-disablesharedenv", cl::init(false),
        cl::desc("Disable loop environment sharing"));
static cl::opt<bool> DisableShrinkInputs(
        "swarm-disableshrinkinputs", cl::init(false),
        cl::desc("Disable live-in sinking"));


/// Return true if the specified value is defined outside of Blocks.
/// These values must be passed in as live-ins to Blocks.
/// Mostly copied from Tapir's definedInCaller().
template <typename SetTy>
static bool definedOutside(Value *V, const SetTy &Blocks) {
  if (isa<Argument>(V)) return true;
  if (auto *I = dyn_cast<Instruction>(V))
    if (!Blocks.count(I->getParent()))
      return true;
  return false;
}


void llvm::findInputsNoOutputs(const SmallSetVector<BasicBlock *, 8> &Blocks,
                               SetVector<Value *> &Inputs) {
  // This code is largely copied from Tapir's findInputsOutputs(),
  // but we use ordered containers that avoid non-determinism.
  for (BasicBlock *BB : Blocks) {
    for (Instruction &II : *BB) {
      // If a used value is defined outside the region, it's a live-in.
      for (User::op_iterator OI = II.op_begin(), OE = II.op_end(); OI != OE;
           ++OI) {
        if (isa<DeepenInst>(*OI)) continue;
        if (definedOutside(*OI, Blocks))
          Inputs.insert(*OI);
      }

      // If an instruction is used outside the region, it's an live-out.
      for (User *U : II.users())
        if (!Blocks.count(cast<Instruction>(U)->getParent())) {
          dbgs() << "Found live-out: " << *U << '\n';
          llvm_unreachable("found unexpected live-out value");
        }
    }
  }
}


Function *llvm::outline(const SetVector<Value *> &Params,
                        const SmallPtrSetImpl<BasicBlock *> &Blocks,
                        BasicBlock *Entry,
                        const StringRef Suffix,
                        ValueToValueMapTy &VMap) {
  assert(Blocks.count(Entry));
  Function *OrigFunc = Entry->getParent();
  assert(all_of(Blocks, [OrigFunc](const BasicBlock *BB) {
    return BB->getParent() == OrigFunc; }));

  // Put Entry first, but otherwise preserve ordering of Blocks from OrigFunc.
  std::vector<BasicBlock *> BlocksVector;
  BlocksVector.push_back(Entry);
  for (BasicBlock &BB : *OrigFunc)
    if (&BB != Entry && Blocks.count(&BB))
      BlocksVector.push_back(&BB);

  // Conservatively remove attributes that may become invalid after outlining
  for (Argument &Arg : OrigFunc->args()) {
    //We may now be capturing pointers even if we weren't before
    Arg.removeAttr(Attribute::NoCapture);
  }

  // Tapir's CreateHelper() utility needs to know what the predecessor of
  // the entry point is, what the unique successor block outside the region is,
  // and which exit-handling blocks might have other incoming edges requiring
  // special handling.
  //victory: The caller likely already knows what these blocks are, but I like
  // traversing the region and checking all incoming and outgoing edges anyway
  // to assert some assumptions, and then if we're already doing all this work
  // for these assertions, recomputing this information is "free" and keeps
  // the interface to this utility clean.
  BasicBlock *PreEntry = nullptr;
  BasicBlock *Exit = nullptr;
  SmallPtrSet<BasicBlock *, 8> EHBlocks;
  SmallPtrSet<const BasicBlock *, 8> VisitedBlocks;
  std::function<bool(BasicBlock *)> DFSHelper = [&](BasicBlock * BB) {
    if (!VisitedBlocks.insert(BB).second)
      return !EHBlocks.count(BB);

    DEBUG(dbgs() << "Outlining block " << BB->getName() << "\n");

    assert(BB != &OrigFunc->getEntryBlock() &&
           "We don't support outlining the entry block");
    assert(!pred_empty(BB) && "Found trivially dead block");

    bool CanReachExit = false;
    for (BasicBlock *Succ : successors(BB)) {
      if (Blocks.count(Succ)) {
        if (Succ != BB)
          CanReachExit |= DFSHelper(Succ);
      } else {
        assert(!Exit || Exit == Succ &&
               "Blocks have multiple outside successors");
        Exit = Succ;
        DEBUG(dbgs() << "Found exit edge from " << BB->getName()
                     << "\n  to " << Exit->getName() << '\n');
        CanReachExit = true;
      }
    }
    if (!CanReachExit) {
      // Besides returns, the only terminators with no successors are
      // unreachables and certain exception-handling terminators.
      // Since BB inevitably reaches one of these dead ends,
      // it is considered error-handling.
      DEBUG(dbgs() << "Found error-handling block " << BB->getName() << '\n');
      EHBlocks.insert(BB);
    }

    for (BasicBlock *Pred : predecessors(BB)) {
      if (!Blocks.count(Pred)) {
        if (BB == Entry) {
          DEBUG(dbgs() << "Entry edge to " << Entry->getName()
                       << "\n  is from " << Pred->getName() << '\n');
          assert(!PreEntry && "Blocks have multiple incoming edges");
          PreEntry = Pred;
        } else {
          DEBUG(dbgs() << "Ignoring incoming edge to non-entry block "
                       << BB->getName() << "\n  from " << Pred->getName() << '\n');
          assert(EHBlocks.count(BB) &&
                 "Found incoming edge to non-entry and non-error-handling block");
        }
      }
    }

    return CanReachExit;
  };
  DFSHelper(Entry);
  assert(PreEntry && "Failed to find entry edge");
  assert(Exit && "Region has no exit");

  // Outlining can separate detaches from preceeding deepens outside of Blocks.
  // After this separation, the detaches have null domain.
  for (BasicBlock *BB : BlocksVector)
    if (auto *Domain = getDomain(dyn_cast<SDetachInst>(BB->getTerminator())))
      if (!Blocks.count(Domain->getParent()))
        VMap[Domain] = ConstantTokenNone::get(OrigFunc->getContext());

  // If we have debug info, add mapping for the metadata nodes that should not
  // be cloned by CloneFunctionInto.
  if (DISubprogram *SP = OrigFunc->getSubprogram()) {
    auto &MD = VMap.MD();
    MD[SP->getUnit()].reset(SP->getUnit());
    MD[SP->getType()].reset(SP->getType());
    MD[SP->getFile()].reset(SP->getFile());
  }

  SmallVector<ReturnInst *, 4> Returns;
  Function *NewFunc = CreateHelper(Params, {}, BlocksVector,
                                   Entry, PreEntry, Exit,
                                   VMap, OrigFunc->getParent(),
                                   OrigFunc->getSubprogram() != nullptr,
                                   Returns, Suffix,
                                   &EHBlocks);
  assert(Returns.empty() && "Returns cloned when outlining.");

  // Move returning block to the end of NewFunc.
  auto *RetBlock = cast<BasicBlock>(VMap[Exit]);
  assert(cast<ReturnInst>(RetBlock->getTerminator())->getReturnValue() == nullptr);
  RetBlock->removeFromParent();
  RetBlock->insertInto(NewFunc);
  assert(RetBlock->getNextNode() == nullptr);
  assert(RetBlock->getPrevNode() == VMap[BlocksVector.back()]);

  assert(VMap[PreEntry] == &NewFunc->getEntryBlock() &&
         "VMap[PreEntry] should be the first block of NewFunc.");
  assert(NewFunc->getEntryBlock().getNextNode() == VMap[Entry] &&
         "The copy of Entry should be the second block of NewFunc.");

  // Make outlined function static (module-local)
  NewFunc->setLinkage(Function::InternalLinkage);

  // Avoid inheriting noinline from OrigFunc
  // (this produces better code due to nested outlines being reinlined)
  NewFunc->removeFnAttr(Attribute::NoInline);

  // Propagate aligment to arguments
  AddAlignmentAssumptions(OrigFunc, Params, VMap,
                          PreEntry->getTerminator(), nullptr, nullptr);

  return NewFunc;
}

void llvm::shrinkInputs(const SmallVectorImpl<BasicBlock *>& Blocks,
                        const SmallPtrSet<Value *, 4>& Blacklist,
                        const TargetTransformInfo &TTI,
                        OptimizationRemarkEmitter* ORE) {
  if (DisableShrinkInputs) return;

  assert(Blocks.size());
  BasicBlock *Entry = Blocks[0];
  const SmallSetVector<BasicBlock *, 8> BBSet(Blocks.begin(), Blocks.end());
  SetVector<Value *> Inputs;
  findInputsNoOutputs(BBSet, Inputs);

  // Try to reduce inputs by sinking them past the detach.
  // TODO: The general solution for minimizing live-ins seems like a variant
  // of set cover, where we want to find the smallest combination of values
  // that allow producing all live-ins (within a certain cost). That is
  // complex but may be worth doing at some point.
  //
  // For now, do two things:
  // 1. If an input is directly computable from other inputs, sink it
  // 2. If multiple inputs can be computed from a single value, and this
  //    value is smaller than the sum of the inputs, sink the inputs them and
  //    make the value the single input.
  auto doNotSink = [&TTI](Instruction *I) {
    // TODO: Loads to read-only data should be sinkable
    return !I || I->mayReadOrWriteMemory() || isa<PHINode>(I) ||
           getInstrCost(I, &TTI) > 2;
  };

  SetVector<Instruction *> ToSink;
  SetVector<Value *> PossibleNewInputs;
  for (Value *V : Inputs) {
    if (Blacklist.count(V))
      continue;
    Instruction *I = dyn_cast<Instruction>(V);
    if (doNotSink(I))
      continue;
    bool Subsumed = true;
    for (auto OI = I->op_begin(); OI != I->op_end(); OI++) {
      Value *Operand = *OI;
      if (!isa<Constant>(Operand) && !Inputs.count(Operand)) {
        Subsumed = false;
        PossibleNewInputs.insert(Operand);
        break;
      }
    }
    if (Subsumed)
      ToSink.insert(I);
  }

  const DataLayout &DL = Entry->getParent()->getParent()->getDataLayout();
  SetVector<Value *> NewInputs;
  for (Value *V : PossibleNewInputs) {
    SmallVector<Instruction *, 8> SubsumedInputs;
    for (User *U : V->users()) {
      if (Blacklist.count(U) || !Inputs.count(U))
        continue;
      Instruction *I = dyn_cast<Instruction>(U);
      if (doNotSink(I))
        continue;
      bool Subsumed = true;
      for (auto OI = I->op_begin(); OI != I->op_end(); OI++) {
        Value *Operand = *OI;
        if (!isa<Constant>(Operand) && !Inputs.count(Operand) && Operand != V) {
          Subsumed = false;
          break;
        }
      }
      if (Subsumed) {
        assert(!ToSink.count(I));
        SubsumedInputs.push_back(I);
      }
    }
    size_t subsumedBytes = 0;
    for (auto *S : SubsumedInputs)
      subsumedBytes += DL.getTypeStoreSize(S->getType());

    if (subsumedBytes > DL.getTypeStoreSize(V->getType())) {
      NewInputs.insert(V);
      for (auto *S : SubsumedInputs)
        ToSink.insert(S);
    }
  }

  for (Value *V : Inputs) {
    Instruction *I = dyn_cast<Instruction>(V);
    if (!I || !ToSink.count(I))
      NewInputs.insert(V);
  }

  if (ORE && NewInputs.size() != Inputs.size()) {
    ORE->emit(OptimizationRemark("group-task-inputs", "shrinkInputs",
                                 Entry->getFirstNonPHI()->getDebugLoc(), Entry)
              << "reduced inputs from "
              << ore::NV("OrigInputs", (unsigned)Inputs.size()) << " to "
              << ore::NV("NewInputs", (unsigned)NewInputs.size()));
  }

  ValueToValueMapTy OldValueMap;

  // Sunk instructions may themselves depend in a sunk instruction, so
  // insertion order matters. Proceed in multiple rounds, deferring values
  // that depend on a not-yet-sunk instruction.
  Instruction *Last = nullptr;
  while (!ToSink.empty()) {
    SetVector<Instruction *> DeferredSinks;
    for (Instruction *I : ToSink) {
      bool Defer = false;
      for (auto OI = I->op_begin(); OI != I->op_end(); OI++) {
        Instruction *I = dyn_cast<Instruction>(*OI);
        if (I && ToSink.count(I)) {
          Defer = true;
          break;
        }
      }
      if (Defer) {
        DeferredSinks.insert(I);
        continue;
      }

      Instruction *C = I->clone();
      if (!Last) {
        C->insertBefore(&*Entry->getFirstInsertionPt());
      } else {
        C->insertAfter(Last);
      }
      Last = C;
      OldValueMap[I] = C;
    }
    ToSink = DeferredSinks;
  }
  remapInstructionsInBlocks(Blocks, OldValueMap);

  // Verify we now have only the live-ins we expected
  SetVector<Value *> VerifyInputs;
  findInputsNoOutputs(BBSet, VerifyInputs);

  assert(VerifyInputs.size() == NewInputs.size());
  VerifyInputs.set_subtract(NewInputs);
  assert(VerifyInputs.empty());
}


StructType *llvm::getClosureType(const Value *Closure) {
  Type *T = cast<PointerType>(Closure->getType())->getElementType();
  return cast<StructType>(T);
}


Instruction *llvm::createClosure(
        ArrayRef<Value *> Captures,
        Instruction *AllocateAndPackBefore,
        StringRef Name,
        unsigned FieldsToSkip) {
  assert(!Captures.empty());
  assert(FieldsToSkip <= Captures.size());
  const DataLayout &DL = AllocateAndPackBefore->getModule()->getDataLayout();
  LLVMContext &Context = AllocateAndPackBefore->getContext();

  // The closure holds all the captured values
  SmallVector<Type *, 8> ClosureFieldTypes;
  for (Value *Capture : Captures)
    ClosureFieldTypes.push_back(Capture->getType());

  auto *ST = StructType::create(ClosureFieldTypes);
  DEBUG(dbgs() << "Closure struct has type: " << *ST << '\n');
  IntegerType *Int32Ty = Type::getInt32Ty(Context);
  uint64_t ClosureSz = DL.getTypeAllocSize(ST);
  Value *AllocSize = ConstantInt::get(Int32Ty, ClosureSz);
  Value *Closure = CallInst::CreateMalloc(AllocateAndPackBefore,
                                          Int32Ty, ST, AllocSize,
                                          nullptr, nullptr,
                                          Name);
  assert(getClosureType(Closure) == ST);

  IRBuilder<> B(AllocateAndPackBefore);
  for (unsigned i = FieldsToSkip; i < Captures.size(); i++) {
    // Store captured values to closure struct
    Value *Pointer = B.CreateConstInBoundsGEP2_32(ST, Closure, 0, i);
    StoreInst *Store = B.CreateStore(Captures[i], Pointer);
    Store->setMetadata(SwarmFlag::Closure, MDNode::get(Context, {}));

    //victory: I'm nervious that LLVM may assign stronger semantics than
    // we want for alignments greater than the size of the type,
    // and I'm not sure there are any benefits to using alignment greater
    // than the natural alignment.
    //// Compute alignment from offset
    //unsigned Offset = DL.getStructLayout(ST)->getElementOffset(i);
    //unsigned Alignment = 1;
    //uint64_t MallocAlignment = SwarmCacheLineSize;
    //while (!(Offset & 1) && Alignment < MallocAlignment) {
    //  Offset >>= 1;
    //  Alignment <<= 1;
    //}
    //Store->setAlignment(Alignment);
    (void)Store;
  }
  return cast<Instruction>(Closure);
}


void llvm::unpackClosure(
        Value *Closure,
        ArrayRef<Value *> Captures,
        std::function<Instruction *(Instruction *)> UnpackBeforeForUser,
        unsigned FieldsToSkip) {
  assert(!Captures.empty());
  assert(FieldsToSkip <= Captures.size());

  StructType *ClosureType = getClosureType(Closure);
  assert(isa<StructType>(ClosureType));
  for (unsigned Capture = FieldsToSkip; Capture < Captures.size(); Capture++) {
    Value *Original = Captures[Capture];
    DenseMap<Instruction *, SmallVector<Use *, 8>> UnpackBeforeMap;
    for (Use &U : Original->uses()) {
      Instruction *Usr = cast<Instruction>(U.getUser());
      Instruction *UsrLoc = Usr;
      if (auto *PN = dyn_cast<PHINode>(Usr))
        UsrLoc = PN->getIncomingBlock(U)->getTerminator();
      if (Instruction *UnpackBefore = UnpackBeforeForUser(UsrLoc)) {
        UnpackBeforeMap[UnpackBefore].push_back(&U);
      }
    }
    for (std::pair<Instruction *, SmallVector<Use *, 8>> &Pair : UnpackBeforeMap) {
      IRBuilder<> Builder(Pair.first);
      // We place this GEP as late as possible, together with the load,
      // to reduce register pressure and closure bloat from carrying the
      // value a greater distance.
      Value *UnpackGEP =
          Builder.CreateConstInBoundsGEP2_32(ClosureType, Closure, 0, Capture);
      LoadInst *Unpacked =
          Builder.CreateLoad(UnpackGEP, Original->getName() + ".unpack");
      Unpacked->setMetadata(SwarmFlag::Closure,
                            MDNode::get(Unpacked->getContext(), {}));
      addSwarmMemArgsMetadata(Unpacked);
      for (Use *U : Pair.second)
        U->set(Unpacked);
    }
  }
}


SetVector<Value *> llvm::getMemArgs(const SetVector<Value *> &Args,
                                    const DataLayout &DL,
                                    const Value *TimeStamp,
                                    uint32_t FieldsToSkip) {
    SetVector<Value *> Result;
    if (Args.empty()) return Result;

    auto argBytes = [TimeStamp, &DL](Value *V) -> size_t {
      // TimeStamp doesn't use an arg register
      return (V == TimeStamp) ? 0ul : DL.getTypeStoreSize(V->getType());
    };

    size_t BytesLeft = (SwarmRegistersTransferred - 1) * 8;
    uint32_t i = 0;
    for (; i < Args.size(); i++) {
      size_t Bytes = argBytes(Args[i]);
      if (Bytes > BytesLeft) {
        // See if we can fit the remaining arg in the MemArgs pointer
        for (uint32_t j = i + 1; j < Args.size(); j++)
          Bytes += argBytes(Args[j]);
        if (Bytes <= BytesLeft + 8)
          return Result;  // we lucked out
        else
          break;  // doesn't fit, cut it here
      } else {
        BytesLeft -= Bytes;
      }
    }

    // If we can't even fit skipped fields, return nothing in contempt
    if (i < FieldsToSkip)
      return Result;

    for (; i < Args.size(); i++) {
      if (Args[i] == TimeStamp)
        continue; // doesn't use an arg register
      Result.insert(Args[i]);
    }
    return Result;
}

void llvm::eraseDominatorSubtree(BasicBlock *Root,
                                 DominatorTree &DT,
                                 LoopInfo *LI) {
  assert(Root);

  // This code bears some similarities to LoopDeletion's deleteDeadLoop()

  std::vector<BasicBlock *> DeadBlocks;
  DeadBlocks.push_back(Root);
  for (unsigned i = 0; i < DeadBlocks.size(); ++i)
    for (DomTreeNode *Child : *DT[DeadBlocks[i]])
      DeadBlocks.push_back(Child->getBlock());

  // In case blocks in the subtree have successors outside the subtree, ensure
  // the successors do not have PHINodes that hold references to the blocks we
  // are erasing.
  for (BasicBlock *BB : DeadBlocks) {
    // successors may appear multiple times, use this set to dedupe them.
    const SmallSetVector<BasicBlock *, 8> Successors(succ_begin(BB),
                                                     succ_end(BB));
    for (BasicBlock *Succ : Successors)
      if (!DT.dominates(Root, Succ))
        Succ->removePredecessor(BB, true);
  }

  for (BasicBlock *BB : DeadBlocks)
    BB->dropAllReferences();

  for (BasicBlock *BB : reverse(DeadBlocks)) {
    DT.eraseNode(BB);
    if (LI) {
      Loop *L = LI->getLoopFor(BB);
      const bool lastInLoop = L && L->getNumBlocks() == 1;
      assert(lastInLoop == (L && L->getHeader() == BB) &&
             "Header should dominate everything in loop");
      LI->removeBlock(BB);
      if (lastInLoop) {
        assert(L->getNumBlocks() == 0);
        assert(L->empty() && "Failed to delete inner loops first?");
        // TODO(victory): In LLVM version 6, LoopInfo::markAsRemoved() is
        // renamed to erase()?
        LI->markAsRemoved(L);
      }
    }
    BB->eraseFromParent();
  }
}


CallInst *llvm::formRecursiveLoop(Function *F) {
  // Find loop header
  BasicBlock *Preheader = &F->getEntryBlock();
  assert(pred_empty(Preheader) && "Function entry must not have predecessors");
  // Preheader should do nothing but branch to the header
  assert(Preheader->size() == 1);
  assert(cast<BranchInst>(Preheader->getTerminator())->isUnconditional());
  BasicBlock *Header = Preheader->getSingleSuccessor();
  assert(Header);

  // Find loop latch
  SmallPtrSet<BasicBlock *, 2> HeaderPreds(pred_begin(Header),
                                           pred_end(Header));
  assert(HeaderPreds.count(Preheader));
  HeaderPreds.erase(Preheader);
  assert(HeaderPreds.size() == 1);
  BasicBlock *Latch = *HeaderPreds.begin();

  // Find end of function
  ReturnInst *Ret = getUniqueReturnInst(*F);
  assert(Ret);
  assert(!Ret->getReturnValue());
  BasicBlock *RetBlock = Ret->getParent();
  assert(RetBlock->size() == 1);

  // Use the parameters that correspond to the PHI nodes.
  assert(F->getFunctionType()->getNumParams()
         >= std::distance(Header->phis().begin(), Header->phis().end()) &&
         "We need a parameter for each header phi node");
  Function::arg_iterator AI = F->arg_begin();
  for (PHINode &PN : Header->phis()) {
    int Idx = PN.getBasicBlockIndex(Preheader);
    assert(Idx >= 0);
    PN.setIncomingValue(Idx, &*AI++);
  }

  // Create a recursive call.
  BasicBlock *Recur = BasicBlock::Create(F->getContext(), "recur", F);
  IRBuilder<> Builder(Recur);
  Builder.SetCurrentDebugLocation(Latch->getTerminator()->getDebugLoc());
  assert(!F->getSubprogram() || Builder.getCurrentDebugLocation()
         && "We are creating an inlinable call, requires debug info.");
  SmallVector<Value *, 8> RecurArgs;
  for (PHINode &PN : Header->phis())
    RecurArgs.push_back(PN.getIncomingValueForBlock(Latch));
  while (AI != F->arg_end())
    RecurArgs.push_back(&*AI++);
  CallInst *RecurCall = Builder.CreateCall(F, RecurArgs);
  RecurCall->setCallingConv(F->getCallingConv());
  Builder.CreateBr(RetBlock);

  // Now redirect the backedge to go to the recursive call.
  Header->removePredecessor(Latch);
  Latch->getTerminator()->replaceUsesOfWith(Header, Recur);
  assert(Header->getSinglePredecessor() == Preheader);
  assert(!isa<PHINode>(Header->front()) &&
         "Header still has PHIs after removal of latch as predecessor?");

  return RecurCall;
}


bool llvm::isDoomedToUnreachableEnd(const BasicBlock *BB) {
  SmallPtrSet<const BasicBlock *, 8> Visited;
  std::function<bool(const BasicBlock *)> DFSHelper =
          [&Visited, &DFSHelper](const BasicBlock *BB) {
    if (!Visited.insert(BB).second) return true;

    const TerminatorInst *TI = BB->getTerminator();
    if (isa<UnreachableInst>(TI))
      return true;
    else if (isa<ReturnInst>(TI) || isa<ReattachInst>(TI))
      return false;
    else if (isa<BranchInst>(TI) || isa<SwitchInst>(TI)
             || isa<IndirectBrInst>(TI))
      return all_of(successors(BB), DFSHelper);
    else if (const auto *DI = dyn_cast<SDetachInst>(TI))
      return DFSHelper(DI->getContinue());
    else
      llvm_unreachable("Unhandled exception-handling terminator?");
  };

  return DFSHelper(BB);
}


const BasicBlock *llvm::getUniqueNonDeadendExitBlock(const Loop &L) {
  SmallVector<BasicBlock *, 8> Exits;
  L.getExitBlocks(Exits);
  const BasicBlock *NonDeadendExit = nullptr;
  for (const BasicBlock *Exit : Exits) {
    if (!isDoomedToUnreachableEnd(Exit)) {
      if (NonDeadendExit && NonDeadendExit != Exit) return nullptr;
      else NonDeadendExit = Exit;
    }
  }
  return NonDeadendExit;
}


void llvm::eraseLoop(
        Loop &L,
        BasicBlock *EndBlock,
        DominatorTree &DT,
        LoopInfo &LI) {
  DEBUG(dbgs() << "Removing old loop code from the original function\n");

  BasicBlock *Preheader = L.getLoopPreheader();
  BasicBlock *Header = L.getHeader();
  assert(Preheader && "Loop does not have a dedicated preheader.");
  assert(Header);

  // Redirect the preheader to branch directly past all the outlined blocks.
  Preheader->getTerminator()->replaceUsesOfWith(Header, EndBlock);
  if (DT.dominates(Preheader, EndBlock))
    DT.changeImmediateDominator(EndBlock, Preheader);

  assert(!isa<PHINode>(EndBlock->front()) &&
         "TODO: Rewrite phis appropriately if they exist");

  Loop *ParentLoop = L.getParentLoop();
  SmallVector<BasicBlock *, 8> BlocksToErase;
  DT.getDescendants(Header, BlocksToErase);
  //TODO(victory): Uncomment this assertion if we keep LoopInfo up to date.
  //assert(!ParentLoop || all_of(BlocksToErase,
  //                             [ParentLoop](const BasicBlock *BB) {
  //                               return ParentLoop->contains(BB); }));

  // Finally, actually perform the deletion and analysis updates.
  eraseDominatorSubtree(Header, DT, &LI);
  assert(std::none_of(ParentLoop ? ParentLoop->begin() : LI.begin(),
                      ParentLoop ? ParentLoop->end() : LI.end(),
                      [&L](const Loop *Sibling) { return Sibling == &L; }) &&
         "Loop not entirely dominated by header?");
  assert(!ParentLoop || (!ParentLoop->isInvalid() && ParentLoop->getNumBlocks())
         && "Parent loop somehow emptied?");

  Function *F = Preheader->getParent();

  // If the loop could reach some error-handling blocks that it does not
  // dominate, the absense of the loop may now change the dominators for
  // those error-handling blocks.
  // TODO(victory): Do an incremental update of the dominator tree instead of
  // throwing it out and recalculating it.
  DT.recalculate(*F);

  DEBUG(assertVerifyFunction(*F, "After removing loop", &DT));
}


// Use a DFS to add to ReachableBBs all basic blocks reachable from Source
// including Source itself, up to but not including End.
// \returns true if you can reach End from Source
template <typename SetTy>
static bool getReachableBlocks(BasicBlock *Source,
                               BasicBlock *End,
                               SetTy &ReachableBBs) {
  if (Source == End)
    return true;
  ReachableBBs.insert(Source);
  bool CanReachEnd = false;
  // Ideally we would strengthen this assertion to also panic if we see a
  // ReattachInst ending the enclosing task, but we can't do that cheaply with
  // the current dominance-based detach-reattach matching.
  assert(!End || !isa<ReturnInst>(Source->getTerminator())
         && "Unexpected exit from region");
  for (BasicBlock *BB : successors(Source))
    if (!ReachableBBs.count(BB))
      CanReachEnd |= getReachableBlocks(BB, End, ReachableBBs);
  return CanReachEnd;
}


BasicBlock *llvm::makeReachableDominated(
        BasicBlock *const Source,
        BasicBlock *const End,
        DominatorTree &DT,
        LoopInfo &LI,
        SmallSetVector<BasicBlock *, 8> &ReachableBBs,
        unsigned *const NumCopiedBlocks) {
  assert(Source != End);
  Function &F = *Source->getParent();
  DEBUG(assertVerifyFunction(F, "Before cloning blocks", &DT, &LI));

  assert(ReachableBBs.empty());
  const bool CanReachEnd = getReachableBlocks(Source, End, ReachableBBs);
  assert(ReachableBBs.count(Source));
  assert(!ReachableBBs.count(End));

  // Clone all non-dominated reachable blocks.
  ValueToValueMapTy IMap;
  DenseMap<const BasicBlock *, BasicBlock *> BBMap;
  SmallVector<BasicBlock *, 8> OldBBs;
  SmallPtrSet<BasicBlock *, 8> NewBBs;
  SmallSetVector<const Loop *, 8> CopiedLoops;
  for (BasicBlock *BB : ReachableBBs) {
    if (DT.dominates(Source, BB))
      continue;

    assert(!BB->hasAddressTaken());
    BasicBlock *NewBB = CloneBasicBlock(BB, IMap, ".dupe", &F);
    //DEBUG(dbgs() << " Cloned block " << BB->getName() << "\n");
    BBMap[BB] = NewBB;
    OldBBs.push_back(BB);
    NewBBs.insert(NewBB);
    if (LI.isLoopHeader(BB))
      CopiedLoops.insert(LI.getLoopFor(BB));
  }

  // Point all predecessors that aren't part of the reachable sub-CFG to the
  // clones. Also update PHINodes appropriately.
  for (BasicBlock *BB : OldBBs) {
    BasicBlock *NewBB = cast<BasicBlock>(BBMap[BB]);
    assert(pred_empty(NewBB));
    //DEBUG(dbgs() << " Redirecting edges to cloned block " << BB->getName() << '\n');
    SmallVector<BasicBlock *, 8> Predecessors(predecessors(BB));
    for (BasicBlock *Pred : Predecessors) {
      if (ReachableBBs.count(Pred)) {
        //DEBUG(dbgs() << "  not from reachable block " << Pred->getName() << '\n');
        if (BBMap.count(Pred)) {
          BasicBlock *ClonedPred = cast<BasicBlock>(BBMap[Pred]);
          assert(ClonedPred);
          assert(NewBBs.count(ClonedPred));
          for (PHINode &PN : NewBB->phis()) {
            int Idx = PN.getBasicBlockIndex(Pred);
            assert(Idx >= 0 && "Non-existant PHINode entry?");
            PN.setIncomingBlock(Idx, ClonedPred);
          }
        } else {
          for (PHINode &PN : NewBB->phis()) {
            assert(PN.getNumIncomingValues() > 1);
            PN.removeIncomingValue(Pred);
          }
        }
      } else {
        //DEBUG(dbgs() << "  from " << Pred->getName() << '\n');
        Pred->getTerminator()->replaceUsesOfWith(BB, NewBB);
        if (!NewBBs.count(Pred)) {
          for (PHINode &PN : BB->phis()) {
            assert(PN.getNumIncomingValues() > 1);
            PN.removeIncomingValue(Pred);
          }
        }
      }
    }
  }
  //TODO(victory): Can we be more intelligent and efficient in this DT update?
  DT.recalculate(F);

  // Fix up operand references for cloned blocks
  for (BasicBlock *BB : OldBBs) {
    BasicBlock *NewBB = cast<BasicBlock>(BBMap[BB]);
    for (Instruction &I : *NewBB)
      RemapInstruction(&I, IMap,
                       RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
  }

  // Update LoopInfo for blocks copied outside copied loops
  for (const BasicBlock *BB : OldBBs) {
    if (CopiedLoops.count(LI.getLoopFor(BB)))
      continue;
    BasicBlock *NewBB = cast<BasicBlock>(BBMap[BB]);
    Loop *OldLoop = LI.getLoopFor(BB);
    if (OldLoop)
      OldLoop->addBasicBlockToLoop(NewBB, LI);
  }

  // Now update LoopInfo for blocks inside copied loops.
  // Our strategy here assumes that if the header of a loop was copied,
  // then the entire loop, including nested inner loops, is also copied.
  assert(!CopiedLoops.count(LI.getLoopFor(End)) &&
         "Uh oh, did we only partially copy some loop?");
  SmallVector<const Loop *, 8> TopCopiedLoops;
  for (const Loop *L : CopiedLoops)
    if (!CopiedLoops.count(L->getParentLoop()))
      TopCopiedLoops.push_back(L);
  for (const Loop *TopOldLoop : TopCopiedLoops) {
    DEBUG(dbgs() << "Updating LoopInfo for copied loop including all "
                 << "levels of nested inner loops:\n" << *TopOldLoop << '\n');
    DenseMap<const Loop *, Loop *> LoopMap;
    for (const Loop *OldLoop : depth_first(TopOldLoop)) {
      bool Erased = CopiedLoops.remove(OldLoop);
      assert(Erased); (void)Erased;
      Loop *NewLoop = new Loop();

      // Set NewLoop's parent
      Loop *NewLoopParent;
      if (OldLoop == TopOldLoop) {
        NewLoopParent = OldLoop->getParentLoop();
      } else {
        NewLoopParent = LoopMap[OldLoop->getParentLoop()];
        assert(NewLoopParent);
      }
      if (NewLoopParent) {
        NewLoopParent->addChildLoop(NewLoop);
      } else {
        LI.addTopLevelLoop(NewLoop);
        NewLoop->setParentLoop(nullptr);
      }

      // Set NewLoop's contents
      for (BasicBlock *BB : OldLoop->blocks()) {
        BasicBlock *NewBB = cast<BasicBlock>(BBMap[BB]);
        NewLoop->addBlockEntry(NewBB);
      }
      assert(NewLoop->getHeader() == BBMap[OldLoop->getHeader()]);
      NewLoop->verifyLoop();
      LoopMap[OldLoop] = NewLoop;
    }

    for (const BasicBlock *BB : TopOldLoop->blocks()) {
      // Set innermost-enclosing-loop mappings
      BasicBlock *NewBB = cast<BasicBlock>(BBMap[BB]);
      Loop *NewLoop = LoopMap[LI.getLoopFor(BB)];
      assert(NewLoop);
      assert(NewLoop->contains(NewBB));
      LI.changeLoopFor(NewBB, NewLoop);

      // Add the basic block to all parent loops...
      Loop *L = TopOldLoop->getParentLoop();
      while (L) {
        L->addBlockEntry(NewBB);
        L = L->getParentLoop();
      }
    }
  }
  assert(CopiedLoops.empty() && "Failed to visit all copied loops?");

  DEBUG(assertVerifyFunction(F, "After cloning blocks", &DT, &LI));

  if (NumCopiedBlocks)
    *NumCopiedBlocks = BBMap.size();

  assert(ReachableBBs.count(Source));
  assert(!ReachableBBs.count(End));

  if (!CanReachEnd) {
    assert(isDoomedToUnreachableEnd(Source));
    return nullptr;
  }

  DEBUG(dbgs() << "Setting up new end block\n");
  SmallVector<BasicBlock *, 8> EndPreds;
  copy_if(predecessors(End), std::back_inserter(EndPreds),
          [&ReachableBBs](BasicBlock *Pred) {
            return ReachableBBs.count(Pred); });
  BasicBlock *NewEndPred = SplitBlockPredecessors(End,
                                                  EndPreds,
                                                  ".newend",
                                                  &DT,
                                                  &LI);
  assert(NewEndPred);
  assert(DT.isReachableFromEntry(NewEndPred));
  assert(DT.dominates(Source, NewEndPred));
  assert(LI.getLoopFor(NewEndPred) == LI.getLoopFor(End));
  DEBUG(assertVerifyFunction(F, "After setting up new end block", &DT, &LI));
  ReachableBBs.insert(NewEndPred);
  return NewEndPred;
}


void llvm::topologicalSort(
        BasicBlock *Entry,
        BasicBlock *Exit,
        const LoopInfo &LI,
        SmallVectorImpl<BasicBlock *> &SortedBBs) {
  assert(SortedBBs.empty());
  const Loop *const L = LI.getLoopFor(Entry);
  assert(LI.getLoopFor(Exit) == L);

  SmallSetVector<const BasicBlock *, 8> Visiting;
  SmallPtrSet<const BasicBlock *, 8> Finished;
  SmallPtrSet<const BasicBlock *, 8> CanReachExit = {Exit};
  SmallVector<BasicBlock *, 8> PostOrderBBs;

  // Recursive helper function does a DFS.
  // Populates PostOrderBBs in post-order.
  // Returns true on success.
  std::function<bool(BasicBlock *)> topoSortDFS =
      [&, L, Exit] (BasicBlock *BB) {
    assert(!L || L->contains(BB));

    //dbgs() << "Visiting: " << BB->getName() << '\n';
    Visiting.insert(BB);

    //SmallVector<const BasicBlock *, 8> SuccessorBlocks;
    SmallVector<BasicBlock *, 8> SuccessorBlocks;
    const Loop *BBLoop = LI.getLoopFor(BB);
    if (BB == Exit) {
      // Ignore any successors past Exit
    } else if (BBLoop == L) {
      TerminatorInst *TI = BB->getTerminator();
      if (auto DI = dyn_cast<SDetachInst>(TI)) {
        SuccessorBlocks = {DI->getContinue()};
      } else {
        auto Successors = BB->getTerminator()->successors();
        SuccessorBlocks.assign(Successors.begin(), Successors.end());
      }
    } else {
      assert(BBLoop->getParentLoop() == L &&
             "Should be entering immediate subloop");
      assert(LI.isLoopHeader(BB) && "entered subloop not at header?");
      BBLoop->getExitBlocks(SuccessorBlocks);
    }

    assert(none_of(SuccessorBlocks, [L](const BasicBlock *Succ) {
      return L && L->getHeader() == Succ;
    }) && "Encountered current loop's backedge during topological sort");
    assert(none_of(SuccessorBlocks, [BB, &LI](const BasicBlock *Succ) {
      const Loop *SuccLoop = LI.getLoopFor(Succ);
      return SuccLoop && SuccLoop->getHeader() == Succ && SuccLoop->contains(BB);
    }) && "Encountered some loop's backedge during topological sort");

    bool CanBBReachExit = BB == Exit;
    for (BasicBlock *Succ : SuccessorBlocks) {
      //dbgs() << "Successor: " << Succ->getName() << '\n';

      if(Visiting.count(Succ)) {
        DEBUG({
          dbgs() << "topological sort found irreducible cycle:\n";
          for (auto I = find(Visiting, Succ), E = Visiting.end(); I != E; ++I) {
            dbgs() << **I << '\n';
          }
        });
        // Irreducible control flow is rare in real-world code.
        return false;
      }

      if (Finished.count(Succ)) {
        CanBBReachExit |= CanReachExit.count(Succ);
        continue;
      }

      // Ignore exits (e.g., error handling exits terminated by unreachable)
      if (L && !L->contains(Succ)) continue;

      if (!topoSortDFS(Succ))
        return false;
      CanBBReachExit |= CanReachExit.count(Succ);
    }

    //dbgs() << "Finished: " << BB->getName() << '\n';
    if (CanBBReachExit) {
      CanReachExit.insert(BB);
      PostOrderBBs.push_back(BB);
    } else {
      DEBUG(dbgs() << "Excluding block that cannot reach exit: "
                   << *BB << '\n');
    }
    assert(Visiting.back() == BB);
    Visiting.pop_back();
    Finished.insert(BB);

    return true;
  };

  if (!topoSortDFS(Entry)) return;
  assert(CanReachExit.count(Exit));
  assert(CanReachExit.count(Entry));

  // topological order is just reverse post-order
  SortedBBs.assign(PostOrderBBs.rbegin(), PostOrderBBs.rend());
  assert(SortedBBs.front() == Entry);
  assert(SortedBBs.back() == Exit);
}
