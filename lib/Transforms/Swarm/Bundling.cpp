//===---------------- Bundling.cpp - Bundling pass ------------------------===//
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
// Replace allocas (stack allocations) with calls to malloc() and free().
// Moving stack allocations to the heap is necessary as SCC will break
// functions into tasks that cannot share stack space.  Specifically, each
// Swarm task discards its local stack space when it finishes, and does not
// wait for later tasks running code originally from the same function that may
// wish to continue using the allocated space.
//
// To reduce overheads, multiple allocas may be "bundled" together
// into a single call to malloc() and free().
//
//===----------------------------------------------------------------------===//

#include "Utils/Flags.h"
#include "Utils/Misc.h"

#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/OptimizationDiagnosticInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Transforms/Swarm.h"

using namespace llvm;

#define PASS_NAME "bundling"
#define DEBUG_TYPE PASS_NAME

static cl::opt<bool> DisableBundlingOpt("swarm-disablebundling",
        cl::init(false),
        cl::desc("Disable bundling of stack allocations"));

static cl::opt<bool> DisablePaddingOpt("swarm-disablepadding",
        cl::init(false),
        cl::desc("Disable padding of bundled stack allocations"));

static cl::opt<bool> DisablePrivatizationOpt("swarm-disableprivatization",
        cl::init(false),
        cl::desc("Disable privatization of loop variables"));

namespace {

class Bundler {
  private:
    Function &F;

    DominatorTree &DT;
    LoopInfo &LI;
    OptimizationRemarkEmitter &ORE;
    Module &M;
    LLVMContext &Context;
    ReturnInst *const Return;
    Function *const Malloc;

  public:
    Bundler(Function &F,
            DominatorTree &DT,
            LoopInfo &LI,
            TargetLibraryInfo &TLI,
            OptimizationRemarkEmitter &ORE)
        : F(F), DT(DT), LI(LI), ORE(ORE),
          M(*F.getParent()), Context(F.getContext()),
          Return(getUniqueReturnInst(F)),
          Malloc(cast<Function>(
              M.getOrInsertFunction("malloc", Type::getInt8PtrTy(Context),
                                    Type::getInt64Ty(Context)))) {
      LibFunc LF; (void) LF;
      assert(TLI.getLibFunc(*Malloc, LF) && (LF == LibFunc_malloc)
             && "Fractalizer must identify allocator calls for special treatment");
    }

    bool bundle();

  private:
    bool replaceEntryStackWithHeapAlloc();
    void replaceStackWithHeapAlloc(ArrayRef<AllocaInst *>,
                                         Instruction* MallocLocation,
                                         Instruction* FreeLocation);
    void replaceIndividualStackWithHeapAlloc(AllocaInst* AllocaI,
                                             Instruction* MallocLocation,
                                             Instruction* FreeLocation);

    bool removeLifetimeIntrinsics();

    void privatize();
    void privatizeLoopVariables(Loop *L);
    void privatizeIfStructOrArray(GetElementPtrInst* Inst,
                                  IntrinsicInst* ParentLifetimeStart);
    SmallVector<IntrinsicInst*, 8> getLifetimeStarts(Loop *);
    SmallVector<IntrinsicInst*, 8> getLifetimeEnds(Loop *);
    SmallVector<IntrinsicInst*, 8> getContainedLifetimeVariables(
                                    SmallVector<IntrinsicInst*, 8>,
                                    SmallVector<IntrinsicInst*, 8>);

    bool isSafeToPrivatizeAlloca(const AllocaInst *I, const Loop *L);
    bool isSafeToPrivatize(const Instruction *I, const Loop *L);

};

bool Bundler::bundle() {
  StringRef FName = F.getName();
  DEBUG(dbgs() << "Bundling function " << FName << "() ...\n");

  DEBUG(assertVerifyFunction(F, "Before bundling", &DT, &LI));
  assert(Return && "F should have a unique return instruction");

  if (!DisablePrivatizationOpt) {
    privatize();
  }
  bool replacedStackAlloc = replaceEntryStackWithHeapAlloc();
  bool removedLifetime = removeLifetimeIntrinsics();

  assertVerifyFunction(F, "After bundlng", &DT, &LI);

  return replacedStackAlloc || removedLifetime;
}

bool Bundler::replaceEntryStackWithHeapAlloc() {
  // We handle AllocaInsts only in the entry block
  // (https://llvm.org/docs/Frontend/PerformanceTips.html#use-of-allocas)

  BasicBlock &EntryBB = F.getEntryBlock();

  SmallVector<AllocaInst*, 8> Allocas;

  for (Instruction &I : EntryBB) {
    // Increment I before it is removed from the instruction iterable
    if (AllocaInst *AllocaI = dyn_cast<AllocaInst>(&I)) {
      Allocas.push_back(AllocaI);
    }
  }

  if (!Allocas.empty()) {
    // replace all allocas with a single malloc and pointers into the malloc
    replaceStackWithHeapAlloc(Allocas,
                                    &*(EntryBB.getFirstInsertionPt()),
                                    Return);
    DEBUG({
      dbgs() << "After replacing Allocas with Malloc in" << EntryBB << "\n"
             << "After adding Free in\n" << *Return->getParent() << "\n";
    });
  }

  assert(none_of(F,
          [] (const BasicBlock &BB) {
            return any_of(BB,
                    [] (const Instruction &I) {
                      return isa<AllocaInst>(&I);
                    });
          }) && "No AllocaInst should remain in the function");

  // return true if we made changes
  return !Allocas.empty();
}

void Bundler::replaceIndividualStackWithHeapAlloc(
                                                  AllocaInst *AllocaI,
                                                  Instruction* MallocLocation,
                                                  Instruction* FreeLocation) {
  const DataLayout &DL = M.getDataLayout();

  Type *AT = AllocaI->getAllocatedType();
  Constant *AllocSize = ConstantInt::get(DL.getIntPtrType(M.getContext()),
                                         DL.getTypeAllocSize(AT));
  assert(AllocSize->getType() == Malloc->getFunctionType()->getParamType(0)
         && "Allocation size vs malloc parameter type mismatch");

  Instruction *MI = CallInst::CreateMalloc(MallocLocation,
                                           AllocSize->getType(),
                                           AT,
                                           AllocSize,
                                           AllocaI->getArraySize(),
                                           Malloc,
                                           AllocaI->getName());
  AllocaI->replaceAllUsesWith(MI);
  AllocaI->eraseFromParent();
  Instruction *FI = CallInst::CreateFree(MI, FreeLocation);
  FI->setDebugLoc(Return->getDebugLoc());
}

void Bundler::replaceStackWithHeapAlloc(ArrayRef<AllocaInst *> Allocas,
                                              Instruction* MallocLocation,
                                              Instruction* FreeLocation) {
  assert(!Allocas.empty() && "Allocas must be nonempty");

  if (DisableBundlingOpt) {
    for (AllocaInst *AllocaI : Allocas) {
      replaceIndividualStackWithHeapAlloc(AllocaI, AllocaI, FreeLocation);
    }
    return;
  }

  const DataLayout &DL = M.getDataLayout();

  // construct a struct type with padding to align each "real" field
  // to a cache line
  SmallVector<Type *, 8> StructFieldTypes;
  SmallVector<int, 8> FieldIndices;
  for (AllocaInst* AllocaI: Allocas) {
    Type* AllocaT = AllocaI->getAllocatedType();
    FieldIndices.push_back(StructFieldTypes.size());
    StructFieldTypes.push_back(AllocaT);

    if (!DisablePaddingOpt) {
      // add padding if necessary
      uint64_t size = DL.getTypeAllocSize(AllocaT);
      if (size % SwarmCacheLineSize != 0) {
        int PaddingSize = SwarmCacheLineSize - (size % SwarmCacheLineSize);
        Type * ByteT = IntegerType::getInt8Ty(AllocaI->getContext());
        ArrayType* PaddingArrayT = ArrayType::get(ByteT, PaddingSize);
        StructFieldTypes.push_back(PaddingArrayT);
      }
    }
  }

  // assert length of fieldIndicies is the same as length of Allocas
  assert(FieldIndices.size() == Allocas.size()
        && "Number of allocas should match number of non-padding fields in malloc struct");

  auto *ST = StructType::create(StructFieldTypes);

  uint64_t StructSz = DL.getStructLayout(ST)->getSizeInBytes();
  Value *StructAllocSize = ConstantInt::get(DL.getIntPtrType(M.getContext()),
                                            StructSz);

  // creates single malloc
  Instruction *MergedMallocInst = CallInst::CreateMalloc(
                                                MallocLocation,
                                                StructAllocSize->getType(),
                                                ST,
                                                StructAllocSize,
                                                nullptr, Malloc,
                                                "allocas_moved_to_heap");

  // replace all allocas with pointers
  IRBuilder<> IRB = IRBuilder<>(MallocLocation);
  auto FieldIndicesI = FieldIndices.begin();
  for (AllocaInst* AllocaI : Allocas) {
    IRB.SetInsertPoint(AllocaI);

    assert(DisablePaddingOpt ||
           DL.getStructLayout(ST)->getElementOffset(*FieldIndicesI)
                  % SwarmCacheLineSize == 0);

    auto* ConstInBounds = IRB.CreateConstInBoundsGEP2_32(ST,
                                                         MergedMallocInst,
                                                         0,
                                                         *(FieldIndicesI++));
    ConstInBounds->takeName(AllocaI);

    AllocaI->replaceAllUsesWith(ConstInBounds);
    AllocaI->eraseFromParent();
  }

  Instruction *FI = CallInst::CreateFree(MergedMallocInst, FreeLocation);
  FI->setDebugLoc(FreeLocation->getDebugLoc());

  DEBUG(assertVerifyFunction(F, "After creating one bundle", &DT, &LI));
}

// Privatizes variables that have been hoisted previously by LLVM.
// For example, it transforms the top piece of code into the bottom:
//
// entry:
//   %a = alloca i64, align 8
//   %0 = bitcast i64* %a to i8*
//   br label %for.body
//
// for.body:
//   <use %0>
//   ...
//   %exitcond = icmp eq i64 %indvars.iv.next, 1000
//   br i1 %exitcond, label %for.cond.cleanup, label %for.body
//
//
//
// entry:
//   br %for.body
//
// for.body:
//   %malloccall = tail call i8* @malloc(i64 64)
//   %allocas_moved_to_heap = bitcast i8* %malloccall to %0*
//   %a = getelementptr inbounds %0, %0* %allocas_moved_to_heap, i32 0, i32 0
//   %3 = bitcast i64* %a to i8*
//   <use %3>
//   ...
//   %exitcond = icmp eq i64 %indvars.iv.next, 1000
//   %4 = bitcast %0* %allocas_moved_to_heap to i8*
//   tail call void @free(i8* %4)
//   br i1 %exitcond, label %for.cond.cleanup, label %for.body
//
//
//
// When loop iterations get split into tasks, this allows each task to have its
// own copy of the variable to work with, reducing the number of potential
// conflicts between tasks.
//
// We call replaceStackWithHeapAlloc to bundle privatized allocas into a single
// malloc and free.
//
void Bundler::privatize() {
  for (Loop *TopLevelLoop : LI) {
    for (Loop *L : depth_first(TopLevelLoop)) {
      privatizeLoopVariables(L);
    }
  }
}

void Bundler::privatizeLoopVariables(Loop *L) {
  DEBUG(dbgs() << "Privatizing loop; " << *L);
  DEBUG(dbgs() << " starting at: ");
  DEBUG(L->getStartLoc().print(dbgs()));
  DEBUG(dbgs() << '\n');

  // For now, only privatize loops with a unique exiting block that's also the
  // unique latch block. If we do privatize a variable, having a unique
  // exiting/latch block allows us to guarantee that we free any privatized
  // variables exactly once each loop iteration.
  BasicBlock* ExitingBB = L->getExitingBlock();
  BasicBlock* LatchBB = L->getLoopLatch();
  if (ExitingBB == nullptr || LatchBB == nullptr || ExitingBB != LatchBB) {
    return;
  }

  SmallVector<IntrinsicInst*, 8> LifetimeStarts = getLifetimeStarts(L);
  SmallVector<IntrinsicInst*, 8> LifetimeEnds = getLifetimeEnds(L);

  auto ContainedLifetimeStarts = getContainedLifetimeVariables(LifetimeStarts,
                                                               LifetimeEnds);

  BasicBlock* Header = L->getHeader();
  IRBuilder<> IRB = IRBuilder<>(Header->getFirstNonPHI());

  SmallVector<AllocaInst*, 8> PrivatizedAllocas;

  // for all contained lifetime intrinsics, check if the variable is privatizable
  for (auto LifetimeStartI : ContainedLifetimeStarts) {
    auto *BitCastI = dyn_cast<BitCastInst>(LifetimeStartI->getArgOperand(1));
    // Don't privatize in cases where this pointer isn't a bitcast.
    // [graceyin] have observed cases where this is a PHINode.
    if (!BitCastI) {
      continue;
    }

    auto *AllocaI = dyn_cast<AllocaInst>(BitCastI->getOperand(0));
    if (!AllocaI) {
      continue;
    }

    // if safe to privatize, move the alloca and bitcast instructions into the
    // loop, followed by any relevant gep instructions.
    if (isSafeToPrivatizeAlloca(AllocaI, L)) {
      std::string Msg;
      raw_string_ostream OS(Msg);
      OS << "Privatizing variable %" << AllocaI->getName()
         << " of type " << *AllocaI->getAllocatedType() << ".";
      ORE.emit(OptimizationRemark(PASS_NAME, "PrivatizedVariable",
                                  LifetimeStartI)
               << OS.str());
      DEBUG(dbgs() << "Privatizing : " << *AllocaI << "  and its uses :\n");
      DEBUG(dbgs() << "  " << *BitCastI << "\n");
      // order matters, because we need the alloca to be before the bitcast
      AllocaI->moveBefore(LifetimeStartI);
      BitCastI->moveBefore(LifetimeStartI);
      for (auto UseIt = AllocaI->user_begin(); UseIt != AllocaI->user_end(); UseIt++) {
        // we have already guaranteed that all uses of this alloca are safe,
        // so just check for bitcasts and GEPs and, if necessary, privatize
        // them as well.
        auto *UseI = cast<Instruction>(*UseIt);
        if (!DT.dominates(AllocaI, UseI)) {
          assert(isa<BitCastInst>(UseI) || isa<GetElementPtrInst>(UseI)
                 || L->contains(UseI)
                 && "guaranteed by isSafeToPrivatizeAlloca()");
          DEBUG(dbgs() << "  " << *UseI << "\n");
          UseI->moveBefore(LifetimeStartI);
        }
      }
      PrivatizedAllocas.push_back(AllocaI);
    } else {
      std::string Msg;
      raw_string_ostream OS(Msg);
      OS << "Not privatizing variable %" << AllocaI->getName()
         << " of type " << *AllocaI->getAllocatedType()
         << " despite it having a lifetime (scope) within a loop body?";
      ORE.emit(DiagnosticInfoOptimizationFailure(
                      PASS_NAME, "NonPrivatizedVariable",
                      LifetimeStartI->getDebugLoc(), L->getHeader())
               << OS.str());
    }
  }

  // merge allocas at this loop depth
  if (!PrivatizedAllocas.empty()) {
    ORE.emit(OptimizationRemark(PASS_NAME, "PrivatizationLoop",
                                L->getStartLoc(), L->getHeader())
             << "Privatization allocates copies for iterations of this loop.");
    Instruction* MallocLocation = &*(L->getHeader()->getFirstInsertionPt());
    Instruction* FreeLocation = ExitingBB->getTerminator();
    replaceStackWithHeapAlloc(PrivatizedAllocas,
                                    MallocLocation,
                                    FreeLocation);
  }
}

bool Bundler::isSafeToPrivatizeAlloca(const AllocaInst *I, const Loop *L) {
  // Checks all uses of an inst, and if a use is a bitcast or gep, recursively
  // checks all uses of the bitcast or gep.
  // It's safe to privatize if all uses (and uses of uses) are within
  // the same loop.

  // check all uses of the alloca
  for (auto UseIt = I->user_begin(); UseIt != I->user_end(); UseIt++) {
    auto *UseI = cast<Instruction>(*UseIt);
    if (isa<GetElementPtrInst>(UseI) || isa<BitCastInst>(UseI)) {
      if (!isSafeToPrivatize(UseI, L)) {
        return false;
      }
    }
    // otherwise, all other uses must be within the loop body.
    else if (!L->contains(cast<Instruction>(*UseIt))) {
      return false;
    }
  }
  return true;
}

bool Bundler::isSafeToPrivatize(const Instruction *I, const Loop *L) {
  if (isa<BitCastInst>(I) || isa<GetElementPtrInst>(I)) {
    for (auto UseIt = I->user_begin(); UseIt != I->user_end(); UseIt++) {
      // if the use is an instruction not contained within this loop,
      // we cannot privatize.
      if (!L->contains(cast<Instruction>(*UseIt))) {
        return false;
      }
    }
    return true;
  }
  return false;
}


SmallVector<IntrinsicInst*, 8> Bundler::getLifetimeStarts(Loop *L) {
  // For now, only look for lifetime starts in loop headers.
  SmallVector<IntrinsicInst*, 8> LifetimeStarts;
  BasicBlock *BB = L->getHeader();
  for (Instruction &I : *BB) {
    if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
      if (II->getIntrinsicID() == Intrinsic::lifetime_start) {
        LifetimeStarts.push_back(II);
      }
    }
  }
  return LifetimeStarts;
}

SmallVector<IntrinsicInst*, 8> Bundler::getLifetimeEnds(Loop *L) {
  // For now, only look for lifetime ends in the unique exiting/latch block
  // (guaranteed that the exiting block is unique and that it's the unique
  // latch block from privatizeLoopVariables() )
  SmallVector<IntrinsicInst*, 8> LifetimeEnds;
  BasicBlock *BB = L->getExitingBlock();
  for (Instruction &I : *BB) {
    if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
      if (II->getIntrinsicID() == Intrinsic::lifetime_end) {
        LifetimeEnds.push_back(II);
      }
    }
  }
  return LifetimeEnds;
}

SmallVector<IntrinsicInst*, 8> Bundler::getContainedLifetimeVariables(
                                SmallVector<IntrinsicInst*, 8> LifetimeStarts,
                                SmallVector<IntrinsicInst*, 8> LifetimeEnds) {
  SmallVector<IntrinsicInst*, 8> toReturn;
  // iterate through all starts and ends to verify we don't have multiple
  // lifetime starts and ends per alloca.
  // If there are multiple lifetime starts or ends per alloca, abort.
  // (We can afford to be "dumb" here, since we expect only a handful of
  // variables to be allocated in each loop)
  // TODO (graceyin): under what circumstances would this ever happen?
  // (Is there any prior pass that would duplicate code in such a way that
  // multiple lifetime starts or ends for a variable would appear?
  // If it doesn't/shouldn't ever happen, turn this into an assertion?)
  for (auto II: LifetimeStarts) {
    for (auto II2: LifetimeStarts) {
      if (II != II2 && II->getArgOperand(1) == II2->getArgOperand(1)) {
        return toReturn;
      }
    }
  }
  for (auto II: LifetimeEnds) {
    for (auto II2: LifetimeEnds) {
      if (II != II2 && II->getArgOperand(1) == II2->getArgOperand(1)) {
        return toReturn;
      }
    }
  }

  for (auto II: LifetimeStarts) {
    for (auto II2: LifetimeEnds) {
      if (II->getArgOperand(1) == II2->getArgOperand(1)) {
        // verify that the lifetime start dominates the lifetime end
        if (DT.dominates(II, II2)) {
            toReturn.push_back(II);
        }
      }
    }
  }
  return toReturn;
}

bool Bundler::removeLifetimeIntrinsics() {
  // Although they have no effect at runtime and are always safe to discard,
  // LLVM must treat llvm.lifetime.*() intrinsics as having side effects in
  // memory to prevent reordering them with respect to some memory operations.
  // They thus present barriers to hoisting/sinking some code, so we remove
  // them to stop them from interfering with later SCC passes.  After this pass
  // eliminates allocas, these lifetime markers are of little value anyway, as
  // the major user of lifetime markers in CodeGen is StackColoring.

  bool containedLifetimeIntrinsic = false;

  for (BasicBlock &BB : F) {
    auto I = BB.begin(), E = BB.end();
    while (I != E) {
      if (auto *II = dyn_cast<IntrinsicInst>(&*(I++))) {
        if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
            II->getIntrinsicID() == Intrinsic::lifetime_end) {
          containedLifetimeIntrinsic = true;
          II->eraseFromParent();
        }
      }
    }
  }

  return containedLifetimeIntrinsic;
}

class Bundling : public FunctionPass {
public:
  static char ID;

  Bundling() : FunctionPass(ID) {
    initializeBundlingPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    if (!F.hasFnAttribute(SwarmFlag::Parallelizable))
      return false;

    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &ORE = getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

    return Bundler(F, DT, LI, TLI, ORE).bundle();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
    AU.setPreservesCFG();
  }
};


} // end anonymous namespace

char Bundling::ID = 0;

INITIALIZE_PASS_BEGIN(Bundling, DEBUG_TYPE,
                      "Bundle stack allocations",
                      false,
                      false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_END(Bundling, DEBUG_TYPE,
                    "Bundle stack allocations",
                    false, false)

Pass *llvm::createBundlingPass() {
  return new Bundling();
}

