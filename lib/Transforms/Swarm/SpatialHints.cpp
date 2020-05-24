//===- SpatialHints.cpp - Generate spatial hints --------------------------===//
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
// The SpatialHints pass adds hints to tasks.
//
//===----------------------------------------------------------------------===//

#include "Utils/Flags.h"
#include "Utils/Misc.h"
#include "Utils/Tasks.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/Swarm.h"

#define PASS_NAME "swarm-hints"
#define DEBUG_TYPE PASS_NAME

using namespace llvm;

static cl::opt<bool> DisableHints("swarm-disablehints",
        cl::init(false),
        cl::desc("Disable automatic generation of spatial hints"));

static cl::opt<bool> AggressiveAutoHints("aggressive-auto-hints",
        cl::init(false),
        cl::desc("Enables more aggressive automatic generation of spatial hints"
                " for speculative parallel programs. With aggressive hints, we "
                "make a single hint even though there may be multiple good "
                "hints."));


// Recursive helper function to gather all instructions that lead to the
// generation of Operand. Adds each to SafeOperands in depth-first order of
// discovery. Returns true if all relevant instructions are safe to be hoisted.
static bool getSafeOperands(const Value *Operand,
                            const SmallPtrSetImpl<const BasicBlock *> &Blocks,
                            SmallPtrSetImpl<const Instruction *> &SafeOperands) {
  DEBUG(dbgs() << "Looking at op: " << *Operand << "\n");
  if (isa<Constant>(Operand))
    return true;
  if (isa<Argument>(Operand))
    return true;
  const Instruction *Inst = cast<Instruction>(Operand);

  // If Inst dominates DI already, no moving needs to be done - unless we are
  // copying an address computation before a detach in a recursive spawn
  //assert (DT->dominates(Inst, DI) == !Blocks.count(Inst->getParent()));
  if (!Blocks.count(Inst->getParent())) {
    DEBUG(dbgs() << "  is instruction from outside of task.\n");
    return true;
  }

  // Don't hoist instructions that have side effects
  if (Inst->mayHaveSideEffects()) {
    DEBUG(dbgs() << "  is unsafe instruction.\n");
    return false;
  }

  bool IsSafe = true;
  if (isa<CastInst>(Inst) || isa<ExtractValueInst>(Inst))
    IsSafe &= getSafeOperands(Inst->getOperand(0), Blocks, SafeOperands);
  else if (isa<BinaryOperator>(Inst)) {
    // TODO(eforde): some binary operators may throw exceptions, like divide by zero
    IsSafe &= getSafeOperands(Inst->getOperand(0), Blocks, SafeOperands);
    IsSafe &= getSafeOperands(Inst->getOperand(1), Blocks, SafeOperands);
  } else if (isa<GetElementPtrInst>(Inst)) {
    for (const Value *InstOp : Inst->operand_values()) {
      IsSafe &= getSafeOperands(InstOp, Blocks, SafeOperands);
    }
  } else if (auto *LI = dyn_cast<LoadInst>(Inst)) {
    if (LI->getMetadata(SwarmFlag::Closure)) {
      const Value *Closure =
          LI->getPointerOperand()->stripInBoundsConstantOffsets();
      DEBUG(dbgs() << " is load from closure " << *Closure << "\n");
      assert(!isa<Constant>(Closure));
      IsSafe = isa<Argument>(Closure) ||
               !Blocks.count(cast<Instruction>(Closure)->getParent());
      DEBUG(if (!IsSafe) dbgs() << " Closure is inside of task.\n");
      if (IsSafe) {
        IsSafe &= getSafeOperands(LI->getPointerOperand(), Blocks, SafeOperands);
        assert(IsSafe);
      }
    } else {
      DEBUG(dbgs() << " is unsafe load.\n");
      IsSafe = false;
    }
  } else {
    DEBUG(dbgs() << "Unhandled instruction type in hint generation trace:\n  "
                 << *Inst << "\n");
    IsSafe = false;
  }

  if (IsSafe)
    SafeOperands.insert(Inst);

  return IsSafe;
}


// First checks if all instructions that lead to the generation of
// AddressComputation can be hoisted above DI. If so, hoists them and returns
// true, otherwise returns false.
static bool tryCreateHintGenerationFromAddress(
        Value *AddressComputation,
        const SmallVectorImpl<BasicBlock *> &Blocks,
        SDetachInst *DI) {
  const SmallPtrSet<const BasicBlock *, 8> BlockSet(Blocks.begin(), Blocks.end());
  SmallPtrSet<const Instruction *, 8> InstructionsToHoist;
  if (!getSafeOperands(AddressComputation, BlockSet, InstructionsToHoist))
    return false;

  for (BasicBlock *BB : Blocks) {
    for (auto II = BB->begin(), E = BB->end(); II != E;) {
      Instruction& I = *II++;
      if (InstructionsToHoist.count(&I)) {
        DEBUG(dbgs() << "Hoisting instruction for hint before detach: "
                     << I << '\n');
        I.moveBefore(DI);
      }
    }
  }
  return true;
}


// Returns true if there is definitely only one instruction from which we are
// able to create a hint.
static void getMemWrites(const SmallVectorImpl<BasicBlock *> &BBs,
                         SetVector<Value *> &PtrOps) {
  for (BasicBlock *BB: BBs)
    for (Instruction& I : *BB) {
      if (I.getMetadata(SwarmFlag::DoneFlag)) continue;
      if (I.getMetadata(SwarmFlag::Closure)) continue;
      // For StoreInsts whose PtrOperand is derived from an input, we can
      // find potentially a hint
      if (StoreInst *SInst = dyn_cast<StoreInst>(&I)) {
        DEBUG(dbgs() << "Considering store: " << *SInst << '\n');
        Value *PtrOp = SInst->getPointerOperand();
        PtrOps.insert(PtrOp);
      }
    }
}


static bool pointsWithinSameStruct(const Value *A, const Value *B) {
  auto *GEP_A = dyn_cast<GetElementPtrInst>(A);
  auto *GEP_B = dyn_cast<GetElementPtrInst>(B);
  if (!GEP_A || !GEP_B)
    return false;
  if (GEP_A->getNumIndices() != GEP_B->getNumIndices())
    return false;
  if (GEP_A->getPointerOperand() != GEP_B->getPointerOperand())
    return false;
  const Use *GEP_A_Idx = GEP_A->idx_begin();
  const Use *GEP_B_Idx = GEP_B->idx_begin();
  for (unsigned i = 0; i < GEP_A->getNumIndices() - 1;
       ++i, ++GEP_A_Idx, ++GEP_B_Idx)
    if (GEP_A_Idx->get() != GEP_B_Idx->get())
      return false;
  if (auto *Final_A_Idx = dyn_cast<ConstantInt>(GEP_A_Idx->get()))
    if (auto *Final_B_Idx = dyn_cast<ConstantInt>(GEP_B_Idx->get())) {
      uint64_t A_Idx = Final_A_Idx->getZExtValue();
      uint64_t B_Idx = Final_B_Idx->getZExtValue();
      if (B_Idx < A_Idx) // Ignore the higher index, use the lower one
        if (A_Idx - B_Idx < 16) { // Only consider small offsets.
          DEBUG(dbgs() << "Address " << *A << "\n  is just above " << *B << "\n");
          return true;
        }
    }
  return false;
}


static bool dependsOnLoadFrom(const Value *V, const Value *OtherPtr,
                              const BasicBlock *Header, const DominatorTree &DT) {
  const Instruction *I = dyn_cast<Instruction>(V);
  // Only consider instructions inside the task.
  if (!I || !DT.dominates(Header, I->getParent())) return false;

  bool Depends = false;
  if (auto *LI = dyn_cast<LoadInst>(I)) {
    if (LI->getPointerOperand() == OtherPtr) {
      Depends = true;
    }
  } else if (isa<CastInst>(I) || isa<ExtractValueInst>(I)) {
    Depends |= dependsOnLoadFrom(I->getOperand(0), OtherPtr, Header, DT);
  } else if (isa<BinaryOperator>(I)) {
    // TODO(eforde): some binary operators may throw exceptions, like divide by zero
    Depends |= dependsOnLoadFrom(I->getOperand(0), OtherPtr, Header, DT);
    Depends |= dependsOnLoadFrom(I->getOperand(1), OtherPtr, Header, DT);
  } else if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
    Depends |= dependsOnLoadFrom(GEP->getPointerOperand(), OtherPtr, Header, DT);
    for (Value *Idx : GEP->indices())
      Depends |= dependsOnLoadFrom(Idx, OtherPtr, Header, DT);
  }
  DEBUG(if (Depends) dbgs() << "Address computation " << *I
                            << "\n  depends on another accessed address "
                            << *OtherPtr << "\n";);
  return Depends;
}


static bool processDetach(SDetachInst *DI, DominatorTree &DT) {
  DEBUG({
    dbgs() << "Examining detach: " << *DI << "\n  from ";
    if (const DebugLoc &DL = DI->getDebugLoc())
      DL->print(dbgs());
    else
      dbgs() << "unknown location";
    dbgs() << '\n';
  });
  if (DI->isSameHint()) {
    DEBUG(dbgs() << "Detach is SAMEHINT.\n\n");
    return false;
  }
  if (DI->hasHint()) {
    DEBUG(dbgs() << "Detach already has hint.\n\n");
    return false;
  }
  DEBUG(dbgs() << "Attempting to find hint for detach...\n");

  BasicBlock *Spawned = DI->getDetached();

  SmallVector<BasicBlock *, 8> BBs;
  getNonDetachDescendants(DT, Spawned, BBs);
  DEBUG({
    dbgs() << "From detached blocks:\n";
    for (BasicBlock *BB : BBs) dbgs() << *BB << '\n';
  });

  SetVector<Value *> PtrOps;
  getMemWrites(BBs, PtrOps);

  SmallVector<Value *, 8> PtrsToRemove;
  for (Value *Ptr : PtrOps) {
    if (any_of(PtrOps, [Ptr, Spawned, &DT](const Value *OtherPtr) {
          return dependsOnLoadFrom(Ptr, OtherPtr, Spawned, DT) ||
                 pointsWithinSameStruct(Ptr, OtherPtr);
        })) {
      DEBUG(dbgs() << "Ignoring access to address " << *Ptr << "\n");
      PtrsToRemove.push_back(Ptr);
    }
  }
  for (Value *Ptr : PtrsToRemove)
    PtrOps.remove(Ptr);

  // If we don't want to be aggressive about finding hints, primitively check
  // that there is only one location from which we could find a hint.
  if (PtrOps.size() > 1) {
    DEBUG(dbgs() << "Too many potential hints (" << PtrOps.size()
                 << " store pointers).  Gave up finding hint.\n\n");
    return false;
  }
  if (PtrOps.size() == 1) {
    DEBUG(dbgs() << "Getting hint from single store address:"
                 << *PtrOps[0] << '\n');
    if (tryCreateHintGenerationFromAddress(PtrOps[0], BBs, DI)) {
      setCacheLineHintFromAddress(DI, PtrOps[0]);
      DEBUG(dbgs() << "Successfully set hint!\n\n");
      return true;
    } else {
      DEBUG(dbgs() << "Failed to set hint based on single store address.\n\n");
      return false;
    }
  }

  return false;
}


namespace {
class SpatialHints : public FunctionPass {
public:
  static char ID; // Pass ID, replacement for typeid
  SpatialHints() : FunctionPass(ID) {
    initializeSpatialHintsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};
} // end anonymous namespace

bool SpatialHints::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  if (DisableHints)
    return false;

  DEBUG(dbgs() << "\n\nGenerating spatial hints in " << F.getName() << "\n");

  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  //auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  //auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  //auto &ORE = getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

  bool Changed = false;

  for (DomTreeNode *Node : depth_first(&DT)) {
    if (auto *DI = dyn_cast<SDetachInst>(Node->getBlock()->getTerminator())) {
      Changed |= processDetach(DI, DT);
    }
  }

  assertVerifyFunction(F, "After SpatialHints pass", &DT);

  return Changed;
}

void SpatialHints::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominatorTreeWrapperPass>();
  //AU.addRequired<LoopInfoWrapperPass>();
  //AU.addRequired<ScalarEvolutionWrapperPass>();
  //AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
  AU.setPreservesCFG();
}

char SpatialHints::ID = 0;

INITIALIZE_PASS_BEGIN(SpatialHints, DEBUG_TYPE,
                      "Generate spatial hints",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
//INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
//INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
//INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_END(SpatialHints, DEBUG_TYPE,
                    "Generate spatial hints",
                    false, false)

Pass *llvm::createSpatialHintsPass() {
  return new SpatialHints();
}
