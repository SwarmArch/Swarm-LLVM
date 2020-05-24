//===-- Tasks.cpp - Tools for tasks and domains ----------------*- C++ -*--===//
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
// Utility methods for manipulating tasks represented inline within parent CFGs
// using a Tapir-like detach-reattach representation.  Includes utilities for
// handling Fractal domains (deepen and undeepen intrinisics) as well.
//
//===----------------------------------------------------------------------===//


#include "Tasks.h"

#include "CFGRegions.h"
#include "Misc.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/SwarmAA.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Swarm.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "tasks"

// This used to be a public utility for when detachTask() isn't a good fit.
// You could manually set up the detach/reattach basic block structure,
// then use this to replace branches with a sdetach and sreattach.
// This has been deprecated pending any use cases where this is needed.
static void replaceWithSDetachAndSReattach(BranchInst *DetachFrom,
                                           BranchInst *ReattachFrom,
                                           Value *Timestamp,
                                           DeepenInst *Domain,
                                           DetachKind = DetachKind::Unknown,
                                           DominatorTree *DT = nullptr,
                                           SDetachInst **rDI = nullptr,
                                           SReattachInst **rRI = nullptr);


void llvm::detachTask(
        Instruction *TaskStart, // inclusive
        Instruction *TaskEnd, // exclusive
        Value *Timestamp,
        DeepenInst *Domain,
        DetachKind Kind,
        const Twine& Name,
        DominatorTree *DT,
        LoopInfo *LI,
        SDetachInst **rDI,
        SReattachInst **rRI) {
  assert(TaskStart != TaskEnd);
  assert(!DT || DT->dominates(TaskStart, TaskEnd));

  BasicBlock *Last = TaskEnd->getParent();
  BasicBlock *Prev = TaskStart->getParent();

  assert((!LI || (*LI)[Last] == (*LI)[Prev]) &&
         "Start and end in different loops; requested LoopInfo update unclear");

  // TaskStart and TaskEnd could be in the same block,
  // so I'm splitting at the (later) reattach point first, and then splitting
  // at the (earlier) detach point.
  BasicBlock *ReattachToBB = SplitBlock(Last, TaskEnd, DT, LI);
  BranchInst *ReattachFrom = cast<BranchInst>(Last->getTerminator());
  BasicBlock *DetachToBB = SplitBlock(Prev, TaskStart, DT, LI);
  BranchInst *DetachFrom = cast<BranchInst>(Prev->getTerminator());

  assert(DetachFrom->getSuccessor(0) == TaskStart->getParent());
  assert(ReattachFrom->getSuccessor(0) == TaskEnd->getParent());

  if (Name.isTriviallyEmpty()) {
    DetachToBB->setName(Prev->getName() + ".det");
    ReattachToBB->setName(Prev->getName() + ".re");
  } else {
    DetachToBB->setName(Name + ".det");
    ReattachToBB->setName(Name + ".re");
  }

  replaceWithSDetachAndSReattach(DetachFrom, ReattachFrom, Timestamp, Domain,
                                 Kind,
                                 DT, rDI, rRI);

  assert(!TaskStart->getPrevNode());
  assert(!rDI ||
         TaskStart->getParent()->getSinglePredecessor() == (*rDI)->getParent());
  assert(!DT || (*DT)[Prev]->getNumChildren() == 2 &&
         "The detached and continue blocks should be immediate children");
  assert(!DT || (*DT)[ReattachToBB]->getIDom()->getBlock() == Prev);
}


static void replaceWithSDetachAndSReattach(
        BranchInst *DetachFrom,
        BranchInst *ReattachFrom,
        Value *Timestamp,
        DeepenInst *Domain,
        DetachKind Kind,
        DominatorTree *DT,
        SDetachInst **rDI,
        SReattachInst **rRI) {
  assert(DetachFrom && ReattachFrom);
  assert(DetachFrom->isUnconditional() &&
         ReattachFrom->isUnconditional() &&
         "Detach/Reattach sites are not unconditional branches");
  assert(DetachFrom->getNumSuccessors() == 1 &&
         ReattachFrom->getNumSuccessors() == 1 &&
         "Detach site has > one succesor");

  BasicBlock *DetachFromBB = DetachFrom->getParent();
  BasicBlock *DetachToBB = DetachFrom->getSuccessor(0);
  BasicBlock *ContinueBB = ReattachFrom->getSuccessor(0);

  assert(DetachFromBB && DetachToBB && ContinueBB);
  assert(DetachFromBB != DetachToBB && "DetachFrom was a loop backedge!");
  assert(DetachToBB != ContinueBB && "ReattachFrom was a backedge?");
  assert(DetachFromBB != ContinueBB && "No creating loops");

  // It doesn't make sense to be performing DominatorTree updates unless
  // these nodes are actually in the DominatorTree.
  assert(!DT || DT->isReachableFromEntry(DetachFromBB));
  assert(!DT || DT->isReachableFromEntry(DetachToBB));
  assert(!DT || DT->isReachableFromEntry(ContinueBB));

  // Invariant inherited from Tapir: the detach must dominate the detached
  // and reattaching blocks.
  assert(!DT || DT->properlyDominates(DetachFromBB, DetachToBB));
  assert(!DT || DT->dominates(DetachToBB, ReattachFrom->getParent()));

  SDetachInst *DI = SDetachInst::Create(Timestamp, DetachToBB, ContinueBB);
  DI->setDomain(Domain);
  setDetachKind(DI, Kind);
  SReattachInst *RI = SReattachInst::Create(ContinueBB);

  ReplaceInstWithInst(DetachFrom, DI);
  ReplaceInstWithInst(ReattachFrom, RI);

  // victory's thoughts on updating the dominator tree.
  // Consider the following CFG:
  //
  // Entry --> DetachingBlock -detach-edge-> Detached -reattach-edge->
  //      \                  \_____________continue_edge_____________> Continue
  //       \_____________some_other_pre-existing_edge________________>
  //
  // that is, when the continue-edge, reattach-edge, and the other pre-existing
  // edge all converge on the Continue block. In this case, Continue's immediate
  // dominator was and should remain as Entry.
  //
  // In general, when the DetachInst is inserted in place of a branch, the only
  // change to the CFG structure is the insertion of the continue-edge which
  // goes directly from DetachingBlock to Continue. This means that only if
  // Continue's was anywhere in the subtree of the dominator tree rooted at
  // DetachingBlock, it should now be hoisted up so that it is an immediate
  // child of DetachingBlock.
  //
  // Through the detachTask utility, the Continue block is formed right before
  // the function is called by splitting, so in these use cases the Continue is
  // dominated by the detach. However, if Continue was reachable (as in the
  // above example) by some other path that does not go through the detach, then
  // its immediate dominator must remain whatever ancestor/dominator that the
  // detach and Continue share, in this case, Entry.
  if (DT && DT->dominates(DI, ContinueBB)) {
    DT->changeImmediateDominator(ContinueBB, DI->getParent());
  }
  assert(!DT || !DT->dominates(DI->getDetached(), ContinueBB));

  assert(!DT || areMatching(DI, RI, *DT) && "Utils out of sync?");

  if (rDI) *rDI = DI;
  if (rRI) *rRI = RI;
}


DeepenInst *llvm::DeepenInst::Create(const Twine &Name,
                                     Instruction *InsertBefore) {
  return cast<DeepenInst>(IRBuilder<>(InsertBefore).CreateCall(
      Intrinsic::getDeclaration(InsertBefore->getModule(), Intrinsic::deepen),
      {},
      Name));
}


UndeepenInst *llvm::UndeepenInst::Create(DeepenInst *Deepen,
                                         Instruction *InsertBefore) {
  return cast<UndeepenInst>(IRBuilder<>(InsertBefore).CreateCall(
      Intrinsic::getDeclaration(InsertBefore->getModule(), Intrinsic::undeepen),
      {Deepen}));
}


const DeepenInst *llvm::DeepenInst::getSuperdomain(const DominatorTree &DT) const {
  return getDomain(getEnclosingTask(this, DT));
}
DeepenInst *llvm::DeepenInst::getSuperdomain(const DominatorTree &DT) {
  return getDomain(getEnclosingTask(this, DT));
}


const DeepenInst *llvm::UndeepenInst::getMatchingDeepen() const {
  return cast<DeepenInst>(getArgOperand(0));
}
DeepenInst *llvm::UndeepenInst::getMatchingDeepen() {
  return cast<DeepenInst>(getArgOperand(0));
}


int llvm::getDomainDepthDiff(const SDetachInst *Task, const DominatorTree &DT,
                             const SDetachInst *ParentTask) {
  assert(!Task->isSubdomain() &&
         "We're not currently using the isSubdomain flag");
  if (!ParentTask) ParentTask = getEnclosingTask(Task, DT);
  const DeepenInst *Domain = getDomain(Task);
  const DeepenInst *ParentDomain = getDomain(ParentTask);

  // First check if this spawn is deepened.
  if (const DeepenInst *Subdomain =
          getPreceedingDeepen(Task, DT, ParentTask->getParent())) {
    assert(DT.dominates(ParentTask->getDetached(), Subdomain->getParent()));
    assert(Subdomain->getSuperdomain(DT) == ParentDomain);
    if (const UndeepenInst *Undeepen = getPreceedingUndeepen(Task, DT, ParentTask->getParent())) {
      assert(Undeepen->getMatchingDeepen() == Subdomain);
      assert(DT.dominates(Subdomain, Undeepen));
      // The deepen and undeepen cancel out.
    } else {
      // This spawn is deepened without undeepen.
      if (Task->isSuperdomain()) {
        assert(Domain == ParentDomain &&
               "Superdomain spawn after deepen goes to the original domain");
        return 0;
      } else {
        assert(Domain != ParentDomain);
        assert(Domain == Subdomain);
        return +1; // ordinary spawn to subdomain
      }
      llvm_unreachable("return statements above");
    }
  }
  // We continue below if this spawn is not deepened.

  if (Task->isSuperdomain()) {
    assert(!Domain || ParentDomain);
    assert(!ParentDomain || ParentDomain->getSuperdomain(DT) == Domain);
    return -1; // ordinary spawn to superdomain
  } else {
    assert(Domain == ParentDomain &&
           "This is an ordinary spawn to the same domain with no deepening");
    return 0;
  }
}


bool llvm::hasAnySDetachInst(const Function &F) {
  for (const BasicBlock &BB : F) {
    if (isa<SDetachInst>(BB.getTerminator())) {
      return true;
    }
  }
  return false;
}


const SDetachInst *llvm::getEnclosingTask(const BasicBlock *BB,
                                          const DominatorTree &DT) {
  const auto *Node = DT.getNode(const_cast<BasicBlock *>(BB));
  assert(Node);
  while ((Node = Node->getIDom()))
    if (auto *DI = dyn_cast<SDetachInst>(Node->getBlock()->getTerminator()))
      if (DT.dominates(DI->getDetached(), BB))
        return DI;
  return nullptr;
}


template<class InstType>
static const InstType *getPreceedingInst(const Instruction *I,
                                         const DominatorTree &DT,
                                         const BasicBlock *UpTo) {
  assert(I->getParent() != UpTo);
  assert(!UpTo || DT.properlyDominates(UpTo, I->getParent()));

  const Instruction *II = I;
  while ((II = II->getPrevNode())) {
    if (auto *Domain = dyn_cast<InstType>(II))
      return Domain;
  }

  const auto *Node = DT.getNode(const_cast<BasicBlock *>(I->getParent()));
  while ((Node = Node->getIDom())) {
    const BasicBlock *BB = Node->getBlock();
    if (BB == UpTo) break;
    assert(!UpTo || DT.properlyDominates(UpTo, BB));
    for (const Instruction &II : reverse(*BB))
      if (auto *Domain = dyn_cast<InstType>(&II))
        return Domain;
  }

  return nullptr;
}

const DeepenInst *llvm::getPreceedingDeepen(const Instruction *I,
                                            const DominatorTree &DT,
                                            const BasicBlock *UpTo) {
  return getPreceedingInst<DeepenInst>(I, DT, UpTo);
}

const UndeepenInst *llvm::getPreceedingUndeepen(const Instruction *I,
                                                const DominatorTree &DT,
                                                const BasicBlock *UpTo) {
  return getPreceedingInst<UndeepenInst>(I, DT, UpTo);
}


void llvm::getNonDetachDescendants(const DominatorTree &DT,
                                   BasicBlock *R,
                                   SmallVectorImpl<BasicBlock *> &Result) {
  // This works similarly to DominatorTree::getDescendants().
  Result.clear();
  const DomTreeNodeBase<BasicBlock> *RN = DT.getNode(R);
  if (!RN)
    return; // If R is unreachable, it will not be present in the DOM tree.
  SmallVector<const DomTreeNodeBase<BasicBlock> *, 8> WL;
  WL.push_back(RN);

  while (!WL.empty()) {
    const DomTreeNodeBase<BasicBlock> *Node = WL.pop_back_val();
    BasicBlock *NodeBlock = Node->getBlock();
    Result.push_back(NodeBlock);
    if (SDetachInst *DI = dyn_cast<SDetachInst>(NodeBlock->getTerminator())) {
      // The only normal children should be the Detached and Continue blocks,
      // but there could also be additional error-handling blocks shared by
      // the detached region and continuation, which we will exclude.
      assert(is_contained(Node->getChildren(), DT[DI->getDetached()]));
      auto ChildIt = find_if(Node->getChildren(),
              [DI](DomTreeNodeBase<BasicBlock> *Child) {
                return Child->getBlock() == DI->getContinue(); });
      if (ChildIt != Node->end())
        WL.push_back(*ChildIt);
    } else {
      WL.append(Node->begin(), Node->end());
    }
  }
}


/// Populate OuterDetaches with detach insturctions from the dominator subtree
/// rooted at Start, ignoring any nested subtrees for whose root isEnd returns
/// true, as well as subtrees corresponding to nested inner detached regions.
/// We only count detaches with timestamps (as from Fractalization).
/// You should think of this as finding the top-level detaches in the CFG
/// region starting at and including Start, and ending at (but excluding)
/// the blocks for which isEnd returns true.
template<class EndPredicate, class OutputContainer>
static void getOuterDetachesHelper(
        BasicBlock *Start,
        EndPredicate isEnd,
        const DominatorTree &DT,
        OutputContainer &OuterDetaches) {
  assert(OuterDetaches.empty());

  // This code below is quite similar to that in getNonDetachDescendants().
  // TODO(victory): Dedupe some of this with getNonDetachDescendants?
  std::vector<BasicBlock *> NonDetachedBlocks; // FIFO queue for BFS of DT
  NonDetachedBlocks.push_back(Start);
  for (unsigned i = 0; i < NonDetachedBlocks.size(); ++i) {
    BasicBlock *BB = NonDetachedBlocks[i];
    if (isEnd(BB)) continue;
    if (auto DI = dyn_cast<SDetachInst>(BB->getTerminator())) {
      if (DI->hasTimestamp())
        OuterDetaches.push_back(DI);
      assert(is_contained(DT[BB]->getChildren(), DT[DI->getDetached()]));
      if (DT[BB]->getNumChildren() == 1) continue;
      assert(is_contained(DT[BB]->getChildren(), DT[DI->getContinue()]));
      NonDetachedBlocks.push_back(DI->getContinue());
      assert(all_of(DT[BB]->getChildren(), [DI](const DomTreeNode *N) {
        return N->getBlock() == DI->getContinue() ||
               N->getBlock() == DI->getDetached() ||
               none_of(depth_first(N), [](const DomTreeNode *N) {
                 return isa<SDetachInst>(N->getBlock()->getTerminator());
               });
      }));
    } else {
      for (DomTreeNode *Child : *DT[BB])
        NonDetachedBlocks.push_back(Child->getBlock());
    }
  }
}


void llvm::getOuterDetaches(
        Function *F,
        const DominatorTree &DT,
        SmallVectorImpl<SDetachInst *> &OuterDetaches) {
  getOuterDetachesHelper(
          &F->getEntryBlock(),
          [](const BasicBlock *BB) { return false; },
          DT,
          OuterDetaches);
}


void llvm::getOuterDetaches(
        Loop &L,
        const DominatorTree &DT,
        SmallVectorImpl<SDetachInst *> &OuterDetaches) {
  getOuterDetachesHelper(
          L.getHeader(),
          [&L](const BasicBlock *BB) { return !L.contains(BB); },
          DT,
          OuterDetaches);
}


void llvm::getOuterDetaches(
        SDetachInst *DI,
        const DominatorTree &DT,
        SmallVectorImpl<SDetachInst *> &OuterDetaches) {
  getOuterDetachesHelper(
          DI->getDetached(),
          [](const BasicBlock *BB) { return false; },
          DT,
          OuterDetaches);
}


void llvm::getOuterDetaches(
        const DeepenInst *Domain,
        const DominatorTree &DT,
        SmallVectorImpl<const SDetachInst *> &OuterDetaches) {
  getOuterDetachesHelper(
          const_cast<BasicBlock *>(Domain->getParent()),
          [Domain](const BasicBlock *BB) {
            return any_of(*BB, [Domain](const Instruction &I) {
              return isa<UndeepenInst>(I) &&
                     cast<UndeepenInst>(I).getMatchingDeepen() == Domain; }); },
          DT,
          OuterDetaches);
}


bool llvm::areMatching(const SDetachInst *DI,
                       const SReattachInst *RI,
                       const DominatorTree &DT) {
  const BasicBlock *ReattachBB = RI->getParent();
  return DI->getContinue() == RI->getDetachContinue()
         && DT.isReachableFromEntry(ReattachBB)
         && DT.dominates(DI->getDetached(), ReattachBB);
}


void llvm::getMatchingReattaches(
        const SDetachInst *DI,
        const DominatorTree &DT,
        SmallVectorImpl<const SReattachInst *> &MatchingReattaches) {
  for (const BasicBlock *ContPred : predecessors(DI->getContinue()))
    if (auto *RI = dyn_cast<SReattachInst>(ContPred->getTerminator()))
      if (areMatching(DI, RI, DT))
        MatchingReattaches.push_back(RI);
}


const SReattachInst *llvm::getUniqueMatchingSReattach(
        const SDetachInst *DI,
        const DominatorTree &DT) {
  const SReattachInst *UniqueRI = nullptr;
  for (const BasicBlock *ContPred : predecessors(DI->getContinue()))
    if (auto *RI = dyn_cast<SReattachInst>(ContPred->getTerminator()))
      if (areMatching(DI, RI, DT)) {
        if (UniqueRI) return nullptr;
        else UniqueRI = RI;
      }

  return UniqueRI;
}


// FIXME(mcj) This function should *require* a DominatorTree, but the
// LoopExpansion late-chain caller doesn't have access to one ...
// TODO(mcj) For many of these assertions, we could opt to early exit
void llvm::hoistDetach(
        SDetachInst *const DI,
        Instruction *MoveBefore,
        DominatorTree *const DT,
        LoopInfo *const LI) {
  DEBUG(dbgs() << "Hoisting " << *DI << '\n');
  DEBUG(if (const DebugLoc &Loc = DI->getDebugLoc()) {
      dbgs() << "  from "; Loc.print(dbgs()); dbgs() << '\n';
  });
  DEBUG(dbgs() << "  before " << *MoveBefore << '\n');
  DEBUG(if (const DebugLoc &Loc = MoveBefore->getDebugLoc()) {
      dbgs() << "  from "; Loc.print(dbgs()); dbgs() << '\n';
  });

  assert(!isa<PHINode>(MoveBefore));
  assert(DI != MoveBefore && "Don't split blocks unnecessarily");
  assert((!LI || (*LI)[DI->getParent()] == (*LI)[MoveBefore->getParent()]) &&
         "For now we do not hoist across loops. Maybe this can be revisited");

  BasicBlock *BeforeMoveBefore = MoveBefore->getParent();
  SplitBlock(MoveBefore->getParent(), MoveBefore, DT, LI);
  BasicBlock *BeforeDI = DI->getParent();
  SplitBlock(DI->getParent(), DI, DT, LI);
  assert(DI->getParent()->size() == 1);

  BasicBlock *DIContinue = DI->getContinue();
  assert(none_of(DIContinue->phis(), [](PHINode&){ return true; }) &&
         "We expect the continue block after the DI detached region "
         "to be absent of any PHINodes. This is a simplifying assumption "
         "that happens to be true for current use cases. It will not hold "
         "as this utility is further generalized for arbitrary detaches with "
         "PHIs in the continue block");
  // The transformation is incorrect if DIContinue has predecessors other than
  // DI and its matching reattaches.
  // e.g., if DIContinue is a loop (header)
  // SplitBlockPredecessors might be part of the solution.
  // FIXME(mcj) The following assertion is still too weak, as it does not
  // confirm that the SReattachInsts are affiliated with DI.
  assert(std::all_of(pred_begin(DIContinue), pred_end(DIContinue),
          [DI] (const BasicBlock *Pred) {
            const TerminatorInst *TI = Pred->getTerminator();
            return TI == DI || isa<SReattachInst>(TI);
          }) &&
          "This transformation assumes the given SDetachInst's continue block "
          "is preceded only by the SDetachInst and its "
          "corresponding reattahes.");
  BasicBlock *Continuation = SplitBlock(DIContinue,
                                        &DIContinue->front(),
                                        DT, LI);
  assert(DIContinue->size() == 1);

  // Clarify the expected state of dominance before transforming
  // (even if some of these assertions are trivially true).
  assert(BeforeMoveBefore == MoveBefore->getParent()->getUniquePredecessor());
  assert(!DT || DT->dominates(MoveBefore->getParent(), BeforeDI));
  assert(BeforeDI == DI->getParent()->getUniquePredecessor());
  assert(!DT || DT->properlyDominates(DI->getParent(), DIContinue));
  assert(DIContinue == Continuation->getUniquePredecessor());
  // TODO(mcj) Move DI's timestamp above MoveBefore. If it depends on
  // MoveBefore's timestamp computation, place it appropriately. Otherwise we
  // can actually hoist DI above MoveBefore's timestamp computation.
  assert((isa<Constant>(DI->getTimestamp()) || !DT ||
          DT->dominates(cast<Instruction>(DI->getTimestamp()), MoveBefore)) &&
         "Detach hoisting does not currently support "
         "moving the hoistee's timestamp computation");
  assert(BeforeMoveBefore->getSingleSuccessor() == MoveBefore->getParent());
  assert(DIContinue->getSingleSuccessor() == Continuation);
  assert(BeforeDI->getSingleSuccessor() == DI->getParent());

  BeforeMoveBefore->getTerminator()->setSuccessor(0, DI->getParent());
  DIContinue->getTerminator()->setSuccessor(0, MoveBefore->getParent());
  BeforeDI->getTerminator()->setSuccessor(0, Continuation);

  if (DT) {
    DT->changeImmediateDominator(DI->getParent(), BeforeMoveBefore);
    DT->changeImmediateDominator(MoveBefore->getParent(), DIContinue);
    DT->changeImmediateDominator(Continuation, BeforeDI);
  }

  // Clarify the expected state of dominance after transforming
  assert(BeforeMoveBefore == DI->getParent()->getUniquePredecessor());
  assert(!DT || DT->properlyDominates(DI->getParent(), DIContinue));
  assert(DIContinue == MoveBefore->getParent()->getUniquePredecessor());
  assert(!DT || DT->dominates(MoveBefore->getParent(), BeforeDI));
  assert(BeforeDI == Continuation->getUniquePredecessor());
}


/// Of all inputs to DI's detached block, return the lowest instruction-based
/// input. If all inputs are non-instructions, return nullptr
static const Instruction *getLowestInput(
        const SDetachInst *DI,
        const DominatorTree &DT) {
  // Gather the detached block's inputs derived from instructions
  SmallVector<const Instruction *, 4> InstructionInputs;
  {
    SmallVector<BasicBlock *, 8> Blocks;
    DT.getDescendants(DI->getDetached(), Blocks);
    SmallSetVector<BasicBlock *, 8> BlockSet(Blocks.begin(), Blocks.end());
    SetVector<Value *> Inputs;
    findInputsNoOutputs(BlockSet, Inputs);
    Inputs.insert(const_cast<Value *>(DI->getTimestamp()));
    if (DI->hasHint())
      Inputs.insert(const_cast<Value *>(DI->getHint()));
    for (Value *V : Inputs)
      if (!isa<Constant>(V) && !isa<Argument>(V))
        // TODO(mcj) we may need to handle other Value base classes?
        InstructionInputs.push_back(cast<Instruction>(V));
  }

  if (InstructionInputs.empty()) return nullptr;

  // All remaining inputs must be instructions and must dominate DI;
  // Find the Lowest.
  const Instruction *Lowest = InstructionInputs[0];
  for (const Instruction *Input : InstructionInputs) {
    if (Input == Lowest) continue;
    auto *I = DT.dominates(Lowest, Input) ? Input
              : DT.dominates(Input, Lowest) ? Lowest
              : nullptr;
    if (I) Lowest = I;
    assert((I ||
            (isa<PHINode>(Input) &&
             isa<PHINode>(Lowest) &&
             Input->getParent() == Lowest->getParent())) &&
           "Either of DI's inputs should dominate the other. "
           "DominatorTree::dominates says PHINodes from the same block "
           "do not dominate each other. "
           "We don't care which one we take in that case");
  }
  assert(DT.dominates(Lowest->getParent(), DI->getParent()));
  return Lowest;
}


const Instruction *llvm::getEarliestHoistPoint(
        const SDetachInst *const DI,
        const DominatorTree &DT,
        const LoopInfo &LI) {
  DEBUG(dbgs() << "Finding earliest safe hoist point of\n  " << *DI << "\n");

  const Instruction *UpperBound = getLowestInput(DI, DT);
  if (!UpperBound) UpperBound = &DI->getFunction()->getEntryBlock().front();

  // Now that we know the lowest input to DI, we can only hoist it as far as we
  // don't change control flow. Traverse up the predecessor chain until we find
  // (i) UpperBoundBB,
  // (ii) a deepen or undeepen instruction, or
  // (iii) a scary predecessor (branching and nontrivial-looping)
  // TODO(mcj) Generalize to use PostDominatorTrees and more cleverly hop over
  // single-entry-single-exit control flow graphs.
  const BasicBlock *UpperBoundBB = UpperBound->getParent();
  const BasicBlock *BB = DI->getParent();
  while (BB != UpperBoundBB) {
    auto II = find_if(*BB, [] (const Instruction &I) {
              return isa<DeepenInst>(I) || isa<UndeepenInst>(I);
            });
    if (II != BB->end()) {
      // Hoist no higher than a deepen or undeepen instruction,
      // as these would change the semantics of the detach.
      // TODO(mcj) handle a single level of Undeepen-then-Deepen instructions
      // like a single-entry stack. Upon finding *one* UndeepenInst, we could
      // change the detach of interest to a super enqueue. If we then
      // encountered a DeepenInst, we can ignore both.
      UpperBound = &*II;
      UpperBoundBB = UpperBound->getParent();
      DEBUG(dbgs() << "  Hoisting stopped at " << *UpperBound << '\n');
    } else if (const BasicBlock *PredBB = BB->getSinglePredecessor()) {
      const BranchInst *BI = dyn_cast<BranchInst>(PredBB->getTerminator());
      if (!BI) {
        // The sole predecessor has a scary terminator. Give up here.
        break;
      } else if (!BI->isConditional()) {
        // Trivially hop up
        BB = PredBB;
      } else if (LI.isLoopHeader(PredBB)) {
        // BB might be the unique exit block of a single-block loop
        const BasicBlock *Header = PredBB;
        const Loop *L = LI.getLoopFor(Header);
        assert(L && Header == L->getHeader());
        const BasicBlock *Preheader = L->getLoopPreheader();
        if (!Preheader) break;
        if (Header != L->getLoopLatch()) break; // Not a single-block loop
        if (BB != L->getUniqueExitBlock()) break;
        // BB is the unique exit block of a single-block loop.
        // Hop up to the preheader.
        // Although the loop might contain Deepen/Undeepen instructions, they
        // ought to appear as a pair that the detach can be safely hoisted past.
        // Otherwise the loop is likely malformed, deepening with every loop
        // iteration.
        //
        // This rather special case is was inspired by 482.sphinx3/cont_mgau.c
        // https://github.mit.edu/swarm/benchmarks/blob/fdc46bca7994757faa4d216e2ad9f6aebf13b1ef/speccpu2006/482.sphinx3/cont_mgau.c#L666-L670
        // TODO(mcj) should these assertions be weakened in the future?
        assert(DT.dominates(UpperBoundBB, Preheader));
        assert(DT.dominates(Preheader, DI->getParent()));
        BB = Preheader;
        DEBUG(dbgs() << "  Hoistable above single-block " << *L // Loops end with \n
                     << "  from ");
        DEBUG(L->getStartLoc().print(dbgs()));
        DEBUG(dbgs() << '\n'; );
      } else {
        break;
      }
    } else if (std::distance(pred_begin(BB), pred_end(BB)) == 2) {
      // BB might be the Continue block of a straightforward detached region
      auto PI = find_if(predecessors(BB), [] (const BasicBlock *B) {
                return isa<SDetachInst>(B->getTerminator());
              });
      if (PI == pred_end(BB)) break; // Nevermind, neither predecessor detached
      const SDetachInst *PredDI = cast<SDetachInst>((*PI)->getTerminator());
      const SReattachInst *PredRI = getUniqueMatchingSReattach(PredDI, DT);
      if (!PredRI) break;
      // Now we know PredDI dominates DI, and its detached region has only
      // one exit that also dominates DI
      assert(is_contained(predecessors(BB), PredDI->getParent()));
      assert(is_contained(predecessors(BB), PredRI->getParent()));
      assert(DT.dominates(UpperBound, PredDI));
      assert(DT.dominates(PredDI, DI));
      BB = PredDI->getParent();
      DEBUG(dbgs() << "  Hoistable above " << *PredDI
                   << "\n  from ");
      DEBUG(PredDI->getDebugLoc().print(dbgs()));
      DEBUG(dbgs() << '\n'; );
    } else {
      // TODO(mcj) handle the case where a block has multiple predecessors
      // whose subgraph exits only to the block and has a common dominator.
      // Solving this case will probably obviate the ugliness above
      break;
    }
  }

  assert((!isa<BranchInst>(UpperBound) ||
          UpperBound->getParent() == &DI->getFunction()->getEntryBlock()) &&
         "UpperBound is strictly higher than the safe hoist point, unless "
         "UpperBound is the sole (branch) instruction in the function's entry "
         "block. We handle the latter case specially below");

  const Instruction *HoistPt = (BB != UpperBoundBB || isa<PHINode>(UpperBound))
          ? BB->getFirstNonPHIOrDbg()
          : isa<BranchInst>(UpperBound)
            ? UpperBound
            : UpperBound->getNextNode();
  assert(HoistPt);

  // Skip llvm.debug.values to keep them close to the values they represent.
  while (isa<DbgValueInst>(HoistPt)) HoistPt = HoistPt->getNextNode();
  DEBUG(dbgs() << "  Hoistable upper bound " << *UpperBound
               << "\n  vs earliest hoistable point " << *HoistPt << '\n');
  assert(DT.dominates(UpperBound, HoistPt));
  return HoistPt;
}


void llvm::setCacheLineHintFromAddress(SDetachInst *DI, Value *Address) {
  DEBUG(dbgs() << "Creating cache-line hint for " << *DI << "\n  from ");
  DEBUG(DI->getDebugLoc().print(dbgs()));
  DEBUG(dbgs() << "\n  using address " << *Address << '\n');

  // We could remove this assertion, but it might catch accidents
  assert(DI->isNoHint() && "Did you mean to overwrite the hint?");

  IRBuilder<> Builder(DI);
  if (Address->getType()->isPointerTy())
    Address = Builder.CreatePointerCast(Address,
                                        Builder.getInt64Ty(),
                                        "hint_int_addr");
  assert(Address->getType()->isIntegerTy(64));

  ConstantInt *CacheLine = Builder.getInt64(SwarmCacheLineSize);
  Value *Hint = Builder.CreateUDiv(Address, CacheLine, "hint");
  DI->setHint(Hint);
}


bool llvm::isExpandableSwarmLoop(const Loop *L, const DominatorTree &DT) {
  DEBUG(dbgs() << "SU: Checking if expandable Swarm loop:\n" << *L);
  const BasicBlock *Header = L->getHeader();

  // 1. Check if Loop's internal control flow graph has needed shape.

  const SDetachInst *DI = dyn_cast<SDetachInst>(Header->getTerminator());
  if (!DI) {
    DEBUG(dbgs() << "loop header not terminated by a detach:\n");
    return false;
  }
  if (!DI->hasTimestamp()) {
    DEBUG(dbgs() << "loop header detach lacks timestamp.\n");
    return false;
  }

  const BasicBlock *Latch = L->getLoopLatch();
  if (!Latch) {
    DEBUG(dbgs() << "loop does not have a unique latch:\n");
    return false;
  }

  if (DI->getContinue() != Latch) {
    DEBUG(dbgs() << "continuation of detach is not the latch:\n");
    return false;
  }

  const SReattachInst *RI = getUniqueSReattachIntoLatch(*L);
  if (!RI) {
    DEBUG(dbgs() << "latch does not have unique reattach predecessor\n");
    return false;
  }
  assert(RI->getDetachContinue() == Latch);
  assert(!isa<PHINode>(Latch->front()));
#ifndef NDEBUG
  const SReattachInst *MatchingReattach = getUniqueMatchingSReattach(DI, DT);
  assert(MatchingReattach && MatchingReattach == RI);
#endif

  // 2. All side effects are in the detached loop body,
  //    and the header must not access memory.

  if (any_of(*Header, [](const Instruction &I) {
              return I.mayReadOrWriteMemory() || I.mayThrow();
             })) {
    DEBUG(dbgs() << "loop header may access memory or throw exceptions.\n");
    return false;
  }

  if (any_of(*Latch, [](const Instruction &I) {
              return mayHaveSideEffects(&I); })) {
    DEBUG(dbgs() << "loop latch may have side effects.\n");
    return false;
  }

  // 3. The loop must have a continuation with no data or control dependencies
  //    on the loop, i.e., there must be a unique non-deadend exit block that
  //    that does not use any values produced in the loop.
  //    Additionally, the unique non-dead-end exit must be reached only by
  //    a simple conditional branch in the latch.

  const auto *LatchBr = dyn_cast<BranchInst>(Latch->getTerminator());
  if (!LatchBr || LatchBr->isUnconditional()) {
    DEBUG(dbgs() << "loop latch does not end in conditional branch:\n");
    return false;
  }

  // Get the exit block from Latch.
  const BasicBlock *Exit = LatchBr->getSuccessor(0);
  if (Header == Exit)
    Exit = LatchBr->getSuccessor(1);
  assert(!L->contains(Exit));

  if (Exit != getUniqueNonDeadendExitBlock(*L)) {
    DEBUG(dbgs() << "loop does not have a unique non-dead-end exit\n");
    return false;
  }

  if (!L->hasDedicatedExits()) {
    DEBUG(dbgs() << "loop has non-dedicated exits\n");
    return false;
  }

  if (any_of(predecessors(Exit), [Latch](const BasicBlock *Pred) {
          return Pred != Latch; })) {
    DEBUG(dbgs() << "loop exits other than through the latch\n");
    return false;
  }

  // This condition bears some similarity to Loop::isLCSSAForm()
  if (any_of(L->blocks(), [L](const BasicBlock *BB) {
        return any_of(*BB, [L](const Instruction &I) {
          return any_of(I.uses(), [L](const Use &U) {
            const Instruction *UI = cast<Instruction>(U.getUser());
            const BasicBlock *UserBB = UI->getParent();
            if (const PHINode *P = dyn_cast<PHINode>(UI))
              UserBB = P->getIncomingBlock(U);

            return (!L->contains(UserBB) && !isDoomedToUnreachableEnd(UserBB));
          });
        });
      })) {
    DEBUG(dbgs() << "loop continuation uses values produced in loop\n");
    return false;
  }

  // 4. LoopExpansion will outline the loop, creating inlinable calls that
  // require DebugLocs. It will depend on finding the DebugLocs it needs at the
  // detach and at the loop latch.
  if (Header->getParent()->getSubprogram() && !DI->getDebugLoc()) {
    DEBUG(dbgs() << "Loop body detach needs debug info\n");
    return false;
  }
  if (Header->getParent()->getSubprogram() && !LatchBr->getDebugLoc()) {
    DEBUG(dbgs() << "Loop backedge needs debug info\n");
    return false;
  }

  DEBUG(dbgs() << "yes!\n");
  assert(isExpandableLoopLatch(LatchBr));
  return true;
}


const SReattachInst *llvm::getUniqueSReattachIntoLatch(const Loop &L) {
  const BasicBlock *Header = L.getHeader();
  const BasicBlock *Latch = L.getLoopLatch();
  assert(Header && Latch);

  if (std::distance(pred_begin(Latch), pred_end(Latch)) != 2)
    return nullptr;

  auto PredIter = pred_begin(Latch);
  const BasicBlock *Pred1 = *(PredIter++);
  const BasicBlock *Pred2 = *(PredIter++);
  assert(Pred1 == Header || Pred2 == Header);

  const BasicBlock *ReattachBlock = (Pred1 == Header) ? Pred2 : Pred1;
  return dyn_cast<const SReattachInst>(ReattachBlock->getTerminator());
}


SDetachInst *llvm::createSuperdomainDetachedFree(Value *Ptr,
                                                 DeepenInst *Superdomain,
                                                 Instruction *InsertBefore,
                                                 DominatorTree *DT,
                                                 LoopInfo *LI) {
  Instruction *Free = CallInst::CreateFree(Ptr, InsertBefore);
  addSwarmMemArgsForceAliasMetadata(cast<CallInst>(Free));
  // Here we detach not just the free() CallInst, but also the pointer cast
  // that CreateFree created, if any.
  Instruction *TaskStart = Free;
  if (Instruction *PtrCast = Free->getPrevNode())
    if (all_of(PtrCast->users(),
               [Free](const User *Usr) { return Usr == Free; }))
      TaskStart = PtrCast;
  SDetachInst *DI;
  detachTask(TaskStart, InsertBefore,
             ConstantInt::getTrue(InsertBefore->getContext()), Superdomain,
             DetachKind::Free, "freeclosure",
             DT, LI, &DI);
  DI->setSuperdomain(true);
  DI->setRelativeTimestamp(true);
  return DI;
}


void llvm::eraseDetach(DetachInst *DI, DominatorTree &DT, LoopInfo *LI) {
  BasicBlock *Detached = DI->getDetached();
  BasicBlock *Continue = DI->getContinue();
  for (BasicBlock *ReattachBB : predecessors(Continue)) {
    if (!DT.dominates(Detached, ReattachBB)) continue;
    assert(all_of(Continue->phis(), [&](const PHINode &PN) {
      return PN.getIncomingValueForBlock(ReattachBB)
             == PN.getIncomingValueForBlock(DI->getParent());
    }) && "Detached region must not produce values used by the continuation");
  }
  ReplaceInstWithInst(DI, BranchInst::Create(DI->getContinue()));
  Function *F = Detached->getParent();
  eraseDominatorSubtree(Detached, DT, LI);
  // If the detach could reach some error-handling blocks that it does not
  // dominate, this erasure may now change the dominators for
  // those error-handling blocks.
  // TODO(victory): Do an incremental update of the dominator tree instead of
  // throwing it out and recalculating it.
  DT.recalculate(*F);
}
