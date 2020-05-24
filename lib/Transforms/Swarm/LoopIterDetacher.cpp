//===- LoopIterDetacher.cpp - detach loop iterations as tasks -------------===//
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
// Attempt to identify as much of the code within a loop as possible, including
// all code with side effects, as the "loop body", to be detached as a separate
// task for each iteration of the loop.
// Keep the minimal amount of "bookkeeping" code outside the loop body.
// This at least attempts to pipeline-parallelize the iterations of a loop,
// even if the bookkeeping to spawn the iterations is executed serially.
//
//===----------------------------------------------------------------------===//

#include "LoopIterDetacher.h"

#include "Utils/CFGRegions.h"
#include "Utils/Flags.h"
#include "Utils/Misc.h"
#include "Utils/Reductions.h"
#include "Utils/Tasks.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/SwarmAA.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Swarm.h"

using namespace llvm;

#define PASS_NAME "fractalization"
#define DEBUG_TYPE PASS_NAME

STATISTIC(LoopsCanonicalized,
          "Number of serial loops transformed into canonical Swarm loop form");
STATISTIC(LoopsChainedEarly,
          "Number of serial loops transformed early into recursive form"
          " because of nontrivial bookkeeping");

static cl::opt<unsigned> ReadOnlyParallelizationThreshold(
    "swarm-readonlyparallelizationthreshold",
    cl::init(50),
    cl::desc("Number of iterations required to trigger breaking read-only loops"
             "into tasks, which may often become chains."));


// Forward declarations.

namespace {


class LoopIterDetacher {
  Loop &L;
  bool ProceedAtAllCosts;
  DominatorTree &DT;
  LoopInfo &LI;
  ScalarEvolution &SE;
  TargetTransformInfo &TTI;
  OptimizationRemarkEmitter &ORE;
  BasicBlock *const Header;
  BasicBlock *const Preheader;
  BasicBlock *Latch;  // == L.getLoopLatch() at all times.
  DeepenInst *const OuterDomain;
  BasicBlock *const ContinuationEnd;
  Function &F;
  Module &M;
  LLVMContext &Context;
  IntegerType *const DynamicTimestampTy;

public:
  LoopIterDetacher(Loop &L, DeepenInst *OuterDomain,
                   BasicBlock *ContinuationEnd, bool ProceedAtAllCosts,
                   DominatorTree &DT, LoopInfo &LI, ScalarEvolution &SE,
                   TargetTransformInfo &TTI,
                   OptimizationRemarkEmitter &ORE)
      : L(L), ProceedAtAllCosts(ProceedAtAllCosts),
        DT(DT), LI(LI), SE(SE), TTI(TTI), ORE(ORE),
        Header(L.getHeader()),
        Preheader(L.getLoopPreheader()), Latch(L.getLoopLatch()),
        OuterDomain(OuterDomain),
        ContinuationEnd(ContinuationEnd), F(*Header->getParent()),
        M(*F.getParent()), Context(M.getContext()),
        DynamicTimestampTy(Type::getInt64Ty(Context)) {}

  BasicBlock *run();

private:
  bool mayReadFromMemory() const;
  bool mayWriteToMemory() const;
  IntegerType *getLoopExpandableTimestampType() const;
  bool sanitizeContinuation();
  void subsumeContinuationIntoLoop(BasicBlock *ContStart);
  void sanitizeExiting(
          const SmallPtrSetImpl<const Instruction *> &LoopCarriedDependencies);
  bool createUnifiedLatch(BasicBlock *ExitBlock,
                          const SetVector<BasicBlock *> &ExitingBlocks);
  Instruction *createDoneVar(BasicBlock *ExitBlock);

  bool canonicalize(IntegerType *TimestampType);
  void addDependencies(
          const Instruction *Target,
          SmallPtrSetImpl<const Instruction *> &Dependencies) const;
  void addLoopCarriedDependencies(
          SmallPtrSetImpl<const Instruction *> &Dependencies) const;
  void addLatchExitComputation(
          SmallPtrSetImpl<const Instruction *> &Dependencies) const;
  bool isSafeToSink(const Instruction *I) const;
  bool isSafeToSink(
          const SmallPtrSetImpl<const Instruction *> &Bookkeeping) const;
  void moveFromLoopBody(
          DomTreeNode *N,
          const SmallPtrSetImpl<const Instruction *> &Bookkeeping);
  bool inSubLoop(const BasicBlock *BB) const {
    assert(L.contains(BB) && "Only valid if BB is IN the loop");
    return LI.getLoopFor(BB) != &L;
  }

  void makeChain(IntegerType *TimestampType);
};


/// Given a loop and its output phi, if the phi's sole incoming value comes
/// from a loop instruction with loop-invariant operands, this class attempts
/// to replace the phi with a copy of the maybe-loop-invariant instruction,
/// thus breaking the dependency between the last loop iteration and the
/// loop continuation.
/// TODO(mcj) Although the contributions of this class are minimal now, I
/// could see it handling a data-flow subgraph of instructions, where the
/// inputs are loop-invariant. If the subgraph is large, we'll need a cost
/// model to determine if it's worthwhile to duplicate execution.
class LoopInvariantOutputReplacer {
  const Loop &L;
  const DominatorTree &DT;
  PHINode *const OutputToReplace;
  Instruction *const InstrToCopy;
public:
  LoopInvariantOutputReplacer(const Loop &L,
                              PHINode *OutputToReplace,
                              const DominatorTree &DT);
  bool run();
private:
  bool canReplace() const;
  void debugNotReplaceable(const Twine &Reason) const;
};


} // anonymous namespace

// End forward declarations.


//FIXME(victory): dedupe with the copy in Fractalizer.cpp
static void finishFractalization(Function &F) {
  assert(all_of(F, [](const BasicBlock &BB) {
    const auto *DI = dyn_cast<SDetachInst>(BB.getTerminator());
    return !DI || DI->hasTimestamp();
  }));
  assert(hasValidSwarmFlags(F));
  StringRef From = SwarmFlag::Parallelizing;
  assert(F.hasFnAttribute(From));
  F.removeFnAttr(From);
  F.addFnAttr(SwarmFlag::Parallelized);
  // Make sure not to panic if we re-encounter some outlined portion of F
  F.removeFnAttr(SwarmAttr::AssertSwarmified);
  assert(hasValidSwarmFlags(F));
}


/// Recursive helper to traverse use-def chains to collect all Instructions
/// within the loop that participate in the dependency graph of Target.
/// That is, collect the set of nodes in the dataflow graph that are inside the
/// loop and from which Target can be reached, including Target itself.
/// Add this set of Instructions to Dependencies.
void LoopIterDetacher::addDependencies(
        const Instruction *Target,
        SmallPtrSetImpl<const Instruction *> &Dependencies) const {
  assert(L.contains(Target->getParent()));

  bool IsNew = Dependencies.insert(Target).second;
  if (!IsNew)
    return;

  for (const Use &U : Target->operands())
    if (const auto *UI = dyn_cast<Instruction>(U.get()))
      if (L.contains(UI->getParent()))
        addDependencies(UI, Dependencies);
}


/// Add to Dependencies those Instructions that should not be detached as part
/// of the loop body because they are involved in carrying Values produced in
/// one iteration of the loop to be used by the next iteration of the loop.
void LoopIterDetacher::addLoopCarriedDependencies(
        SmallPtrSetImpl<const Instruction *> &Dependencies) const {
  for (const PHINode &PN : Header->phis())
    addDependencies(&PN, Dependencies);
}


/// Add to Dependencies those Instructions that should not be detached as part
/// of the loop body because they are involved in computing whether the loop
/// should exit at the end of an iteration.
void LoopIterDetacher::addLatchExitComputation(
        SmallPtrSetImpl<const Instruction *> &Dependencies) const {
  assert(is_contained(successors(Latch), Header));
  assert(!all_of(successors(Latch),
                 [this](const BasicBlock *Succ) { return L.contains(Succ); }) &&
         "Loop latch is not exiting.");
  const TerminatorInst *LatchTerm = Latch->getTerminator();
  assert(isa<BranchInst>(LatchTerm) || isa<SwitchInst>(LatchTerm) &&
         "Loop latch not terminated by instruction with condition operand.");
  addDependencies(LatchTerm, Dependencies);
}


// Walk dominator tree in post-order, guaranteeing we see uses before defs,
// sinking the Instructions in Bookkeeeping so that the values they produce
// will remain available in the latch after the body gets detached.
// Assumes caller already checked that Bookkeeping contains only instructions
// that are safe to sink, such as by using isSafeToSink().
void LoopIterDetacher::moveFromLoopBody(
        DomTreeNode *N,
        const SmallPtrSetImpl<const Instruction *> &Bookkeeping) {
  BasicBlock *BB = N->getBlock();
  assert(N && BB);
  assert(BB != Latch);
  assert(L.contains(BB));

  // We are processing blocks in reverse dfo, so process children first.
  const std::vector<DomTreeNode *> &Children = N->getChildren();
  for (DomTreeNode *Child : Children) {
    // Only look at those loop body blocks we aim to detach.
    const BasicBlock *B = Child->getBlock();
    if (B == Latch || !L.contains(B))
      continue;

    moveFromLoopBody(Child, Bookkeeping);
  }

  //DEBUG(dbgs() << "Visiting DT node on sinking pass: " << BB->getName() << '\n');

  // Only need to process the contents of this block if it is not part of a
  // subloop (which will be processed later).
  if (inSubLoop(BB))
    return;

  //DEBUG(dbgs() << "Scanning for instructions to sink from BB: " << BB->getName() << '\n');

  for (auto II = BB->rbegin(), E = BB->rend(); II != E;) {
    Instruction &I = *II++; // to avoid iterator invalidation
    if (!Bookkeeping.count(&I)) continue;
    // Header phis never need to move
    if (BB == Header && isa<PHINode>(I)) continue;
    DEBUG(dbgs() << "Checking whether to sink or to copy instruction: "
                 << I << '\n');
    assert(isSafeToSink(&I));

    SmallVector<Use *, 8> BodyUses;
    for (Use &U : I.uses()) {
      auto *Usr = cast<Instruction>(U.getUser());
      BasicBlock *UsrBB = Usr->getParent();
      assert(DT.dominates(Header, UsrBB) &&
             !DT.dominates(getUniqueNonDeadendExitBlock(L), UsrBB) &&
             "continuation should be sanitized, should not use loop values");
      bool IsUsrInBody =
          UsrBB != Latch && !(UsrBB == Header && isa<PHINode>(Usr));
      assert(IsUsrInBody == !Bookkeeping.count(Usr) &&
             "Bookkeeping users are already sunk/copied to the latch");
      if (IsUsrInBody) {
        DEBUG(dbgs() << "User in loop body, in %" << UsrBB->getName() << ": "
                     << *Usr <<'\n');
        BodyUses.push_back(&U);
      }
    }
    if (BodyUses.empty()) {
      // The instruction is pure bookkeeping, i.e., it is used only in the
      // latch or in the header (by the next iteration), so we can just sink it
      // to the latch.
      assert(!I.use_empty() && "Bookkeeping is used in latch or next iteration");
      DEBUG(dbgs() << "Sinking instruction to latch: " << I << '\n');
      I.moveBefore(*Latch, Latch->getFirstInsertionPt());
    } else {
      // The instruction plays double duty in producing a value used in both
      // the body as well as as bookkeeping. Duplicate it: one copy can stay in
      // the body, and one copy can produce the value needed in the latch.
      DEBUG(dbgs() << "Copying instruction to latch: " << I << '\n');
      Instruction *New = I.clone();
      // The original copy (I) is in the Bookkeeping set, and the new copy
      // (New) is not. So have the new copy take the place of the original
      // copy, being used in the body, and sink the original copy to the latch,
      // to be used for bookkeeping purposes.
      New->insertBefore(&I);
      New->takeName(&I);
      for (Use *U : BodyUses)
        U->set(New);
      I.setName(New->getName() + ".latchclone");
      I.moveBefore(*Latch, Latch->getFirstInsertionPt());
    }
  }
}


bool LoopIterDetacher::mayReadFromMemory() const {
  for (const BasicBlock *BB : L.blocks())
    for (const Instruction &I : *BB)
      if (I.mayReadFromMemory())
        return true;
  return false;
}


bool LoopIterDetacher::mayWriteToMemory() const {
  for (const BasicBlock *BB : L.blocks())
    for (const Instruction &I : *BB)
      if (I.mayWriteToMemory())
        return true;
  return false;
}


/// Return true if it is safe to sink I to the end of the loop iteration,
/// ignoring dominance constraints for any uses of I as a Value.
bool LoopIterDetacher::isSafeToSink(const Instruction *I) const {
  // Can't move instructions that are closely tied to the local CFG structure.
  if (isa<TerminatorInst>(I))
    return false;
  if (isa<PHINode>(I))
    return false;

  // We don't want to move things with side effects, they should stay in the loop body!
  if (mayHaveSideEffects(I))
    return false;

  // Easy case: some instructions don't care where they are.
  if (isSafeToSinkOrHoist(I))
    return true;
  // Okay, now we are considering instructions that may care about their
  // ordering with respect to other instructions due to possible dependency
  // on memory values or because they have the ability to trap (e.g., division
  // where the divisor could be zero).

  // If I executes only under some conditions within each iteration, it may be
  // unsafe to execute it unconditionally at the end of each iteration.
  assert(DT.isReachableFromEntry(I->getParent()));
  assert(DT.isReachableFromEntry(Latch));
  if (!DT.dominates(I, Latch->getTerminator()))
    return false;

  // If I has no interactions with memory and always executes on every
  // iteration of the loop that reaches the latch, it's fine to put it at the
  // latch, even if it has the potential to trap.
  if (!I->mayReadFromMemory()) {
    assert(!I->mayReadOrWriteMemory());
    return true;
  }

  // Now we consider an instruction that may read from memory.  It is safe to
  // sink only if there are no intervening stores to the same memory.
  // TODO(victory): Check for an invariant.start dominating the load.
  auto *LI = dyn_cast<LoadInst>(I);
  return isInvariantInPath(LI ? LI->getPointerOperand() : nullptr,
                           I, Latch->getTerminator());
}


bool LoopIterDetacher::isSafeToSink(
        const SmallPtrSetImpl<const Instruction *> &Bookkeeping) const {
  // Iterating through this unordered set means debug output will get printed
  // in random order, but at least we will show all of them.
  // victory: We could iterate through blocks in the loop to print debug output
  // in a deterministic order, but that would complicate the code and slow
  // down non-debugging compile jobs for no reason. And it's not worth writing
  // two versions of this loop.
  bool ret = true;
  for (const Instruction *I : Bookkeeping) {
    // Header phis will stay in the header and never need to be moved.
    if (I->getParent() == Header && isa<PHINode>(I))
      continue;
    // The latch backedge doesn't need to move because it's already at the
    // end of the loop iteration
    if (I == Latch->getTerminator())
      continue;

    if (!isSafeToSink(I)) {
      DEBUG(dbgs() << "Bookkeeping cannot be sunk to end of loop iteration: "
                   << *I << '\n');
      ret = false;
    }
  }
  return ret;
}


static bool isSaneLoopBackedgeTakenCount(const ConstantInt *Limit) {
  if (Limit->isZero()) {
    // SCCP + SimplifyCFG seems to eliminate these.
    DEBUG(dbgs() << "Loop backedge never taken?\n");
    return false;
  } else if (Limit->isMinusOne()) {
    //victory: It seems like ScalarEvolution sometimes produces a value of -1
    // for some loops after this pass runs, which is nonsense.
    // This is rare, and it appears to happen for unreachable paths.
    // Perhaps if we ran some existing LLVM passes such as ADCE and/or
    // loop-deletion, these loops would be eliminated.
    DEBUG(dbgs() << "Loop backedge taken count == -1.\n");
    return false;
  } else if (Limit->isNegative()) {
    DEBUG(dbgs() << "Loop backedge taken count is negative.\n");
    return false;
  }

  uint64_t limit = Limit->getLimitedValue();
  assert(limit != 0 && "Should have been caught in earlier check");
  assert(limit != UINT64_MAX && "Should have been caught in earlier check");
  assert(limit <= INT64_MAX && "Negative tripcount wraparound?");
  return true;
}


// Return a type for a timestamp variable that won't cause LoopExpansion to
// crash, or null if this loop might cause LoopExpansion to crash anyway.
IntegerType *LoopIterDetacher::getLoopExpandableTimestampType() const {
  const SCEV *Limit = SE.getBackedgeTakenCount(&L);
  DEBUG(dbgs() << "loop limit: " << *Limit << "\n");
  assert(isa<SCEVCouldNotCompute>(Limit) || SE.isLoopInvariant(Limit, &L));
  assert(!isa<SCEVConstant>(Limit)
         || isSaneLoopBackedgeTakenCount(cast<SCEVConstant>(Limit)->getValue()));
  if (!isa<SCEVCouldNotCompute>(Limit) && !Limit->getType()->isIntegerTy()) {
    ORE.emit(OptimizationRemarkMissed(PASS_NAME, "BadLimitType",
                                      L.getStartLoc(), Header)
             << "ScalarEvolution computed a non-integer limit expression");
    return nullptr;
  }

  if (PHINode *IV = L.getCanonicalInductionVariable()) {
    DEBUG(dbgs() << "Loop has preexisting canonical IV: " << *IV << '\n');
    assert(IV->getType()->isIntegerTy());
    if (!isa<SCEVCouldNotCompute>(Limit) && IV->getType() != Limit->getType()) {
      ORE.emit(OptimizationRemarkMissed(PASS_NAME, "BadIVType",
                                        L.getStartLoc(), Header)
               << "loop induction variable width mismatch");
      return nullptr;
    }
  }

  if (isa<SCEVCouldNotCompute>(Limit))
    return DynamicTimestampTy;
  return cast<IntegerType>(Limit->getType());
}


/// Ensure the continuation outside the loop has no dependencies on the loop,
/// so the external continuation can be spawned in parallel with the loop.
/// In particular, any continuation outside the loop must not have SSA data
/// dependencies on any values produced inside the loop, and the continuation
/// must not have control dependencies on the loop (i.e., it must only be
/// possible to exit the loop to a unique external continuation, a.k.a. a
/// unique exit block).  To ensure these properties, we may rewrite some loop
/// continuations so that they will be spawned inside the loop body, on the
/// last loop iteration.
///
/// [victory] I found it interesting that there's a lot of hacks ranging from
///           brute-force to very clever that have been implemented in practical
///           tools that need to impose structure on control flow graphs,
///           such as the ones mentioned here: https://perma.cc/UWL2-CXGA
///           However, we seem to have taken on a unique challenge in seeking
///           to structure loops so that their can be no control *or* data
///           dependence between them and their continuations, and I have not
///           found anything out there like our subsumption-based approach.
///
/// \return true on success.
bool LoopIterDetacher::sanitizeContinuation() {
  assert(L.hasDedicatedExits());
  BasicBlock *ExitBlock = nullptr;

  bool haveSubsumedContinuations = false;
  auto ensureDedicatedExitBeforeContinuationEnd = [&]() -> void {
    SmallSetVector<BasicBlock *, 8> ExitingBlocks;
    for (BasicBlock *Pred : predecessors(ContinuationEnd))
      if (L.contains(Pred)) ExitingBlocks.insert(Pred);
    assert(!haveSubsumedContinuations || !ExitingBlocks.empty()
           && "Any continuation subsumption will make the loop exit directly "
              "to ContinuationEnd");
    if (!ExitingBlocks.empty()) {
      // We now create a dedicated ExitBlock, which will be a safe place to
      // insert code that should run after the loop exits.  This new ExitBlock
      // between the loop and ContinuationEnd will be spawned with a timestamp
      // ordered after the loop by Fractalizer.
      // We must do this even if sanitization is failing, as makeChain()
      // still depends on dedicated exits to ensure there is a safe place
      // to call free() after the loop.
      DEBUG(dbgs() << "Creating new continuation block outside the loop.\n");
      ExitBlock = SplitBlockPredecessors(ContinuationEnd,
                                         ExitingBlocks.getArrayRef(),
                                         ".newexit", &DT, &LI);
      assert(ExitBlock); (void)ExitBlock;
    }
    assert(!ExitBlock || !L.contains(ExitBlock));
    assert(!ExitBlock || all_of(predecessors(ExitBlock),
                  [this](const BasicBlock *BB) { return L.contains(BB); }));

    // If we subsumed a continuation, that continuation may have exit to
    // to non-dominated doomed blocks.  Let's restore LoopSimplify form
    // because we like to use getUniqueExitBlocks.
    formDedicatedExitBlocks(&L, &DT, &LI, true);
    // The way this can fail is if a programmer used the (rarely used)
    // nonstandard GCC "labels as values" extension with goto statements for
    // fatal error handling.  We might never encounter this, and certainly
    // won't encounter it if we're only trying to compile portable C/C++ code.
    assert(L.hasDedicatedExits());
  };

  // Phase 1: Ensure the loop has a single continuation outside the loop
  //          without control dependencies on the loop.

  SmallVector<BasicBlock *, 8> ExitBlocks;
  L.getUniqueExitBlocks(ExitBlocks);
  erase_if(ExitBlocks, [](const BasicBlock *Exit) {
    if (isDoomedToUnreachableEnd(Exit)) {
      DEBUG(dbgs() << "Ignoring exit block terminated by unreachable: " << *Exit);
      return true;
    }
    return false;
  });
  if (ExitBlocks.empty()) {
    ORE.emit(OptimizationRemarkMissed(PASS_NAME, "NoExit",
                                      L.getStartLoc(), Header)
             << "loop has no exit");
    //return false;
    llvm_unreachable("loop has no CFG-visible exit: infinite loop? "
                     "or exits via non-returning call/exception handling?");
  } else if (ExitBlocks.size() == 1) {
    ExitBlock = ExitBlocks[0];
  } else { // ExitBlocks.size() > 1
    ORE.emit(OptimizationRemarkAnalysis(PASS_NAME, "MultiExit",
                                        L.getStartLoc(), Header)
             << "loop can reach multiple continuation paths");
    // Since the continuations have a control dependency on the loop, we
    // spawn whichever continuation should actually be reached only on the
    // last itartion of the loop.
    for (BasicBlock *Exit : ExitBlocks) subsumeContinuationIntoLoop(Exit);
    haveSubsumedContinuations = true;
    // Cannot call getUniqueExitBlocks() because ContinuationEnd may be a non-dedicated exit
    ExitBlocks.clear();
    L.getExitBlocks(ExitBlocks);
    std::sort(ExitBlocks.begin(), ExitBlocks.end());
    ExitBlocks.erase(std::unique(ExitBlocks.begin(), ExitBlocks.end()),
                     ExitBlocks.end());
    erase_if(ExitBlocks, isDoomedToUnreachableEnd);
    assert(ExitBlocks.size() == 1);
    assert(ExitBlocks[0] == ContinuationEnd);
    ExitBlock = getUniqueNonDeadendExitBlock(L);
    assert(ExitBlock);
    assert(ExitBlock == ContinuationEnd);
  }
  DEBUG(dbgs() << "Loop exit block is: " << *ExitBlock);
  assert(ExitBlock && getUniqueNonDeadendExitBlock(L) == ExitBlock);

  // Phase 2. Ensure the continuation outside the loop has no data dependencies
  //          on the loop.

  formLCSSA(L, DT, &LI, nullptr);
  DEBUG(for (const PHINode &PN : ExitBlock->phis()) {
    for (unsigned i = 0; i < PN.getNumIncomingValues(); ++i) {
      if (!L.contains(PN.getIncomingBlock(i))) continue;
      const Value *LoopOutput = PN.getIncomingValue(i);
      if (!L.isLoopInvariant(LoopOutput))
        dbgs() << "Loop continuation uses value produced in loop";
      else
        dbgs() << "Loop continuation (conditionally?) uses loop-invariant value";
      dbgs() << " as %" << PN.getName() << ": " << *LoopOutput << '\n';
    }
  });

  // TODO(mcj) If there are multiple outputs, we should attempt all-or-nothing
  // transformation of outputs, since there's no point transforming just some.
  auto PI = ExitBlock->phis().begin(), PE = ExitBlock->phis().end();
  while (PI != PE) {
    assert(!haveSubsumedContinuations);
    PHINode &Output = *(PI++); // Carefully crafted to avoid iterator invalidation
    bool Replaced = replaceLoopOutputWithReductionCalls(L, &Output, DT, ORE);
    if (Replaced)
      // SCCRT update calls spawn tasks that update the reduction value,
      // and rely on the collapse call in the continuation later being spawned
      // with a timestamp that orders them after the loop.
      ProceedAtAllCosts = true;
    else
      Replaced = LoopInvariantOutputReplacer(L, &Output, DT).run();
    if (!Replaced) break;
  }
  if (ExitBlock->phis().begin() != ExitBlock->phis().end()) {
    // There are SSA values produced in the loop, and we need to keep
    // these values available to the continuation to use while allowing
    // the body of the loop to be spawned.
    // To satsify the continuation's data dependency, we spawn the continuation
    // as part of the loop's last iteration.
    ORE.emit(OptimizationRemarkAnalysis(PASS_NAME, "LoopOutput",
                                        L.getStartLoc(), Header)
             << "loop produces value used in continuation that cannot be "
                "trivially eliminated or converted into a reduction");
    assert(!haveSubsumedContinuations);
    assert(ExitBlock == getUniqueNonDeadendExitBlock(L));
    subsumeContinuationIntoLoop(ExitBlock);
    haveSubsumedContinuations = true;
    ExitBlock = getUniqueNonDeadendExitBlock(L);
    assert(ExitBlock);
    assert(ExitBlock == ContinuationEnd);
  }

  // Phase 3: Ensure a dedicated exit, see comment in definition of
  // ensureDedicatedExitBeforeContinuationEnd above.

  ensureDedicatedExitBeforeContinuationEnd();

  assert(ExitBlock && getUniqueNonDeadendExitBlock(L) == ExitBlock);

  return true;
}


/// In order to transform this loop into our canonical Swarm loop form,
/// we need to only exit the loop via a single edge from the latch,
/// which must be a simple conditional branch.
void LoopIterDetacher::sanitizeExiting(
        const SmallPtrSetImpl<const Instruction *> &LoopCarriedDependencies) {
  BasicBlock *const PrevLatch = Latch;
  BasicBlock *const ExitBlock = getUniqueNonDeadendExitBlock(L);
  assert(ExitBlock && "Guaranteed by sanitizeContinuation");
  const SmallSetVector<BasicBlock *, 8> ExitingBlocks(pred_begin(ExitBlock),
                                                      pred_end(ExitBlock));
  assert(all_of(ExitingBlocks, [this](const BasicBlock *Pred) {
    return L.contains(Pred); }) && "Exit should be dedicated");
  assert(!ExitingBlocks.empty());

  if (ExitingBlocks.count(Latch) && ExitingBlocks.size() == 1
      && isa<BranchInst>(Latch->getTerminator())) {
    assert(ExitBlock->getSinglePredecessor() == Latch);
    // No need to do anything!
    return;
  }
  //TODO(victory): Transforming to use a done variable is very general,
  // but it's really not necessary just to deal with latches that terminate
  // in switch instructions, and it can be expensive if those loops are short
  // and should have been unexpanded or balanced-tree expanded.
  ORE.emit(OptimizationRemark(PASS_NAME, "EarlyDoneVar",
                              L.getStartLoc(), Header)
           << "Rewriting loop to exit only via latch using a done variable "
              "in the hopes of enabling progressive expansion");

  // Create a flag that is initialized to false and set to true precisely when
  // you would exit from the loop.
  Instruction *DoneVar = createDoneVar(ExitBlock);
  BasicBlock *NewExitingBlock =
          SplitBlockPredecessors(ExitBlock, ExitingBlocks.getArrayRef(),
                                 ".exitingto", &DT, &LI);
  assert(NewExitingBlock);
  IRBuilder<> Builder(NewExitingBlock->getTerminator());
  StoreInst *StoreDone = Builder.CreateStore(Builder.getTrue(), DoneVar);
  StoreDone->setMetadata(SwarmFlag::DoneFlag, MDNode::get(Context, {}));

  DEBUG(dbgs() << "Creating a unified latch that we will make "
                  "into the only exiting block.\n");
  Latch = SplitBlockPredecessors(Header, {Latch}, ".unified", &DT, &LI);
  assert(Latch);

  // Sink/copy loop carried dependencies into the latch.
  moveFromLoopBody(DT[Header], LoopCarriedDependencies);
  assert(all_of(LoopCarriedDependencies, [this](const Instruction *I) {
    const BasicBlock *BB = I->getParent();
    return BB == Latch || (BB == Header && isa<PHINode>(I));
  }));

  DEBUG(dbgs() << "Setting up latch to exit or not based on the done flag.\n");
  TerminatorInst *LatchTerm = Latch->getTerminator();
  Builder.SetInsertPoint(LatchTerm);
  Builder.SetCurrentDebugLocation(PrevLatch->getTerminator()->getDebugLoc());
  assert(Builder.getCurrentDebugLocation()
         && "Recursive chains will require latch branch to have a DebugLoc");
  LoadInst *DoneVal = Builder.CreateLoad(DoneVar, "done");
  assert(!isa<PHINode>(ExitBlock->front())
         && "Guranteed by sanitizeContinuation");
  assert(ExitBlock->getSinglePredecessor() == NewExitingBlock);
  Builder.CreateCondBr(DoneVal, ExitBlock, Header);
  LatchTerm->eraseFromParent();

  DEBUG(dbgs() << "Routing path to exit loop through the latch.\n");
  // This subsumes NewExitingBlock into the loop.
  NewExitingBlock->getTerminator()->replaceUsesOfWith(ExitBlock, Latch);
  assert(LI.getLoopFor(Latch) == &L);
  assert(LI.getLoopFor(NewExitingBlock) == L.getParentLoop());
  LI.changeLoopFor(NewExitingBlock, &L);
  L.addBlockEntry(NewExitingBlock);


  DEBUG(dbgs() << "New loop latch is: " << *Latch);

  assert(ExitBlock->getSinglePredecessor() == Latch);
  assert(ExitBlock && getUniqueNonDeadendExitBlock(L) == ExitBlock);
  assert(DT.dominates(Header, PrevLatch));
  assert(DT.dominates(Header, NewExitingBlock));
  assert(DT.dominates(Header, Latch));
  assert(DT.dominates(Header, ExitBlock));
  DT.changeImmediateDominator(Latch,
          DT.findNearestCommonDominator(NewExitingBlock, PrevLatch));
  DT.changeImmediateDominator(ExitBlock, Latch);
  DEBUG(assertVerifyFunction(F, "After setting up unified latch", &DT, &LI));
}


/// Populate CanReachEnd with the basic blocks in the CFG from which there is
/// a path to End that does not include any node outside of those in Blocks.
/// That is, CanReachEnd will be the subset of Blocks from which End is
/// reachable, in the CFG subgraph induced by Blocks.
template <typename SetTy>
static void getBlocksThatCanReachEnd(
        const SetTy &Blocks,
        BasicBlock *End,
        SmallVectorImpl<BasicBlock *> &CanReachEnd) {
  assert(CanReachEnd.empty());
  assert(Blocks.count(End));
  // Perform a DFS traversing control-flow edges backwards from End.
  SetVector<BasicBlock *> Visited;
  std::function<void(BasicBlock *)> DFSHelper =
          [&Blocks, &Visited, &DFSHelper](BasicBlock *BB) {
    if (bool IsNew = Visited.insert(BB))
      for (BasicBlock *Pred : predecessors(BB))
        if (Blocks.count(Pred))
          DFSHelper(Pred);
  };
  DFSHelper(End);
  // It seems preferable to reverse the order of the vector so that End
  // is at the back, to put the blocks in something closer to program order.
  std::copy(Visited.rbegin(), Visited.rend(), std::back_inserter(CanReachEnd));
  assert(CanReachEnd.back() == End);
}


/// Take the continuation of the loop starting from ContStart and rewrite the
/// control flow so that the continuation falls inside the loop.  This is a
/// clever/dirty hack to convince later passes to see the continuation as part
/// of the loop, and more specifically as something the loop just spawns within
/// its body on its last iteration.  This transformation inserts some extra
/// branch overhead, and will be undone by jump threading.
///
/// In pseudocode, we're taking a thing that looks approximately like this:
///
///   while (true) {
///     bool some_condition = some_stuff_above_exiting();
///     if (some_condition) break;
///     some_stuff_below_exiting();
///   }
///   continuation();
///
/// And we're rewriting it to look a bit like this:
///
///   do {
///     bool some_condition = some_stuff_above_exiting();
///     if (some_condition) {
///       spawn { continuation() };
///     } else {
///       some_stuff_below_exiting();
///     }
///   } while (!some_condition);
///
/// See further comments inside the function implementation for a more
/// accurate picture of what we're really doing.
///
/// \return true on success.
void LoopIterDetacher::subsumeContinuationIntoLoop(BasicBlock *ContStart) {
  DEBUG(dbgs() << "Attempting to rewrite continuation starting at "
               << ContStart->getName() << '\n');
  assert(!L.contains(ContStart) && "ContStart must be external to the loop");

  // We need to ensure the entire continuation is dedicated so we're not
  // changing behavior if the continuation is reached by some other path than
  // from the loop.
  SmallSetVector<BasicBlock *, 8> ContBlocks;
  BasicBlock *NewContEnd =
      makeReachableDominated(ContStart, ContinuationEnd, DT, LI, ContBlocks);
  assert(NewContEnd && "Impossible to subsume region with no outgoing edges");

  // Find the exiting block inside the loop that goes to ContStart.
  assert(all_of(predecessors(ContStart), [this](const BasicBlock *Pred) {
           return L.contains(Pred); })
         && "ContStart must be a dedicated exit");
  BasicBlock *ExitingBlock = ContStart->getUniquePredecessor();
  if (!ExitingBlock) {
    DEBUG(dbgs() << "Continuation is not reached from a unique predecessor.\n");
    const SmallSetVector<BasicBlock *, 8> ExitingBlocks(pred_begin(ContStart),
                                                        pred_end(ContStart));
    if (ContBlocks.size() * ExitingBlocks.size() > 10) {
      // To avoid this multiplicative blowup in code size,
      // eagerly outline the continuation
      DEBUG(dbgs() << "Eagerly outlining continuation.\n");
      BasicBlock *NewContStart =
              SplitBlock(ContStart, ContStart->getFirstNonPHI(), &DT, &LI);
      ContBlocks.remove(ContStart);
      ContBlocks.insert(NewContStart);
      SetVector<Value *> Inputs;
      findInputsNoOutputs(ContBlocks, Inputs);
      Function *ContFunc = outline(Inputs, ContBlocks, NewContStart, ".eager");
      ContBlocks.clear();
      ContFunc->setCallingConv(CallingConv::Fast);
      finishFractalization(*ContFunc);
      assertVerifyFunction(*ContFunc, "Eagerly outlined continuation task");
      CallInst *Call = IRBuilder<>(ContStart->getTerminator()).CreateCall(
          ContFunc, Inputs.getArrayRef());
      Call->setCallingConv(CallingConv::Fast);
      if (!Call->getDebugLoc())
        Call->setDebugLoc(getSafeDebugLoc(L));
      ContStart->getTerminator()->replaceUsesOfWith(NewContStart,
                                                    ContinuationEnd);
      if (DT.dominates(ContStart, ContinuationEnd)) {
        assert(DT.dominates(NewContStart, ContinuationEnd));
        DT.changeImmediateDominator(
                ContinuationEnd, ContStart);
        assert(!DT.dominates(NewContStart, ContinuationEnd));
      }
      eraseDominatorSubtree(NewContStart, DT, &LI);
    }
    SmallVector<BasicBlock *, 8> NewContStarts;
    for (BasicBlock *ExitingBlock : ExitingBlocks)
      NewContStarts.push_back(
              SplitBlockPredecessors(ContStart, {ExitingBlock}, ".loopexit",
                                     &DT, &LI));
    for (BasicBlock *NewContStart : NewContStarts) {
      subsumeContinuationIntoLoop(NewContStart);
    }
    return;
  }
  DEBUG(dbgs() << "Continuation is reached from exiting block "
               << ExitingBlock->getName() << '\n');
  assert(L.contains(ExitingBlock));

  // The exiting block might be in a nested inner loop, in which case we are
  // really subsuming the continuation into that nested inner loop.
  Loop *SubsumingLoop = LI.getLoopFor(ExitingBlock);
  DEBUG(dbgs() << "Subsuming continuation into loop: " << *SubsumingLoop);
  assert(L.contains(SubsumingLoop));

  // See here for a picture of how we're going to rewrite control flow:
  // https://photos.app.goo.gl/8JELeFEwpojcZxSz5
  //
  // By definition, if the exiting block is inside the loop, it must have some
  // path to stay in the loop, so it must have some other successor inside the
  // loop.  So in fact the exiting block must have at least two successors,
  // i.e., it must end in a conditional branch of some sort, and we call its
  // successor that is inside the loop "next".  This is the situation in the
  // left portion of the linked picture, which is something like this:
  //
  //  // Inside the loop:
  //  exiting:
  //    ... some stuff above ...
  //    if (%cond) then { goto cont; } else { goto next; }
  //  next:
  //    ... some stuff below ...
  //
  //  // Outside the loop:
  //  cont:
  //    ... some continuation stuff ...
  //    goto end;
  //
  // We create a new "join" block that splits the edge from the exiting block
  // to next, duplicate the conditional branch into the join block, and reroute
  // some control flow edges as shown in the right portion of the linked
  // picture so that the continuation ends by passing back through the join
  // block, like this:
  //
  //  exiting:
  //    ... some stuff above ...
  //    if (%cond) then { goto cont; } else { goto join; }
  //  join:
  //    if (%cond) then { goto end; } else { goto next; }
  //  next:
  //    ... some stuff below ...
  //
  //  cont:
  //    ... some continuation stuff ...
  //    goto join;
  //
  // This makes all the computation that was in the continuation now inside of
  // the loop, since there is now a false control flow path from the
  // continuation, through join, to next.
  //
  // Although it may seem like this is introducing additional branch overhead,
  // getting rid of exactly this kind of branch overhead can easily be done
  // later by running the well-known "jump threading" compiler optimization:
  // https://en.wikipedia.org/wiki/Jump_threading
  // https://llvm.org/docs/Passes.html#jump-threading-jump-threading

  // Find the next block after the exiting block inside the loop we are subsuming into.
  auto NextBlockIter = find_if(
          successors(ExitingBlock),
          [SubsumingLoop](BasicBlock *Succ) { return SubsumingLoop->contains(Succ); });
  assert(NextBlockIter != succ_end(ExitingBlock) &&
         "Since ExitingBlock is in the loop, it must have a successor in the loop");
  BasicBlock *NextBlock = *NextBlockIter;

  // Create the join block and set up the two conditional branches.
  TerminatorInst *ExitingTerm = ExitingBlock->getTerminator();
  DEBUG(dbgs() << "Conditional branch at the end of exiting block:\n  "
               << *ExitingTerm << '\n');
  assert(isa<BranchInst>(ExitingTerm) || isa<SwitchInst>(ExitingTerm)
         && "Exception-handling terminators should have been caught earlier");
  // N.B. ExitingTerm could be a switch instruction with multiple edges to
  // each of NextBlock and ContStart, and both NextBlock and ContStart could
  // have phis which need one entry for each incoming edge.
  // Therefore, we do not use SplitBlock() or SplitBlockPredecessors() which
  // would do violence to the phis, and we instead carefully construct the
  // join block and perform all the analysis and phi updates ourselves.
  BasicBlock *JoinBlock =
          BasicBlock::Create(Context, NextBlock->getName() + ".join", &F,
                             NextBlock);
  BranchInst *TempBranch = BranchInst::Create(NextBlock, JoinBlock);
  auto *JoinTerm = cast<TerminatorInst>(ExitingTerm->clone());
  JoinTerm->insertBefore(TempBranch);
  TempBranch->eraseFromParent();
  ExitingTerm->replaceUsesOfWith(NextBlock, JoinBlock);
  DT.addNewBlock(JoinBlock, ExitingBlock);
  for (unsigned i = 0; i < JoinTerm->getNumSuccessors(); ++i) {
    if (JoinTerm->getSuccessor(i) == ContStart) {
      JoinTerm->setSuccessor(i, ContinuationEnd);
    } else if (JoinTerm->getSuccessor(i) != NextBlock) {
      // If the condition was such that ExitingTerminator would have gone to
      // some other successor, then JoinBlock will not be reached and it does
      // not matter where JoinTerm goes.
      BasicBlock *UnreachableBlock =
              BasicBlock::Create(Context, "unreachable", &F);
      new UnreachableInst(Context, UnreachableBlock);
      JoinTerm->setSuccessor(i, UnreachableBlock);
      DT.addNewBlock(UnreachableBlock, JoinBlock);
    }
  }
  assert(!isa<PHINode>(ContinuationEnd->front()));
  for (PHINode &Phi : NextBlock->phis()) {
    int Idx;
    while ((Idx = Phi.getBasicBlockIndex(ExitingBlock)) != -1)
      Phi.setIncomingBlock(Idx, JoinBlock);
  }
  DEBUG(dbgs() << "Created join block:" << *JoinBlock);
  BasicBlock *NextPred = NextBlock->getUniquePredecessor();
  assert(!!NextPred == (NextPred == JoinBlock));
  if (NextPred == JoinBlock)
    DT.changeImmediateDominator(NextBlock, JoinBlock);
  assert(DT[ContStart]->getIDom()->getBlock() == ExitingBlock);
  DT.changeImmediateDominator(ContinuationEnd,
          DT.findNearestCommonDominator(ContinuationEnd, JoinBlock));
  SubsumingLoop->addBasicBlockToLoop(JoinBlock, LI);
  DEBUG(assertVerifyFunction(F, "After setting up join block in preparation"
                                " for continuation subsumption", &DT, &LI));
  assert((NextBlock == Header) == (ExitingBlock == Latch));
  if (ExitingBlock == Latch) {
    Latch = JoinBlock;
    DEBUG(dbgs() << "Latch is now: " << *Latch);
    assert(Latch == L.getLoopLatch());
  }

  // Now, change the end of the continuation to branch up to the join block.
  // This is the action that makes the continuation subsumed into the loop.
  DEBUG(dbgs() << "Rewriting end of continuation "
               << NewContEnd->getName() << '\n');
  assert(NewContEnd->getUniqueSuccessor() == ContinuationEnd);
  NewContEnd->getTerminator()->replaceUsesOfWith(ContinuationEnd, JoinBlock);
  // Update DominatorTree: removed NewContEnd as a predecessor of ContinuationEnd
  BasicBlock *EndDominator = JoinBlock;
  for (BasicBlock *Pred : predecessors(ContinuationEnd)) {
    EndDominator = DT.findNearestCommonDominator(EndDominator, Pred);
  }
  DT.changeImmediateDominator(ContinuationEnd, EndDominator);
  // Update LoopInfo to reflect that the continuation was subsumed into the loop
  Loop *EnclosingLoop = L.getParentLoop();
  SmallVector<BasicBlock *, 8> ContBlocksThatCanReachEnd;
  getBlocksThatCanReachEnd(ContBlocks, NewContEnd, ContBlocksThatCanReachEnd);
  for (BasicBlock *BB : ContBlocksThatCanReachEnd) {
    Loop *BBLoop = LI.getLoopFor(BB);
    assert(!EnclosingLoop || BBLoop && EnclosingLoop->contains(BBLoop)
           && "Entire continuation should be contained in L's parent loop");
    Loop *AncestorLoop = SubsumingLoop;
    do {
      AncestorLoop->addBlockEntry(BB);
      AncestorLoop = AncestorLoop->getParentLoop();
    } while (AncestorLoop != EnclosingLoop);
    if (BBLoop == EnclosingLoop)
      LI.changeLoopFor(BB, SubsumingLoop);
    else if (BBLoop
        && BBLoop->getHeader() == BB
        && BBLoop->getParentLoop() == EnclosingLoop) {
      if (EnclosingLoop)
        // TODO(victory): LLVM version 6 gets an alternative version of
        // removeChildLoop() that eliminates the need to call find().
        EnclosingLoop->removeChildLoop(find(*EnclosingLoop, BBLoop));
      else
        LI.removeLoop(find(LI, BBLoop));
      SubsumingLoop->addChildLoop(BBLoop);
    }
  }
  DEBUG(dbgs() << "Subsumed continuation into loop: " << *SubsumingLoop);
  assert(SubsumingLoop->contains(JoinBlock));
  assert(LI.getLoopFor(JoinBlock) == SubsumingLoop);
  assert(SubsumingLoop->contains(ContStart));
  assert(LI.getLoopFor(ContStart) == SubsumingLoop);
  assert(SubsumingLoop->contains(NewContEnd));
  assert(LI.getLoopFor(NewContEnd) == SubsumingLoop);
  assert(!SubsumingLoop->contains(ContinuationEnd));
  assert(LI.getLoopFor(ContinuationEnd) == EnclosingLoop);
  SubsumingLoop->verifyLoop();
  L.verifyLoop();
  DEBUG(assertVerifyFunction(F, "After subsuming continuation into loop", &DT, &LI));

  // To ensure that continuations always end up in the right domain,
  // create a single, easy-to-find continuation detach for Fractalizer
  // to later retarget to the correct superdomain.
  SDetachInst *DI = nullptr;
  detachTask(ContStart->getFirstNonPHI(), NewContEnd->getTerminator(),
             ConstantInt::getTrue(Context), OuterDomain,
             DetachKind::SubsumedCont, "subsumed_cont", &DT, &LI, &DI);
  DI->setRelativeTimestamp(true);
  DEBUG(assertVerifyFunction(F, "After detaching continuation", &DT, &LI));
  // Now that the continuation is detached to timestamp +1,
  // We must proceed with detaching the loop with a timestamp to ensure
  // the continuation runs at a sensible, unique timestamp.
  ProceedAtAllCosts = true;

  addStringMetadataToLoop(&L, SwarmFlag::LoopSubsumedCont);
}


/// For when there's some computation for whether to exit from the loop, and
/// that computation will stay in the loop body to be detached, allocate a
/// Boolean flag in memory to communicate when to exit from the loop.
/// Initialize the flag to false.
Instruction *LoopIterDetacher::createDoneVar(BasicBlock *ExitBlock) {
  IntegerType *BoolTy = Type::getInt1Ty(Context);
  Constant *AllocSize = ConstantExpr::getSizeOf(BoolTy);
  Instruction *DoneVar = CallInst::CreateMalloc(Preheader->getTerminator(),
                                                AllocSize->getType(),
                                                BoolTy,
                                                AllocSize,
                                                nullptr, nullptr,
                                                "done-var");
  auto *Store = new StoreInst(ConstantInt::getFalse(Context), DoneVar,
                              Preheader->getTerminator());
  Store->setMetadata(SwarmFlag::DoneFlag, MDNode::get(Context, {}));
  assert(all_of(predecessors(ExitBlock),
                [this](const BasicBlock *Pred) { return L.contains(Pred); })
         && "Exit must be dedicated so malloc dominates free.");
  CallInst::CreateFree(DoneVar, &*ExitBlock->getFirstInsertionPt());
  return DoneVar;
}


// \return the start of the loop's continuation on success, or null otherwise.
BasicBlock *LoopIterDetacher::run() {
#ifndef NDEBUG
  SmallVector<BasicBlock *, 8> ExitingBlocks;
  L.getExitingBlocks(ExitingBlocks);
  assert(all_of(ExitingBlocks, [](const BasicBlock *BB) {
    const TerminatorInst *TI = BB->getTerminator();
    return isa<BranchInst>(TI) || isa<IndirectBrInst>(TI)
           || isa<SwitchInst>(TI);
  }) && "Should be guaranteed by prior checks for exception handling, and"
        " we will assume this when rewriting control flow around the loop");
#endif

  DEBUG(dbgs() << "Attempting to detach each iteration of loop.\n");
  DEBUG(dbgs() << "Initial loop header is: " << *Header);

  // LoopSimplify may fail to ensure a dedicated preheader or single latch
  // in the presense of IndirectBrInsts. (See LoopSimplify's
  // InsertPreheaderForLoop() and insertUniqueBackedgeBlock().)
  // However, LLVM's indirectbr instruction exists only to support the non-
  // standard GNU extension that allows local labels to be used as values:
  // http://blog.llvm.org/2010/01/address-of-label-and-indirect-branches.html
  // And, AFAIK, nobody really uses this GNU extension, not even across SPECCPU.
  assert(Preheader && "LoopSimplify should ensure a dedicated preheader.");
  assert(Latch && "LoopSimplify should ensure a single backedge.");

  DEBUG(dbgs() << "Initial loop latch is: " << *Latch);

  // Basic profitability analysis: only parallelize a loop if it accesses
  // memory (or it has subloops or function calls that could access memory).
  if (!ProceedAtAllCosts && !mayWriteToMemory() && !mayReadFromMemory()) {
    ORE.emit(OptimizationRemarkAnalysis(PASS_NAME, "NotProfitable",
                                        L.getStartLoc(), Header)
             << "Not parallelizing loop as it is not profitable");
    return nullptr;
  }

  if (!L.hasDedicatedExits()) {
    formDedicatedExitBlocks(&L, &DT, &LI, true);
    if (!L.hasDedicatedExits()) {
      // Note that even makeChain() calls Loop::getUniqueExitBlocks which expects
      // dedicated exit blocks.
      ORE.emit(OptimizationRemarkMissed(PASS_NAME, "SharedExit",
                                        L.getStartLoc(), Header)
               << "Unable to rewrite loop to have dedicated exits.");
      return nullptr;
    }
  }

  DEBUG(assertVerifyFunction(F, "Before detaching loop iterations", &DT, &LI));

  DEBUG(dbgs() << "Detaching each iteration of loop: " << L);

  // We will insert new instructions that inherit DebugLocs from the
  // terminators of the Preheader and Latch.
  // Let's try to ensure they have non-null DebugLocs.
  DebugLoc Loc = getSafeDebugLoc(L);
  if (!Loc) {
    ORE.emit(OptimizationRemark(PASS_NAME, "NoDebugLoc",
                                L.getStartLoc(), Header)
             << "Could not find a safe DebugLoc for loop");
  }
  assert((Loc || !F.getSubprogram()) && "Loop lacks needed debuginfo");
  if (!Preheader->getTerminator()->getDebugLoc())
    Preheader->getTerminator()->setDebugLoc(Loc);
  if (!Latch->getTerminator()->getDebugLoc())
    Latch->getTerminator()->setDebugLoc(Loc);

  IntegerType *TimestampType = getLoopExpandableTimestampType();
  if (TimestampType && canonicalize(TimestampType)) {
    return getUniqueNonDeadendExitBlock(L);
  } else if (ProceedAtAllCosts) {
    if (!TimestampType)
      TimestampType = DynamicTimestampTy;
    makeChain(TimestampType);
    return ContinuationEnd;
  } else {
    ORE.emit(OptimizationRemarkAnalysis(PASS_NAME, "BailingChain",
                                        L.getStartLoc(), Header)
             << "Not parallelizing loop that would be a pointless serial chain");
    return nullptr;
  }
}


/// Attempt to detach each iteration of the loop body in the canonical form
/// expected by LoopExpansion. Each task should have timestamp equal to the
/// iteration index and of type TimestampType.
/// A minimal set of "bookkeeping" instructions will not be detached, but
/// will be sunk to the latch, because they are needed there to determine
/// the spawning of the next iteration of the loop.
/// \return true on success.
bool LoopIterDetacher::canonicalize(IntegerType *TimestampType) {
  SmallPtrSet<const Instruction *, 8> Bookkeeping;
  // Gather computation slices that produce values the next iteration will
  // need.  For example, this may be pointer-chasing, or updating an
  // accumulator variable with partial results on each iteration.
  addLoopCarriedDependencies(Bookkeeping);
  if (!isSafeToSink(Bookkeeping)) {
    ORE.emit(OptimizationRemarkMissed(PASS_NAME, "ImmovableCarriedValues",
                                      L.getStartLoc(), Header)
             << "Unable to canonicalize loop for high parallelism "
                "because of immovable loop-carried dependencies.");
    return false;
  }

  if (!sanitizeContinuation())
    return false;

  // Some loop carried dependencies may have been eliminated by replacing loop
  // output computations (e.g., by rewriting reductions for parallelism).
  {
    SmallPtrSet<const Instruction *, 8> NewBookkeeping;
    addLoopCarriedDependencies(NewBookkeeping);
    assert(all_of(NewBookkeeping, [&Bookkeeping](const Instruction *I) {
                                    return Bookkeeping.count(I); }) &&
           "Bookkeeping after eliminating dependencies should be subset");
    assert(isSafeToSink(NewBookkeeping));
    Bookkeeping = std::move(NewBookkeeping);
  }

  sanitizeExiting(Bookkeeping);

  assert(isSafeToSink(Bookkeeping) &&
         "Latch unification left loop-carried dependencies immovable?");

  // Gather the computation slice that determines whether the loop will exit or
  // continue on to the next iteration.  This may include, e.g., loading an
  // in-memory variable to check an exit condition on each iteration.
  addLatchExitComputation(Bookkeeping);
  // Now we have identified all "bookkeeping" computation that is needed to
  // produce values needed at the end of each iteration to enable the launch of
  // the next iteration.  Can we safely rewrite the loop to put (possibly a
  // copy of) these instructions at the end of each iteration?  This bookeeping
  // won't form part of the body of the loop that we spawn, so we'll count on
  // LoopExpansion to ensure any latch loads are correctly timestamp-ordered
  // after the loop body.
  if (!isSafeToSink(Bookkeeping)) {
    ORE.emit(OptimizationRemarkMissed(PASS_NAME, "ImmovableExitCond",
                                      L.getStartLoc(), Header)
             << "TODO: canonicalize loops with immovable exit comdition "
                "computation via done variables?");
    return false;
  }

  // Create a new latch block that initially contains only the backedge branch.
  // More bookkeeping may be sunk into the latch later.
  BasicBlock *const OldLatch = Latch;
  Latch = SplitBlock(Latch, Latch->getTerminator(), &DT, &LI);
  Latch->setName(OldLatch->getName() + ".newlatch");
  assert(Latch->size() == 1u);
  // N.B. OldLatch might be the same as Header.
  auto LatchBranch = cast<BranchInst>(OldLatch->getTerminator());

  // Ensure the header initially contains only its phis.
  BasicBlock* Detached = SplitBlock(Header, Header->getFirstNonPHI(), &DT, &LI);

  assert(L.getHeader() == Header && "Loop didn't track new header?");
  assert(L.getLoopLatch() == Latch && "Loop didn't track new latch?");
  DEBUG(assertVerifyFunction(F, "After splitting header and latch", &DT, &LI));

  // Some bookkeeping instructions may have already been ordered at the end of
  // the loop iteration, after any code we want to keep in the loop body that
  // may have side effects, in the original serial code's program order.
  // Take that code and directly move it to the latch, without any need for
  // checking whether any of that code needs a copy to stay in the loop.
  for (Instruction *I = LatchBranch->getPrevNode(); !!I;) {
    Instruction *Prev = I->getPrevNode();  // to avoid iterator invalidation
    DEBUG(dbgs() << "Considering whether to move instruction to latch: "
                 << *I << '\n');
    assert(all_of(I->users(), [this](const User *U) {
      auto *I = cast<Instruction>(U);
      const BasicBlock* BB = I->getParent();
      return BB == Latch || (BB == Header && isa<PHINode>(I));
    }));
    if (isa<PHINode>(I) || mayHaveSideEffects(I)) {
      DEBUG(dbgs() << "Stopping latch movement at: " << *I << '\n');
      break;
    }
    assert(isSafeToSink(I));
    if (isInstructionTriviallyDead(I)) {
      // Induction variable substitution may have left dead "increment"
      // computations near the latch.
      DEBUG(dbgs() << "Removing dead instruction: " << *I << "\n");
      I->eraseFromParent();
    } else {
      DEBUG(dbgs() << "Moving instruction to latch: " << *I << '\n');
      assert(Bookkeeping.count(I) || isa<DbgInfoIntrinsic>(I) &&
             "Since this instruction is not used for any side effects within"
             "its loop iteration, it must used for the next iteration");
      I->moveBefore(*Latch, Latch->getFirstInsertionPt());
    }
    I = Prev;
  }

  // Move or copy any remaining bookkeeping computations out of the body.
  moveFromLoopBody(DT[Detached], Bookkeeping);
  assert(all_of(Bookkeeping, [this](const Instruction *I) {
    const BasicBlock *BB = I->getParent();
    return BB == Latch || (BB == Header && isa<PHINode>(I));
  }));
  DEBUG(assertVerifyFunction(F, "After sinking bookkeeping", &DT, &LI));

  // Now that we've identified the loop body vs. the loop bookkeeping,
  // we can actually detach the body.

  SCEVExpander Exp(SE, M.getDataLayout(), "ts");
  PHINode* LoopIndex =
          Exp.getOrInsertCanonicalInductionVariable(&L, TimestampType);
  DEBUG(dbgs() << "Timestamp induction variable " << *LoopIndex << "\n");

  DEBUG(assertVerifyFunction(F, "After computing timestamps", &DT, &LI));

  SDetachInst* DI;
  detachTask(Header->getTerminator(),
             Latch->getFirstNonPHI(),
             LoopIndex,
             nullptr,
             DetachKind::UnexpandedIter,
             Header->getName() + ".loopiter",
             &DT, &LI,
             &DI);
  DI->setMetadata(SwarmFlag::TempNullDomain, MDNode::get(F.getContext(), {}));
  Latch = DI->getContinue();
  assert(L.getLoopLatch() == Latch);
  if (!DI->getDebugLoc())
    DI->setDebugLoc(Preheader->getTerminator()->getDebugLoc());

  DEBUG(dbgs() << "canonicalized loop header:" << *Header);
  DEBUG(dbgs() << "canonicalized loop latch:" << *Latch);

  DEBUG(assertVerifyFunction(F, "After canonicalization", &DT, &LI));
  assert(isExpandableSwarmLoop(&L, DT) && "Should be in canonical form");

  ORE.emit(OptimizationRemark(PASS_NAME, "CanonicalLoop",
                              L.getStartLoc(), Header)
           << "detached each iteration of this loop body in canonical form");

  LoopsCanonicalized++;

  return true;
}


/// Unpack the closure before instruction UnpackBefore, but do not substitute
/// any users. Fill OldValueMap with new load -> old value
static void unpackClosureNoSubsitutes(
        Value *Closure,
        ArrayRef<Value *> Captures,
        Instruction *UnpackBefore,
        ValueToValueMapTy *OldValueMap) {
  assert(!Captures.empty());

  StructType *ClosureType = getClosureType(Closure);
  IRBuilder<> B(UnpackBefore);
  for (unsigned i = 0; i < Captures.size(); i++) {
    // Unpack captured values from the closure struct
    auto UnpackGEP = B.CreateConstInBoundsGEP2_32(ClosureType, Closure, 0, i);
    Value *Original = Captures[i];
    LoadInst *Unpacked = B.CreateLoad(UnpackGEP, Original->getName() + ".unpack");
    Unpacked->setMetadata(SwarmFlag::Closure,
                          MDNode::get(Unpacked->getContext(), {}));
    addSwarmMemArgsMetadata(Unpacked);
    (*OldValueMap)[Unpacked] = Original;
  }
}


// TODO(victory): Deduplicate this with LoopExpander's version, either by
//                making it a shared utility function, or implemented in a
//                utility base class that we inherit from.
// Transform the loop into a recursive form, and detach each entire iteration.
// It will not be transformed by later passes that expect canonical form.
void LoopIterDetacher::makeChain(IntegerType *TimestampType) {
  ORE.emit(OptimizationRemark(PASS_NAME, "Chain", L.getStartLoc(), Header)
           << "pipelining chain of iterations");

  BasicBlock *Preheader = L.getLoopPreheader();

  // Compute the timestamp
  //TODO(victory): Deduplicate this computation with any existing canonical IV.
  IRBuilder<> Builder(Header->getFirstNonPHI());
  PHINode *TS = Builder.CreatePHI(TimestampType, 2, "ts");
  TS->addIncoming(ConstantInt::get(TimestampType, 0), Preheader);
  Builder.SetInsertPoint(Latch->getTerminator());
  Value *NextIterTS = Builder.CreateAdd(TS, ConstantInt::get(TimestampType, 1),
                                        "next_ts");
  TS->addIncoming(NextIterTS, Latch);
  DEBUG(assertVerifyFunction(F, "After creating recursive timestamp", &DT, &LI));

  // Get the set of blocks associated with the current loop to be outlined.
  SmallSetVector<BasicBlock *, 8> BlockSet(L.block_begin(), L.block_end());

  // Include continuation blocks, up to but not including ContinuationEnd.
  SmallVector<BasicBlock *, 8> ExitBlocks;
  L.getUniqueExitBlocks(ExitBlocks);
  SmallVector<BasicBlock *, 8> EHExits;
  for (BasicBlock *Exit : ExitBlocks) {
    DEBUG(dbgs() << "Examining loop exit " << Exit->getName() << '\n');
    assert(DT.dominates(Header, Exit) && "Exits should still be dedicated");

    if (Exit == ContinuationEnd)
      continue;

    SmallSetVector<BasicBlock *, 8> ReachableBBs;
    //TODO(victory): Avoid doing so much copying. This is potentially
    // wastefully making many copies of shared continuation blocks.
    BasicBlock *ReattachingBlock = makeReachableDominated(Exit, ContinuationEnd,
                                                          DT, LI,
                                                          ReachableBBs);
    if (!ReattachingBlock) {
      EHExits.push_back(Exit);
      continue;
    }

    DEBUG(dbgs() << "Detaching loop continuation back to superdomain.\n");
    assert(!ReachableBBs.count(Header));
    assert(none_of(ExitBlocks, [&ReachableBBs, Exit](BasicBlock *BB) {
      return BB != Exit && ReachableBBs.count(BB); }));
    SDetachInst *DI = nullptr;
    detachTask(Exit->getFirstNonPHI(),
               ReattachingBlock->getTerminator(),
               ConstantInt::get(TimestampType, 1),
               OuterDomain,
               DetachKind::SubsumedCont,
               "recursive_loop_cont",
               &DT,
               &LI,
               &DI);
    DI->setRelativeTimestamp(true);
    DEBUG(assertVerifyFunction(F, "After detaching continuation", &DT, &LI));

    assert(!ReachableBBs.count(DI->getDetached()));
    ReachableBBs.insert(DI->getDetached());
    assert(!ReachableBBs.count(DI->getContinue()));
    ReachableBBs.insert(DI->getContinue());

    assert(none_of(ReachableBBs, [&BlockSet](BasicBlock *Reachable) {
      return BlockSet.count(Reachable); }));
    BlockSet.insert(ReachableBBs.begin(), ReachableBBs.end());
  }
  for (BasicBlock *EHExit : EHExits) {
    SmallSetVector<BasicBlock *, 8> ReachableBBs;
    bool CanReachEnd = makeReachableDominated(EHExit, ContinuationEnd,
                                              DT, LI,
                                              ReachableBBs);
    assert(!CanReachEnd); (void)CanReachEnd;
    BlockSet.insert(ReachableBBs.begin(), ReachableBBs.end());
  }
  assert(!isa<PHINode>(ContinuationEnd->front()));
  assert(!BlockSet.count(ContinuationEnd));
  assert(any_of(predecessors(ContinuationEnd),
                [&BlockSet](BasicBlock *Pred) {
                  return BlockSet.count(Pred);
                }));
  DEBUG(assertVerifyFunction(F, "After detaching continuations", &DT, &LI));

  // Shrink task inputs, but avoid sinking non-loop-invariant ones
  SmallPtrSet<Value *, 4> Blacklist;
  for (const PHINode &PN : Header->phis()) {
    Value *In = PN.getIncomingValueForBlock(Preheader);
    if (!isa<Constant>(In))
      Blacklist.insert(In);
  }
  Blacklist.insert(NextIterTS);  // used as timestamp, which is always an input
  //TODO(victory): Change shrinkInputs() API to look more like our outline()
  // utility, allowing us to reduce the boilerplate code here.
  {
    SmallVector<BasicBlock *, 32> BlocksVec;
    BlocksVec.push_back(Header);
    for (BasicBlock *BB : BlockSet)
      if (BB != Header) {
        assert(DT.dominates(Header, BB));
        BlocksVec.push_back(BB);
      }
    shrinkInputs(BlocksVec, Blacklist, TTI, &ORE);
  }

  // If loop-invariant inputs don't fit in registers, use a shared closure.
  // NOTE: Based on code for LoopExpansion's makeChain(), but much simpler
  // since there's a single task that aligns with the outlined call.
  // However, since some of the outlined blocks are not dominated by the
  // header, we replace the uses of all values in the outlined function, not in
  // the original code. This is because replacing the uses beforehand, then
  // restoring the uses of non-dominated blocks to the original values after
  // outlining caused strange compiler crashes.
  ValueToValueMapTy OldValueMap;
  SmallVector<Instruction *, 8> ClosureFrees;
  if (!DisableEnvSharing) {
    SetVector<Value *> Inputs;
    findInputsNoOutputs(BlockSet, Inputs);

    // Gather non-loop-invariant inputs first
    SetVector<Value *> Args;
    for (const PHINode &PN : Header->phis()) {
      Value *In = PN.getIncomingValueForBlock(Preheader);
      assert(isa<Constant>(In) || Inputs.count(In));
      Args.insert(In);
    }
    Args.insert(TS);
    uint32_t NLIArgs = Args.size();

    // Add loop-invariant inputs
    for (Value *Input : Inputs)
      Args.insert(Input);

    //DEBUG({
    //  dbgs() << "Candidate MemArgs (NLI=" << NLIArgs << "):\n";
    //  for (Value *Arg : Args)
    //    dbgs() << "Arg: " << *Arg << "\n";
    //});

    SetVector<Value *> MemArgs =
        getMemArgs(Args, F.getParent()->getDataLayout(), NextIterTS, NLIArgs);

    if (MemArgs.size()) {
      DEBUG({
        dbgs() << "Creating shared loop closure (NLI=" << NLIArgs << "):\n";
        for (Value *Arg : Args) dbgs() << "  Arg: " << *Arg << "\n";
        for (Value *MemArg : MemArgs) dbgs() << "  MemArg: " << *MemArg << "\n";
      });
      Instruction *MemArgsPtr = createClosure(
          MemArgs.getArrayRef(), Preheader->getTerminator(),
          "chain_mem_args");
      unpackClosureNoSubsitutes(
          MemArgsPtr, MemArgs.getArrayRef(),
          Header->getFirstNonPHI(),
          &OldValueMap);

      // Finally, free MemArgsPtr at all exit blocks
      // [victory] This depends on dedicated exits to avoid double frees.
      for (BasicBlock *ExitBlock : ExitBlocks) {
        if (!BlockSet.count(ExitBlock)) {
          // corner case for a loop with no continuation
          assert(ExitBlock == ContinuationEnd);
          const SmallSetVector<BasicBlock *, 8> ExitingBlocks(
              pred_begin(ContinuationEnd), pred_end(ContinuationEnd));
          assert(all_of(ExitingBlocks, [this](const BasicBlock *Pred) {
            return L.contains(Pred); }));
          ExitBlock = SplitBlockPredecessors(ContinuationEnd,
                                             ExitingBlocks.getArrayRef(),
                                             ".newexit", &DT, &LI);
          assert(ExitBlock); (void)ExitBlock;
          BlockSet.insert(ExitBlock);
        }
        Instruction *Free =
            CallInst::CreateFree(MemArgsPtr, ExitBlock->getTerminator());
        addSwarmMemArgsForceAliasMetadata(cast<CallInst>(Free));
        ClosureFrees.push_back(Free);
      }
    }
  }

  // Get live-in values that need to be passed because they are used by Blocks.
  SetVector<Value *> Inputs;
  findInputsNoOutputs(BlockSet, Inputs);

  SetVector<Value *> Params;
  SmallVector<Value *, 8> StartArgs;

  // The parameters shall start with one parameter for each header phi.
  SmallVector<Instruction *, 8> ParamDummies;
  for (const PHINode &PN : Header->phis()) {
    Value *In = PN.getIncomingValueForBlock(Preheader);
    assert(isa<Constant>(In) || Inputs.count(In));
    StartArgs.push_back(In);
    Instruction *ParamDummy = createDummyValue(PN.getType(), PN.getName(),
                                               Preheader->getTerminator());
    ParamDummies.push_back(ParamDummy);
    Params.insert(ParamDummy);
  }

  // Add the other loop-invariant inputs as parameters
  for (Value *In : Inputs)
    Params.insert(In);

  DEBUG(assertVerifyFunction(F, "Before outlining loop", &DT, &LI));

  ValueToValueMapTy VMap;
  Function *RecursiveLoop = outline(Params, BlockSet, Header, ".chain", VMap);
  RecursiveLoop->addFnAttr(Attribute::AlwaysInline); // inline into task function
  RecursiveLoop->setCallingConv(CallingConv::Fast);
  assert(RecursiveLoop->hasFnAttribute(SwarmFlag::Parallelizing));
  DEBUG(assertVerifyFunction(*RecursiveLoop, "Outlined loop is invalid"));

  for (Instruction *ParamDummy : ParamDummies)
    ParamDummy->eraseFromParent();

  // If using a shared closure, replace the uses of values in the closure in
  // the outlined function. Note the outlined function may have a bunch of dead
  // arguments; this is OK, as the function is inlined (and the args would be
  // dead-arg eliminated otherwise).
  if (!OldValueMap.empty()) {
    for (auto ValuePair : OldValueMap) {
      Value *New = VMap[ValuePair.first];
      Value *Old = VMap[ValuePair.second];
      auto UI = Old->use_begin(), E = Old->use_end();
      while (UI != E) {
        Use &U = *(UI++); // Carefully crafted to avoid iterator invalidation
        U.set(New);
      }
    }
  }

  // Similarly, since the outlined function always frees the closure, remove
  // all calls to free() from the original, maybe non-dominated blocks.
  for (Instruction* I : ClosureFrees) {
    auto *FreeCall = cast<CallInst>(I);
    auto *PointerCast = cast<BitCastInst>(FreeCall->getArgOperand(0));
    FreeCall->eraseFromParent();
    PointerCast->eraseFromParent();
  }

  // Now that the loop is outlined, transform it into a chain of tasks.
  CallInst *RecurCall = formRecursiveLoop(RecursiveLoop);
  SDetachInst *RecurDetach;
  detachTask(RecurCall,
             RecurCall->getNextNode(),
             VMap[NextIterTS],
             nullptr,
             DetachKind::EarlyChainIter,
             "recursive_loop_iter",
             nullptr, nullptr,
             &RecurDetach);
  RecurDetach->setIsSameHint(true);

  assertVerifyFunction(*RecursiveLoop, "Transformed recursive loop is invalid");

  // Detach first iteration with timestamp zero.
  CallInst *TopCall;
  {
    // Setup arguments for call.
    SmallVector<Value *, 4> TopCallArgs;
    TopCallArgs.append(StartArgs.begin(), StartArgs.end());
    //TODO(victory): Be more selective to avoid redundant args
    TopCallArgs.append(Inputs.begin(), Inputs.end());
    //DEBUG(for (Value *TCArg : TopCallArgs)
    //        dbgs() << "Top call arg: " << *TCArg << "\n";);

    // Create call instruction.
    IRBuilder<> Builder(Preheader->getTerminator());
    TopCall = Builder.CreateCall(RecursiveLoop, TopCallArgs);
    TopCall->setCallingConv(CallingConv::Fast);
    assert(TopCall->getCallingConv() == RecursiveLoop->getCallingConv());
    DEBUG(dbgs() << "Created call to outlined loop:\n " << *TopCall << '\n');
  }
  SDetachInst* DI;
  detachTask(TopCall, TopCall->getNextNode(),
             ConstantInt::get(TimestampType, 0), nullptr,
             DetachKind::EarlyChainIter, "first_loop_iter",
             &DT, &LI, &DI);
  DI->setMetadata(SwarmFlag::TempNullDomain, MDNode::get(F.getContext(), {}));

  DEBUG(assertVerifyFunction(F, "Transformed to chain of tasks", &DT, &LI));

  SE.forgetLoop(&L);
  eraseLoop(L, ContinuationEnd, DT, LI);

  ++LoopsChainedEarly;
}


LoopInvariantOutputReplacer::LoopInvariantOutputReplacer(
        const Loop &L,
        PHINode *OutputToReplace,
        const DominatorTree &DT)
    : L(L), DT(DT)
    , OutputToReplace(OutputToReplace)
    , InstrToCopy(dyn_cast_or_null<Instruction>(
                          OutputToReplace->hasConstantValue()))
{
  SmallVector<BasicBlock *, 4> Exits;
  L.getExitBlocks(Exits);
  assert(is_contained(Exits, OutputToReplace->getParent())
         && "Output is not in any of this loop's exit blocks");
}


bool LoopInvariantOutputReplacer::canReplace() const {
  if (!InstrToCopy) {
    debugNotReplaceable("because the loop output phi does not have a "
                        "sole incoming instruction from the loop");
    return false;
  }

  if (!L.contains(InstrToCopy)) {
    debugNotReplaceable("because the loop output phi takes a loop-invariant "
                        "value? TODO: investigate why this phi wasn't "
                        "optimized away. Should we run InstCombine?");
    return false;
  }

  DEBUG(dbgs() << "  InstrToCopy: " << *InstrToCopy << "\n");
  DEBUG(if (const DebugLoc &Loc = InstrToCopy->getDebugLoc()) {
      dbgs() << "  from ";
      Loc.print(dbgs());
      dbgs() << '\n';
  });

  if (!L.hasLoopInvariantOperands(InstrToCopy)) {
    debugNotReplaceable("because the instruction to copy does not have "
                        "loop-invariant operands");
    // TODO(mcj) maybe explore a data-flow subgraph of instructions?
    return false;
  }

  if (isSafeToSinkOrHoist(InstrToCopy))
    return true;

  if (auto *Load = dyn_cast<LoadInst>(InstrToCopy)) {
    DEBUG(dbgs() << "  Checking if load can be copied for loop continuation: "
                 << *Load << '\n');
    if (isInvariantInPath(Load->getPointerOperand(), Load, OutputToReplace)) {
      DEBUG(dbgs() << "  Load can be copied.\n");
      return true;
    }
  }

  debugNotReplaceable("because the instruction to copy may be unsafe to move "
                      "to the continuation");
  return false;
}


bool LoopInvariantOutputReplacer::run() {
  DEBUG(dbgs() << "Checking if output phi can be replaced: "
               << *OutputToReplace << "\n");

  if (!canReplace()) return false;

  DEBUG(dbgs() << "Copying InstrToCopy to break dependence between last "
                  "iteration and loop continuation.\n");

  Instruction *Copy = InstrToCopy->clone();
  Copy->setName(InstrToCopy->getName() + ".loop-invar-copy");
  Copy->setDebugLoc(InstrToCopy->getDebugLoc());
  if (!Copy->getDebugLoc()) Copy->setDebugLoc(OutputToReplace->getDebugLoc());
  assert(L.hasLoopInvariantOperands(Copy));
  Copy->insertBefore(&*OutputToReplace->getParent()->getFirstInsertionPt());
  OutputToReplace->replaceAllUsesWith(Copy);
  OutputToReplace->eraseFromParent();

  // InstrToCopy may now be dead if its only user was OutputToReplace.
  // This cleanup could be left to a later DCE pass, but we do it now to make
  // it simpler to debug subsequent movement of instructions around the latch.
  if (isInstructionTriviallyDead(InstrToCopy))
    InstrToCopy->eraseFromParent();

  DEBUG(assertVerifyFunction(*Copy->getFunction(),
                             "Replaced loop output phi with a copy of its "
                             "incoming instruction",
                             &DT));
  return true;
}


void LoopInvariantOutputReplacer::debugNotReplaceable(
        const Twine &Reason) const {
  DEBUG(dbgs() << "instruction in loop cannot be copied to break dependency "
               << "between the last iteration and the loop continuation "
               << Reason << ".\n");
}


BasicBlock *llvm::detachLoopIterations(
        Loop &L, DeepenInst *OuterDomain, BasicBlock *ContinuationEnd,
        bool ProceedAtAllCosts,
        AssumptionCache &AC, DominatorTree &DT, LoopInfo &LI,
        TargetLibraryInfo &TLI, TargetTransformInfo &TTI,
        OptimizationRemarkEmitter &ORE) {
  ScalarEvolution SE(*L.getHeader()->getParent(), TLI, AC, DT, LI);
  return LoopIterDetacher(L, OuterDomain, ContinuationEnd, ProceedAtAllCosts,
                          DT, LI, SE, TTI, ORE).run();
}
