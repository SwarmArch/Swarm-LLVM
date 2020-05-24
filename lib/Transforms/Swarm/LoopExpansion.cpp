//===- LoopExpansion.cpp - Launch parallel loop iterations efficiently ----===//
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
// Restructure Swarm loop task trees to launch many iterations in parallel.
//
// Before this pass runs, a Swarm loop looks like a flat tree, where the
// parent directly enqueues every loop iteration as a child task.
// After this pass runs, the loop will generate a taller task tree,
// with a smaller fanout per node, that may better dynamically expand
// to fill the system with parallel work.
//
//===----------------------------------------------------------------------===//

#include "LoopCoarsen.h"
#include "Utils/CFGRegions.h"
#include "Utils/Flags.h"
#include "Utils/Misc.h"
#include "Utils/Reductions.h"
#include "Utils/SCCRT.h"
#include "Utils/SwarmABI.h"
#include "Utils/Tasks.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationDiagnosticInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/SwarmAA.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Swarm.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define LE_NAME "swarm-loop-expansion"
#define DEBUG_TYPE LE_NAME

#define LOOP_EXPANDED "SwarmEnqueuer"
#define BALANCED_ENQUEUER "SwarmBalancedEnqueuer"

STATISTIC(LoopsAnalyzed, "Number of Swarm loops analyzed");
STATISTIC(LoopsConvertedToBalanced,
          "Number of Swarm loops converted to balanced tree expansion");
STATISTIC(LoopsConvertedToProgressive,
          "Number of Swarm loops converted to progressive expansion");
STATISTIC(LoopsConvertedToChain,
          "Number of Swarm loops converted to pipelined chain");

static cl::opt<unsigned> EnqueuesPerTask("swarm-enqspertask", cl::init(8),
    cl::desc("Number of loop iterations enqueued by each leaf enqueuer task"));
static cl::opt<unsigned> SpawnerFanout("swarm-fanout", cl::init(4),
    cl::desc("Number of children spawners per parent spawner"));
static cl::opt<bool> ChainExpansionOnly("swarm-chainexpandonly", cl::init(false),
    cl::desc("Disable balanced tree and progressive expansion"));
static cl::opt<bool> DisableUnboundedTrees("swarm-disableunboundedtrees", cl::init(false),
    cl::desc("Disable progressive expansion of unbounded loops"));
static cl::opt<unsigned> LateChainLocalEnqueueMask("swarm-latechainlocalmask",
    cl::init(0x07),
    cl::desc("Mask applied to late-chain body timestamps to "
             "determine when to enqueue their spawners locally"));

namespace {
// Forward declarations.

/// LoopExpander implements transforming a Swarm loop that serially
/// enqueues tasks into an expanding tree of parallel task enqueuing.
class LoopExpander {
private:
  /// The original loop.
  Loop *const L;
  Loop *const Prolog;
  Loop *const Epilog;

  DominatorTree *const DT;
  LoopInfo *const LI;
  ScalarEvolution &SE;
  // PredicatedScalarEvolution &PSE;
  const TargetTransformInfo& TTI;
  OptimizationRemarkEmitter &ORE;

  Function *const F;
  Module *const M;
  BasicBlock *const Header;
  BasicBlock *const Latch;
  BasicBlock *const ExitBlock;
  SmallVector<BasicBlock *, 4> ExitBlocks;

  SCEVExpander Exp;

  const SCEV *Limit;

  DeepenInst *const Domain;

  const DebugLoc DLoc;

  bool MustSpawnLatch = false;

public:
  LoopExpander(Loop *L,
               DominatorTree *DT,
               LoopInfo *LI,
               ScalarEvolution &SE,
               const TargetTransformInfo& TTI,
               OptimizationRemarkEmitter &ORE)
    : L(L),
      Prolog(getProlog(*L, *LI)),
      Epilog(getEpilog(*L, *LI)),
      DT(DT), LI(LI), SE(SE), TTI(TTI), ORE(ORE),
      F(L->getHeader()->getParent()),
      M(F->getParent()),
      Header(L->getHeader()),
      Latch(L->getLoopLatch()),
      ExitBlock(getUniqueNonDeadendExitBlock(*L)),
      Exp(SE, M->getDataLayout(), "le"),
      Limit(SE.getBackedgeTakenCount(L)),
      Domain(getDomain(cast<SDetachInst>(Header->getTerminator()))),
      DLoc(L->getStartLoc())
  {
    assert(Header);
    assert(Latch && "No single loop latch found for loop.");
    assert(ExitBlock && "No unique non-dead-end exit block for loop.");
    assert(Domain || !F->hasFnAttribute(SwarmFlag::Parallelized) &&
           "Normal autoparallelized code should know its domains.");

    TerminatorInst *TI = Latch->getTerminator();
    assert(TI->getNumSuccessors() == 2 && "Guaranteed by canonical form");
    assert(is_contained(TI->successors(), ExitBlock) &&
           "The unique non-dead-end exit block should be a latch successor");

    assert(L->hasDedicatedExits() && "Guaranteed by canonical form");
    L->getUniqueExitBlocks(ExitBlocks);

    assert(!isa<PHINode>(ExitBlock->front()));
  }

  bool run();

private:
  bool isNondecreasing(const SCEV* S) const;
  bool isIncreasing(const SCEV* S) const;

  ICmpInst *canonicalizeLoopLatch(PHINode *IV, Value *Limit);
  void getSanitizedBlocks(SmallSetVector<BasicBlock *, 8> &Blocks);
  Instruction *makeClosure(ArrayRef<Value *> Inputs,
                           const SmallSetVector<BasicBlock *, 8> &Blocks);
  Function *outline(const SetVector<Value *> &Params,
                    const SmallSetVector<BasicBlock *, 8> &Blocks,
                    ValueToValueMapTy &VMap) const;
  void eraseLoop();

  CallInst *makeChain();
  CallInst *makeParallelExpansion();
  CallInst *makeBalanced();
  CallInst *makeProgressive();
  bool isSmallLoop() const;
  bool mustSpawnLatch(SDetachInst *IterDetach) const;
  Instruction *createChainClosure();

  // victory: The following methods are static because they act on newly created
  // Functions, and therefore should not have access to the non-static fields
  // of this class, which refer to parts of the original Function the code was
  // outlined from. When these methods run, there are no existing analyses
  // for the new Functions that should be accessed or updated.

  static void implementRecursiveTree(BasicBlock *Preheader,
                                     PHINode *CanonicalIV,
                                     Argument *Limit,
                                     bool CanonicalIVFlagNUW,
                                     bool CanonicalIVFlagNSW);
  static void implementRecursiveChain(Function *Enqueuer);
};

} // end anonymous namespace


// Returns true if we can guarantee S never decreases.
// Returns false if S may decrease or we aren't sure.
bool LoopExpander::isNondecreasing(const SCEV* S) const {
  if (isa<SCEVConstant>(S))
    return true;

  if (auto SN = dyn_cast<SCEVAddRecExpr>(S)) {
    const SCEV *Step = SN->getStepRecurrence(SE);
    if (const auto *ConstStep = dyn_cast<SCEVConstant>(Step))
      return !ConstStep->getValue()->isNegative();
    if (const auto *StepRecurrence = dyn_cast<SCEVAddRecExpr>(Step)) {
      // This check is a bit too strict, but it will do.
      // If all the coefficients on the polynomial are nonnegative,
      // then the polynomial can never evaluate to a negative value
      // (over the domain of nonnegative integers)
      return all_of(StepRecurrence->operands(), [] (const SCEV *Op) {
        auto *Coefficient = dyn_cast<SCEVConstant>(Op);
        return Coefficient && !Coefficient->getValue()->isNegative();
      });
    }
  }

  return false;
}


// Returns true if we can guarantee S increases from each iteration to the next.
// Returns false if S might not increase or we aren't sure.
bool LoopExpander::isIncreasing(const SCEV* S) const {
  if (isa<SCEVConstant>(S))
    return false;

  if (const auto *SN = dyn_cast<SCEVAddRecExpr>(S)) {
    const SCEV *Step = SN->getStepRecurrence(SE);
    if (const auto *ConstStep = dyn_cast<SCEVConstant>(Step))
      return ConstStep->getAPInt().isStrictlyPositive();
    if (const auto *StepRecurrence = dyn_cast<SCEVAddRecExpr>(Step)) {
      // This check is a bit too strict, but it will do.
      auto *StepStart = dyn_cast<SCEVConstant>(StepRecurrence->getStart());
      return StepStart && StepStart->getAPInt().isStrictlyPositive()
             && isNondecreasing(StepRecurrence);
    }
  }

  return false;
}


/// \brief Replace the latch of the loop to check that IV is always less than or
/// equal to the limit.
///
/// This method assumes that the loop has a single loop latch.
ICmpInst* LoopExpander::canonicalizeLoopLatch(PHINode *IV, Value *Limit) {
  IRBuilder<> Builder(&*Latch->getFirstInsertionPt());

  // This process assumes that IV's increment is in Latch.

  // Create comparison between IV and Limit at top of Latch.
  auto *NewCondition = cast<ICmpInst>(Builder.CreateICmpULT(IV, Limit));

  // Replace the conditional branch at the end of Latch.
  BranchInst *LatchBr = cast<BranchInst>(Latch->getTerminator());
  assert(LatchBr->isConditional() &&
         "Latch does not terminate with a conditional branch.");
  Value *OldCondition = LatchBr->getCondition();

  BranchInst *NewLatchBr = BranchInst::Create(Header, ExitBlock, NewCondition);
  ReplaceInstWithInst(LatchBr, NewLatchBr);

  // Erase the old conditional branch.
  if (!OldCondition->hasNUsesOrMore(1))
    if (Instruction *OldCondInst = dyn_cast<Instruction>(OldCondition))
      OldCondInst->eraseFromParent();

  return NewCondition;
}


// Populate Blocks with the set of blocks associated with the current loop
// that should be outlined. May duplicate blocks to ensure a dedicated copy
// to be outlined exists that can be deleted after outlining.
void LoopExpander::getSanitizedBlocks(SmallSetVector<BasicBlock *, 8> &Blocks) {
  Blocks.insert(L->block_begin(), L->block_end());
  // Add doomed blocks.  There should not be any other exit blocks in the loop.
  for (BasicBlock *Exit : ExitBlocks) {
    if (Exit == ExitBlock) continue;
    SmallSetVector<BasicBlock *, 8> DoomedBBs;
    bool ExitCanReachEnd =
        makeReachableDominated(Exit, ExitBlock, *DT, *LI, DoomedBBs);
    assert(!ExitCanReachEnd);
    Blocks.insert(DoomedBBs.begin(), DoomedBBs.end());
  }
  assert(!Blocks.count(ExitBlock));
}


/// Allocate and pack a shared closure holding loop-invariant live-in values.
/// All iterations of the loop, including any prolog and epilog, shall load
/// the value from the shared closure as late as possible.  This is done to
/// reduce register pressure and mem_runners.
Instruction *LoopExpander::makeClosure(
        ArrayRef<Value *> Inputs,
        const SmallSetVector<BasicBlock *, 8> &Blocks) {
  // Find an instruction at which to allocate and pack the closure that is
  // guaranteed to dominate the loop including its prolog and epilog, if the
  // loop was coarsened.
  BasicBlock *Dominator = L->getLoopPreheader();
  if (Epilog)
    Dominator = DT->findNearestCommonDominator(Dominator,
                                               Epilog->getLoopPreheader());
  if (Prolog)
    Dominator = DT->findNearestCommonDominator(Dominator,
                                               Prolog->getLoopPreheader());
  // [victory] If there was a prolog, Dominator will be PrologStart.
  DEBUG(dbgs() << "Will allocate and pack shared closure at end of %"
               << Dominator->getName() << "\n");
  Instruction *AllocateAndPackBefore = Dominator->getTerminator();
  // [victory] The assertion below documents our reliance on a fragile
  // condition met by the careful structure of LoopCoarsen:  LoopCoarsen
  // minimizes values that need to be captured as arguments, and values that
  // need to be captured should be available in PrologStart.
  assert(all_of(Inputs, [this, AllocateAndPackBefore] (Value *V) {
    return !isa<Instruction>(V) ||
           DT->dominates(cast<Instruction>(V), AllocateAndPackBefore); }));
  Instruction *Closure =
      createClosure(Inputs, AllocateAndPackBefore, "shared_loop_closure");

  SmallVector<SDetachInst *, 4> IterDetaches;
  IterDetaches.push_back(cast<SDetachInst>(Header->getTerminator()));
  if (Prolog) IterDetaches.push_back(getPrologSDetach(*Prolog));
  if (Epilog) IterDetaches.push_back(getEpilogSDetach(*Epilog));
  assert(all_of(IterDetaches, [this](const SDetachInst *DI) {
    return DI->getDomain() == Domain; }));
  for (SDetachInst *IterDetach : IterDetaches) {
    unpackClosure(
        Closure, Inputs, [this, IterDetach](Instruction *Usr) -> Instruction * {
          if (!DT->dominates(IterDetach->getDetached(), Usr->getParent()))
            return nullptr;

          // Unfortunately progressive expansion requires loop environment
          // sharing.  We can only reduce the benefits for nested tasks
          // by placing the unpacking load as early as possible,
          // at the start of the iteration task.
          if (DisableEnvSharing)
            return IterDetach->getDetached()->getFirstNonPHI();

          SDetachInst *UsrTask = getEnclosingTask(Usr, *DT);
          assert(UsrTask);
          assert(UsrTask == IterDetach
              || DT->dominates(IterDetach->getDetached(), UsrTask->getParent()));

          // Good case: load immediately before user if in the loop domain.
          DeepenInst *UsrDomain = getDomain(UsrTask);
          if (UsrDomain && (Domain == UsrDomain))
            return Usr;

          assert(UsrTask != IterDetach);

          // Place load before the first superdomain detach
          // that might have exited the loop's domain.
          SmallVector<SDetachInst *, 8> TaskLineage;
          SDetachInst *Task = UsrTask;
          do {
            TaskLineage.push_back(Task);
            Task = getEnclosingTask(Task, *DT);
            assert(Task);
          } while (Task != IterDetach);
          int DomainDepth = 0;
          SDetachInst *PrevTask = IterDetach;
          for (SDetachInst *Task : reverse(TaskLineage)) {
            DomainDepth += getDomainDepthDiff(Task, *DT, PrevTask);

            if (DomainDepth < 0) {
              assert((!Task->getDomain() && !Domain) ||
                     (Task->getDomain() == Domain->getSuperdomain(*DT)));
              assert(getDomain(getEnclosingTask(Task, *DT)) == Domain);
              assert(Task->isSuperdomain());
              return Task;
            }

            PrevTask = Task;
          }

          // The user must actually be in the loop domain or a subdomain.
          // It's safe to place load immediately before user.
          assert(DomainDepth >= 0);
          assert(!Domain || UsrDomain);
          assert(!UsrDomain || DT->dominates(IterDetach->getDetached(),
                                             UsrDomain->getParent()));
          // Note: That UsrDomain is dominated by IterDetach->getDetached(),
          //       which is dominated by the loop domain, is not a sufficient
          //       condition to guarantee UsrDomain is a subdomain of the
          //       loop domain.  Consider the counterexample:
          //    %loopdomain = deepen();
          //    for (...) {
          //      spawn (i within %loopdomain) { // iter detach
          //        ...
          //        spawn_super { // subsumed continuation
          //          %another_domain = deepen(); // sibling of %loopdomain
          //          spawn (ts within %another_domain) { ... }
          //        }
          //      }
          //    }
          return Usr;
        });
  }

  // Handle remaining users outside the detached body, i.e. in the header,
  // latch, or continuation.
  unpackClosure(
      Closure, Inputs, [&Blocks, this](Instruction *Usr) -> Instruction * {
        BasicBlock *Block = Usr->getParent();
        if (!Blocks.count(Block)) return nullptr;

        // For Header and Latch, we generate per-BB loads
        if (Block == Header || Block == Latch)
          return Block->getFirstNonPHI();

        // For other users (i.e., the continuation), we load the value in
        // ExitBlock. Otherwise we'd have to free the closure somewhere in the
        // continuation...
        return ExitBlock->getFirstNonPHI();
      });

  return Closure;
}


// Create a new function containing a copy of Blocks, with parameters
// corresponding to Params, and populate VMap with a mapping from values
// associated with the old Blocks to the new copies.
Function *LoopExpander::outline(const SetVector<Value *> &Params,
                                const SmallSetVector<BasicBlock *, 8> &Blocks,
                                ValueToValueMapTy &VMap) const {
  Function *OutlinedLoop = llvm::outline(Params, Blocks, Header, ".le", VMap);
  assert(OutlinedLoop->hasFnAttribute(LOOP_EXPANDED));
  OutlinedLoop->setCallingConv(CallingConv::Fast);
  DEBUG(assertVerifyFunction(*OutlinedLoop, "Outlined loop is invalid"));
  return OutlinedLoop;
}


/// Delete blocks associated with the current loop.  Update DominatorTree and
/// LoopInfo.  Note that after this runs, the current loop is gone, so some
/// member variables of LoopExpander will be invalid.
void LoopExpander::eraseLoop() {
  SE.forgetLoop(L);
  Exp.clear();
  llvm::eraseLoop(*L, ExitBlock, *DT, *LI);
}


/// \brief Method to help makeBalanced() transform a function containing
/// a Swarm loop to recursively enqueue itself with subranges of iterations.
///
/// In pseudocode, given a function of the following form:
///
/// void f(iter_t start, iter_t end, ...) {
///   iter_t i = start;
///   ... Other loop setup ...
///   do {
///     spawn (ts(i)) { ... loop body ... };
///   } while (i++ < end);
/// }
///
/// Then this method transforms the function into the following form:
///
/// void f(iter_t start, iter_t end, ...) {
///   iter_t i = start;
///   ... Other loop setup ...
///
///   iter_t itercount = end - start + 1;
///   if (itercount > EnqueuesPerTask) {
///     count_t miditer = start + itercount / 2;
///     spawn (ts(start)) f(start, miditer-1, ...);
///     spawn (ts(miditer)) f(miditer, end, ...);
///     return;
///   }
///
///   do {
///     spawn (ts(i)) { ... Loop Body ... };
///   } while (i++ < end);
/// }
///
void LoopExpander::implementRecursiveTree(BasicBlock *Preheader,
                                          PHINode *const CanonicalIV,
                                          Argument *const Limit,
                                          bool CanonicalIVFlagNUW,
                                          bool CanonicalIVFlagNSW) {
  Function *const F = Preheader->getParent();
  assert(&(F->getEntryBlock()) == Preheader &&
         "Function does not start with Preheader");
  BasicBlock *const Entry = Preheader;
  Entry->setName("entry");
  BasicBlock *const Header = Preheader->getSingleSuccessor();

  DEBUG(dbgs() << "LE CanonicalIV: " << *CanonicalIV << "\n");
  assert(CanonicalIV->getParent() == Header &&
         "CanonicalIV does not belong to header");
  Value *CanonicalIVStart = CanonicalIV->getIncomingValueForBlock(Preheader);

  auto *OrigDetach = cast<SDetachInst>(Header->getTerminator());
  assert(!F->hasFnAttribute(SwarmFlag::Parallelized)
         || getDetachKind(OrigDetach) == DetachKind::UnexpandedIter);
  setDetachKind(OrigDetach, DetachKind::BalancedIter);
  assert(!OrigDetach->getDomain() && "Outlined detach separated from deepen");
  assert(!OrigDetach->isSubdomain() && !OrigDetach->isSuperdomain() &&
         "At this point the detach must not be subdomain nor superdomain.");
  const DebugLoc &DLoc = OrigDetach->getDebugLoc();
  assert((DLoc || !F->getSubprogram()) &&
         "Loop header detach lacks debug info");

  Constant *One = ConstantInt::get(Limit->getType(), 1);

  // Create branch to a new block based on itercount.
  IRBuilder<> Builder(Preheader->getTerminator());
  assert(EnqueuesPerTask >= 1U &&
         "-swarm-enqspertask must be at least 1");
  Value* MaxLeafEnqueues = ConstantInt::get(Limit->getType(), EnqueuesPerTask);
  Value *IterCount = Builder.CreateAdd(
          Builder.CreateSub(Limit, CanonicalIVStart, "limitdiff"),
          One, "itercount");
  TerminatorInst *ThenTerm = SplitBlockAndInsertIfThen(
          Builder.CreateICmpUGT(IterCount, MaxLeafEnqueues, "should_recur"),
          Entry->getTerminator(), /*Unreachable=*/true);
  Preheader = Entry->getTerminator()->getSuccessor(1);
  assert(Preheader->getSingleSuccessor() == Header);
  Preheader->setName("new_preheader");
  BasicBlock *const ThenBlock = ThenTerm->getParent();
  ThenBlock->setName("recur");

  // This new block to contain the recursive calls should end by returning.
  assert(isa<UnreachableInst>(ThenTerm));
  ReplaceInstWithInst(ThenTerm, ReturnInst::Create(F->getContext()));
  ThenTerm = nullptr;

  // Inside the new block, compute the subrange limits that will be passed
  // to the recursive calls.
  SmallVector<Value *, 8> SubrangeStarts = {CanonicalIVStart};
  SmallVector<Value *, 8> SubrangeEnds;
  Builder.SetInsertPoint(ThenBlock->getTerminator());
  assert(SpawnerFanout <= EnqueuesPerTask);
  Value *SubrangeSize = Builder.CreateUDiv(
      IterCount, ConstantInt::get(IterCount->getType(), SpawnerFanout),
      "subrange_size");
  for (unsigned i = 1; i < SpawnerFanout; ++i) {
    SubrangeStarts.push_back(
        Builder.CreateAdd(SubrangeStarts.back(), SubrangeSize, "subrange_start",
                          CanonicalIVFlagNUW, CanonicalIVFlagNSW));
    SubrangeEnds.push_back(
        Builder.CreateSub(SubrangeStarts.back(), One, "subrange_end",
                          CanonicalIVFlagNUW, CanonicalIVFlagNSW));
  }
  SubrangeEnds.push_back(Limit);

  // Now we will create the recursive calls and detach them with timestamps
  // based on those of the iteration tasks.
  Value *OrigTimestamp = OrigDetach->getTimestamp();
  DEBUG(dbgs() << "Original timestamp computation: " << *OrigTimestamp << '\n');

  // First, set up a small array to hold arguments for recursive calls.
  SmallVector<Value *, 8> RecurArgs(F->arg_size(), nullptr);
  // Recursive calls pass all but the first two parameters without change.
  for (unsigned i = 2; i < F->arg_size(); i++)
    RecurArgs[i] = &F->arg_begin()[i];

  BasicBlock *Prev = ThenBlock;
  for (unsigned i = 0; i < SpawnerFanout; ++i) {
    // Compute the right timestamp for the recursive detach.
    ValueToValueMapTy BaseMap;
    BaseMap[CanonicalIVStart] = SubrangeStarts[i];
    Builder.SetInsertPoint(Prev->getTerminator());
    Value *IterTS = copyComputation(OrigTimestamp,
                                    BaseMap,
                                    Builder,
                                    Preheader);

    // Create the recursive call instruction.
    RecurArgs[0] = SubrangeStarts[i];
    RecurArgs[1] = SubrangeEnds[i];
    CallInst *RecurCall = Builder.CreateCall(F, RecurArgs);
    RecurCall->setDebugLoc(DLoc);
    RecurCall->setCallingConv(CallingConv::Fast);
    assert(RecurCall->getCallingConv() == F->getCallingConv());

    // Detach the call.
    SDetachInst *DI;
    SReattachInst *RI;
    detachTask(RecurCall, RecurCall->getNextNode(), IterTS, nullptr,
               DetachKind::BalancedSpawner, "baltreechild" + Twine(i),
               nullptr, nullptr, &DI, &RI);
    DI->setDebugLoc(DLoc);
    // The left-side tasks of the tree can/should use EnqFlags::SAMEHINT to
    // 1) reduce the latency in starting the lowest-timestamp work
    // 2) marginally reduce task bandwidth
    // Note that swarm::__enqueuer trees use the same approach.
    if (i == 0) DI->setIsSameHint(true);
    DEBUG(dbgs() << "Built detach with timestamp based on:\n  "
                 << *SubrangeStarts[i] << "\nfor recursive call:\n"
                 << *RecurCall << '\n');
    RI->setDebugLoc(DLoc);

    Prev = RI->getDetachContinue();
  }
}


/// Outline the loop, transforming it so it recursively enqueues subranges
/// of its iterations in a balanced tree.
/// \returns the call instruction in the loop's original function that
/// initiates the enqueuing of iterations.
CallInst *LoopExpander::makeBalanced() {
  DEBUG(dbgs() << "Checking whether we can transform to balanced tree\n");

  if (isa<SCEVCouldNotCompute>(Limit)) {
    ORE.emit(OptimizationRemarkAnalysis(LE_NAME, "UnknownLoopLimit",
                                        DLoc,
                                        Header)
             << "could not compute limit");
    return nullptr;
  }
  if (auto ConstLimit = dyn_cast<SCEVConstant>(Limit)) {
    uint64_t limit = ConstLimit->getValue()->getLimitedValue();
    assert(limit < UINT64_MAX);
  }

  SDetachInst *OrigDetach = cast<SDetachInst>(Header->getTerminator());

  auto TimestampSCEV = SE.getSCEV(OrigDetach->getTimestamp());
  if (!isNondecreasing(TimestampSCEV)) {
    // TODO(victory): Eventually we should handle this case by using
    // swarm::timestamp() for the enqueuer timestamps.
    ORE.emit(OptimizationRemarkAnalysis(LE_NAME, "DecreasingTS", DLoc, Header)
             << "Timestamp is not guaranteed to be nondecreasing");
    DEBUG(dbgs() << "Timestamp SCEV: " << *TimestampSCEV << '\n');
    assert(!F->hasFnAttribute(SwarmFlag::Parallelized) &&
           "Auto-parallelized loop should have increasing timestamps");
    return nullptr;
  }

  // Checks complete, we can now begin transformation

  ORE.emit(OptimizationRemark(LE_NAME, "BalancedExpansion",
                              DLoc, Header)
           << "expanding loop using balanced tree");

  assert(!MustSpawnLatch
         && "since canonicalizeIVs eliminated loop-carried dependencies"
            "and since we can determine in advance when the loop will exit,"
            "-indvars and ADCE should have eliminated any loads in the latch."
            "If you see this, try more DCE.");
  OrigDetach->setMetadata(SwarmFlag::MustSpawnLatch, nullptr);

  BasicBlock *Preheader = L->getLoopPreheader();

  PHINode *CanonicalIV = cast<PHINode>(&Header->front());
  const SCEVAddRecExpr *CanonicalSCEV =
    cast<const SCEVAddRecExpr>(SE.getSCEV(CanonicalIV));
  assert(SE.isLoopBackedgeGuardedByCond(L, ICmpInst::ICMP_ULT,
                                        CanonicalSCEV, Limit) &&
         "Loop backedge is not guarded by canonical comparison with limit.");

  // Insert the computation for the loop limit into the Preheader.
  Value *LimitVar = Exp.expandCodeFor(Limit, Limit->getType(),
                                      Preheader->getTerminator());
  DEBUG(dbgs() << "LimitVar: " << *LimitVar << "\n");

  // Rewrite canonical IV to start at the start iteration argument
  Instruction *StartParam = createDummyValue(CanonicalIV->getType(), "start",
                                             Preheader->getTerminator());
  int PreheaderIdx = CanonicalIV->getBasicBlockIndex(Preheader);
  assert(PreheaderIdx >= 0);
  assert(cast<Constant>(CanonicalIV->getIncomingValue(PreheaderIdx)
                       )->isNullValue());
  CanonicalIV->setIncomingValue(PreheaderIdx, StartParam);
  DEBUG(dbgs() << "Rewritten header: " << *Header);

  // Rewrite the loop to exit when a end parameter value is reached.
  Instruction *EndParam = createDummyValue(LimitVar->getType(), "end",
                                           Preheader->getTerminator());
  ICmpInst *NewCond = canonicalizeLoopLatch(CanonicalIV, EndParam);
  DEBUG(dbgs() << "Rewritten Latch: " << *Latch);
  assert(NewCond->getOperand(0) == CanonicalIV);
  assert(NewCond->getOperand(1) == EndParam);

  DEBUG(assertVerifyFunction(*F, "Before gathering inputs", DT, LI));

  // Get the set of loop blocks to outline and their inputs.
  SmallSetVector<BasicBlock *, 8> Blocks;
  getSanitizedBlocks(Blocks);
  SetVector<Value*> Inputs;
  findInputsNoOutputs(Blocks, Inputs);
  assert(Inputs.count(StartParam));
  Inputs.remove(StartParam);
  assert(Inputs.count(EndParam));
  Inputs.remove(EndParam);
  DEBUG({
    for (Value *V : Inputs)
      dbgs() << "Loop input: " << *V << "\n";
  });

  // Allocate a shared closure to avoid repeatedly packing and unpacking
  // loop-invariant values through mem_runners for each spawner and iteration.
  // TODO(victory): Unify this with Daniel's MemArgs code to replace this with
  // a proper check on the ultimate size of params, and minimize the amount of
  // args passed through closures.
  if (Inputs.size() > 2 && !DisableEnvSharing) {
    DEBUG(dbgs() << "Capturing loop-invariant inputs to avoid repeatedly "
                    "passing them through mem_runner heap arguments.\n");
    Instruction *Closure = makeClosure(Inputs.getArrayRef(), Blocks);
    SDetachInst *DI = createSuperdomainDetachedFree(
        Closure, Domain ? Domain->getSuperdomain(*DT) : nullptr,
        Preheader->getTerminator(), DT, LI);
    Preheader = DI->getContinue();
    assert(Preheader == L->getLoopPreheader());
    DEBUG(assertVerifyFunction(*F, "After setting up closure", DT, LI));

    // Inputs are all captured, only need to pass closure now.
    Inputs.clear();
    findInputsNoOutputs(Blocks, Inputs);
    assert(Inputs.size() == 3);
    Inputs.remove(StartParam);
    Inputs.remove(EndParam);
    assert(Inputs.size() == 1);
    assert(Inputs.count(Closure));
  }

  /// Clone the loop into a new function.

  // The parameters start with the start and end iteration.
  SetVector<Value *> Params;
  Params.insert(StartParam);
  Params.insert(EndParam);
  assert(none_of(Params, [&Inputs](Value *P) -> bool {
    return Inputs.count(P); }));
  for (Value *V : Inputs)
    Params.insert(V);
  DEBUG({
    for (Value *V : Params)
      dbgs() << "Balanced spawner param: " << *V << "\n";
  });
  assert(Params.size() == 2 + Inputs.size());

  DEBUG(assertVerifyFunction(*F, "Before outlining", DT, LI));

  ValueToValueMapTy VMap;
  Function *Enqueuer = outline(Params, Blocks, VMap);

  Enqueuer->addFnAttr(BALANCED_ENQUEUER);
  Enqueuer->addFnAttr(Attribute::AlwaysInline);

  implementRecursiveTree(cast<BasicBlock>(VMap[Preheader]),
                         cast<PHINode>(VMap[CanonicalIV]),
                         cast<Argument>(VMap[EndParam]),
                         CanonicalSCEV->getNoWrapFlags(SCEV::FlagNUW),
                         CanonicalSCEV->getNoWrapFlags(SCEV::FlagNSW));

  assertVerifyFunction(*Enqueuer, "Transformed tree enqueuer is invalid");

  // Add call to enqueuer function in original function.
  CallInst *TopCall;
  {
    // Setup arguments for call.
    SmallVector<Value *, 4> TopCallArgs;
    // Add start iteration 0.
    assert(CanonicalSCEV->getStart()->isZero() &&
           "Canonical IV does not start at zero.");
    TopCallArgs.push_back(ConstantInt::get(CanonicalIV->getType(), 0));
    // Add loop limit.
    TopCallArgs.push_back(LimitVar);
    // Add the rest of the arguments.
    for (Value *V : Inputs)
      TopCallArgs.push_back(V);
    DEBUG({
      for (Value *TCArg : TopCallArgs)
        dbgs() << "Top call arg: " << *TCArg << "\n";
    });

    // Create call instruction.
    TopCall = CallInst::Create(Enqueuer,
                               TopCallArgs,
                               /*Name=*/Twine(),
                               Preheader->getTerminator());
    TopCall->setCallingConv(CallingConv::Fast);
    assert(TopCall->getCallingConv() == Enqueuer->getCallingConv());
    TopCall->setDebugLoc(Header->getTerminator()->getDebugLoc());
    DEBUG(dbgs() << "Created call to outlined loop:\n " << *TopCall << '\n');
  }

  eraseLoop();
  StartParam->eraseFromParent();
  EndParam->eraseFromParent();

  DEBUG(assertVerifyFunction(*F, "Expanded into balanced tree", DT, LI));

  ++LoopsConvertedToBalanced;

  return TopCall;
}



/// In pseudocode, we are taking this:
///  void IterTask(...) {
///    if (!done) {
///      swarm::deepen();
///      ... loop body containing spawns ...
///      swarm::undeepen();
///      if (loop_exit_condition()) {
///        done = true;
///        ... whatever continuation ...
///      }
///    }
///  }
/// And we are turning it into this:
///  void IterTask(...) {
///    if (!done) {
///      swarm::deepen();
///      ... loop body containing spawns ...
///      spawn (UINT64_MAX - 1) {
///        if (loop_exit_condition()) {
///          done = true;
///          spawn_super (swarm::superTimestamp()) {
///            ... whatever continuation ...
///          }
///        }
///      }
///      swarm::undeepen();
///    }
///  }
/// where UINT64_MAX-1 was chosen to be big enough to exceed the
/// dynamic tripcount of any loop we care about.
///
///TODO(victory): This is ugly: depending on being able to find and move
/// an Undeepen instruction, having to re-spawning code to a superdomain,
/// as well as depending on 64-bit timestamps are all uncomfortable.
/// We are depending Fractalizer creating domains that may be unnecessary.
/// Since SCCRT to spawns IterTasks to even timestamps, in the future we could
/// try to use odd timestamps for exit-condition-checking latch tasks.
static void spawnProgressiveLatch(
        BasicBlock *Header,
        BasicBlock *Latch,
        BasicBlock *RetBlock,
        UndeepenInst *Undeepen,
        StoreInst *SetDone,
        DominatorTree &DT) {
  DEBUG(dbgs() << "Spawning latch at end of loop body domain.\n");
  SmallVector<BasicBlock *, 4> RetPreds(pred_begin(RetBlock),
                                        pred_end(RetBlock));
  assert(is_contained(RetPreds, Header) &&
         "If done flag is set, function entry goes directly to exit");
  RetPreds.erase(find(RetPreds, Header));
  assert(is_contained(RetPreds, Latch) &&
         "If loop iteration is not the last, latch goes directly to exit");
  BasicBlock *NewRetPred =
        SplitBlockPredecessors(RetBlock, RetPreds, ".latch_task_end", &DT);
  Undeepen->moveBefore(NewRetPred->getTerminator());

  Type *TimestampTy = Type::getInt64Ty(Undeepen->getModule()->getContext());
  SDetachInst *LatchDI;
  detachTask(Latch->getFirstNonPHI(),
             Undeepen,
             ConstantInt::get(TimestampTy, UINT64_MAX - 1),
             Undeepen->getMatchingDeepen(),
             DetachKind::ProgressiveLatch, "progressive_latch",
             &DT, nullptr, &LatchDI);

  RetPreds.erase(find(RetPreds, Latch));
  assert(!RetPreds.empty());
  BasicBlock *ContinuationEnd =
      SplitBlockPredecessors(NewRetPred, RetPreds, ".latch_cont_task_end", &DT);
  SDetachInst *LatchContDI;
  detachTask(SetDone->getNextNode(),
             ContinuationEnd->getTerminator(),
             ConstantInt::get(TimestampTy, 0), nullptr,
             DetachKind::RetargetSuperdomain, "latch_cont_task",
             &DT, nullptr, &LatchContDI);
  LatchContDI->setRelativeTimestamp(true);
  LatchContDI->setSuperdomain(true);

  // Hoist the latch detach as early as possible to (speculatively) reduce the
  // critical path of setting the "done" variable and killing later useless
  // iterations. Recall this does not change program order.
  LoopInfo LI(DT);
  Instruction *HoistPt = getEarliestHoistPoint(LatchDI, DT, LI);
  if (HoistPt != LatchDI) hoistDetach(LatchDI, HoistPt, &DT);
  assert(DT.dominates(Undeepen->getMatchingDeepen(), LatchDI) &&
         "The Deepen instruction, which dominated LatchDI, "
         "should still dominate it. This assert should change once we "
         "have strategic deepen/undeepen hopping, or latch spawning "
         "is enabled for all loops");
}


/// Transforms and outlines the loop and uses SCCRT to expand the
/// iterations of a Swarm loop in a progressive fashion that does not
/// require knowing the trip count at the start of the loop.
/// \returns the call to SCCRT that initiates the progressive expansion.
CallInst *LoopExpander::makeProgressive() {
  DEBUG(dbgs() << "Checking whether we can transform to progressive expansion\n");

  auto *DI = cast<SDetachInst>(Header->getTerminator());
  Value *const Timestamp = DI->getTimestamp();

  // Progressive expansion depends on timestamp-ordering of writes to and reads
  // from the done variable across iterations.
  auto TimestampSCEV = SE.getSCEV(Timestamp);
  if (!isIncreasing(TimestampSCEV)) {
    ORE.emit(OptimizationRemarkAnalysis(LE_NAME, "NonIncreasingTS",
                                        DLoc, Header)
             << "Timestamp is not guaranteed to be strictly increasing");
    DEBUG(dbgs() << "Timestamp SCEV: " << *TimestampSCEV << '\n');
    assert(!F->hasFnAttribute(SwarmFlag::Parallelized) &&
           "Auto-parallelized loop should have increasing timestamps");
    return nullptr;
  }

  UndeepenInst *Undeepen;
  if (MustSpawnLatch) {
    // To support the current implementation of latch spawning below,
    // we need to find the undeepen that marks the end of each loop iteration
    // domain, so that we can move the latch into that domain.
    // See comments and TODO on spawnProgressiveLatch for explanation of the
    // current strategy and future plans to replace this.
    SReattachInst *RI = getUniqueSReattachIntoLatch(*L);
    Undeepen = dyn_cast_or_null<UndeepenInst>(RI->getPrevNode());
    if (!Undeepen) {
      ORE.emit(DiagnosticInfoOptimizationFailure(LE_NAME, "NoBodyTaskDomain",
                                                 DLoc, Header)
               << "Failed to find domain set up by Parallelizer?");
      DEBUG(dbgs() << "As of when this code is written, this is extremely "
                   << "rare.\n It seems to only happen if there's a "
                   << "topological sort failure inside a canonical loop.\n");
      return nullptr;
    }
    if (Timestamp->getType()->getIntegerBitWidth() < 64) {
      ORE.emit(DiagnosticInfoOptimizationFailure(LE_NAME, "<64BitTimestamp",
                                                 DLoc, Header)
               << "When we must spawn the latch, we assert the timestamp "
                  "is 64 bits. This is a soft failure to avoid crashes.");
      return nullptr;
    }
  }

  // Checks complete, we can now begin transformation

  ORE.emit(OptimizationRemark(LE_NAME, "ProgressiveExpansion",
                              DLoc, Header)
           << "expanding iterations progressively");

  BasicBlock *Preheader = L->getLoopPreheader();

  PHINode *CanonicalIV = cast<PHINode>(&Header->front());
  const SCEVAddRecExpr *CanonicalSCEV =
    cast<const SCEVAddRecExpr>(SE.getSCEV(CanonicalIV));

  /// Clone the loop iteration into a new function.

  // Get the set of loop blocks to outline and their inputs.
  SmallSetVector<BasicBlock *, 8> Blocks;
  getSanitizedBlocks(Blocks);
  SetVector<Value*> Inputs;
  findInputsNoOutputs(Blocks, Inputs);

  DEBUG(assertVerifyFunction(*F, "Before progressive expansion", DT, LI));

  // The implementation of progressive expansion in SCCRT requires that
  // all loop-invariant live-ins are passed through memory.
  DEBUG(for (Value *V : Inputs) dbgs() << "Captured input: " << *V << "\n");
  Instruction *Closure = nullptr;
  if (!Inputs.empty()) {
    Closure = makeClosure(Inputs.getArrayRef(), Blocks);
    DEBUG(assertVerifyFunction(*F, "After setting up closure", DT, LI));
  }

  // Allocate and initialize the done flag.
  // Insert early-exit check to avoid running iterations after the flag is set.
  Value *DonePtr;
  {
    IntegerType *DoneFieldTy = Type::getInt32Ty(M->getContext());
    Instruction *DoneAllocation = createClosure(
        {ConstantInt::get(DoneFieldTy, 0)},
        L->getLoopPreheader()->getTerminator(), "progressive_done");
    IRBuilder<> B(DoneAllocation->getNextNode());
    DonePtr = B.CreateBitCast(DoneAllocation,
                              PointerType::getUnqual(DoneFieldTy), "done.ptr");

    B.SetInsertPoint(Header->getFirstNonPHIOrDbgOrLifetime());
    LoadInst *Done = B.CreateLoad(DonePtr, "done");
    Done->setMetadata(SwarmFlag::DoneFlag, MDNode::get(M->getContext(), {}));
    Value *DoneBool = B.CreateIsNotNull(Done, "done.bool");
    BasicBlock *NewBB = SplitBlock(Header, &*B.GetInsertPoint(), DT, LI);
    Blocks.insert(NewBB);
    ReplaceInstWithInst(Header->getTerminator(),
                        BranchInst::Create(ExitBlock, NewBB, DoneBool));
    //DT->insertEdge(Header, ExitBlock);
    if (DT->dominates(Header, ExitBlock))
      DT->changeImmediateDominator(ExitBlock, Header);
  }

  DEBUG(assertVerifyFunction(*F, "Before outlining", DT, LI));

  // Since SCCRT will handle all enqueues of loop iteration tasks,
  // we won't need the detach any more.  If were weren't deleting this detach,
  // we'd need to replace UnexpandedIter with some other DetachKind,
  // and remove the SwarmFlag::MustSpawnLatch tag.
  assert(!F->hasFnAttribute(SwarmFlag::Parallelized)
         || getDetachKind(DI) == DetachKind::UnexpandedIter);
  SerializeDetachedCFG(DI, DT);
  DI = nullptr;
  // Note: At this point nested detaches may no longer be consistent with
  // any preceeding deepens before the loop.  We cannot verify until the loop is erased.

  // The only parameters of the iteration task function will be the timestamp,
  // a pointer to the done flag, and a pointer to the struct of captured values.
  SetVector<Value *> Params;
  Instruction *TimestampDummy =
      createDummyValue(CanonicalIV->getType(), CanonicalIV->getName(),
                       Preheader->getTerminator());
  Params.insert(TimestampDummy);
  Params.insert(DonePtr);
  Instruction *ClosureDummy = nullptr;
  if (Closure) {
    Params.insert(Closure);
  } else {
    ClosureDummy =
        createDummyValue(Type::getInt1Ty(M->getContext()),
                         "progressive_closure", Preheader->getTerminator());
    Params.insert(ClosureDummy);
  }

  ValueToValueMapTy VMap;
  Function *IterTask = outline(Params, Blocks, VMap);

  BasicBlock *NewHeader = cast<BasicBlock>(VMap[Header]);
  BasicBlock *NewLatch = cast<BasicBlock>(VMap[Latch]);
  BasicBlock *RetBlock = cast<BasicBlock>(VMap[ExitBlock]);
  assert(RetBlock->size() == 1);
  assert(!cast<ReturnInst>(RetBlock->getTerminator())->getReturnValue());

  // Run only one iteration of the loop, and if the loop would exit,
  // 1) set the done variable
  // 2) spawn a task in the super domain that frees the done flag and closure
  StoreInst *SetDone;
  {
    auto DoneBlock = BasicBlock::Create(M->getContext(), "done", IterTask);
    IRBuilder<> Builder(DoneBlock);
    SetDone = Builder.CreateStore(Builder.getInt32(1), VMap[DonePtr]);
    SetDone->setMetadata(SwarmFlag::DoneFlag, MDNode::get(M->getContext(), {}));
    Instruction *ExitBranch = Builder.CreateBr(RetBlock);

    SDetachInst *DI = createSuperdomainDetachedFree(
        VMap[DonePtr], nullptr, ExitBranch);
    if (Closure) {
      Instruction *Free = CallInst::CreateFree(
          VMap[Closure], DI->getDetached()->getTerminator());
      addSwarmMemArgsForceAliasMetadata(cast<CallInst>(Free));
    }

    auto BackEdgeBranch = cast<BranchInst>(NewLatch->getTerminator());
    assert(BackEdgeBranch->isConditional());
    assert(((BackEdgeBranch->getSuccessor(0) == RetBlock)
            && (BackEdgeBranch->getSuccessor(1) == NewHeader))
           || ((BackEdgeBranch->getSuccessor(1) == RetBlock)
               && (BackEdgeBranch->getSuccessor(0) == NewHeader)));

    BackEdgeBranch->replaceUsesOfWith(RetBlock, DoneBlock);
    NewHeader->removePredecessor(NewLatch, true);
    BackEdgeBranch->replaceUsesOfWith(NewHeader, RetBlock);
    DEBUG(dbgs() << "Modified latch to conditionally branch to done:"
                 << *NewLatch);
  }

  DominatorTree *IterTaskDT = nullptr;
  if (MustSpawnLatch) {
    // We need to ensure any load in the latch, which may be used to compute
    // the loop exit condition, is correctly ordered after any tasks
    // detached within the loop body.
    assert(Timestamp->getType()->getIntegerBitWidth() == 64);
    IterTaskDT = new DominatorTree(*IterTask);
    spawnProgressiveLatch(NewHeader, NewLatch, RetBlock,
                          cast<UndeepenInst>(VMap[Undeepen]),
                          SetDone,
                          *IterTaskDT);
  }

  DEBUG(assertVerifyFunction(*IterTask, "After de-looping the iteration task",
                             IterTaskDT));

  // Rewrite clone of canonical IV to start at the start iteration argument
  {
    PHINode *NewCanonicalIV = cast<PHINode>(VMap[CanonicalIV]);
    BasicBlock *NewPreheader = cast<BasicBlock>(VMap[Preheader]);
    int NewPreheaderIdx = NewCanonicalIV->getBasicBlockIndex(NewPreheader);
    assert(isa<Constant>(NewCanonicalIV->getIncomingValue(NewPreheaderIdx)) &&
           "Cloned canonical IV does not inherit a constant value from cloned preheader.");
    auto *Timestamp = cast<Argument>(VMap[TimestampDummy]);
    Value *CanonicalIV = IRBuilder<>(NewPreheader->getTerminator())
                             .CreateLShr(Timestamp, 1, "iter_index");
    NewCanonicalIV->setIncomingValue(NewPreheaderIdx, CanonicalIV);
  }

  TimestampDummy->eraseFromParent();
  if (ClosureDummy) ClosureDummy->eraseFromParent();

  swarm_abi::optimizeTaskFunction(IterTask);

  //DEBUG(dbgs() << "Final iterTask with outlined loop code:" << *IterTask);
  assertVerifyFunction(*IterTask, "Transformed iterTask is invalid",
                       IterTaskDT);
  if (IterTaskDT) delete IterTaskDT;

  // Call the progressive enqueuer with the new task function.
  CallInst *CI;
  {
    // Setup arguments for call.
    SmallVector<Value *, 4> TopCallArgs;
    // Add start iteration 0.
    assert(CanonicalSCEV->getStart()->isZero() &&
           "Canonical IV does not start at zero.");
    TopCallArgs.push_back(ConstantInt::get(CanonicalIV->getType(), 0));
    // Add the rest of the arguments.
    for (Value *V : Inputs)
      TopCallArgs.push_back(V);
    DEBUG({
        for (Value *TCArg : TopCallArgs)
          dbgs() << "Top call arg: " << *TCArg << "\n";
      });

    // Create call instruction.
    IRBuilder<> Builder(Preheader->getTerminator());

    Function *F;
    if (CanonicalIV->getType()->getIntegerBitWidth() == 64)
      F = RUNTIME_FUNC(__sccrt_enqueue_progressive_64, M);
    else if (CanonicalIV->getType()->getIntegerBitWidth() == 32)
      F = RUNTIME_FUNC(__sccrt_enqueue_progressive_32, M);
    else
      llvm_unreachable("Bad timestamp/canonical IV width");
    ArrayRef<Type *> ParamTypes = F->getFunctionType()->params();
    assert(ParamTypes.size() == 3);
    Value *args[3] = {
        Builder.CreatePointerCast(IterTask, ParamTypes[0]),
        Builder.CreatePointerCast(DonePtr, ParamTypes[1]),
        Closure ? Builder.CreatePointerCast(Closure, ParamTypes[2])
                : Constant::getNullValue(ParamTypes[2])};
    CI = Builder.CreateCall(F, args);
    assert(CI->getCallingConv() == CallingConv::C);
    assert(F->getCallingConv() == CallingConv::C);
    DEBUG(dbgs() << "Created runtime library call:\n " << *CI << '\n');
  }

  DEBUG(dbgs() << "Preheader with launch of parallel enqueuers:" << *Preheader);

  eraseLoop();

  DEBUG(assertVerifyFunction(*F, "Transformed to progressively expand loop", DT, LI));

  ++LoopsConvertedToProgressive;

  return CI;
}


/// Given a function containing nothing but a canonical Swarm loop,
/// transform it to run only one iteration and then recursively spawn itself
/// for the next iteration. Assumes that if the header has N phi nodes,
/// then the first N parameters correspond to those N phi nodes, respectively.
///
/// In pseudocode, given a function of the following form:
///
/// void f(phi0, phi1, ...) {
///   do {
///     Timestamp ts = ts();
///     spawn (ts) { ... loop body ... };
///   } while (some_cond());
/// }
///
/// Then this method transforms the function into the following form:
///
/// void f(phi0, phi1, ...) {
///   Timestamp ts = ts();
///   spawn (ts+1) {
///     if (some_cond())
///       f(phi0_next, phi1_next, ...);
///   }
///   spawn (ts) { ... loop body ... };
/// }
///
void LoopExpander::implementRecursiveChain(Function *Enqueuer) {
  BasicBlock *Header = Enqueuer->getEntryBlock().getSingleSuccessor();
  assert(Header);
  auto *BodyDetach = cast<SDetachInst>(Header->getTerminator());
  assert(!BodyDetach->getDomain() && "Outlined detach separated from deepen");
  BasicBlock *Latch = BodyDetach->getContinue();
  ReturnInst *Ret = getUniqueReturnInst(*Enqueuer);
  assert(Ret && !Ret->getReturnValue());
  BasicBlock *RetBlock = Ret->getParent();
  assert(RetBlock->size() == 1);
  assert(Latch == RetBlock->getSinglePredecessor());
  assert(!Enqueuer->hasFnAttribute(SwarmFlag::Parallelized)
         || getDetachKind(BodyDetach) == DetachKind::UnexpandedIter);
  setDetachKind(BodyDetach, DetachKind::ExpandedChainIter);
  BodyDetach->setMetadata(SwarmFlag::MustSpawnLatch, nullptr);

  formRecursiveLoop(Enqueuer);

  // Now spawn the latch together with the recursive call as an enqueuer
  // for the next iteration.
  IRBuilder<> Builder(BodyDetach);
  //TODO(victory): While this is correct for autoparallelized loops,
  // we ought to figure out what the semantics of bookkeeping that accesses
  // memory is for manual parallelization and consider whether this timestamp
  // makes sense in that case.
  Value *NextTS = Builder.CreateAdd(
          BodyDetach->getTimestamp(),
          ConstantInt::get(BodyDetach->getTimestamp()->getType(), 1),
          "enqueuer_ts");
  // [victory] We sometimes send spawners in a chain to a random tile.
  // We mainly use SAMEHINT to expand the chain of tasks quickly.
  // However, keeping all spawners at one tile will quickly fill
  // the tile's commit queue and block further spawners from running.
  // Distributing spawners across the system lets us instead use the full
  // capacity of all commit queues to hold these ordered tasks.
  // I hope groups of 8 spawners won't overwhelm our 64-entry commit queues.
  Value *MaskedTimestamp = Builder.CreateAnd(
          BodyDetach->getTimestamp(),
          LateChainLocalEnqueueMask,
          "masked_timestamp");
  Value *UseSameHint = Builder.CreateIsNotNull(MaskedTimestamp, "use_samehint");
  SDetachInst *EnqueuerDetach;
  detachTask(Latch->getFirstNonPHI(), RetBlock->getTerminator(),
             NextTS, nullptr, DetachKind::ExpandedChainSpawner,
             "chain_enqueuer", nullptr, nullptr, &EnqueuerDetach);
  EnqueuerDetach->setSameHintCondition(UseSameHint);

  // Hoist the enqueuer detach above the body to reduce the critical path of
  // task spawning and maximize parallelism.
  hoistDetach(EnqueuerDetach, BodyDetach);
}


/// If loop-invariant inputs don't fit in registers, use a closure.
///
/// TODO(victory): This method mainly contains logic to figure out exactly
/// whether the loop reall needs an in-memory closure does not apply just to
/// chains, we should bring these benefits to trees as well.
///
/// NOTE: This is tricky. We want to reduce inputs across the detaches,
/// **not** across the outlined function that wraps them. We'll re-inline the
/// outlined function, so it being bloated is irrelevant.
///
/// Moreover, we're using a weird set of params that includes ALL the header
/// phis (inputs to the loop task) + other loop-variant inputs (e.g., header
/// non-phis) + loop-invariant inputs to the body task. This is because we
/// want to avoid mem_runners for **both** the loop and the body tasks.
///
/// The source for all this complexity is that we need to infer the inputs to
/// the different tasks far before we actually outline the function and
/// produce the tasks. Unfortunately, I don't see a way out because recursion
/// implies we must outline.
Instruction *LoopExpander::createChainClosure() {
  if (DisableEnvSharing) return nullptr;

  // First, consider the loop body task
  auto *IterDetach = cast<SDetachInst>(Header->getTerminator());
  BasicBlock *Spawned = IterDetach->getDetached();
  SmallVector<BasicBlock *, 8> DetachedBlocks;
  DT->getDescendants(Spawned, DetachedBlocks);
  const SmallSetVector<BasicBlock *, 8> DetachedBBSet(DetachedBlocks.begin(),
                                                      DetachedBlocks.end());
  SetVector<Value *> DetachedInputs;
  findInputsNoOutputs(DetachedBBSet, DetachedInputs);

  // Gather non-loop-invariant inputs first
  SetVector<Value *> Args;
  for (PHINode &Phi : Header->phis())
    Args.insert(&Phi);
  for (Value *Input : DetachedInputs)
    if (Instruction *I = dyn_cast<Instruction>(Input))
      if (L == LI->getLoopFor(I->getParent())) {
        DEBUG(dbgs() << "loop-variant body task arg:" << *Input << "\n");
        Args.insert(Input);
      }
  uint32_t NLIArgs = Args.size();

  // Add loop-invariant inputs from the body task
  for (Value *Input : DetachedInputs) {
    DEBUG(dbgs() << "body task arg:" << *Input << "\n");
    Args.insert(Input);
  }

  // Now consider the loop task. The header and latch may have other
  // loop-invariant inputs that we should also include in the closure to
  // avoid mem-runners.
  SmallSetVector<BasicBlock *, 8> Blocks;
  getSanitizedBlocks(Blocks);
  SetVector<Value*> Inputs;
  findInputsNoOutputs(Blocks, Inputs);

  // We need to avoid double-counting Phis and their inputs (we already
  // counted phis for the body task)
  SetVector<Value *> PhiInputs;
  for (const PHINode &PN : Header->phis()) {
    Value *In = PN.getIncomingValueForBlock(L->getLoopPreheader());
    if (!isa<Constant>(In))
      PhiInputs.insert(In);
  }

  for (Value *Input : Inputs)
    if (!PhiInputs.count(Input)) {
      DEBUG(dbgs() << "external arg:" << *Input << "\n");
      Args.insert(Input);
    }

  SetVector<Value *> MemArgs =
      getMemArgs(Args, F->getParent()->getDataLayout(), nullptr, NLIArgs);

  if (!MemArgs.size()) return nullptr;

  DEBUG({
    dbgs() << "Chosen MemArgs:\n";
    for (Value *Arg : MemArgs)
      dbgs() << "Arg: " << *Arg << "\n";
  });

  Instruction *MemArgsPtr = makeClosure(MemArgs.getArrayRef(), Blocks);

  return MemArgsPtr;
}

/// Transforms and outlines the loop into a TLS-style linked list of tasks.
/// This can exploit pipeline parallelism between iteration tasks as well as
/// parallelism within the loop body, if it exists, but cannot exploit
/// parallelism in the loop bookkeeping.
/// \returns the call to the first task inserted into the original function.
CallInst *LoopExpander::makeChain() {
  DEBUG(dbgs() << "Checking whether we can transform to chain\n");

  auto *DI = cast<SDetachInst>(Header->getTerminator());
  auto TimestampSCEV = SE.getSCEV(DI->getTimestamp());
  if (!isNondecreasing(TimestampSCEV)) {
    // TODO(victory): Eventually we should handle this case by using
    // swarm::timestamp() for the enqueuer timestamps.
    ORE.emit(OptimizationRemarkAnalysis(LE_NAME, "DecreasingTS", DLoc, Header)
             << "Timestamp is not guaranteed to be nondecreasing");
    DEBUG(dbgs() << "Timestamp SCEV: " << *TimestampSCEV << '\n');
    assert(!F->hasFnAttribute(SwarmFlag::Parallelized) &&
           "Auto-parallelized loop should have increasing timestamps");
    return nullptr;
  }

  // If the latch contains loads, for example, we need to ensure those loads
  // are timestamp-ordered after each iteration of the loop body.
  assert(!MustSpawnLatch || isIncreasing(TimestampSCEV));

  ORE.emit(OptimizationRemark(LE_NAME, "Chain", DLoc, Header)
           << "pipelining chain of iterations");

  // Eliminate some induction variables to save on task communication costs.
  unsigned TimestampWidth = DI->getTimestamp()->getType()->getIntegerBitWidth();
  canonicalizeIVs(*L, TimestampWidth, *DT, SE);

  /// Clone the loop into a new function.

  DEBUG(assertVerifyFunction(*F, "Before gathering inputs", DT, LI));

  BasicBlock *Preheader = L->getLoopPreheader();

  if (Instruction *Closure = createChainClosure()) {
    SDetachInst *DI = createSuperdomainDetachedFree(
        Closure, Domain ? Domain->getSuperdomain(*DT) : nullptr,
        Preheader->getTerminator(), DT, LI);
    Preheader = DI->getContinue();
    assert(Preheader == L->getLoopPreheader());
    DEBUG(assertVerifyFunction(*F, "After freeing closure", DT, LI));
  }

  // Get the set of loop blocks to outline and their inputs.
  SmallSetVector<BasicBlock *, 8> Blocks;
  getSanitizedBlocks(Blocks);
  SetVector<Value*> Inputs;
  findInputsNoOutputs(Blocks, Inputs);

  // The parameters shall start with one parameter for each header phi.
  SetVector<Value *> Params;
  SmallVector<Value *, 8> StartArgs;
  SmallVector<Instruction *, 8> ParamDummies;
  for (const PHINode &PN : Header->phis()) {
    Value *In = PN.getIncomingValueForBlock(Preheader);
    assert(isa<Constant>(In) || Inputs.count(In));
    StartArgs.push_back(In);

    Instruction *PhiParam = createDummyValue(PN.getType(), PN.getName(),
                                             Preheader->getTerminator());
    ParamDummies.push_back(PhiParam);
    Params.insert(PhiParam);
  }

  // Add the other loop-invariant inputs as parameters
  for (Value *In : Inputs)
    Params.insert(In);

  DEBUG(assertVerifyFunction(*F, "Before outlining loop", DT, LI));

  ValueToValueMapTy VMap;
  Function *Enqueuer = outline(Params, Blocks, VMap);
  Enqueuer->addFnAttr(Attribute::AlwaysInline); // inline into task function

  for (Instruction *ParamDummy : ParamDummies)
    ParamDummy->eraseFromParent();

  // Now that the loop is outlined, transform it into a chain of tasks.
  implementRecursiveChain(Enqueuer);
  //DEBUG(dbgs() << "Final progressive enqueuer with outlined loop code:"
  //             << *Enqueuer);
  assertVerifyFunction(*Enqueuer, "Transformed chain enqueuer is invalid");

  // Add call to enqueuer function in original function.
  CallInst *TopCall;
  {
    // Setup arguments for call.
    SmallVector<Value *, 4> TopCallArgs(StartArgs.begin(), StartArgs.end());
    //TODO(victory): Be more selective to avoid redundant args
    TopCallArgs.append(Inputs.begin(), Inputs.end());
    DEBUG(for (Value *TCArg : TopCallArgs)
            dbgs() << "Top call arg: " << *TCArg << "\n";);

    // Create call instruction.
    IRBuilder<> Builder(Preheader->getTerminator());
    TopCall = Builder.CreateCall(Enqueuer, TopCallArgs);
    TopCall->setCallingConv(CallingConv::Fast);
    assert(TopCall->getCallingConv() == Enqueuer->getCallingConv());
    DEBUG(dbgs() << "Created call to outlined loop:\n " << *TopCall << '\n');
  }

  eraseLoop();

  DEBUG(assertVerifyFunction(*F, "Transformed to chain of tasks", DT, LI));

  ++LoopsConvertedToChain;

  return TopCall;
}


/// \return string containing a file name and a line # for the given loop.
static std::string getDebugLocString(const Loop *L) {
  std::string Result;
  if (L) {
    raw_string_ostream OS(Result);
    if (const DebugLoc LoopDbgLoc = L->getStartLoc())
      LoopDbgLoc.print(OS);
    else
      // Just print the module name.
      OS << L->getHeader()->getParent()->getParent()->getModuleIdentifier();
    OS.flush();
  }
  return Result;
}


static bool expandCanonicalSwarmLoops(Function &F,
                                      DominatorTree &DT,
                                      LoopInfo &LI,
                                      ScalarEvolution &SE,
                                      const TargetTransformInfo& TTI,
                                      OptimizationRemarkEmitter &ORE) {
  // Build up a worklist of loops to examine and expand.
  // This is necessary as expanding loops outlines and removes them,
  // which can invalidate iterators across the loops.
  SmallVector<Loop *, 4> PreorderLoops = LI.getLoopsInPreorder();

  DEBUG(dbgs() << "LE Function " << F.getName() << "() has "
               << PreorderLoops.size() << " loops.\n");

  // Walk the loops in post order, i.e., process inner loops before outer loops.
  bool Changed = false;
  while (!PreorderLoops.empty()) {
    Loop *L = PreorderLoops.pop_back_val();

    assert(!L->isInvalid() && "Loop was deleted?");

    DEBUG(dbgs() << "\nLE: Checking a loop in \""
                 << F.getName() << "\"\n from "
                 << getDebugLocString(L) << ":\n " << *L);

    if (isExpandableSwarmLoop(L, DT)) {
      Changed |= LoopExpander(L, &DT, &LI, SE, TTI, ORE).run();
      ++LoopsAnalyzed;
    } else {
      DEBUG(dbgs() << "Leaving loop alone as it is not a canonical swarm loop.\n");
#ifndef NDEBUG
      SmallVector<SDetachInst *, 8> InternalDetaches;
      getOuterDetaches(*L, DT, InternalDetaches);
      assert((!F.hasFnAttribute(SwarmFlag::Parallelized) ||
              findStringMetadataForLoop(L, SwarmCoarsened::InnerLoop) ||
              all_of(InternalDetaches, [](const SDetachInst *DI) {
                return DI->isSuperdomain() && (
                        getDetachKind(DI) == DetachKind::SubsumedCont ||
                        getDetachKind(DI) == DetachKind::RetargetSuperdomain);
              })) &&
             "Autoparallelized loop no longer in canonical form?");
#endif
    }
  }
  return Changed;
}


bool LoopExpander::isSmallLoop() const {
  Optional<const MDOperand *> ExpandEnable =
      findStringMetadataForLoop(L, "llvm.loop.expand.enable");
  if (ExpandEnable && mdconst::extract<ConstantInt>(**ExpandEnable)->isZero()) {
    DEBUG(dbgs() << "Not expanding loop with expand(disable) pragma:\n  "
                 << *L->getStartLoc() << '\n');
    ORE.emit(OptimizationRemarkAnalysis(LE_NAME, "ExpandDisable", DLoc, Header)
                 << "LoopExpansion disabled, loop not expanded");
    return true;
  }

  if (auto ConstLimit = dyn_cast<SCEVConstant>(Limit)) {
    ConstantInt *CI = ConstLimit->getValue();
    if (CI->isZero()) {
      ORE.emit(OptimizationRemarkAnalysis(LE_NAME, "BackEdge", DLoc, Header)
                   << "loop not taking backedge not expanded");
      return true;
    }
  }
  return false;
}


bool LoopExpander::mustSpawnLatch(SDetachInst *DI) const {
  // If the latch doesn't access memory and performs only arithmatic on
  // registers, then there's no need to spawn it as a separate task.
  if (none_of(*Latch,
              [](const Instruction &I) { return I.mayReadOrWriteMemory(); }))
    return false;
  DEBUG(dbgs() << "Found latch that accesses memory.\n");
  assert(!F->hasFnAttribute(SwarmFlag::Parallelized) ||
         DI->getMetadata(SwarmFlag::MustSpawnLatch));

  if (DI->getMetadata(SwarmFlag::Coarsenable)) {
    DEBUG(dbgs() << "Loop body spawns no timestamp-sensitive internal tasks. "
                 << "No latch spawning needed.\n");
#ifndef NDEBUG
    // N.B. while the metadata tag ought to be sufficient to guarantee this
    // assertion condition, the converse is not true. As a counterexample,
    // an outer swarm loop, by the time we are processing it here, may have
    // had inner loops that were lowered into SCCRT calls, so it may not
    // have compiler-visible internal detaches left, although it is
    // not a leaf task.
    SmallVector<SDetachInst *, 8> InternalDetaches;
    getOuterDetaches(DI, *DT, InternalDetaches);
    assert(all_of(InternalDetaches, [](const SDetachInst *DI) {
      return DI->isSuperdomain();
    }) && "Unsafe to coarsen loops with internal detaches in the loop's domain");
#endif
    // By not spawning the latch as a separate task, we are effectively
    // coarsening the loop iteration task to include the loop latch.
    // The Coarsenable flag tells us this is safe.
    DI->setMetadata(SwarmFlag::MustSpawnLatch, nullptr);
    return false;
  }

  // In general, if the latch might read from memory, it must be spawned
  // with a timestamp that orders it after any tasks from the loop body,
  // in case those loop body tasks have stores that might change the value
  // loaded in the latch.
  // This is the common case for progressive expansion, and for pointer-
  // chasing loops that will be expanded with a spawner chain.
  // TODO(victory): We could use alias analysis to check if the load is actually
  // accessing loop-invariant data in memory.
  return true;
}


CallInst *LoopExpander::makeParallelExpansion() {
  // Strengthen the loop's induction variables to eliminate loop-carried
  // SSA register dependencies.
  //FIXME(victory): At this point, if subsequent checks fail, we still want
  // to return nullptr to indicate we didn't succeed, but it would
  // be preferable in that case to truely have not mutated anything.
  // It would be good to delay this IV strengthening until after all checks
  // have passed.
  unsigned TimestampWidth = cast<SDetachInst>(Header->getTerminator())
                            ->getTimestamp()->getType()->getIntegerBitWidth();
  //FIXME(victory): This assumes the timestamp is wide enough to avoid wraparounds.
  PHINode *CanonicalIV = canonicalizeIVsAllOrNothing(*L, TimestampWidth,
                                                     *DT, SE);
  if (!CanonicalIV) {
    ORE.emit(OptimizationRemarkAnalysis(LE_NAME, "CannotCanonicalizeAllIVs",
                                        DLoc, Header)
             << "failed to strengthen all IVs other than the canonical one.");
    return nullptr;
  }
  DEBUG(dbgs() << "Rewrote IVs in stronger form:" << *Header);

  // Post IV canonicalization, ensure that the Limit's bitwidth at least
  // matches that of CanonicalIV.
  if (!isa<SCEVCouldNotCompute>(Limit)) {
    unsigned CanonicalIVWidth = CanonicalIV->getType()->getIntegerBitWidth();
    if (CanonicalIVWidth > Limit->getType()->getIntegerBitWidth())
      Limit = SE.getZeroExtendExpr(Limit, IntegerType::get(M->getContext(),
                                                           CanonicalIVWidth));
    assert(CanonicalIVWidth == Limit->getType()->getIntegerBitWidth());
  }

  DEBUG(assertVerifyFunction(*F, "After initial IV analysis", DT, LI));

  CallInst *NewCall = makeBalanced();
  if (!NewCall && !DisableUnboundedTrees)
    NewCall = makeProgressive();

  if (!NewCall) {
    ORE.emit(DiagnosticInfoOptimizationFailure(DEBUG_TYPE, "FailedTreeExpansion",
                                               DLoc, Header)
             << "Swarm loop not expanded into parallel tree:"
                " falling back to a serial chain of enqueuers");
    return nullptr;
  }

  return NewCall;
}


bool LoopExpander::run() {
  DEBUG(assertVerifyFunction(*F, "Before restructuring loop", DT, LI));

  DEBUG(dbgs() << "LE loop header:" << *Header);
  DEBUG(dbgs() << "LE loop latch:" << *Latch);
  DEBUG(dbgs() << "LE SE backedge taken count: " << *(SE.getBackedgeTakenCount(L)) << "\n");
  DEBUG(dbgs() << "LE SE max backedge taken count: " << *(SE.getMaxBackedgeTakenCount(L)) << "\n");
  DEBUG(dbgs() << "LE SE exit count: " << *(SE.getExitCount(L, Latch)) << "\n");
  // PredicatedScalarEvolution PSE(SE, *L);
  // const SCEV *PLimit = PSE.getExitCount(L, Latch);

  if (auto *ConstLimit = dyn_cast<SCEVConstant>(Limit)) {
    if (ConstLimit->getValue()->isNegative()) {
      //victory: It seems like ScalarEvolution sometimes produces a value of -1
      // for some loops after some previous Swarm passes, which is nonsense.
      // This is rare, and it appears to only happen for unreachable paths.
      int64_t limit = ConstLimit->getValue()->getSExtValue();
      assert(limit == -1LL || limit == -2LL
             && "We never see other negative numbers?");
      ORE.emit(OptimizationRemark(DEBUG_TYPE, "NegativeLimit", DLoc, Header)
               << "Deleting loop that is unreachable or ill-defined?");

      // Once we call eraseLoop(), L is deleted and is no longer valid,
      // so get all the info we will need about its surroundings first.
      BasicBlock *Preheader = L->getLoopPreheader();
      Loop *ParentLoop = L->getParentLoop();

      eraseLoop();

      // Mark the end of the preheader unreachable.
      // This ensures a crash if this loop would have been reached at runtime.
      auto *PrevTerm = cast<BranchInst>(Preheader->getTerminator());
      assert(PrevTerm->isUnconditional());
      assert(PrevTerm->getSuccessor(0) == ExitBlock);
      auto *Unreachable = new UnreachableInst(F->getContext());
      ReplaceInstWithInst(PrevTerm, Unreachable);
      if (DT->dominates(Preheader, ExitBlock))
        eraseDominatorSubtree(ExitBlock, *DT, LI);
      //TODO(victory): Do an incremental update of the dominator tree instead
      // of throwing it out and recalculating it.
      DT->recalculate(*F);
      assert(ParentLoop == LI->getLoopFor(Preheader));
      LI->removeBlock(Preheader);
      while (ParentLoop) {
        Loop *CurrentLoop = ParentLoop;
        ParentLoop = CurrentLoop->getParentLoop();
        DEBUG({
          dbgs() << "Checking parent loop at depth "
                 << CurrentLoop->getLoopDepth()
                 << " with header: ";
          CurrentLoop->getHeader()->printAsOperand(dbgs(), false);
          dbgs() << "\n";
        });
        SmallVector<BasicBlock *, 4> Latches;
        CurrentLoop->getLoopLatches(Latches);
        if (Latches.empty()) {
          DEBUG(dbgs() << "  This loop no longer has a reachable backedge.\n");
          LI->markAsRemoved(CurrentLoop);
        } else {
          DEBUG(dbgs() << "  Loop still exists.\n");
        }
      }

      // Print out a message before crashing.
      createPrintString("\nReached code that LoopExpansion deleted"
                        " because a loop had a negative bound.\n",
                        "le_reached_negative_bound",
                        Unreachable);

      DEBUG(dbgs() << "LE preheader marked unreachable:" << *Preheader);

      DEBUG(assertVerifyFunction(*F, "After marking loop unreachable", DT, LI));

      return true;
    }
  }

  SDetachInst *OrigDetach = cast<SDetachInst>(Header->getTerminator());
  assert(!F->hasFnAttribute(SwarmFlag::Parallelized)
         || getDetachKind(OrigDetach) == DetachKind::UnexpandedIter);

  MustSpawnLatch = mustSpawnLatch(OrigDetach);

  if (!MustSpawnLatch && isSmallLoop())
    return false;

  // If loop detaches to a different domain, we will handle this by wrapping
  // the initial call that sets off the loop enqueuing in a detach to the right
  // domain. The loop will then no longer need to detach to a different domain.
  auto TimestampTy = cast<IntegerType>(OrigDetach->getTimestamp()->getType());
  const bool isSubdomain = OrigDetach->isSubdomain();
  const bool isSuperdomain = OrigDetach->isSuperdomain();
  OrigDetach->setSubdomain(false);
  OrigDetach->setSuperdomain(false);

  assert(none_of(*Header, [](const Instruction &I) {
                  return I.mayReadOrWriteMemory() || I.mayThrow(); })
         && "Guaranteed by canonical form");

  assert(none_of(*Latch, [] (const Instruction &I) {
                           return mayHaveSideEffects(&I); })
         && "Guaranteed by canonical form");

  // Shrink inputs to loop tasks
  //TODO(victory): Change shrinkInputs() API to look more like our outline()
  // utility, allowing us to reduce the boilerplate code here.
  {
    SmallVector<BasicBlock *, 8> DetachedBlocks;
    BasicBlock *Detached = OrigDetach->getDetached();
    DT->getDescendants(Detached, DetachedBlocks);

    std::vector<BasicBlock *> Blocks;
    for (BasicBlock *BB : DetachedBlocks)
      Blocks.push_back(BB);
    assert(Blocks[0] == Detached);

    // Blacklist non-loop-invariant inputs and the timestamp
    SmallPtrSet<Value *, 4> Blacklist;
    for (Instruction& I : *Header)
      Blacklist.insert(&I);
    Blacklist.insert(OrigDetach->getTimestamp());

    shrinkInputs(Blocks, Blacklist, TTI, &ORE);
  }

  CallInst *NewCall = nullptr;
  if (!ChainExpansionOnly) NewCall = makeParallelExpansion();
  if (!NewCall) NewCall = makeChain();

  if (!NewCall) {
    ORE.emit(DiagnosticInfoOptimizationFailure(LE_NAME, "FailedExpansion",
                                               DLoc, Header)
             << "Swarm loop not expanded: serially enqueuing "
                "all iterations in a single serial task");

    assert(!F->hasFnAttribute(SwarmFlag::Parallelized) &&
           "Failed to expand autoparallelized loop");

    // restore original domain handling
    OrigDetach->setSubdomain(isSubdomain);
    OrigDetach->setSuperdomain(isSuperdomain);
    assert(OrigDetach->getDomain() == Domain);
    DEBUG(assertVerifyFunction(*F, "After restoring loop", DT, LI));

    return false;
  }

  if (isSubdomain || isSuperdomain) {
    DEBUG(dbgs() << "Detaching the newly created call to right domain.\n");
    SDetachInst *OuterDetach;
    detachTask(NewCall,
               NewCall->getNextNode(),
               ConstantInt::get(TimestampTy, 0),
               Domain,
               DetachKind::LoopStart,
               "",
               DT,
               LI,
               &OuterDetach);
    OuterDetach->setRelativeTimestamp(true);
    OuterDetach->setSubdomain(isSubdomain);
    OuterDetach->setSuperdomain(isSuperdomain);
    DEBUG(assertVerifyFunction(*F, "After detaching top call", DT, LI));
  }

  return true;
}


namespace {
struct LoopExpansion : public FunctionPass {
  /// Pass identification, replacement for typeid
  static char ID;

  explicit LoopExpansion() : FunctionPass(ID) {
    initializeLoopExpansionPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    if (F.getName().startswith("_ZN5swarm") ||
        F.getName().startswith("_ZN3pls"))
      return false;

    // Avoid reprocessing loops
    if (F.hasFnAttribute(LOOP_EXPANDED))
      return false;
    F.addFnAttr(LOOP_EXPANDED);

    // Just a compiler performance optimization
    if (!llvm::hasAnySDetachInst(F))
      return false;

    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    // auto *AA = &getAnalysis<AAResultsWrapperPass>(*F).getAAResults();
    auto &ORE = getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

    const bool Changed = expandCanonicalSwarmLoops(F, DT, LI, SE, TTI, ORE);
    assertVerifyFunction(F, "After running LoopExpansion on function", &DT, &LI);
    DEBUG(dbgs() << "\nFinished running LoopExpansion on function "
                 << F.getName() << "()\n\n\n");
    return Changed;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredID(LoopSimplifyID);
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    // AU.addRequired<LoopAccessLegacyAnalysis>();
    // getAAResultsAnalysisUsage(AU);
    // AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
  }
};
}

char LoopExpansion::ID = 0;
// static RegisterPass<LoopExpansion> X(LE_NAME, "Transform Swarm loops into expanding task trees", false, false);
static const char le_name[] = "Swarm Loop Expansion";
INITIALIZE_PASS_BEGIN(LoopExpansion, LE_NAME, le_name, false, false)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
// INITIALIZE_PASS_DEPENDENCY(LoopAccessLegacyAnalysis)
// INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_END(LoopExpansion, LE_NAME, le_name, false, false)

namespace llvm {
Pass *createLoopExpansionPass() {
  return new LoopExpansion();
}
}
