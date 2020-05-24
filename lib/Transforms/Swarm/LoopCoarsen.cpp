//===-- LoopCoarsen.cpp - Loop coarsening pass ----------------------------===//
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
// This pass coarsens canonical Swarm loops. When detached loop bodies write
// data in a stride pattern that is denser than a cache line, several iterations
// are lumped together into a single detach.
//
//===----------------------------------------------------------------------===//

#include "LoopCoarsen.h"

#include "Utils/Flags.h"
#include "Utils/InstructionCost.h"
#include "Utils/Misc.h"
#include "Utils/Reductions.h"
#include "Utils/Tasks.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationDiagnosticInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Swarm.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;

#define DEBUG_TYPE "loop-coarsen"

#define LOOPS_COARSENED_ATTR "SwarmLoopsCoarsened"

// TODO(eforde): verify that 64 cache lines is a good default cap
static cl::opt<uint64_t> MaxCoarsenCacheLines("max-coarsen-cache-lines",
        cl::init(64),
        cl::desc("Places a limit on the coarsening factor of a parllel loop by "
                 "capping the number of cache lines each iteration may "
                 "access."));

static cl::opt<bool> CoarsenLoads("swarm-coarsenloads", cl::init(false),
    cl::desc("Enable coarsening of loops with striding loads when there are no striding stores"));

static cl::opt<bool> DisableLoopCoarsen("swarm-disableloopcoarsen", cl::init(false),
    cl::desc("Disable coarsening of loop iteration tasks"));

namespace {

class LoopCoarsen : public FunctionPass {
public:
  static char ID; // Pass ID, replacement for typeid
  LoopCoarsen() : FunctionPass(ID) {
    initializeLoopCoarsenPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

class LoopCoarsener {
  Loop &L;
  LoopInfo &LI;
  ScalarEvolution &SE;
  DominatorTree &DT;
  TargetTransformInfo &TTI;
  OptimizationRemarkEmitter &ORE;
  Function &F;

  static uint64_t CoarseningID;

  // Populated by analyze(), and constant afterward.
  SmallVector<LoadInst *, 8> Loads;
  SmallVector<const SCEVAddRecExpr *, 8> LoadPtrSCEVs;
  SmallVector<StoreInst *, 8> Stores;
  SmallVector<const SCEVAddRecExpr *, 8> StorePtrSCEVs;

public:
  LoopCoarsener(Loop& L,
                LoopInfo &LI,
                ScalarEvolution &SE,
                DominatorTree &DT,
                TargetTransformInfo &TTI,
                OptimizationRemarkEmitter &ORE)
      : L(L), LI(LI), SE(SE), DT(DT), TTI(TTI), ORE(ORE),
        F(*L.getHeader()->getParent())
  {}

  bool coarsen();

private:
  bool isPermitted() const;

  void analyze();

  template <class MemoryAccessInst>
  const SCEVAddRecExpr *getPtrSCEV(MemoryAccessInst *AccessInst) const;

  int64_t getStrideInBytes(const SCEVAddRecExpr *PtrSCEV) const;
  int64_t getOffsetInBytes(const SCEVAddRecExpr *PtrSCEV) const;

  uint32_t iterationsToAlignToCacheLines() const;

  // Returns the store instruction to prioritize for cache line alignment (via
  // prolog iterations) and spatial hints.  Must be strided.
  const SCEVAddRecExpr *getCriticalStoreSCEV() const;

  void debugNotCoarsenable(const Twine &Reason) const;
  void remarkAbandonedCoarsening(unsigned CoarseningFactor,
                                 StringRef RemarkName,
                                 const Twine &Reason,
                                 const Instruction *Inst = nullptr) const;

  static uint32_t strideToCoarseningFactor(int64_t);
};

} // end anonymous namespace


// Counter to generate unique IDs. Note these IDs won't be unique during LTO.
uint64_t LoopCoarsener::CoarseningID = 0;


// These are the per-loop checks (different from the per-function checks).
bool LoopCoarsener::isPermitted() const {
  if (!isExpandableSwarmLoop(&L, DT)) {
    debugNotCoarsenable("as it is not a canonical Swarm loop");
    return false;
  }

  if (L.getHeader()->getTerminator()->getMetadata(SwarmFlag::Coarsenable)) {
#ifndef NDEBUG
    BasicBlock *Header = const_cast<BasicBlock *>(L.getHeader());
    auto *IterDetach = cast<SDetachInst>(Header->getTerminator());
  
    SmallVector<SDetachInst *, 8> InternalDetaches;
    getOuterDetaches(IterDetach,
                     DT,
                     InternalDetaches);
    assert(all_of(InternalDetaches, [](const SDetachInst *DI) {
      return DI->isSuperdomain();
    }) && "Unsafe to coarsen loops with internal detaches in the loop's domain");
  
    SmallVector<BasicBlock *, 8> LoopBlocksWithoutContinuation;
    getNonDetachDescendants(DT, IterDetach->getDetached(), LoopBlocksWithoutContinuation);
    LoopBlocksWithoutContinuation.push_back(Header);
    LoopBlocksWithoutContinuation.push_back(L.getLoopLatch());
    assert(none_of(LoopBlocksWithoutContinuation, [](const BasicBlock *BB) {
      return any_of(*BB, [](const Instruction &I) { return isa<DeepenInst>(I); });
    }) && "The loop should not create any internal subdomains");
#endif

    DEBUG(dbgs() << "Will coarsen loop: " << L);
    return true;
  }

  if (findStringMetadataForLoop(&L, "llvm.loop.coarsen.factor")) {
    ORE.emit(DiagnosticInfoOptimizationFailure(DEBUG_TYPE,
                                               "UnsafeCoarsenPragma",
                                               L.getStartLoc(), L.getHeader())
             << "coarsen_factor pragma on loop with internal detaches. "
                "This may break sequential semantics!");
    return true;
  }

  return false;
}


static uint32_t gcd(uint32_t a, uint32_t b) {
  if (a == b)
    return a;
  if (a > b)
    return gcd(a - b, a);
  return gcd(a, b - a);
}


static uint32_t lcm(uint32_t a, uint32_t b) {
  uint32_t ret = a * b / gcd(a, b);
  assert((ret % a == 0) && (ret % b == 0) &&
         "integer overflow in lcm computation?");
  return ret;
}

template <class MemoryAccessInst>
const SCEVAddRecExpr *LoopCoarsener::getPtrSCEV(MemoryAccessInst *AccessInst) const {
  Value *PtrOp = AccessInst->getPointerOperand();

  const SCEV *PtrSCEV = SE.getSCEV(PtrOp);
  const SCEVAddRecExpr *PtrSCEVAdd = dyn_cast<SCEVAddRecExpr>(PtrSCEV);
  if (PtrSCEVAdd) {
    DEBUG(dbgs() << "Found additive recurrence " << *PtrSCEVAdd
                 << " for: " << *AccessInst << '\n');
    if (PtrSCEVAdd->getLoop() != &L) {
      //TODO(victory): Why does this sometimes happen? What does it mean?
      // For now, let's conservatively discard these SCEVs and treat these
      // memory accesses as non-striding.
      DEBUG(dbgs() << "Ignoring recurrence as it isn't associated with the current loop.\n");
      return nullptr;
    }
  } else {
    DEBUG(dbgs() << "No additive recurrence for: " << *AccessInst << '\n');
  }
  return PtrSCEVAdd;
}

int64_t LoopCoarsener::getStrideInBytes(const SCEVAddRecExpr *PtrSCEV) const {
  const SCEV *StepRec = PtrSCEV->getStepRecurrence(SE);
  if (const SCEVConstant *ConstStepRec = dyn_cast<SCEVConstant>(StepRec)) {
    int64_t Stride = ConstStepRec->getValue()->getSExtValue();
    DEBUG(dbgs() << "Found stride of " << Stride << " bytes.\n");
    assert(Stride != 0 &&
           "PtrSCEV should be an SCEVConstant rather than an SCEVAddRecExpr");
    return Stride;
  }
  DEBUG(dbgs() << "Found non-constant stride.\n");
  return 0;
}

int64_t LoopCoarsener::getOffsetInBytes(const SCEVAddRecExpr *PtrSCEV) const {
  const SCEV *StartSCEV = PtrSCEV->getStart();
  // Handle constant + pointer case
  // TODO: Consider other cases? (e.g., multiple constants if they ever happen)
  if (const SCEVAddExpr *AddSCEV = dyn_cast<SCEVAddExpr>(StartSCEV)) {
    StartSCEV = AddSCEV->getOperand(0);
  }
  if (const SCEVConstant *ConstStartSCEV = dyn_cast<SCEVConstant>(StartSCEV)) {
    int64_t Offset = ConstStartSCEV->getValue()->getSExtValue();
    DEBUG(dbgs() << "Found " << Offset << "-byte offset.\n");
    return Offset;
  }
  DEBUG(dbgs() << "Did not find constant offset.\n");
  return 0;
}


uint32_t LoopCoarsener::strideToCoarseningFactor(int64_t Stride) {
  if (Stride == 0) return 1;

  Stride = std::abs(Stride);
  if (Stride >= (int64_t)INT32_MAX) {  // Avoid integer truncation.
    DEBUG(dbgs() << "Skip huge stride of " << Stride << " bytes.\n");
    return 1;
  }
  // If each iteration touches 1.5 cache lines, makes sense to coarsen by two
  uint32_t CoarsenedStride = lcm(Stride, SwarmCacheLineSize);
  uint32_t CF = CoarsenedStride / Stride;
  assert(Stride * CF == CoarsenedStride);
  if (CoarsenedStride / SwarmCacheLineSize > MaxCoarsenCacheLines) {
    DEBUG(dbgs() << "skip bytes stride " << Stride
                 << " as it requires a CoarseningFactor of "
                 << CF
                 << " which exceeds MaxCoarsenCacheLines of "
                 << Twine(MaxCoarsenCacheLines));
    return 1;
  } else {
    return CF;
  }
}


void LoopCoarsener::analyze() {
  assert(Loads.empty() && LoadPtrSCEVs.empty()
         && Stores.empty() && StorePtrSCEVs.empty());
  DEBUG(dbgs() << "Trying to find memory access patterns in\n  " << L);

  BasicBlock *Spawned =
      cast<SDetachInst>(L.getHeader()->getTerminator())->getDetached();
  SmallVector<BasicBlock *, 8> DetachedBlocks;
  getNonDetachDescendants(DT, Spawned, DetachedBlocks);
  for (BasicBlock *BB: DetachedBlocks) {
    DEBUG(dbgs() << "Looking through BB for stride: " << *BB << '\n');
    for (Instruction &I : *BB) {
      if (LoadInst *LInst = dyn_cast<LoadInst>(&I)) {
        Loads.push_back(LInst);
        LoadPtrSCEVs.push_back(getPtrSCEV(LInst));
      } else if (StoreInst *SInst = dyn_cast<StoreInst>(&I)) {
        Stores.push_back(SInst);
        StorePtrSCEVs.push_back(getPtrSCEV(SInst));
      }
    }
  }
}


uint32_t LoopCoarsener::iterationsToAlignToCacheLines() const {
  SmallSetVector<uint32_t, 8> LoadFactors;
  SmallSetVector<uint32_t, 8> StoreFactors;

  for (auto *PtrSCEV : LoadPtrSCEVs) {
    if (PtrSCEV) {
      uint32_t CF = strideToCoarseningFactor(getStrideInBytes(PtrSCEV));
      if (CF > 1) LoadFactors.insert(CF);
    }
  }
  for (auto *PtrSCEV : StorePtrSCEVs) {
    if (PtrSCEV) {
      uint32_t CF = strideToCoarseningFactor(getStrideInBytes(PtrSCEV));
      if (CF > 1) StoreFactors.insert(CF);
    }
  }
  if (StoreFactors.empty() && LoadFactors.empty()) {
    debugNotCoarsenable("as no stride could be found for the loop");
    return 1;
  }

  uint32_t CoarseningFactor = 1;
  if (CoarsenLoads && StoreFactors.empty()) {
    DEBUG(dbgs() << "No stride found for stores; using loads.\n");
    for (uint32_t CF : LoadFactors) {
      CoarseningFactor = std::max(CF, CoarseningFactor);
    }
  } else {
    for (uint32_t CF : StoreFactors) {
      CoarseningFactor = std::max(CF, CoarseningFactor);
    }
  }

  assert(CoarseningFactor >= 1
         && "Doesn't make sense to detach less than one iteration");
  return CoarseningFactor;
}


const SCEVAddRecExpr *LoopCoarsener::getCriticalStoreSCEV() const {
    // For now, select the first strided store
    // TODO(dsm): In principle, better selection strategies are possible, e.g.,
    // using dominator/postdominator trees and/or coordinating with hint
    // selection. However, I suspect that the right solution is to split stores
    // with non-concordant offsets into separate tasks.
    const SCEVAddRecExpr *StridingPtrSCEV = nullptr;
    for (auto I : zip(Stores, StorePtrSCEVs)) {
      StoreInst *SInst;
      const SCEVAddRecExpr *PtrSCEV;
      std::tie(SInst, PtrSCEV) = I;
      if (!PtrSCEV) continue;
      if (!getStrideInBytes(PtrSCEV)) continue;

      if (!StridingPtrSCEV) StridingPtrSCEV = PtrSCEV;
      else {
        ORE.emit(DiagnosticInfoOptimizationFailure(DEBUG_TYPE, "CoarsenMultiStore",
                                                   L.getStartLoc(), L.getHeader())
                 << "Loop contains multiple striding stores.  Coarsened tasks "
                    "will be aligned to cache lines for one of them.");
        break;
      }
    }
    return StridingPtrSCEV;
}


/// Set hint of DI based on (StartAddr + Stride*IterIdx)
static void setHintFromStridingAddr(SDetachInst *DI, Value *StartAddr,
                                    int64_t Stride = 0,
                                    Value *IterIdx = nullptr) {
  assert(Stride || !IterIdx);

  // Despite the fact that this value is loop-invariant, we do not want to do
  // this bitcast in the preheader, because that would add to the number of
  // values that must be communicated to the loop iteration tasks.
  IRBuilder<> Builder(DI);
  Value *StartHintAddrInt = Builder.CreatePointerCast(
      StartAddr, Builder.getInt64Ty(), "start_hint_addr");

  Value *HintAddrInt =
      IterIdx ? Builder.CreateAdd(StartHintAddrInt,
                                  Builder.CreateNSWMul(IterIdx,
                                                       Builder.getInt64(Stride),
                                                       "offset_from_start"),
                                  "hint_addr")
              : StartHintAddrInt;
  setCacheLineHintFromAddress(DI, HintAddrInt);
}


/// \brief Coarsen the tasks in a Swarm loop to each execute multiple iterations.
///
/// For certain autoparallellized loops where all IVs are eliminated
/// except for one canonical IV, transform it into two nested loops:
/// an inner serial loop that does a small fixed number of iterations,
/// and an outer (still canonical) Swarm loop that detaches the inner loop
/// to run each small group of iterations as a task.
///
/// Example: given the following Swarm loop:
///
///   i = 0;
///   do {
///     spawn (i)
///       loop_body(i);
///   } while (i++ != Limit);
///
/// This method transforms it to the following:
///
///   i = 0;
///   do {
///     spawn (i) {
///       j = i;
///       do {
///         loop_body(j);
///       } while (j++ != min(Limit, i + CoarseningFactor-1));
///     }
///   } while (i < Limit; i += CoarseningFactor);
///
/// Note that this does mean the loop now uses fewer timestamps, and this
/// will only work safely on certain autoparallelized loops.
///
bool LoopCoarsener::coarsen() {
  DEBUG(dbgs() << "Checking whether to coarsen loop:\n " << L);
  DEBUG(dbgs() << "  starting at: ");
  DEBUG(L.getStartLoc().print(dbgs()));
  DEBUG(dbgs() << "\n");

  if (!isPermitted())
    return false;

  analyze();

  unsigned CoarseningFactor = iterationsToAlignToCacheLines();

  Optional<const MDOperand *> PragmaCoarsenFactor =
      findStringMetadataForLoop(&L, "llvm.loop.coarsen.factor");
  if (PragmaCoarsenFactor) {
    CoarseningFactor =
        mdconst::extract<ConstantInt>(**PragmaCoarsenFactor)->getZExtValue();
    ORE.emit(OptimizationRemarkAnalysis("swarm-loop-coarsen", "CoarsenFactor",
                                        L.getStartLoc(), L.getHeader())
             << "Coarsen factor overriden to "
             << ore::NV("NewCoarsen", CoarseningFactor));
  }

  if (CoarseningFactor == 1)
    return false;
  DEBUG(dbgs() << "Attempting to coarsen by " << CoarseningFactor << ":\n  "
               << L << " in " << F.getName() << "()\n");

  BasicBlock *Preheader = L.getLoopPreheader();
  BasicBlock *const OuterHeader = L.getHeader();
  SDetachInst *const DI = cast<SDetachInst>(OuterHeader->getTerminator());
  DeepenInst *const Domain = getDomain(DI);
  BasicBlock *const BodyStart = DI->getDetached();
  BasicBlock *const InnerLatch = DI->getContinue();
  assert(InnerLatch == L.getLoopLatch());
  SReattachInst *RI = getUniqueSReattachIntoLatch(L);
  assert(RI);
  BasicBlock *const BodyEnd = RI->getParent();
  const bool HasReductionCalls = hasReductionCalls(L);

  const SCEV *LimitSCEV = SE.getBackedgeTakenCount(&L);
  DEBUG(dbgs() << "Loop limit: " << *LimitSCEV << "\n");
  if (isa<SCEVCouldNotCompute>(LimitSCEV)) {
    remarkAbandonedCoarsening(CoarseningFactor, "UnknownLimit",
                              "because the loop limit is unknown");
    return false;
  }

  PHINode *const CanonicalIV = canonicalizeIVsAllOrNothing(
      L, DI->getTimestamp()->getType()->getIntegerBitWidth(), DT, SE);
  if (!CanonicalIV) {
    remarkAbandonedCoarsening(CoarseningFactor, "FailedToCanonicalize",
                              "because canonicalization failed");
    return false;
  }
  DEBUG(dbgs() << "Canonical IV: " << *CanonicalIV << "\n");

  if (DI->getTimestamp() != L.getCanonicalInductionVariable()) {
    // FIXME(mcj) Does LoopCoarsen require that the timestamp matches the
    // canonical induction variable? Note that isExpandableSwarmLoop does not
    // check for this condition.
    // An old assertion assumed that this is true for all canonical Swarm loops
    remarkAbandonedCoarsening(CoarseningFactor, "TSDoesNotMatchCanonicalIV",
                              "because the timestamp does not "
                              "match the canonical IV");
    return false;
  }

  ORE.emit(OptimizationRemark(DEBUG_TYPE, "CoarsenLoop", L.getStartLoc(),
                              OuterHeader)
           << "coarsening " << ore::NV("CoarseningFactor", CoarseningFactor)
           << " loop iterations per task");
  if (auto ConstLimit = dyn_cast<SCEVConstant>(LimitSCEV)) {
    unsigned IterCount = ConstLimit->getValue()->getLimitedValue(~0U) + 1;
    if (IterCount <= CoarseningFactor) {
      // TODO(victory): Instead of coarsening, should we just serialize tiny loops?
      ORE.emit(DiagnosticInfoOptimizationFailure(DEBUG_TYPE, "CoarsenTinyLoop",
                                                 L.getStartLoc(), OuterHeader)
               << "tiny loop of " << ore::NV("IterCount", IterCount)
               << " iterations coarsened to 1 or 2 tasks");
    } else if (IterCount <= CoarseningFactor * 8) {
      ORE.emit(OptimizationRemarkAnalysis(DEBUG_TYPE, "CoarsenShortLoop",
                                          L.getStartLoc(), OuterHeader)
               << "short loop of " << ore::NV("IterCount", IterCount)
               << " iterations coarsened to few tasks");
    }
  }

  {
    unsigned CIVWidth = CanonicalIV->getType()->getIntegerBitWidth();
    if (CIVWidth > LimitSCEV->getType()->getIntegerBitWidth())
      LimitSCEV = SE.getZeroExtendExpr(
          LimitSCEV, IntegerType::get(OuterHeader->getContext(), CIVWidth));
    assert(CIVWidth == LimitSCEV->getType()->getIntegerBitWidth());
  }
  SCEVExpander Exp(SE, OuterHeader->getModule()->getDataLayout(), "lc");
  Value *Limit = Exp.expandCodeFor(LimitSCEV, CanonicalIV->getType(),
                                   Preheader->getTerminator());

  // Set up blocks
  assert(CanonicalIV == &OuterHeader->front() &&
         !isa<PHINode>(CanonicalIV->getNextNode()) &&
         "Other phis were eliminated by induction variable substitution");
  BasicBlock *const InnerPreheader =
      SplitBlock(OuterHeader, CanonicalIV->getNextNode(), &DT, &LI);
  BasicBlock *const InnerHeader =
      SplitBlock(InnerPreheader, InnerPreheader->getFirstNonPHI(), &DT, &LI);
  InnerHeader->setName(OuterHeader->getName() + ".loopcoarse.inner");
  InnerPreheader->setName(OuterHeader->getName() + ".coarse.innerprehead");
  OuterHeader->setName(OuterHeader->getName() + ".coarse.outer");
  BasicBlock *const Reattaching =
      SplitBlock(InnerLatch, InnerLatch->getTerminator(), &DT, &LI);
  BasicBlock *const OuterLatch =
      SplitBlock(Reattaching, Reattaching->getTerminator(), &DT, &LI);
  OuterLatch->setName(InnerLatch->getName() + ".outer");
  Reattaching->takeName(InnerLatch);
  InnerLatch->setName(Reattaching->getName() + ".inner");

  // Restore canonical structure to outer loop
  // Move detach towards outer header
  BranchInst::Create(BodyStart, DI);
  TerminatorInst *TI = OuterHeader->getTerminator();
  //victory: Right now, all autoparallelized loop's detach timestamps
  // are computed simply from the canonical IV, but if there are ever more
  // instructions involved in computing the timestamp, they may need to be
  // hoisted here as well. (TODO)
  //victory: At least if we fail to hoist those instructions, it should generate
  // a nice clear instruction-doesn't-dominate-uses verification failure, so
  // it should be easy to notice when we need to fix this.
  DI->moveBefore(TI);
  TI->eraseFromParent();
  DI->replaceUsesOfWith(BodyStart, InnerPreheader);
  DI->replaceUsesOfWith(InnerLatch, OuterLatch);
  DT.changeImmediateDominator(InnerLatch, BodyEnd);
  DT.changeImmediateDominator(OuterLatch, OuterHeader);
  // Move reattach towards outer latch
  ReplaceInstWithInst(RI, BranchInst::Create(InnerLatch));
  RI = SReattachInst::Create(OuterLatch);
  ReplaceInstWithInst(Reattaching->getTerminator(), RI);

  assert(isExpandableSwarmLoop(&L, DT) &&
         "LoopCoarsener broke loop canonicalness");

  // Before fixing up the latch terminators, let's pick the arbitrary convention
  // that the back edge is always successor 0 (the branch target if the
  // condition is true) and the exit edge is successor 1 (the false target).
  // This means the condition used by the branch will have the same sense as
  // the conditions in the pseudocode in the function comment above.
  auto *const OuterBackedge = cast<BranchInst>(OuterLatch->getTerminator());
  if (OuterBackedge->getSuccessor(0) != OuterHeader) {
    IRBuilder<> Builder(InnerLatch->getTerminator());
    OuterBackedge->setCondition(
        Builder.CreateNot(OuterBackedge->getCondition(), "backedge_cond"));
    OuterBackedge->swapSuccessors();
  }
  assert(OuterBackedge->getSuccessor(0) == OuterHeader);
  BasicBlock *const ExitBlock = OuterBackedge->getSuccessor(1);
  assert(ExitBlock->getSinglePredecessor() == OuterLatch &&
         "Guaranteed by canonical form");

  // Make the inner loop a loop
  auto *const InnerBackedge = cast<BranchInst>(OuterBackedge->clone());
  ReplaceInstWithInst(InnerLatch->getTerminator(), InnerBackedge);
  InnerBackedge->replaceUsesOfWith(ExitBlock, Reattaching);
  InnerBackedge->replaceUsesOfWith(OuterHeader, InnerHeader);
  Loop *InnerLoop = new Loop();
  // LoopSimplify's separateNestedLoop() demonstrates how to use some of these
  // LoopInfo-updating APIs.
  L.addChildLoop(InnerLoop);
  for (BasicBlock *BB : L.blocks()) {
    if (BB != OuterHeader && BB != InnerPreheader &&
        BB != Reattaching && BB != OuterLatch) {
      InnerLoop->addBlockEntry(BB);
      LI.changeLoopFor(BB, InnerLoop);
    }
  }
  InnerLoop->moveToHeader(InnerHeader);
  assert(InnerLoop->contains(InnerLatch));
#ifndef NDEBUG
  InnerLoop->verifyLoop();
  DenseSet<const Loop *> Loops;
  L.verifyLoopNest(&Loops);
  if (Loop *Parent = L.getParentLoop()) Parent->verifyLoopNest(&Loops);
#endif

  // Useful constants for IV maniputlations
  auto *const CoarseningFactorVal =
      ConstantInt::get(CanonicalIV->getType(), CoarseningFactor);
  auto *const CFMinusOneVal =
      ConstantInt::get(CanonicalIV->getType(), CoarseningFactor - 1);
  auto *const ZeroVal = ConstantInt::get(CanonicalIV->getType(), 0);
  auto *const OneVal = ConstantInt::get(CanonicalIV->getType(), 1);

  // Set the timestamp of each outer iteration as the original iteration index
  // (we could use CanonicalIV as the timestamp but it generates worse task
  // code).
  IRBuilder<> Builder(OuterHeader->getFirstNonPHI());
  Value *IterTS =
      Builder.CreateNUWMul(CanonicalIV, CoarseningFactorVal, "iter_ts");
  DI->setTimestamp(IterTS);

  // Patch up canonical IV phis and the inner loop increment.
  Builder.SetInsertPoint(InnerHeader->getFirstNonPHI());
  PHINode *const InnerCanonicalIV =
      Builder.CreatePHI(CanonicalIV->getType(), 2, "inner_canonical_iv");
  Value *NextInnerIV =
      Builder.CreateNUWAdd(InnerCanonicalIV, OneVal, "inner_inc");
  InnerCanonicalIV->addIncoming(ZeroVal, InnerPreheader);
  InnerCanonicalIV->addIncoming(NextInnerIV, InnerLatch);

  // Replace uses of CanonicalIV with InnerIter inside the loop.
  Value *InnerIter =
      Builder.CreateNUWAdd(IterTS, InnerCanonicalIV, "inner_iter");
  auto UI = CanonicalIV->use_begin(), E = CanonicalIV->use_end();
  while (UI != E) {
    Use &U = *(UI++); // Carefully crafted to avoid iterator invalidation
    if (U.getUser() != IterTS) {
      assert(cast<Instruction>(U.getUser())->getParent() != OuterHeader);
      U.set(InnerIter);
    }
  }

  // Set up the increment and exit condition of the outer loop
  Builder.SetInsertPoint(OuterHeader->getTerminator());
  Value *OuterInc = Builder.CreateNUWAdd(CanonicalIV, OneVal, "outer_inc");
  int Idx = CanonicalIV->getBasicBlockIndex(OuterLatch);
  assert(Idx >= 0);
  CanonicalIV->setIncomingValue(Idx, OuterInc);

  Builder.SetInsertPoint(Preheader->getTerminator());
  Value *FinalIterIVValue = Builder.CreateUDiv(Limit, CoarseningFactorVal,
                                               "outer_backedge_taken_count");
  Builder.SetInsertPoint(OuterBackedge);
  Value *OuterBackedgeCond = Builder.CreateICmpNE(CanonicalIV, FinalIterIVValue,
                                                  "outer_backedge_cond");
  OuterBackedge->setCondition(OuterBackedgeCond);
  // Verify that ScalarEvolution can understand this thing.
  SE.forgetLoop(&L);
  assert(!isa<SCEVCouldNotCompute>(SE.getBackedgeTakenCount(&L)) &&
         "LoopCoarsener broke ScalarEvolution's tripcount analysis");

  // Set up exit condition for the inner loop
  Builder.SetInsertPoint(InnerPreheader->getTerminator());
  // This limit value is invariant w.r.t. the inner loop, so compute it once
  // in the preheader.
  Value *InnerIterLimit = Builder.CreateNUWSub(Limit, IterTS);
  Value *InnerLimit =
      Builder.CreateSelect(Builder.CreateICmpULT(InnerIterLimit, CFMinusOneVal),
                           InnerIterLimit, CFMinusOneVal, "inner_limit");
  Builder.SetInsertPoint(InnerBackedge);
  InnerBackedge->setCondition(Builder.CreateICmpNE(InnerCanonicalIV, InnerLimit,
                                                   "inner_backedge_cond"));

  // Verify that ScalarEvolution can understand inner loop too
  SE.forgetLoop(InnerLoop);
  assert(!isa<SCEVCouldNotCompute>(SE.getBackedgeTakenCount(InnerLoop)) &&
         "LoopCoarsener broke ScalarEvolution's analysis for inner loop");

  ++CoarseningID;
  addStringMetadataToLoop(&L, SwarmCoarsened::OuterLoop, CoarseningID);
  addStringMetadataToLoop(InnerLoop, SwarmCoarsened::InnerLoop, CoarseningID);

  DEBUG(assertVerifyFunction(F, "After initial coarsening of loop", &DT));

  // Pick a striding store to generate hints and align tasks to cache lines
  const SCEVAddRecExpr *CriticalSCEV = getCriticalStoreSCEV();
  const int64_t Stride = CriticalSCEV ? getStrideInBytes(CriticalSCEV) : 0;
  Exp.setInsertPoint(Preheader->getTerminator());
  Value *const StartAddr =
      CriticalSCEV ? Exp.expandCodeFor(CriticalSCEV->getStart()) : nullptr;
  const bool GenerateHint = CriticalSCEV && DI->isNoHint();
  if (GenerateHint) {
    DEBUG(dbgs() << "Generating hints for coarsened loop tasks based on stores\n"
                 << "  that start at ptr " << *StartAddr << '\n'
                 << "  and stride by " << Stride << "bytes per iteration.\n");
    setHintFromStridingAddr(DI, StartAddr, Stride, IterTS);
  }
  DEBUG(assertVerifyFunction(F, "After picking critical striding access addr",
                             &DT));

  // Expandable (canonical) loop form guarantees that the loop produces no
  // SSA values used by the continuation.
  assert(!isa<PHINode>(ExitBlock->front()) && "guaranteed by expandable form");
  {
    // Epilog needs to know how many iterations the prolog takes
    Value *PrologIterationsVal = ZeroVal;

    // Use a prolog if there is a critical striding memory access to align to.
    bool UseProlog = CriticalSCEV;
    if (UseProlog) {
      // Transform the general structure from:
      // Preheader->OuterLoop(<->InnerLoop)->ExitBlock
      // to:
      // PrologStart->PrologLoop->OuterCheck->Prehead->OuterLoop->ExitBlock
      //       \___________________/      \________________________/
      // PrologLoop executes the first min(Limit, PrologIterations) iters
      // OuterCheck runs OuterLoop only when Limit >= PrologIterations
      // PrologStart may compute the number of prolog iterations dynamically.
      // If it discovers 0 prolog iterations are needed, it skips PrologLoop.
      //
      // NOTE: PrologLoop is short (< 1 coarsened task) and it's common for code
      // immediately preceding the loop to touch the same cache line(s) as the
      // prolog (e.g., a typical reason to emit a prolog is that the code before
      // the loop sets the first value of an array, and the loop starts at the
      // second element). Therefore, PrologLoop is detached with SAMEHINT so
      // that any conflict on unaligned data in the prolog do not kill the
      // spawning of the rest of the loop.

      // Move preheader code (except the terminator) to PrologStart. This
      // includes limit computation and deepening.
      BasicBlock *PrologStart =
          SplitBlock(Preheader, Preheader->getTerminator(), &DT, &LI);
      std::swap(Preheader, PrologStart);
      PrologStart->setName(OuterHeader->getName() + ".prolog_start");

      IRBuilder<> Builder(PrologStart->getTerminator());
      // Compute Prolog iteration count
      {
        Builder.SetInsertPoint(PrologStart->getTerminator());
        Value *StoreAddrInt = Builder.CreatePtrToInt(
            StartAddr, CanonicalIV->getType(), "start_addr");
        // Ensure lines are a power-of-2 bytes...
        assert((SwarmCacheLineSize & (SwarmCacheLineSize - 1)) == 0);
        Value *LineMask = ConstantInt::get(CanonicalIV->getType(),
                                           SwarmCacheLineSize - 1);
        Value *Offset =
            Builder.CreateAnd(StoreAddrInt, LineMask, "cacheline_offset");

        // NOTE: Stride may be signed
        assert(Stride);
        Value *StrideVal = ConstantInt::get(CanonicalIV->getType(), Stride);
        Value *OffsetIters =
            Builder.CreateSDiv(Offset, StrideVal, "offset_iters");
        // Offset is positive, but OffsetIters may be negative if Stride is
        // negative. In any case, abs(OffsetIters) < CoarseningFactorVal, so
        // X = CoarseningFactorVal - OffsetIters is always positive and
        // X mod CoarseningFactor should be the number of iterations to get to
        // the next cache line, regardless of #lines per coarsened loop or
        // stride sign.
        // TODO(dsm): Test thoroughly with negative strides and multi-line CFs!
        // These are untested b/c the test fails somewhere else...
        PrologIterationsVal = Builder.CreateURem(
            Builder.CreateNUWSub(CoarseningFactorVal, OffsetIters),
            CoarseningFactorVal, "prolog_iters");
      }
      // NOTE: this may go to -1
      auto *const PIMinusOneVal =
          Builder.CreateNUWSub(PrologIterationsVal, OneVal);
      // Supplant Limit (used in both prolog and epilog below).
      // It's important for this to be in PrologStart so it can be captured
      // into a shared loop closure by LoopExpansion.
      Value *OldLimit = Limit;
      Limit = Builder.CreateNUWSub(OldLimit, PrologIterationsVal, "new_limit");

      Value *PrologLimit =
          Builder.CreateSelect(Builder.CreateICmpULT(OldLimit, PIMinusOneVal),
                               OldLimit, PIMinusOneVal, "prolog_limit");
      // the prolog is conditional now
      Value *PrologCond =
          Builder.CreateICmpNE(PrologIterationsVal, ZeroVal, "prolog_cond");

      // Set up OuterCheck BB
      BasicBlock *OuterCheck =
          SplitBlock(PrologStart, PrologStart->getTerminator(), &DT, &LI);
      OuterCheck->setName(OuterHeader->getName() + ".outer_check");

      // Compute outer check condition and new OuterLoop limit (taken
      // backedges). These are overflow-safe.
      Builder.SetInsertPoint(OuterCheck->getTerminator());
      Value *OuterCond =
          Builder.CreateICmpUGE(OldLimit, PrologIterationsVal, "outer_cond");
      Builder.SetInsertPoint(OuterBackedge);
      Value *OuterLimit =
          Builder.CreateUDiv(Limit, CoarseningFactorVal, "outer_limit");
      Value *OuterBackedgeCond =
          Builder.CreateICmpNE(CanonicalIV, OuterLimit, "outer_backedge_cond");
      OuterBackedge->setCondition(OuterBackedgeCond);

      // Adjust and supplant IterTS
      Builder.SetInsertPoint(OuterHeader->getFirstNonPHI());
      Instruction *NewIterTS = BinaryOperator::Create(
          Instruction::Add,
          Builder.CreateNUWMul(CanonicalIV, CoarseningFactorVal),
          PrologIterationsVal);
      NewIterTS->setHasNoUnsignedWrap();
      NewIterTS->setName("iter_ts");
      ReplaceInstWithInst(cast<Instruction>(IterTS), NewIterTS);
      IterTS = NewIterTS;

      // Create PrologLoop from InnerLoop
      ValueToValueMapTy OldValueMap;
      SmallVector<BasicBlock *, 8> PrologLoopBlocks;
      Loop *PrologLoop = cloneLoopWithPreheader(
          OuterCheck, PrologStart, InnerLoop, OldValueMap, ".prolog", &LI, &DT,
          PrologLoopBlocks, L.getParentLoop() /* PrologLoop's parent */);
      // Change OldValueMap to replace uses of InnerTS/Limit with prolog values
      OldValueMap[IterTS] = ZeroVal;
      OldValueMap[InnerLimit] = PrologLimit;
      remapInstructionsInBlocks(PrologLoopBlocks, OldValueMap);

      // Link basic blocks
      BasicBlock *const PrologPreheader = PrologLoop->getLoopPreheader();
      ReplaceInstWithInst(PrologStart->getTerminator(),
              BranchInst::Create(PrologPreheader, OuterCheck, PrologCond));
      DT.changeImmediateDominator(PrologPreheader, PrologStart);
      DT.changeImmediateDominator(OuterCheck, PrologStart);

      BasicBlock *const PrologLatch = PrologLoop->getLoopLatch();
      auto *const PrologBackedge =
          cast<BranchInst>(PrologLatch->getTerminator());
      assert(PrologBackedge->getSuccessor(0) == PrologLoop->getHeader());
      PrologBackedge->setSuccessor(1, OuterCheck);

      BasicBlock *OuterExit =
          SplitBlock(ExitBlock, &ExitBlock->front(), &DT, &LI);
      ReplaceInstWithInst(OuterCheck->getTerminator(),
                          BranchInst::Create(Preheader, OuterExit, OuterCond));
      DT.changeImmediateDominator(OuterExit, OuterCheck);

      // Detach prolog loop
      // FIXME(mcj) this code is very similar to the epilog variant.
      // De-duplicate. Why this coarsen() method is so enormous is beyond me.
      // Class variables, they're amazing.
      BasicBlock *PrologExit = SplitBlockPredecessors(OuterCheck, {PrologLatch},
                                                      ".prolog_exit", &DT, &LI);
      SDetachInst *PrologDI;
      SReattachInst *PrologRI;
      detachTask(PrologPreheader->getTerminator(), PrologExit->getTerminator(),
                 ZeroVal, Domain, DetachKind::CoarsenProlog,
                 OuterHeader->getName() + ".prolog_task",
                 &DT, &LI, &PrologDI, &PrologRI);
      PrologDI->setDebugLoc(DI->getDebugLoc());
      PrologRI->setDebugLoc(RI->getDebugLoc());
      if (GenerateHint) setHintFromStridingAddr(PrologDI, StartAddr);
      else PrologDI->setIsSameHint(true);

      addStringMetadataToLoop(PrologLoop, SwarmCoarsened::Prolog, CoarseningID);
      assert(getProlog(L, LI) == PrologLoop &&
             "getProlog() has fallen out of sync!");
      assert(getPrologSDetach(*PrologLoop) == PrologDI &&
             "getPrologSDetach() has fallen out of sync!");
      if (HasReductionCalls) moveReductionCallsAfterLoop(*PrologLoop);

      DEBUG(assertVerifyFunction(F, "After generating prolog", &DT));
#ifndef NDEBUG
      PrologLoop->verifyLoop();
      DenseSet<const Loop *> Loops;
      L.verifyLoopNest(&Loops);
      if (Loop *Parent = L.getParentLoop()) Parent->verifyLoopNest(&Loops);
#endif
    }

    // Split last few iterations into an epilog to simplify steady-state tasks.
    // TODO: Make epilog generation depend on opt level or unrolling.
    bool useEpilog = true;
    if (useEpilog) {
      // Transform the general structure from:
      // Preheader->OuterLoop(<->InnerLoop)->ExitBlock
      // to:
      // CoarseCheck->Prehead->OuterLoop->EpilogCheck->EpilogLoop->ExitBlock
      //          \_______________________/        \_________________/
      // InnerLoop is changed to run a fixed CoarseningFactor iterations per loop
      // EpilogLoop is a copy of InnerLoop that handles the last few iterations
      // CoarseCheck runs OuterLoop only when there are >= CoarseningFactor iters
      // EpilogCheck runs EpilogLoop only when the iteration count is not a
      //   multiple of CoarseningFactor
      // EpilogLoop, like InnerLoop, is detached with the appropriate timestamp

      // Move preheader code (except the terminator) to CoarseCheck (this
      // includes Limit computation and deepening, so that EpilogLoop is on the
      // same domain).
      // NOTE: If a prolog was generated, this code has already been hoisted to
      // PrologStart.
      BasicBlock *CoarseCheck =
          SplitBlock(Preheader, Preheader->getTerminator(), &DT, &LI);
      std::swap(Preheader, CoarseCheck);
      CoarseCheck->setName(OuterHeader->getName() + ".coarse_check");

      // Compute coarse check condition and new OuterLoop limit (taken
      // backedges). These are crafted to be overflow-safe. Specifically, with
      // coarsening factor CF:
      //    CoarseCond = Limit >=u (CF - 1)
      //    CoarseLimit = (Limit - (CF - 1)) / CF
      // CoarseLimit never overflows (subtraction), and if CoarseCond is true,
      // it never underflows either.
      IRBuilder<> Builder(CoarseCheck->getTerminator());
      Value *CoarseCond =
          Builder.CreateICmpUGE(Limit, CFMinusOneVal, "coarse_cond");
      Builder.SetInsertPoint(OuterLatch->getTerminator());
      Value *CoarseLimit =
          Builder.CreateUDiv(Builder.CreateNUWSub(Limit, CFMinusOneVal),
                             CoarseningFactorVal, "coarse_limit");

      // Change outer and inner loop iteration counts
      Value *OuterBackedgeCond =
          Builder.CreateICmpNE(CanonicalIV, CoarseLimit, "outer_backedge_cond");
      OuterBackedge->setCondition(OuterBackedgeCond);
      Builder.SetInsertPoint(InnerBackedge);
      InnerBackedge->setCondition(Builder.CreateICmpNE(
          InnerCanonicalIV, CFMinusOneVal, "inner_backedge_cond"));

      // Set up EpilogCheck BB
      // NOTE: Because Limit == Number of taken backedges == Iterations - 1,
      // we run the epilog iff (Limit % CF) != (CF -1) <=> (Iters % CF) != 0
      BasicBlock *EpilogCheck = SplitBlockPredecessors(
          ExitBlock, {OuterLatch}, ".epilog_check", &DT, &LI);
      assert(EpilogCheck);
      Builder.SetInsertPoint(EpilogCheck->getTerminator());
      Value *EpilogLimit =
          Builder.CreateURem(Limit, CoarseningFactorVal, "epilog_limit");
      Value *EpilogCond =
          Builder.CreateICmpNE(EpilogLimit, CFMinusOneVal, "epilog_cond");
      Value *EpilogTS = Builder.CreateNUWSub(Limit, EpilogLimit, "epilog_ts");
      EpilogTS = // must account for prolog iterations
          Builder.CreateNUWAdd(EpilogTS, PrologIterationsVal, "epilog_ts");

      // Preserve dedicated OuterLoop exit (required for loop canonicalness)
      BasicBlock *OuterExit = SplitBlockPredecessors(EpilogCheck, {OuterLatch},
                                                     ".outer_exit", &DT, &LI);
      assert(OuterExit);
      (void)OuterExit;

      // Create EpilogLoop from InnerLoop
      ValueToValueMapTy OldValueMap;
      SmallVector<BasicBlock *, 8> EpilogLoopBlocks;
      Loop *EpilogLoop = cloneLoopWithPreheader(
          ExitBlock, EpilogCheck, InnerLoop, OldValueMap, ".epilog", &LI, &DT,
          EpilogLoopBlocks, L.getParentLoop() /* EpilogLoop's parent */);
      OldValueMap[IterTS] = EpilogTS; // to replace uses with epilog timestamp
      remapInstructionsInBlocks(EpilogLoopBlocks, OldValueMap);

      BasicBlock *const EpilogPreheader = EpilogLoop->getLoopPreheader();
      BasicBlock *const EpilogLatch = EpilogLoop->getLoopLatch();

      // Change epilog termination condition
      auto *const EpilogBackedge =
          cast<BranchInst>(EpilogLatch->getTerminator());
      Builder.SetInsertPoint(EpilogBackedge);
      Value *EpilogIV = OldValueMap[InnerCanonicalIV];
      EpilogBackedge->setCondition(
          Builder.CreateICmpNE(EpilogIV, EpilogLimit, "epilog_backedge_cond"));
      assert(EpilogBackedge->getSuccessor(0) == EpilogLoop->getHeader());
      EpilogBackedge->setSuccessor(1, ExitBlock);

      // Link basic blocks
      ReplaceInstWithInst(
          CoarseCheck->getTerminator(),
          BranchInst::Create(Preheader, EpilogCheck, CoarseCond));
      ReplaceInstWithInst(
          EpilogCheck->getTerminator(),
          BranchInst::Create(EpilogPreheader, ExitBlock, EpilogCond));
      DT.changeImmediateDominator(EpilogCheck, CoarseCheck);
      DT.changeImmediateDominator(ExitBlock, EpilogCheck);

      // Detach epilog loop
      BasicBlock *EpilogExit = SplitBlockPredecessors(ExitBlock, {EpilogLatch},
                                                      ".epilog_exit", &DT, &LI);
      assert(EpilogExit);
      SDetachInst *EpilogDI;
      SReattachInst *EpilogRI;
      detachTask(EpilogPreheader->getTerminator(), EpilogExit->getTerminator(),
                 EpilogTS, Domain, DetachKind::CoarsenEpilog,
                 OuterHeader->getName() + ".epilog_task",
                 &DT, &LI, &EpilogDI, &EpilogRI);
      EpilogDI->setDebugLoc(DI->getDebugLoc());
      EpilogRI->setDebugLoc(RI->getDebugLoc());
      if (GenerateHint)
        setHintFromStridingAddr(EpilogDI, StartAddr, Stride, EpilogTS);

      addStringMetadataToLoop(EpilogLoop, SwarmCoarsened::Epilog, CoarseningID);
      assert(getEpilog(L, LI) == EpilogLoop &&
             "getEpilog() has fallen out of sync!");
      assert(getEpilogSDetach(*EpilogLoop) == EpilogDI &&
             "getEpilogSDetach() has fallen out of sync!");
      if (HasReductionCalls) moveReductionCallsAfterLoop(*EpilogLoop);

      DEBUG(assertVerifyFunction(F, "After generating epilog", &DT));
#ifndef NDEBUG
      EpilogLoop->verifyLoop();
      DenseSet<const Loop *> Loops;
      L.verifyLoopNest(&Loops);
      if (Loop *Parent = L.getParentLoop()) Parent->verifyLoopNest(&Loops);
#endif
    }
  }

  // Push any call of __sccrt_reduction* within the inner loop to the inner
  // loop's exit block, by setting up an inner loop reduction.
  // Do not make this change until after the Prolog and Epilog loops have been
  // copied from InnerLoop, so that the original calls of __sccrt_reduction*
  // can be copied correctly.
  if (HasReductionCalls) moveReductionCallsAfterLoop(*InnerLoop);

  assert(isExpandableSwarmLoop(&L, DT) &&
         "LoopCoarsener broke loop canonicalness");
  DEBUG(assertVerifyFunction(F, "After coarsening one loop", &DT));

  return true;
}


void LoopCoarsener::debugNotCoarsenable(const Twine &Reason) const {
  DEBUG(dbgs() << "Loop will not be coarsened\n  "
               << L << Reason << ".\n");
}


void LoopCoarsener::remarkAbandonedCoarsening(unsigned CoarseningFactor,
                                              StringRef RemarkName,
                                              const Twine &Reason,
                                              const Instruction *Inst) const {
  std::string Msg;
  raw_string_ostream OS(Msg);
  OS << "abandoned coarsening by "
     << CoarseningFactor << " "
     << Reason << ".";
  OS.flush();
  ORE.emit((Inst
            ? OptimizationRemarkMissed(DEBUG_TYPE, RemarkName, Inst)
            : OptimizationRemarkMissed(DEBUG_TYPE, RemarkName,
                                       L.getStartLoc(), L.getHeader()))
           << Msg);
}


bool LoopCoarsen::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  if (DisableLoopCoarsen)
    return false;

  // Don't second guess the programmer's manual parallelization
  if (!F.hasFnAttribute(SwarmFlag::Parallelized))
    return false;

  // Don't re-process loops. This condition could occur due to other Swarm
  // FunctionPasses that break the rules and outline new functions.
  if (F.hasFnAttribute(LOOPS_COARSENED_ATTR))
    return false;
  F.addFnAttr(LOOPS_COARSENED_ATTR);

  // A compiler performance optimization
  if (!llvm::hasAnySDetachInst(F))
    return false;

  DEBUG(dbgs() << "Coarsening loops in function: " << F.getName() << "\n");

  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  auto &ORE = getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

  SmallVector<Loop *, 8> Worklist;
  for (Loop *TopLevelLoop : LI)
    for (Loop *L : depth_first(TopLevelLoop)) {
      assert(none_of(Worklist,
                     [L](Loop *OtherL) { return L->contains(OtherL); }) &&
             "Pre-order traversal guarantees we visit outer loops first");
      Worklist.push_back(L);
    }

  // A coarsenable loop may be subsumed as part of the continuation of an
  // another coarsenable loop.  To handle this case, we walk the worklist in
  // reverse so that we process inner loops first.
  bool Changed = false;
  for (Loop *L : reverse(Worklist)) {
    LoopCoarsener LC(*L, LI, SE, DT, TTI, ORE);
    Changed |= LC.coarsen();
  }

  DEBUG(dbgs() << "Done coarsening loops in " << F.getName() << "()\n");
  assertVerifyFunction(F, "After loop coarsening", &DT);

  return Changed;
}


void LoopCoarsen::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequiredID(LoopSimplifyID);
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();
  AU.addRequired<TargetTransformInfoWrapperPass>();
  AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
}


char LoopCoarsen::ID = 0;

INITIALIZE_PASS_BEGIN(LoopCoarsen, DEBUG_TYPE,
                      "Coarsen canonical Swarm loops",
                      false,
                      false)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_END(LoopCoarsen, DEBUG_TYPE,
                    "Coarsen canonical Swarm loops",
                    false, false)

Pass *llvm::createLoopCoarsenPass() {
  return new LoopCoarsen();
}


/// Populate SiblingLoops with all loops that share a parent with L other than
/// L itself.
static void getSiblingLoops(const Loop &L, const LoopInfo &LI,
                            SmallVectorImpl<const Loop *> &SiblingLoops) {
  if (const Loop *Parent = L.getParentLoop()) {
    for (const Loop *Sibling : *Parent)
      if (Sibling != &L)
        SiblingLoops.push_back(Sibling);
  } else {  // L is an outer loop
    for (const Loop *OuterLoop : LI)
      if (OuterLoop != &L)
        SiblingLoops.push_back(OuterLoop);
  }
}


static const Loop *getPrologOrEpilog(
        const Loop &CoarsenedOuterLoop,
        const LoopInfo &LI,
        bool ReturnProlog) {
  Optional<const MDOperand *> MDOpt =
          findStringMetadataForLoop(&CoarsenedOuterLoop,
                                    SwarmCoarsened::OuterLoop);
  if (!MDOpt) return nullptr;
  uint64_t CoarseningID =
          mdconst::extract<ConstantInt>(**MDOpt)->getZExtValue();

  SmallVector<const Loop *, 8> SiblingLoops;
  getSiblingLoops(CoarsenedOuterLoop, LI, SiblingLoops);
  for (const Loop *L : SiblingLoops) {
    Optional<const MDOperand *> MDOpt = findStringMetadataForLoop(L,
            (ReturnProlog
             ? SwarmCoarsened::Prolog
             : SwarmCoarsened::Epilog));
    if (MDOpt && mdconst::extract<ConstantInt>(**MDOpt)
                 ->getZExtValue() == CoarseningID) {
      return L;
    }
  }

  return nullptr;
}


const Loop *llvm::getProlog(const Loop &COL, const LoopInfo &LI) {
  return getPrologOrEpilog(COL, LI, true);
}


const Loop *llvm::getEpilog(const Loop &COL, const LoopInfo &LI) {
  return getPrologOrEpilog(COL, LI, false);
}


const SDetachInst *llvm::getPrologSDetach(const Loop &Prolog) {
  if (!findStringMetadataForLoop(&Prolog, SwarmCoarsened::Prolog))
    return nullptr;
  return cast<const SDetachInst>(Prolog.getLoopPreheader()
                                       ->getUniquePredecessor()
                                       ->getTerminator());
}


const SDetachInst *llvm::getEpilogSDetach(const Loop &Epilog) {
  if (!findStringMetadataForLoop(&Epilog, SwarmCoarsened::Epilog))
    return nullptr;
  return cast<const SDetachInst>(Epilog.getLoopPreheader()
                                       ->getUniquePredecessor()
                                       ->getTerminator());
}
