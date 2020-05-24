//===------ Profitability.cpp - Cost and profitability analysis -----------===//
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
// Use information about the structure and size of code regions (e.g., loops)
// to drive decisions on whether or how to parallelize the code.
//
//===----------------------------------------------------------------------===//

#include "Utils/Flags.h"
#include "Utils/InstructionCost.h"
#include "Utils/Misc.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationDiagnosticInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Swarm.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;

#define PASS_NAME "profitability"
#define DEBUG_TYPE PASS_NAME

static cl::opt<unsigned> ShortLoopSerializingThreshold(
    "swarm-shortloopserializingthreshold", cl::init(1000),
    cl::desc("Cost of loop execution below which loops will be serialized."));

namespace {

class ProfitabilityAnalyzer {
private:
  Function &F;
  DominatorTree &DT;
  LoopInfo &LI;
  ScalarEvolution &SE;
  TargetTransformInfo &TTI;
  OptimizationRemarkEmitter &ORE;
  Module &M;
  LLVMContext &Context;

  unsigned LoopIDCounter = 0;

public:
  ProfitabilityAnalyzer(
          Function &F,
          DominatorTree &DT, LoopInfo &LI, ScalarEvolution &SE,
          TargetTransformInfo &TTI,
          OptimizationRemarkEmitter &ORE)
      : F(F), DT(DT), LI(LI), SE(SE), TTI(TTI), ORE(ORE),
        M(*F.getParent()), Context(F.getContext()) { }

  bool run();

private:
  bool analyzeLoop(Loop &L);
};

// This class undoes some serial optimization that reduces the number of loads
// by carrying values loaded from memory across loop iterations in registers.
//
// Consider serial code that starts out like this C-like pseudocode:
//
//     i = 0;
//   loop:
//     foo(some_pointer->limit);
//     if (some_condition(i)) {
//       bar(some_pointer);
//     }
//     ++i;
//     if (i < some_pointer->limit) goto loop;
//
// Suppose foo() is known not to have any side effects on some_pointer->limit,
// but the compiler must assume that some_pointer->limit's value could change
// on each call to bar(). Some standard LLVM optimization (mem2reg?) would try
// to minimize the number of loads from memory by producing optimized code
// like:
//
//     %limit = %some_pointer->limit;
//     %i = 0;
//   loop:
//     foo(%limit);
//     if (some_condition(%i)) {
//       bar(%some_pointer);
//       %limit = %some_pointer->limit;
//     }
//     ++%i;
//     if (%i < %limit) goto loop;
//
// That is, on iterations of the loop where do_something() is NOT called, LLVM
// will generate code that does NOT reload the value from %some_pointer->limit
// into the register %limit, since on those iterations the value couldn't have
// changed. But this complex bookkeeping gets in the way of parallelization:
// it creates an explicit dependency between each iteration and the next
// since the value of %limit may be carried across iterations in a register.
// So we transform the code to look more like:
//
//     %i = 0;
//   loop:
//     %limit = %some_pointer->limit;
//     foo(%limit);
//     if (some_condition(%i)) {
//       bar(%some_pointer);
//     }
//     %limit = %some_pointer->limit;
//     ++%i;
//     if (%i < %limit) goto loop;
//
// That is, unconditionally loading the value of %some_pointer->limit in the
// loop header and latch.
//TODO(victory): Think harder about whether this is just a specific case of a
// broader class of transformations we may want to do to break explicit
// dependencies by duplicating loads.  Certainly we don't want to just run the
// reg2mem pass and hide all dependencies in memory, as most of the time
// respecting explicit dependencies is useful in avoiding overspeculation.
class LoopCarriedDependenciesBreaker {
  Loop *const L;
  BasicBlock *const Predecessor;
  BasicBlock *const Header;
  BasicBlock *const Latch;
public:
  LoopCarriedDependenciesBreaker(Loop *L)
    : L(L)
    , Predecessor(L->getLoopPredecessor())
    , Header(L->getHeader())
    , Latch(L->getLoopLatch())
  {
    assert(Predecessor &&
           "We require the header have a unique predecessor outside the loop");
    assert(Latch && "We depend on the loop having a single backedge");
  }
  void run();
private:
  void runOnHeaderPhi(PHINode *HeaderPhi);
};

} // end anonymous namespace


// See comment on LoopCarriedDependenciesBreaker class definition.
void LoopCarriedDependenciesBreaker::run() {
  SmallPtrSet<const BasicBlock *, 2> HeaderPreds(pred_begin(Header),
                                                 pred_end(Header));
  assert(HeaderPreds.size() == 2);
  assert(HeaderPreds.count(Predecessor) && HeaderPreds.count(Latch));

  Instruction *I = &Header->front();
  while (auto *PN = dyn_cast<PHINode>(I)) {
    I = I->getNextNode();  // avoid pointer invalidation
    runOnHeaderPhi(PN);
  }
}


// Check if the PN's value is loaded from memory if the control-flow edge
// from Pred to PN's parent is taken, and check if, when that edge is taken,
// if PN's value is still the current value in memory at the address from
// which the value was loaded. If we can guarantee this, return the address.
// Otherwise, return null.
static Value *getLoadPtrForPhiPredecessor(const PHINode *PN,
                                          const BasicBlock *Pred) {
  Value *PNVal = PN->getIncomingValueForBlock(Pred);
  auto *Load = dyn_cast<LoadInst>(PNVal);
  if (!Load) {
    DEBUG(dbgs() << "  Phi takes value from non-load: " << *PNVal << '\n');
    return nullptr;
  }
  DEBUG(dbgs() << "  Phi takes value from associated load in "
               << Load->getParent()->getName() << ": " << *Load << '\n');
  if (!Load->isUnordered()) {
    DEBUG(dbgs() << "  Load is ordered (volatile or atomic).\n");
    return nullptr;
  }
  Value *LoadPtr = Load->getPointerOperand();
  // Check if the phi's value is still the current value in memory
  // at LoadPtr when the edge from Pred to PN is taken.
  if (!isInvariantInPath(LoadPtr, Load, Pred->getTerminator())) {
    DEBUG(dbgs() << "  Load followed by intervening store?\n");
    return nullptr;
  }
  return LoadPtr;
}


static LoadInst *replacePhiWithLoad(PHINode *PN, Value *LoadPtr,
                                    Instruction *InsertBefore = nullptr) {
  if (!InsertBefore) InsertBefore = PN->getParent()->getFirstNonPHI();
  assert(InsertBefore->getParent() == PN->getParent());

  auto *NewLoad =
      new LoadInst(LoadPtr,
                   LoadPtr->getName() + ".copyval",
                   InsertBefore);
  NewLoad->setDebugLoc(PN->getDebugLoc());
  PN->replaceAllUsesWith(NewLoad);
  DEBUG(dbgs() << "  In block " << PN->getParent()->getName()
               << ", replaced " << *PN
               << "\n  with: " << *NewLoad << '\n');
  PN->eraseFromParent();
  return NewLoad;
}


// Try to break the loop-carried dependency of HeaderPhi (the value that
// HeaderPhi carries along the backedge of the loop) by replacing HeaderPhi
// with re-loading its value from memory.
void LoopCarriedDependenciesBreaker::runOnHeaderPhi(PHINode *HeaderPhi) {
  assert(HeaderPhi->getParent() == Header);

  DEBUG(dbgs() << "Examining header phi: " << *HeaderPhi << '\n');

  if (HeaderPhi == L->getCanonicalInductionVariable()) {
    // For DEBUG clarity, we're not concerned about breaking the canonical IV.
    // It would be caught by the following if (!LoadPtr) condition anyway.
    DEBUG(dbgs() << "  This is the canonical IV.\n");
    return;
  }

  // Check if the phi's value is loaded from memory for the first iteration.
  Value *LoadPtr = getLoadPtrForPhiPredecessor(HeaderPhi, Predecessor);
  if (!LoadPtr) {
    DEBUG(dbgs() << "  Header phi does not take initial value from memory.\n");
    return;
  }
  assert(L->isLoopInvariant(LoadPtr));

  // Check if the phi's value is loaded from memory for subsequent iterations.
  if (Value *BodyLoadPtr = getLoadPtrForPhiPredecessor(HeaderPhi, Latch)) {
    // We only need to add a load to the header.
    if (!L->contains(cast<LoadInst>(
            HeaderPhi->getIncomingValueForBlock(Latch)))) {
      // TODO(mcj) Reason about whether we can handle this case using the code
      // below. i.e. does it matter that the latch load is not contained in the
      // loop? This early exit is replacing an earlier assertion. Maybe it was
      // too strict?
      DEBUG(dbgs() << "  Header phi's latch load is not in the loop\n");
    } else if (BodyLoadPtr == LoadPtr) {
      // The address to load is loop invariant
      replacePhiWithLoad(HeaderPhi, LoadPtr);
    } else if (auto *LatchGEP = dyn_cast<GetElementPtrInst>(BodyLoadPtr)) {
      DEBUG(dbgs() << "  Examining address computation for body load: "
                   << *BodyLoadPtr << '\n');
      auto *PredecessorGEP = dyn_cast<GetElementPtrInst>(LoadPtr);
      if (LatchGEP->getPointerOperand() != LoadPtr &&
          (!PredecessorGEP ||
           PredecessorGEP->getPointerOperand() != LatchGEP->getPointerOperand())
          ) {
        DEBUG(dbgs() << "  Load in body doesn't use the same base address as "
                        "the predecessor load\n");
      } else if (LatchGEP->getNumIndices() != 1) {
        DEBUG(dbgs() << "  Load in body doesn't use one index.\n");
      } else if (PredecessorGEP && PredecessorGEP->getNumIndices() != 1) {
        DEBUG(dbgs() << "  Load in predecessor doesn't use one index.\n");
      } else if (PredecessorGEP &&
                 PredecessorGEP->getOperand(1)->getType() !=
                 LatchGEP->getOperand(1)->getType()) {
        DEBUG(dbgs() << "  Load in predecessor and body don't have matching "
                        "index types.\n");
      } else {
        // The predecessor load's base address is equal to the body load's base
        // address. Both addresses to load are computed with a simple index.
        // We construct a Phi to hold the index:
        // * If entering from the latch, just use the body's load index value.
        // * On the first iteration, use the predecessor's index
        //   (this will be zero if the predecessor has a simple load)
        // We then hope the new Phi will be strengthened away in LoopExpansion
        Value *BodyIndex = LatchGEP->getOperand(1);
        Value *PredecessorIndex = PredecessorGEP
                ? PredecessorGEP->getOperand(1)
                : Constant::getNullValue(BodyIndex->getType());
        IRBuilder<> Builder(HeaderPhi);
        auto *NewIndex = Builder.CreatePHI(BodyIndex->getType(), 2,
                                           BodyIndex->getName() + ".index");
        NewIndex->addIncoming(BodyIndex, Latch);
        NewIndex->addIncoming(PredecessorIndex, Predecessor);
        Builder.SetInsertPoint(&*Header->getFirstInsertionPt());
        Value *HeaderGEP = Builder.CreateGEP(LatchGEP->getPointerOperand(),
                                             NewIndex,
                                             LatchGEP->getName() + ".dupe");
        replacePhiWithLoad(HeaderPhi, HeaderGEP,
                           cast<Instruction>(HeaderGEP)->getNextNode());
      }
    } else {
      DEBUG(dbgs() << "  Load in body uses different addresses: "
                   << *BodyLoadPtr << '\n');
    }
    return;
  }

  // Check if the phi's value comes from a latch phi on subsequent iterations.
  if (Header == Latch) {
    DEBUG(dbgs() << "  Giving up on phi in single-block loop.\n");
    return;
  }
  Value *BackedgeVal = HeaderPhi->getIncomingValueForBlock(Latch);
  auto *LatchPhi = dyn_cast<PHINode>(BackedgeVal);
  if (!LatchPhi || LatchPhi->getParent() != Latch) {
    DEBUG(dbgs() << "  Header phi does not directly take value of latch phi.\n");
    return;
  }
  DEBUG(dbgs() << "  Associated latch phi: " << *LatchPhi << '\n');
  // Assuming the latch phi always holds the then-current value at LoadPtr,
  // check if the header phi's value also still holds the current value
  // at LoadPtr on a subsequent iteration of the loop.
  if (!isInvariantInPath(LoadPtr,
                         &Latch->front(), Latch->getTerminator())) {
    DEBUG(dbgs() << "  Intervening store in latch?\n");
    return;
  }

  // All that remains is to check if the latch phi always holds the current
  // value at LoadPtr.
  for (unsigned i = 0; i < LatchPhi->getNumIncomingValues(); ++i) {
    Value *V = LatchPhi->getIncomingValue(i);
    const BasicBlock *LatchIncomingBlock = LatchPhi->getIncomingBlock(i);

    if (V == HeaderPhi) {
      // Assuming the header phi always holds the then-current value at LoadPtr,
      // check if the latch phi still holds the current value at LoadPtr if,
      // on any iteration, the latch phi takes the value from the header phi.
      if (!isInvariantInPath(LoadPtr,
                             &Header->front(),
                             LatchIncomingBlock->getTerminator())) {
        DEBUG(dbgs() << "  Load followed by intervening store "
                     << "in path from header to latch?\n");
        return;
      }
    } else if (auto *BodyLoadPtr =
                  getLoadPtrForPhiPredecessor(LatchPhi, LatchIncomingBlock)) {
      assert(L->contains(cast<LoadInst>(V)));
      if (BodyLoadPtr != LoadPtr) {
        DEBUG(dbgs() << "  Load in body uses different addresses: "
                     << *BodyLoadPtr << '\n');
        return;
      }
    } else {
      DEBUG(dbgs() << "  Latch phi takes value not from header phi or load in body: "
                   << *V << '\n');
      return;
    }
  }

  DEBUG(dbgs() << "  Phi value always equals the value in memory "
               << "at loop-invariant address: " << *LoadPtr << '\n');

  // All preflight checks complete, time to make the transformation.

  replacePhiWithLoad(HeaderPhi, LoadPtr);
  replacePhiWithLoad(LatchPhi, LoadPtr);
}


static void breakLoopOutputDependencies(Loop &L, DominatorTree &DT) {
  assert(L.isLCSSAForm(DT));
  BasicBlock *ExitBlock = L.getExitBlock();
  BasicBlock *ExitingBlock = L.getExitingBlock();
  if (!ExitBlock || !ExitingBlock) return;
  DEBUG(dbgs() << "Examining outputs of loop that exits from "
               << ExitingBlock->getName() << " to " << ExitBlock->getName()
               << "\n");

  Instruction *I = &ExitBlock->front();
  while (auto *PN = dyn_cast<PHINode>(I)) {
    I = I->getNextNode();  // avoid pointer invalidation

    DEBUG(dbgs() << "Examining phi: " << *PN << "\n");
    if (Value *Ptr = getLoadPtrForPhiPredecessor(PN, ExitingBlock)) {
      DEBUG(dbgs() << "  Replacing phi with load from: " << *Ptr << "\n");
      replacePhiWithLoad(PN, Ptr);
    }
  }
}


bool ProfitabilityAnalyzer::analyzeLoop(Loop &L) {
  const unsigned LoopID = LoopIDCounter++;
  DEBUG(dbgs() << "\nStarting analysis for loop " << LoopID
               << " in " << F.getName() << "\n  starting at ");
  DEBUG(L.getStartLoc().print(dbgs()));
  DEBUG(dbgs() << '\n' << L);

  const SCEV *LimitSCEV = SE.getBackedgeTakenCount(&L);
  DEBUG(dbgs() << "Backedge taken count: " << *LimitSCEV << "\n");

  const unsigned InstructionCost = getLoopBodyCost(L, TTI);
  DEBUG(dbgs() << "Cost of instructions in loop: " << InstructionCost << "\n");

  uint64_t AvgIterations = 0;

  // Remove dependences on SSA values by inserting more loads.
  LoopCarriedDependenciesBreaker(&L).run();
  breakLoopOutputDependencies(L, DT);

  bool HighCost = false;
  for (Loop *SubLoop : L)
    HighCost |= analyzeLoop(*SubLoop);

  DEBUG(dbgs() << "\nFinishing analysis for loop " << LoopID
               << " in " << F.getName() << "\n");

  if (auto ConstLimit = dyn_cast<SCEVConstant>(LimitSCEV))
    AvgIterations = ConstLimit->getValue()->getLimitedValue() + 1;
  if (!AvgIterations) { // # of iterations in loop unknown
    HighCost = true;
  } else {
    uint64_t TotalCost = InstructionCost * AvgIterations;
    assert(AvgIterations == 0 || TotalCost / AvgIterations == InstructionCost
           && "64-bit overflow shouldn't happen here");
    DEBUG(dbgs() << "Loop has total execution cost: " << TotalCost << "\n");
    if (TotalCost > ShortLoopSerializingThreshold)
      HighCost = true;
  }

  if (!HighCost) {
    ORE.emit(OptimizationRemark(PASS_NAME, "SerializingShortLoop",
                                L.getStartLoc(), L.getHeader())
             << "Serializing short loop.");
    addStringMetadataToLoop(&L, SwarmFlag::LoopUnprofitable);
  }

  return HighCost;
}


bool ProfitabilityAnalyzer::run() {
  for (Loop *L : LI) analyzeLoop(*L);

  bool Changed = false;
  return Changed;
}


class Profitability : public FunctionPass {
public:
  static char ID;

  Profitability() : FunctionPass(ID) {
    initializeProfitabilityPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    if (!F.hasFnAttribute(SwarmFlag::Parallelizable))
      return false;

    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    auto &ORE = getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

    return ProfitabilityAnalyzer(F, DT, LI, SE, TTI, ORE).run();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredID(LoopSimplifyID);
    AU.addRequiredID(LCSSAID);
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
    AU.setPreservesCFG();
  }
};

char Profitability::ID = 0;

INITIALIZE_PASS_BEGIN(Profitability, DEBUG_TYPE,
                      "Cost and profitability analysis",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LCSSAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_END(Profitability, DEBUG_TYPE,
                    "Cost and profitability analysis",
                    false, false)

Pass *llvm::createProfitabilityPass() {
  return new Profitability();
}
