//===- LowerToSwarm.cpp - Convert Tapir into Swarm instructions -----------===//
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
// This pass lowers (i.e., translates) Tapir-like tasks and related intrinsics
// to standard LLVM IR constructs (e.g., inline assembly or runtime calls) that
// can be processed by the target machine code generation machinery.  This
// includes task lifting: the conversion of functions that include nested task
// code into separate functions for each task, and the generation of explicit
// closures to capture live-in values (eliminate free variables) for each task.
//
//===----------------------------------------------------------------------===//

#include "Utils/CFGRegions.h"
#include "Utils/Flags.h"
#include "Utils/Misc.h"
#include "Utils/SwarmABI.h"
#include "Utils/Tasks.h"

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Swarm.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "lower2swarm"

using namespace llvm;

static cl::opt<std::string> DebugUnloweredFuncOpt("swarm-debugunloweredfunc",
        cl::init(""),
        cl::desc("Name of function for which IR will be displayed before lowering."));

namespace {

struct LowerTapirToSwarm : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  LowerTapirToSwarm()
      : ModulePass(ID) {
  }
  StringRef getPassName() const override {
    return "Simple Lowering of Tapir to Swarm ABI";
  }

  // bool runOnFunction(Function &F) override;

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }
private:
  bool processFunction(Function *F);
};
}  // End of anonymous namespace

char LowerTapirToSwarm::ID = 0;
INITIALIZE_PASS_BEGIN(LowerTapirToSwarm, "lower2swarm",
                      "Simple Lowering of Tapir to Swarm ABI",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(LowerTapirToSwarm, "lower2swarm",
                    "Simple Lowering of Tapir to Swarm ABI",
                    false, false)


// Build the absolute timestamp value for the child task.
static void absolutizeTimestamp(SDetachInst *DI) {
  Value *Timestamp = DI->getTimestamp();
  assert(Timestamp && "sdetach without timestamp?");

  IntegerType *Int64Ty = Type::getInt64Ty(DI->getContext());

  // If necessary, build the current app-level Timestamp in the target domain
  Value *CurrentTimestamp =
          !DI->isRelativeTimestamp() ? nullptr
          : DI->isSubdomain() ? cast<Value>(ConstantInt::get(Int64Ty, 0))
          : cast<Value>(swarm_abi::createGetTimestampInst(DI,
                                                          DI->isSuperdomain()));

  IRBuilder<> Builder(DI);
  if (Timestamp->getType() != Int64Ty)
    Timestamp = Builder.CreateZExtOrTrunc(Timestamp, Int64Ty, "timestamp");
  if (DI->isRelativeTimestamp())
    Timestamp = Builder.CreateAdd(CurrentTimestamp, Timestamp, "abstimestamp");

  DI->setTimestamp(Timestamp);
  DI->setRelativeTimestamp(false);
}


bool LowerTapirToSwarm::processFunction(Function *F) {
  bool Changed = false;

  DEBUG(dbgs() << "\n\nLowering function " << F->getName() << "()\n");
  if (F->getName().equals(DebugUnloweredFuncOpt)) {
    dbgs() << "\n\n Function before lowering:\n" << *F << "\n\n";
    F->viewCFG();
  }

  const bool hasDetachesToLower = llvm::hasAnySDetachInst(*F);
  DominatorTree *const DT =
          hasDetachesToLower
          ? &getAnalysis<DominatorTreeWrapperPass>(*F).getDomTree()
          : nullptr;
  DEBUG(assertVerifyFunction(*F, "Before lowering", DT));

  SmallVector<Function *, 8> ResultingFunctions;

  if (hasDetachesToLower) {
    // Lower Tapir-like detaches and reattaches in this function.
    // Traverse the DT in post-order to ensure inner detaches are outlined first.
    // Since lowering may alter the DT, we need to grab the blocks first.
    SmallVector<BasicBlock *, 8> PreOrderBlocks;
    for (DomTreeNode *Node : depth_first(DT))
      PreOrderBlocks.push_back(Node->getBlock());
    const auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(*F);
    for (BasicBlock *BB : reverse(PreOrderBlocks)) {
      if (auto *DI = dyn_cast<SDetachInst>(BB->getTerminator())) {
        DEBUG(dbgs() << "\nProcessing detach: " << *DI << '\n');
        const DebugLoc DL = DI->getDebugLoc();
        DEBUG(if (DL) {
          dbgs() << "  from ";
          DL.print(dbgs());
          dbgs() << '\n';
        });
        assert(!DI->getMetadata(SwarmFlag::MustSpawnLatch));
        assert(getDetachKind(DI) != DetachKind::SubsumedCont
               || DI->isSuperdomain());
        assert(!DI->getMetadata(SwarmFlag::TempNullDomain));

        if (!DI->hasTimestamp()) {
          DEBUG(dbgs() << "Marking untimestamped detach unreachable.\n");
          assert(!F->hasFnAttribute(SwarmFlag::Parallelized) &&
                 "Fractalization should have inserted timestamps");
          // Ensures a crash if the detach would have been reached at runtime.
          eraseDetach(DI, *DT);
          auto *PrevTerm = cast<BranchInst>(BB->getTerminator());
          assert(PrevTerm->isUnconditional());
          BasicBlock *Continue = PrevTerm->getSuccessor(0);
          auto *Unreachable = new UnreachableInst(F->getContext());
          ReplaceInstWithInst(PrevTerm, Unreachable);
          if (DT->dominates(BB, Continue))
            eraseDominatorSubtree(Continue, *DT);
          // TODO(victory): Do an incremental update of the dominator tree
          // instead of throwing it out and recalculating it.
          DT->recalculate(*F);

          // Print out a message before crashing.
          createPrintString("\nReached code that LowerToSwarm deleted"
                            " because a detach lacks a timestamp.\n",
                            "swarmabi_notimestamp_detach", Unreachable);

          DEBUG(
              assertVerifyFunction(*F, "After marking detach unreachable", DT));

          continue;
        }

        absolutizeTimestamp(DI);
        Function *OutlinedTask = swarm_abi::lowerDetach(*DI, *DT, TTI);
        if (OutlinedTask) {
          ResultingFunctions.push_back(OutlinedTask);
        }
      }
    }
    Changed = true;
  }

  ResultingFunctions.push_back(F);

  for (Function *ResultingFunction : ResultingFunctions) {
    // Lower deepen and undeepen intrinsics
    SmallVector<DeepenInst *, 8> DeepenIntrinsics;
    auto II = inst_begin(ResultingFunction), E = inst_end(ResultingFunction);
    while (II != E) {
      Instruction *I = &*II++; // To avoid iterator invalidation
      if (auto *DeepenIntrinsic = dyn_cast<DeepenInst>(I)) {
        swarm_abi::createDeepenInst(DeepenIntrinsic);
        assert(all_of(DeepenIntrinsic->users(), [](const User *U) {
                        return isa<UndeepenInst>(U); })
               && "Domain token passed as argument or captured in a closure?");
        // We can't erase DeepenIntrinsic while it still has users,
        // collect it to be erased after all the users are lowered.
        DeepenIntrinsics.push_back(DeepenIntrinsic);
        Changed = true;
      } else if (auto *UndeepenIntrinsic = dyn_cast<UndeepenInst>(I)) {
        swarm_abi::createUndeepenInst(UndeepenIntrinsic);
        assert(isa<DeepenInst>(UndeepenIntrinsic->getArgOperand(0))
               && "Undeepen receives token parameter or closure load?");
        UndeepenIntrinsic->eraseFromParent();
        Changed = true;
      }
    }
    // Now that the user undeepens are gone, we can erase the deepen intrinsics.
    for (DeepenInst *DeepenIntrinsic : DeepenIntrinsics) {
      assert(DeepenIntrinsic->use_empty());
      DeepenIntrinsic->eraseFromParent();
    }

    // If Swarm passes are later re-run, do nothing more with this function.
    ResultingFunction->addFnAttr(SwarmFlag::Parallelized);

    if (ResultingFunction == F) {
      assertVerifyFunction(*F, "After lowering", DT);
    } else {
      assertVerifyFunction(*ResultingFunction, "After lowering");
    }
  }

  DEBUG(dbgs() << "\nDone lowering function " << F->getName() << "()\n\n");

  return Changed;
}

bool LowerTapirToSwarm::runOnModule(Module &M) {
  if (skipModule(M))
    return false;

  // Find functions that detach for processing.
  SmallVector<Function*, 4> OriginalFunctions;
  for (Function &F : M)
    OriginalFunctions.push_back(&F);

  bool Changed = false;
  for (Function* F : OriginalFunctions)
    Changed |= processFunction(F);

  return Changed;
}

// createLowerTapirToSwarmPass - Provide an entry point to create this pass.
//
ModulePass *llvm::createLowerTapirToSwarmPass() {
  return new LowerTapirToSwarm();
}
