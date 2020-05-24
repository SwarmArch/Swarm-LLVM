//===- InsertHeartbeats.cpp - Instrumentation to track application work ---===//
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
// Instrument the tops of functions if they contain any loops with a heartbeat
// instruction, which can be used by the simulator to track the amount of
// work that has been done by the application.
//
//===----------------------------------------------------------------------===//

#include "Utils/SwarmABI.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Swarm.h"

using namespace llvm;

#define IH_NAME "insert-heartbeats"
#define DEBUG_TYPE IH_NAME

#define INSERTED_HEARTBEAT_ATTR "SwarmHeartbeatInserted"

STATISTIC(FunctionHeartbeats,
          "Number of functions instrumented with a top-of-function heartbeat");

static cl::opt<bool> InsertHeartbeatsOpt("swarm-insertheartbeats",
        cl::init(false),
        cl::desc("Minimum number of basic blocks duplicated that will "
                 "trigger eager task outlining during parallelization"));


namespace {
struct InsertHeartbeats : public FunctionPass {
  /// Pass identification, replacement for typeid
  static char ID;

  explicit InsertHeartbeats() : FunctionPass(ID) {
    initializeInsertHeartbeatsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    if (!InsertHeartbeatsOpt)
      return false;

    // Prevent reprocessing of functions or their outlined parts
    if (F.hasFnAttribute(INSERTED_HEARTBEAT_ATTR))
      return false;
    F.addFnAttr(INSERTED_HEARTBEAT_ATTR);

    LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    if (LI.empty())
      return false;

    // Avoid adding heartbeats to runtime functions that may be different
    // between the SCC and Comptetition runtimes.
    //TODO(victory): This could be made more efficient by making this a pass a
    // CallGraphSCCPass
    if (F.getName().startswith("_ZN5swarm") ||
        F.getName().startswith("_ZN3pls") ||
        (F.getName().startswith("_ZN3scc") &&
         !F.getName().startswith("_ZN3scc17callROILambdaFunc"))) {
      for (const Instruction &I : instructions(&F))
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (Function *Callee = CI->getCalledFunction())
            if (!Callee->isDeclaration()
                && !Callee->getName().startswith("_ZN5swarm")
                && !Callee->getName().startswith("_ZN3pls")
                && !Callee->getName().startswith("_ZN3scc")) {
              std::string Msg;
              raw_string_ostream OS(Msg);
              OS << "Call from runtime function "
                 << F.getName() << "() may affect heartbeats\n";
              F.getContext().diagnose(
                  DiagnosticInfoUnsupported(F, OS.str(), CI->getDebugLoc()));
            }
      return false;
    }

    DEBUG(dbgs() << "Inserting heartbeat at top of function: "
                 << F.getName() << "()\n");
    swarm_abi::createHeartbeatInst(&F.getEntryBlock().front());
    FunctionHeartbeats++;
    return true;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.setPreservesCFG();
  }
};
} // anonymous namespace

char InsertHeartbeats::ID = 0;

INITIALIZE_PASS_BEGIN(InsertHeartbeats, IH_NAME,
                      "Instrumentation to track work done by the application",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(InsertHeartbeats, IH_NAME,
                    "Instrumentation to track work done by the application",
                    false, false)

namespace llvm {
Pass *createInsertHeartbeatsPass() {
  return new InsertHeartbeats();
}
}
