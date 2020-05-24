//===-- InstructionCost.cpp - Cost modeling based on LLVM instructions ----===//
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
// Simple static cost modeling that estimates the amount of work.
//
//===----------------------------------------------------------------------===//

#include "InstructionCost.h"

#include "llvm/Analysis/CostModel.h"
#include "llvm/Analysis/LoopInfo.h"

using namespace llvm;

#define DEBUG_TYPE "cost-model"


unsigned llvm::getCost(const Instruction *I, const TargetTransformInfo &TTI) {
  // Function calls are potentially large, so they are worth parallelizing.
  if (isa<CallInst>(I) && !isa<IntrinsicInst>(I))
    return -1U;

  // llvm::getInstrCost() doesn't know about these.
  // Let's treat detaches as infinite cost for now, to avoid skipping them.
  if (isa<SDetachInst>(I) || isa<SReattachInst>(I))
    return -1U;

  // In our version of LLVM, getInstrCost() crashes on
  // llvm.*.with.overflow.*() intrinsics because they return a struct.
  if (isa<IntrinsicInst>(I) && I->getType()->isStructTy())
    return 2;

  return llvm::getInstrCost(I, &TTI);
}


unsigned llvm::getLoopBodyCost(const Loop &L, const TargetTransformInfo &TTI) {
  //TODO(victory): Summing up the cost across all conditional branches within
  // the loop doesn't make the most sense. We should instead sum along each
  // possible control flow path from the header to the latch.
  unsigned TotalCost = 0;
  for (const BasicBlock *BB : L.blocks())
    for (const Instruction &I : *BB) {
      unsigned InstrCost = getCost(&I, TTI);
      DEBUG(dbgs() << "Cost " << InstrCost << " for " << I << "\n");
      if (InstrCost == -1U) return -1U;
      TotalCost += InstrCost;
      assert(TotalCost >= InstrCost && "32-bit overflow?");
    }
  return TotalCost;
}
