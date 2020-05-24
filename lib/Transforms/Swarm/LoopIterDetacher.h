//===- LoopIterDetacher.h - detach loop iterations as tasks ---------------===//
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
// The interface by which Fractalizer can access LoopIterDetacher.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationDiagnosticInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"

namespace llvm {

class DeepenInst;

// Detach each iteration of the loop body as a separate task,
// attempting to produce a "canonical" Swarm loop.
// \return the start of the loop's continuation on success, or null otherwise.
BasicBlock *detachLoopIterations(
        Loop &L, DeepenInst *OuterDomain, BasicBlock *ContinuationEnd,
        bool ProceedAtAllCosts,
        AssumptionCache &AC, DominatorTree &DT, LoopInfo &LI,
        TargetLibraryInfo &TLI, TargetTransformInfo &TTI,
        OptimizationRemarkEmitter &ORE);

} // namespace llvm
