//===-- Reductions.h - Swarm Reductions -------------------------*- C++ -*-===//
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
// This header file defines prototypes for Swarm runtime-assisted reductions
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SWARM_REDUCTIONS_H
#define LLVM_TRANSFORMS_SWARM_REDUCTIONS_H

namespace llvm {


class DominatorTree;
class Loop;
class OptimizationRemarkEmitter;
class PHINode;


/// If Output is the result of a (presently supported) reduction in loop L,
/// replace key instructions in L's preheader, header, body, and exit with calls
/// to the __sccrt_reduction* functions.
/// \return true iff the Output was successfully replaced
bool replaceLoopOutputWithReductionCalls(Loop &L, PHINode *Output,
                                         const DominatorTree &DT,
                                         OptimizationRemarkEmitter &ORE);


/// \return true iff L performs a reduction via __sccrt_reduction* calls
bool hasReductionCalls(const Loop &L);


/// If L's body calls an __sccrt_reduction* update function, this function
/// pushes that call to the exit block of L, and replaces it with the
/// appropriate reduction instructions. The function assumes L has no detaches,
/// nor reattaches, it is intended to be used on coarsened loops.
void moveReductionCallsAfterLoop(Loop &L);


}  // end of llvm namespace


#endif //LLVM_TRANSFORMS_SWARM_REDUCTIONS_H
