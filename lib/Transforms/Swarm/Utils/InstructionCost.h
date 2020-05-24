//===-- InstructionCost.h - Cost modeling based on LLVM instructions ------===//
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

#ifndef LLVM_TRANSFORMS_SWARM_INSTRUCTIONCOST_H
#define LLVM_TRANSFORMS_SWARM_INSTRUCTIONCOST_H

namespace llvm {

class Instruction;
class Loop;
class TargetTransformInfo;

/// Returns an integer denoting the amount of work to execute the instruction I.
unsigned getCost(const Instruction *I, const TargetTransformInfo &TTI);

/// Returns an integer denoting the amount of work to execute one iteration
/// of loop L.
unsigned getLoopBodyCost(const Loop &L, const TargetTransformInfo &TTI);

}

#endif
