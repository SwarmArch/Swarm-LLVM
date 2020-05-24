//===-- Swarm.h - Swarm Transformations -------------------------*- C++ -*-===//
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
// This header file defines prototypes for accessor functions that expose passes
// in the Swarm transformations library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SWARM_H
#define LLVM_TRANSFORMS_SWARM_H

#include "llvm/Support/CommandLine.h"

namespace llvm {

class Pass;
class ModulePass;

// These annotations affect the behavior of Swarm passes
struct SwarmAttr {
  static constexpr const char* NoSwarmify = "NoSwarmify";
  static constexpr const char* Swarmify = "Swarmify";
  static constexpr const char* AssertSwarmified = "AssertSwarmified";
};

extern cl::opt<unsigned> SwarmRegistersTransferred;
extern cl::opt<uint64_t> SwarmCacheLineSize;

Pass *createBundlingPass();
ModulePass *createCreateParallelizableCopyPass();
Pass *createFractalizationPass();
Pass *createInsertHeartbeatsPass();
Pass *createLoopCoarsenPass();
Pass *createLoopExpansionPass();
ModulePass *createLowerTapirToSwarmPass();
Pass *createProfitabilityPass();
Pass *createSpatialHintsPass();

} // End llvm namespace

#endif
