//===-- Flags.h - Swarm Transformations Markers -----------------*- C++ -*-===//
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
// Tags used to record information about code processed by Swarm passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SWARM_FLAGS_H
#define LLVM_TRANSFORMS_SWARM_FLAGS_H

#include "llvm/IR/Function.h"
#include "llvm/Transforms/Swarm.h"

namespace llvm {

struct SwarmFlag {
  // Function attributes
  static constexpr const char* Parallelizable = "SwarmParallelizable";
  static constexpr const char* Parallelizing = "SwarmParallelizing";
  static constexpr const char* Parallelized = "SwarmParallelized";

  // Parameter attributes
  static constexpr const char* Continuation = "SwarmContinuation";

  // Metadata for...

  // Function calls
  static constexpr const char* ParallelCall = "SwarmParallelCall";

  // Detaches
  // TODO(victory): These flags affect detach semantics and maybe should be
  // carried in the detach instruction itself (similar to hint- and
  // domain-related flags) instead of attached as metadata.
  static constexpr const char* Coarsenable = "SwarmCoarsenable";
  static constexpr const char* MustSpawnLatch = "SwarmLoopMustSpawnLatch";
  static constexpr const char* TempNullDomain = "SwarmTempNullDomain";

  // Allocations and pointers
  static constexpr const char* Closure = "SwarmClosure";
  static constexpr const char* DoneFlag = "SwarmDoneFlag";

  // Loops
  static constexpr const char* LoopUnprofitable = "SwarmUnprofitableLoop";
  static constexpr const char* LoopSubsumedCont = "SwarmLoopSubsumedCont";
};


/// Return true if F does not simulataneously have incongruous SwarmFlags
inline bool hasValidSwarmFlags(const Function &F) {
  if (F.hasFnAttribute(SwarmFlag::Parallelizable)) {
    return !F.hasFnAttribute(SwarmAttr::NoSwarmify) &&
           !F.hasFnAttribute(SwarmFlag::Parallelizing) &&
           !F.hasFnAttribute(SwarmFlag::Parallelized);
  } else if (F.hasFnAttribute(SwarmFlag::Parallelizing)) {
    return !F.hasFnAttribute(SwarmFlag::Parallelized);
  } else if (F.hasFnAttribute(SwarmFlag::Parallelized)) {
    // Fractalizer should not be able to panic if it
    // re-encounters some outlined portion of F
    return !F.hasFnAttribute(SwarmAttr::AssertSwarmified);
  } else {
    return true;
  }
}


} // End llvm namespace

#endif
