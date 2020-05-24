//===- LoopCoarsen.h -----------------------------------------------C++ -*-===//
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
// Internal interfaces used by LoopExpansion to find the components of loops
// processed by LoopCoarsener.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SWARM_LOOPCOARSEN_H
#define LLVM_TRANSFORMS_SWARM_LOOPCOARSEN_H

namespace llvm {

class Loop;
class LoopInfo;
class SDetachInst;

namespace SwarmCoarsened {
static const constexpr char *OuterLoop = "SwarmCoarsenedOuterLoop";
static const constexpr char *InnerLoop = "SwarmCoarsenedInnerLoop";
static const constexpr char *Prolog = "SwarmCoarsenedPrologLoop";
static const constexpr char *Epilog = "SwarmCoarsenedEpilogLoop";
}

/// If CoarsenedOuterLoop is a Swarm loop that has been automatically coarsened,
/// and if it has an prolog generated during coarsening, return the prolog loop.
const Loop *getProlog(const Loop &CoarsenedOuterLoop, const LoopInfo &LI);
inline Loop *getProlog(Loop &CoarsenedOuterLoop, LoopInfo &LI) {
  return const_cast<Loop *>(
          getProlog(static_cast<const Loop &>(CoarsenedOuterLoop),
                    static_cast<const LoopInfo &>(LI)));
}

const Loop *getEpilog(const Loop &CoarsenedOuterLoop, const LoopInfo &LI);
inline Loop *getEpilog(Loop &CoarsenedOuterLoop, LoopInfo &LI) {
  return const_cast<Loop *>(
          getEpilog(static_cast<const Loop &>(CoarsenedOuterLoop),
                    static_cast<const LoopInfo &>(LI)));
}

/// If Prolog would be returned by getProlog() above, then
/// return the SDetachInst for this coarsened loop's epilog task, otherwise
/// return nullptr if Prolog is not a coarsened loop's epilog
const SDetachInst *getPrologSDetach(const Loop &Prolog);
inline SDetachInst *getPrologSDetach(Loop &Prolog) {
  const Loop &CE = static_cast<const Loop &>(Prolog);
  return const_cast<SDetachInst *>(getPrologSDetach(CE));
}

const SDetachInst *getEpilogSDetach(const Loop &Epilog);
inline SDetachInst *getEpilogSDetach(Loop &Epilog) {
  const Loop &CE = static_cast<const Loop &>(Epilog);
  return const_cast<SDetachInst *>(getEpilogSDetach(CE));
}


} // end namespace llvm

#endif // LLVM_TRANSFORMS_SWARM_LOOPCOASEN_H
