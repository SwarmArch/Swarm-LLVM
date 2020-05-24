//===- SwarmABI.h - Swarm hardware interface ------------------------------===//
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
// Provides the interface to generate calls into hardware or simulators
// for features implemented by Swarm extensions to a conventional ISA.
// This interface does the low-level dirty work of passes such as LowerToSwarm.
//
//===----------------------------------------------------------------------===//
#ifndef SWARM_ABI_H_
#define SWARM_ABI_H_

namespace llvm {

class CallInst;
class DominatorTree;
class Function;
class Instruction;
class LoopInfo;
class SDetachInst;
class TargetTransformInfo;

namespace swarm_abi {

  // Outlines the detached region to a function that is enqueued to hardware.
  // Updates DT, and updates LI if it is given.
  // Returns a new function which may contain further detach instructions.
  Function *lowerDetach(SDetachInst &Detach, DominatorTree &DT,
                        const TargetTransformInfo &TTI, LoopInfo *LI = nullptr);

  void optimizeTaskFunction(Function *TaskFn);

  // Creating and inserting calls to the Swarm ABI
  CallInst *createGetTimestampInst(Instruction *InsertBefore,
                                   bool isSuper=false);
  CallInst *createDeepenInst(Instruction *InsertBefore);
  CallInst *createUndeepenInst(Instruction *InsertBefore);
  CallInst *createHeartbeatInst(Instruction *InsertBefore);

  // Detecting calls into the SwarmABI
  bool isSwarmEnqueueInst(const Instruction *I);
  bool isGetTimestampInst(const Instruction *I);
  bool isDeepenInst(const Instruction *I);
  bool isUndeepenInst(const Instruction *I);
  bool isDequeueInst(const Instruction *I);
  bool isHeartbeatInst(const Instruction *I);

  const Function *getEnqueueFunction(const Instruction *EnqueueI);
  inline Function *getEnqueueFunction(Instruction *EnqueueI) {
    const Instruction *CI = static_cast<const Instruction *>(EnqueueI);
    return const_cast<Function*>(getEnqueueFunction(CI));
  }

}  // end of swarm_abi namespace
}  // end of llvm namespace

#endif
