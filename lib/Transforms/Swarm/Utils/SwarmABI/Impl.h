//===- Impl.h - Interface between ABI-independent and ABI-depenent code ---===//
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
// Interface for code that is common across different implementations of Swarm
// to make calls to code that is specific to particular implementations of
// Swarm hardware or simulations.
//
//===----------------------------------------------------------------------===//

#ifndef SWARM_ABI_IMPL_H_
#define SWARM_ABI_IMPL_H_

#include "llvm/ADT/ArrayRef.h"

namespace llvm {

class CallInst;
class Constant;
class Function;
class Instruction;
class SDetachInst;
class Value;

struct ImplBase {
  virtual ~ImplBase() = default;


  // APIs for building code.

  virtual CallInst *createGetTimestampInst(Instruction *InsertBefore,
                                           bool isSuper) = 0;
  virtual CallInst *createDeepenInst(Instruction *InsertBefore) = 0;
  virtual CallInst *createUndeepenInst(Instruction *InsertBefore) = 0;
  virtual CallInst *createHeartbeatInst(Instruction *InsertBefore) = 0;

  // Exploit specific ABI to minimize task function's prolog and epilog
  virtual void optimizeTaskFunction(Function *TaskFn) = 0;

  /// Build code in the parent task to enqueue the child to hardware,
  /// and insert the new code immediately before the detach instruction.
  virtual CallInst *createSwarmEnqueue(Function *TaskFn,
                                       ArrayRef<Value *> Args,
                                       SDetachInst *DI) = 0;


  // The APIs below are for recognizing code built using the APIs above.

  virtual const Constant *getEnqueueFunction(const CallInst *EnqueueI) = 0;
  virtual bool isSwarmEnqueueInst(const CallInst *CI) = 0;
  virtual bool isGetTimestampInst(const CallInst *CI) = 0;
  virtual bool isDeepenInst(const CallInst *CI) = 0;
  virtual bool isUndeepenInst(const CallInst *CI) = 0;
  virtual bool isDequeueInst(const CallInst *CI) = 0;
  virtual bool isHeartbeatInst(const CallInst *CI) = 0;
};

struct oss_v1 final : public ImplBase {
  CallInst *createGetTimestampInst(Instruction *InsertBefore,
                                   bool isSuper) override;
  CallInst *createDeepenInst(Instruction *InsertBefore) override;
  CallInst *createUndeepenInst(Instruction *InsertBefore) override;
  CallInst *createHeartbeatInst(Instruction *InsertBefore) override;
  void optimizeTaskFunction(Function *TaskFn) override;
  CallInst *createSwarmEnqueue(Function *TaskFn,
                               ArrayRef<Value *> Args,
                               SDetachInst *DI) override;
  const Constant *getEnqueueFunction(const CallInst *EnqueueI) override;
  bool isSwarmEnqueueInst(const CallInst *CI) override;
  bool isGetTimestampInst(const CallInst *CI) override;
  bool isDeepenInst(const CallInst *CI) override;
  bool isUndeepenInst(const CallInst *CI) override;
  bool isDequeueInst(const CallInst *CI) override;
  bool isHeartbeatInst(const CallInst *CI) override;
};

struct sccrt_serial final : public ImplBase {
  CallInst *createGetTimestampInst(Instruction *InsertBefore,
                                   bool isSuper) override;
  CallInst *createDeepenInst(Instruction *InsertBefore) override;
  CallInst *createUndeepenInst(Instruction *InsertBefore) override;
  CallInst *createHeartbeatInst(Instruction *InsertBefore) override;
  void optimizeTaskFunction(Function *TaskFn) override {}
  CallInst *createSwarmEnqueue(Function *TaskFn,
                               ArrayRef<Value *> Args,
                               SDetachInst *DI) override;
  const Constant *getEnqueueFunction(const CallInst *EnqueueI) override;
  bool isSwarmEnqueueInst(const CallInst *CI) override;
  bool isGetTimestampInst(const CallInst *CI) override;
  bool isDeepenInst(const CallInst *CI) override;
  bool isUndeepenInst(const CallInst *CI) override;
  bool isDequeueInst(const CallInst *CI) override;
  bool isHeartbeatInst(const CallInst *CI) override;
};

}  // end namespace llvm

#endif
