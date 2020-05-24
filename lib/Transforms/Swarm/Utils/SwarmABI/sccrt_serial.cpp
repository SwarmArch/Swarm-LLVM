//===- sccrt_serial.cpp - Queuing tasks in userspace software -------------===//
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
// Generate calls to libsccrt to use software priority queues to run tasks,
// instead using hardware or simulator support for task queuing and dispatch.
// This will result in running all tasks serially and without speculation.
//
//===----------------------------------------------------------------------===//

#include "Impl.h"

#include "Utils/SCCRT.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "swarmabi"


CallInst *llvm::sccrt_serial::createGetTimestampInst(Instruction *InsertBefore,
                                                     bool isSuper) {
  Module *M = InsertBefore->getModule();
  Function *Callee = isSuper
                         ? RUNTIME_FUNC(__sccrt_serial_get_timestamp_super, M)
                         : RUNTIME_FUNC(__sccrt_serial_get_timestamp, M);
  CallInst *CI = CallInst::Create(Callee, "timestamp", InsertBefore);
  CI->setDebugLoc(InsertBefore->getDebugLoc());
  return CI;
}

CallInst *llvm::sccrt_serial::createDeepenInst(Instruction *InsertBefore) {
  Module *M = InsertBefore->getModule();
  CallInst *CI = CallInst::Create(RUNTIME_FUNC(__sccrt_serial_deepen, M), "",
                                  InsertBefore);
  CI->setDebugLoc(InsertBefore->getDebugLoc());
  return CI;
}

CallInst *llvm::sccrt_serial::createUndeepenInst(Instruction *InsertBefore) {
  Module *M = InsertBefore->getModule();
  CallInst *CI = CallInst::Create(RUNTIME_FUNC(__sccrt_serial_undeepen, M), "",
                                  InsertBefore);
  CI->setDebugLoc(InsertBefore->getDebugLoc());
  return CI;
}

CallInst *llvm::sccrt_serial::createHeartbeatInst(Instruction *InsertBefore) {
  Module *M = InsertBefore->getModule();
  CallInst *CI = CallInst::Create(RUNTIME_FUNC(__sccrt_serial_heartbeat, M), "",
                                  InsertBefore);
  CI->setDebugLoc(InsertBefore->getDebugLoc());
  return CI;
}


CallInst *llvm::sccrt_serial::createSwarmEnqueue(Function *TaskFn,
                                                 ArrayRef<Value *> Args,
                                                 SDetachInst *DI) {
  Module *M = DI->getModule();
  Function *Callee = DI->isSuperdomain()
                         ? RUNTIME_FUNC(__sccrt_serial_enqueue_super, M)
                         : RUNTIME_FUNC(__sccrt_serial_enqueue, M);
  DEBUG(dbgs() << "Generating call to enqueue function:" << *Callee);

  IRBuilder<> Builder(DI);

  SmallVector<Value *, 8> Inputs;
  {
    Inputs.push_back(Builder.CreatePointerCast(
        TaskFn, Callee->getFunctionType()->getParamType(Inputs.size()),
        "taskfn"));

    assert(!DI->isRelativeTimestamp());
    Inputs.push_back(Builder.CreateZExt(
        DI->getTimestamp(),
        cast<IntegerType>(
            Callee->getFunctionType()->getParamType(Inputs.size())),
        "timestamp"));

    assert(Args.size() <= __SCCRT_SERIAL_MAX_ARGS);
    for (Value *Arg : Args) {
      Type *DesiredTy = cast<IntegerType>(
          Callee->getFunctionType()->getParamType(Inputs.size()));
      Inputs.push_back((Arg->getType()->isPointerTy())
                           ? Builder.CreatePtrToInt(Arg, DesiredTy, "swarmarg")
                           : Builder.CreateZExt(Arg, DesiredTy, "swarmarg"));
    }

    while (Inputs.size() < Callee->getFunctionType()->getNumParams()) {
      Type *DesiredTy = cast<IntegerType>(
          Callee->getFunctionType()->getParamType(Inputs.size()));
      Inputs.push_back(ConstantInt::get(DesiredTy, 0xDEADFEED));
    }
  }

  CallInst *Call = Builder.CreateCall(Callee, Inputs);
  Call->setDebugLoc(DI->getDebugLoc());
  return Call;
}


bool llvm::sccrt_serial::isSwarmEnqueueInst(const CallInst *CI) {
  if (Function *CF = CI->getCalledFunction())
    if (CF->getName().startswith("__sccrt_serial_enqueue"))
      return true;
  return false;
}

const Constant *llvm::sccrt_serial::getEnqueueFunction(const CallInst *EnqueueI) {
  return cast<Constant>(EnqueueI->getArgOperand(0));
}

bool llvm::sccrt_serial::isGetTimestampInst(const CallInst *CI) {
  if (Function *CF = CI->getCalledFunction())
    if (CF->getName().startswith("__sccrt_serial_get_timestamp"))
      return true;
  return false;
}

bool llvm::sccrt_serial::isDeepenInst(const CallInst *CI) {
  if (Function *CF = CI->getCalledFunction())
    if (CF->getName().startswith("__sccrt_serial_deepen"))
      return true;
  return false;
}

bool llvm::sccrt_serial::isUndeepenInst(const CallInst *CI) {
  if (Function *CF = CI->getCalledFunction())
    if (CF->getName().startswith("__sccrt_serial_undeepen"))
      return true;
  return false;
}

bool llvm::sccrt_serial::isDequeueInst(const CallInst *CI) {
  return false;
}

bool llvm::sccrt_serial::isHeartbeatInst(const CallInst *CI) {
  if (Function *CF = CI->getCalledFunction())
    if (CF->getName().startswith("__sccrt_serial_heartbeat"))
      return true;
  return false;
}
