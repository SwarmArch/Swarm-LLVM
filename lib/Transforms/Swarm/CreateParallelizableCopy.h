//===- CreateParallelizableCopy.h ----------------------------------C++ -*-===//
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
// CreateParallelizableCopy provides the construction of parallel functions and
// their type signatures and callsites. Fractalizer uses these APIs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SWARM_CREATEPARALLELIZABLECOPY_H
#define LLVM_TRANSFORMS_SWARM_CREATEPARALLELIZABLECOPY_H

#include "llvm/IR/Attributes.h"

namespace llvm {

class CallInst;
class Function;
class FunctionType;
class Instruction;
class Value;

/// For indirect calls (calls that use function pointers),
/// Generate code that will determine at runtime if the serial function
/// pointed to by the function pointer has a parallel version,
Instruction *createHasParVersion(Value *SerialCalleePtr, Instruction *InsertBefore);
/// For indirect calls (calls that use function pointers),
/// Generate code that will obtain at runtime a pointer to the parallel version
/// of the serial function pointed to by the function pointer.
Instruction *createGetParFuncPtr(Value *SerialCalleePtr, Instruction *InsertBefore);

/// Build a call to ParallelCallee based on SerialCallInst. The new call will
/// use argument values and attributes taken from SerialCallInst.
/// ParallelCallee must be a Function or a function pointer.
/// If ContClosure is supplied, pass it to ParallelCallee as a continuation.
/// \return the new parallel CallInst.
CallInst *createParallelCall(Value *ParallelCallee,
                             CallInst *SerialCallInst,
                             Value *ContClosure,
                             Instruction *InsertBefore);
inline CallInst *createParallelCall(Value *ParallelCallee,
                                    CallInst *SerialCallInst,
                                    Instruction *InsertBefore) {
  return createParallelCall(ParallelCallee, SerialCallInst,
                            nullptr, InsertBefore);
}

/// Return the parallel version of SerialFunc.
/// If the parallel version may be external, create and return a thunk with
/// weak linkage to act as the parallel version, which calls SerialFunc
/// using CallAttributes and then sequentially spawns any continuation it is
/// suppied, passing along the serial callee's return value.
/// The thunk may be replaced with a real parallel version at link time.
Function *getOrInsertParallelVersion(
        Function *SerialFunc,
        AttributeList CallAttributes = AttributeList());

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SWARM_CREATEPARALLELIZABLECOPY_H
