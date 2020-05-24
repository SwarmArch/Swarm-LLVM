//===-- Misc.h - Generic utilities for LLVM --------------------*- C++ -*--===//
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
// Small, generic utilities for LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SWARM_UTILS_MISC_H
#define LLVM_TRANSFORMS_SWARM_UTILS_MISC_H

#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

namespace llvm {

class BasicBlock;
class DominatorTree;
class Function;
class Instruction;
class Loop;
class LoopInfo;
class ReturnInst;
class ScalarEvolution;
class Value;


//===---------------Generic Utilies----------------------------------------===//

/// Return an iterator to the single element of Range for which P returns true.
/// Assert that Range contains exactly one such element.
template <typename R, typename UnaryPredicate>
auto findUnique(R &&Range, UnaryPredicate P) -> decltype(std::begin(Range)) {
  auto Ret = find_if(Range, P);
  assert(Ret != Range.end());
  assert(std::none_of(std::next(Ret), Range.end(), P));
  return Ret;
}


//===---------------Generic LLVM IR Utilies--------------------------------===//

// Note that Instruction::mayHaveSideEffects and Instruction::mayWriteToMemory
// are misleadingly named: they return true for volatile and atomic loads,
// which is a convenient way of ensuring they are not reordered with any other
// memory operations, although it is clearly a lie.  This API is more honest:
// it returns false for all loads.  This is useful for task deliniation, where
// there's no need to treat volatile loads any differently from ordinary loads.
inline bool mayHaveSideEffects(const Instruction *I) {
  return I->mayHaveSideEffects() && !isa<LoadInst>(I);
}

/**
 * Return the function's unique return instruction. N.B. we're relying on
 * UnifyFunctionExitNodes (a.k.a. -mergereturn) to have produced the single
 * returning block.
 * @return nullptr if the function has no ReturnInst
 */
const ReturnInst *getUniqueReturnInst(const Function &F);
inline ReturnInst *getUniqueReturnInst(Function &F) {
  const Function &CF = static_cast<const Function &>(F);
  return const_cast<ReturnInst *>(getUniqueReturnInst(CF));
}

/// \brief Generate a dummy call instruction that returns the desired type.
///
/// These instructions can be useful as temporary placeholder values during a
/// transformation, but must be erased before the transformation is finished.
Instruction *createDummyValue(Type *T,
                              const Twine &Name,
                              Instruction *InsertBefore);

/// Insert code that creates a call to the C standard library's puts(),
/// which will print out Str followed by a newline.
/// Name will be the name given to the string global variable in the IR.
CallInst *createPrintString(StringRef Str,
                            const Twine &Name,
                            Instruction *InsertBefore);
inline CallInst *createPrintString(StringRef Str,
                                   Instruction *InsertBefore) {
  return createPrintString(Str, "", InsertBefore);
}

/// Return true if I is safe to duplicate to or move to anywhere that dominates
/// its users and is dominated by its operands.
bool isSafeToSinkOrHoist(const Instruction *I);

/// Return true if we can guarantee that there are no intervening stores to
/// Ptr between Begin and End (exclusive), so the value in memory at Ptr
/// does not change as control flow proceeds from Begin to End.
/// Assumes Begin dominates End.
/// If Ptr is null, it refers to all application data in memory.
bool isInvariantInPath(const Value *Ptr, const Instruction *Begin,
                       const Instruction *End);

/**
 * Recursively copy operands of Expr and Expr itself using Builder.
 * If a value in BaseMap is encountered, substitute according to BaseMap.
 * Only works on some simple expressions that are safe to copy or hoist.
 * If PhiPred is given, this can find operands through Phis in the successor
 * of PhiPred by using the operand those Phis use when incoming from PhiPred.
 */
Value *copyComputation(Value *Expr,
                       const ValueToValueMapTy &BaseMap,
                       IRBuilder<> &Builder,
                       const BasicBlock *PhiPred = nullptr);


/// Return true if PN is an induction variable that ScalarEvolution can rewrite.
bool canStrengthenIV(PHINode *PN, ScalarEvolution &SE);


/// Canonicalize as many induction variables in the loop as possible.
/// Return the canonical induction variable created or inserted by the scalar
/// evolution expander. Ensure the width of the returned canonical IV is at
/// least MinWidth. Given that canonical IV, strengthen as many other IVs as
/// possible to rewrite/eliminate them.
PHINode *canonicalizeIVs(
        Loop &L,
        unsigned MinBitWidth,
        const DominatorTree &DT,
        ScalarEvolution &SE);

/// Similar to canonicalizeIVs, but canonicalizes either all the
/// induction variables in the loop or none of them.
/// Returns nullptr if there exists any non-canonicalizable header PHI node.
PHINode *canonicalizeIVsAllOrNothing(
        Loop &L,
        unsigned MinBitWidth,
        const DominatorTree &DT,
        ScalarEvolution &SE);

/// \returns a DebugLoc that is safe to pass to the setDebugLoc() method of
/// 1) an instruction within L, or
/// 2) an instruction within the same function as L.
/// Much like Loop::getStartLoc(), but does not get a DebugLoc that refers to
/// a different function from which the loop was originally inlined.
DebugLoc getSafeDebugLoc(const Loop &L);


//===---------------Verification Utilies-----------------------------------===//

/**
 * Print a user-friendly error and return true if F has the
 * `assertswarmified` attribute. Otherwise, return false.
 */
bool errorIfAssertSwarmified(const Function &F);

/**
 * Crash if F's contents are invalid. If DT is passed, also crash if DT is not
 * up to date for F. If DT and LI are passed, also crash if LI is stale.
 * Before crashing, this attempts to dump information useful for debugging.
 * While the caller is welcome to use
 *   assert(!verifyFunction(F, &dbgs()) && msg);
 *   DT->verify();
 *   LI->verify(*DT);
 * this function combines the functionality of the three
 * and dumps out more information about F on failure.
 */
void assertVerifyFunction(const Function &F, const Twine &msg,
                          const DominatorTree *DT = nullptr,
                          const LoopInfo *LI = nullptr);

} // namespace llvm

#endif
