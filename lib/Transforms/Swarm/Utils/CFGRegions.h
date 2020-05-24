//===-- CFGRegions.h - Manipulation of CFG subgraphs -----------*- C++ -*--===//
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
// Utility methods for traversing and finding subgraph structure within a CFG,
// for forming such subgraphs into single-entry, single-valid-exit regions,
// for manipulating the communication of live-in values into such regions, and
// for restructuring such regions by separating them into new functions (a.k.a.
// "outlining", roughly the reverse of function inlining).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SWARM_UTILS_CFGREGIONS_H
#define LLVM_TRANSFORMS_SWARM_UTILS_CFGREGIONS_H

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

namespace llvm {

class BasicBlock;
class DominatorTree;
class Function;
class Instruction;
class Loop;
class LoopInfo;
class OptimizationRemarkEmitter;
class TargetTransformInfo;
class StructType;
class CallInst;

extern cl::opt<bool> DisableEnvSharing;

/// Find the live-ins for Blocks, which must have no live-outs. In other words,
/// determine the values that would need to be passed as arguments if Blocks
/// were outlined into a new function.
void findInputsNoOutputs(const SmallSetVector<BasicBlock *, 8> &Blocks,
                         SetVector<Value *> &Inputs);

/// Create and return a new Function containing a copy of Blocks, with
/// parameters corresponding to Params, and populate VMap with a mapping
/// from Values associated with the old Blocks to the new copies.
/// The new Function will be appropriate to call from the original Function
/// as a replacement for the original copy of Blocks.
/// Blocks must be a region of the CFG with a single incoming edge and
/// a unique successor outside of Blocks, ignoring error-handling Blocks.
/// Entry must be the element of Blocks that is the entry point of the region.
/// Right now, this utility does not support outlining the entry block of
/// a function or any return instructions.
//TODO(victory): Settle on SmallPtrSet- or SmallSetVector-based interface.
Function *outline(const SetVector<Value *> &Params,
                  const SmallPtrSetImpl<BasicBlock *> &Blocks,
                  BasicBlock *Entry,
                  const StringRef Suffix,
                  ValueToValueMapTy &VMap);
inline Function *outline(const SetVector<Value *> &Params,
                         const SmallSetVector<BasicBlock *, 8> &Blocks,
                         BasicBlock *Entry,
                         const StringRef Suffix,
                         ValueToValueMapTy &VMap) {
  const SmallPtrSet<BasicBlock *, 8> BlockSet(Blocks.begin(), Blocks.end());
  return outline(Params, BlockSet, Entry, Suffix, VMap);
}
inline Function *outline(const SetVector<Value *> &Params,
                         const SmallPtrSetImpl<BasicBlock *> &Blocks,
                         BasicBlock *Entry,
                         const StringRef Suffix) {
  ValueToValueMapTy VMap;
  return outline(Params, Blocks, Entry, Suffix, VMap);
}
inline Function *outline(const SetVector<Value *> &Params,
                         const SmallSetVector<BasicBlock *, 8> &Blocks,
                         BasicBlock *Entry,
                         const StringRef Suffix) {
  ValueToValueMapTy VMap;
  const SmallPtrSet<BasicBlock *, 8> BlockSet(Blocks.begin(), Blocks.end());
  return outline(Params, BlockSet, Entry, Suffix, VMap);
}

/// Shrinks inputs (live-ins) to Blocks by sinking cheap instructions into them.
/// Blocks must all be dominated by Blocks[0].
/// Values in Blacklist are not considered for sinking.
/// Useful to reduce the number of arguments to outlined functions.
void shrinkInputs(const SmallVectorImpl<BasicBlock *> &Blocks,
                  const SmallPtrSet<Value *, 4> &Blacklist,
                  const TargetTransformInfo &TTI,
                  OptimizationRemarkEmitter *ORE = nullptr);
inline void shrinkInputs(const std::vector<BasicBlock *> &BBs,
                         const SmallPtrSet<Value *, 4> &Blacklist,
                         const TargetTransformInfo &TTI,
                         OptimizationRemarkEmitter *ORE = nullptr) {
  SmallVector<BasicBlock *, 32> Blocks;
  for (BasicBlock *BB : BBs)
    Blocks.push_back(BB);
  shrinkInputs(Blocks, Blacklist, TTI, ORE);
}

/// Allocate a struct (closure) using the types of each value in Captures,
/// before instruction AllocateAndPackBefore. Pack the closure before that
/// instruction, using the values in Captures, skipping the packing of
/// the first "FieldsToSkip" fields. If FieldsToSkip == 0, all values in
/// Captures are packed into the closure.
/// Return the Instruction/Value that yields the allocated struct
Instruction *createClosure(
        ArrayRef<Value *> Captures,
        Instruction *AllocateAndPackBefore,
        StringRef Name = "closure",
        unsigned FieldsToSkip = 0);

/// Unpack the values in Closure, assumed to be generated by createClosure()
/// from Captures, again skipping the first FieldsToSkip elements of Captures.
/// Specifically, for each use of a non-skipped value in Captures, replace the
/// use with an unpacking load from Closure if UnpackBeforeForUser returns
/// non-null.  Loads shall be inserted before the instruction indicated by
/// UnpackBeforeForUser.
void unpackClosure(
        Value *Closure,
        ArrayRef<Value *> Captures,
        std::function<Instruction *(Instruction *)> UnpackBeforeForUser,
        unsigned FieldsToSkip = 0);

/// Helper to get the StructType of a Closure pointer
StructType *getClosureType(const Value *Closure);

/// Return the subset of arguments that will not fit in a task's registers and
/// would have to be passed through memory. Args are considered in the order
/// they are given; the code does not try to do any bin-packing.
SetVector<Value *> getMemArgs(const SetVector<Value *> &Args,
                              const DataLayout &DL,
                              const Value *TimeStamp = nullptr,
                              uint32_t FieldsToSkip = 0);

/// Erase all code dominated by Root. Also remove from DT.
/// If LI is given, also remove blocks from LI.
/// This is intended to be called immediately after Root is made dead,
/// in order to clean up code during a transform pass.
/// This will fail if Root (and its subtree) are not dead.
/// This does not update the DT other than removing nodes.
/// If Root does not dominate everything it can reach, it is the caller's
/// responsibility to update the immediate dominators of any continuations
/// that are not deleted.
void eraseDominatorSubtree(BasicBlock *Root, DominatorTree &DT,
                           LoopInfo *LI = nullptr);

/// Delete everything associated with the loop starting with the header,
/// up to but not including EndBlock, and update DT and LI accordingly.
/// Assumes the loop has a dedicated preheader.
/// After this runs, the preheader with branch directly to EndBlock.
void eraseLoop(Loop &L,
               BasicBlock *EndBlock,
               DominatorTree &DT,
               LoopInfo &LI);

/// Given a function that starts with a simple loop, transform it into a
/// recursive loop by transforming it to run only one iteration and then
/// recursively call itself for the next iteration.
/// Assumes that if the header has N phi nodes, the first N parameters
/// correspond to those N phi nodes, respectively.
/// \returns the newly created recursive call.
///
/// In pseudocode, given a function of the following form:
///
/// void f(phi0, phi1, ...) {
///   do {
///     ... loop body ...
///   } while (some_cond());
///   ... continuation ...
/// }
///
/// Then this method transforms the function into the following form:
///
/// void f(phi0, phi1, ...) {
///   ... loop body ...
///   if (some_cond()) {
///     f(phi0_next, phi1_next, ...);
///   } else {
///     ... continuation ...
///   }
/// }
///
/// If building with debug info, the caller must ensure the loop's backedge has
/// a non-null DebugLoc, which will be used for the recursive call.
CallInst *formRecursiveLoop(Function *F);

/// Return true if every control flow path from BB inevitably reaches an
/// unreachable instruction.  Return false if there is a path from BB to a
/// return or reattach instruction
bool isDoomedToUnreachableEnd(const BasicBlock *BB);

/// If the loop has exactly one exit block that is not a dead end, return it.
/// Otherwise return nullptr.
/// Like a version of Loop::getUniqueExitBlock() that excludes doomed exits.
const BasicBlock *getUniqueNonDeadendExitBlock(const Loop &L);
inline BasicBlock *getUniqueNonDeadendExitBlock(Loop &L) {
  const Loop &CL = static_cast<const Loop &>(L);
  return const_cast<BasicBlock *>(getUniqueNonDeadendExitBlock(CL));
}

/// Duplicate blocks reachable from Source up until End if they are not
/// dominated, to ensure that Source dominates them.  The duplicates will not
/// be reachable from Source.  Populate ReachableBBs with all the blocks
/// reachable from (and dominated by) Source up to and excuding End.
/// If NumCopiedBlocks is given, set it to the number of blocks duplicated.
/// If End is reachable from Source, create and return a new predecessor for
/// End that is dominated by Source and that all paths from Source to End pass
/// through, which will be included in ReachableBBs.  Otherwise, return null.
BasicBlock *makeReachableDominated(BasicBlock *Source,
                                   BasicBlock *End,
                                   DominatorTree &DT,
                                   LoopInfo &LI,
                                   SmallSetVector<BasicBlock *, 8> &ReachableBBs,
                                   unsigned *NumCopiedBlocks = nullptr);
inline BasicBlock *makeReachableDominated(BasicBlock *Source,
                                          BasicBlock *End,
                                          DominatorTree &DT,
                                          LoopInfo &LI,
                                          unsigned *NumCopiedBlocks = nullptr) {
  SmallSetVector<BasicBlock *, 8> ReachableBBs;
  return makeReachableDominated(Source, End, DT, LI,
                                ReachableBBs, NumCopiedBlocks);
}

/// Attempt a topological sort for the CFG subgraph reachable from Entry and
/// that can reach Exit.  Exclude any detached subregions.  Entry and Exit must
/// share their nearest enclosing loop (or both have no enclosing loop).
/// Collapse each inner loop to a single node, represented by its header.  If
/// this modified subgraph is a DAG, populate SortedBBs in a topological order.
/// Otherwise, there is irreducible control flow, leave SortedBBs empty.
void topologicalSort(
        BasicBlock *Entry,
        BasicBlock *Exit,
        const LoopInfo &LI,
        SmallVectorImpl<BasicBlock *> &SortedBBs);

} // namespace llvm

#endif
