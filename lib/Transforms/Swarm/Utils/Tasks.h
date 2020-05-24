//===-- Tasks.h - Tools for tasks and domains ------------------*- C++ -*--===//
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
// Utility methods for manipulating tasks represented inline within parent CFGs
// using a Tapir-like detach-reattach representation.  Includes utilities for
// handling Fractal domains (deepen and undeepen intrinisics) as well.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SWARM_UTILS_TASKS_H
#define LLVM_TRANSFORMS_SWARM_UTILS_TASKS_H

#include "llvm/ADT/Twine.h"
#include "llvm/IR/IntrinsicInst.h"

namespace llvm {

class BasicBlock;
class DominatorTree;
class Function;
class Instruction;
class Loop;
class LoopInfo;
class SDetachInst;
class SReattachInst;
class Value;

/// Wrapper class for deepen intrinsics
/// with an LLVM-style static Create() method.
/// Compatible with isa, cast, and dyn_cast template functions.
class DeepenInst : public IntrinsicInst {
public:
  DeepenInst() = delete;
  DeepenInst(const DeepenInst &) = delete;
  DeepenInst &operator=(const DeepenInst &) = delete;
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::deepen;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
  static DeepenInst *Create(const Twine& Name, Instruction *InsertBefore);
  static DeepenInst *Create(Instruction *InsertBefore) {
    return Create(Twine(), InsertBefore);
  }

  const DeepenInst *getSuperdomain(const DominatorTree &DT) const;
  DeepenInst *getSuperdomain(const DominatorTree &DT);
};

/// Wrapper class for undeepen intrinsics
/// with an LLVM-style static Create() method.
/// Compatible with isa, cast, and dyn_cast template functions.
class UndeepenInst : public IntrinsicInst {
public:
  UndeepenInst() = delete;
  UndeepenInst(const UndeepenInst &) = delete;
  UndeepenInst &operator=(const UndeepenInst &) = delete;
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::undeepen;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
  static UndeepenInst *Create(DeepenInst *Deepen, Instruction *InsertBefore);

  const DeepenInst *getMatchingDeepen() const;
  DeepenInst *getMatchingDeepen();
};

/// Detach kinds are tracked for the purpose of collecting stats.
/// We sometimes also write assertions to document the kind of detaches.
/// Beyond that, let's avoid giving DetachKind any semantic effects.
enum class DetachKind {
  Unknown,
  Annotation,
  AnnotationCont,
  Write,
  WriteCont,
  Call,
  CallIndependentCont,
  CallPassedCont,
  SerialLoop,
  UnexpandedIter,
  LoopStart,
  LoopIndependentCont,
  SubsumedCont,
  EarlyChainIter,
  BalancedSpawner,
  BalancedIter,
  ExpandedChainSpawner,
  ExpandedChainIter,
  ProgressiveLatch,
  CoarsenProlog,
  CoarsenEpilog,
  RetargetSuperdomain,
  Free
};

static constexpr const char* DetachKindStringKey = "SwarmDetachKind";

inline void setDetachKind(SDetachInst *DI, DetachKind Kind) {
  LLVMContext &Context = DI->getContext();
  Metadata *MD[] = {ConstantAsMetadata::get(
      ConstantInt::get(Type::getInt32Ty(Context), static_cast<int>(Kind)))};
  DI->setMetadata(DetachKindStringKey, MDNode::get(Context, MD));
}

inline DetachKind getDetachKind(const SDetachInst *DI) {
  if (MDNode *MD = DI->getMetadata(DetachKindStringKey)) {
    return static_cast<DetachKind>(mdconst::extract<ConstantInt>(MD->getOperand(0))->getSExtValue());
  }
  return DetachKind::Unknown;
}

/// Create a task with the given timestamp, detaching just before TaskStart,
/// reattaching into TaskEnd.
void detachTask(Instruction *TaskStart, // inclusive
                Instruction *TaskEnd, // exclusive
                Value *Timestamp,
                DeepenInst *Domain = nullptr,
                DetachKind Kind = DetachKind::Unknown,
                const Twine& Name = Twine(),
                DominatorTree *DT = nullptr,
                LoopInfo *LI = nullptr,
                SDetachInst **rDI = nullptr,
                SReattachInst **rRI = nullptr);

/// Return true if F contains any SDetachInsts
bool hasAnySDetachInst(const Function &F);

/// Returns +1 if this spawns a child task to a subdomain of the parent task,
/// -1 if this spawns a child task to the superdomain of the domain in which
/// the parent task started, and 0 if the child and parent tasks are in the
/// same domain.
int getDomainDepthDiff(const SDetachInst *Task, const DominatorTree &DT,
                       const SDetachInst *ParentTask=nullptr);

/// Return the nearest enclosing sdetach instruction.
/// Consider a detach to enclose BB if BB is dominated by the detach edge.
/// That is, don't consider non-dominated error-handling blocks reachable
/// from a task to be enclosed by the task.
const SDetachInst *getEnclosingTask(const BasicBlock *BB, const DominatorTree &DT);
inline SDetachInst *getEnclosingTask(BasicBlock *BB, const DominatorTree &DT) {
  return const_cast<SDetachInst *>(
      getEnclosingTask(const_cast<const BasicBlock *>(BB), DT));
}
inline const SDetachInst *getEnclosingTask(const Instruction *I, const DominatorTree &DT) {
  return getEnclosingTask(I->getParent(), DT);
}
inline SDetachInst *getEnclosingTask(Instruction *I, const DominatorTree &DT) {
  return getEnclosingTask(I->getParent(), DT);
}

/// Return the nearest DeepenInst/UndeepenInst that strictly dominates I,
/// but is strictly dominated by UpTo.
/// Return null if no such instruction exists.
const DeepenInst *getPreceedingDeepen(const Instruction *I,
                                      const DominatorTree &DT,
                                      const BasicBlock *UpTo=nullptr);
const UndeepenInst *getPreceedingUndeepen(const Instruction *I,
                                          const DominatorTree &DT,
                                          const BasicBlock *UpTo=nullptr);

/// For convenience, a wrapper around SDetachInst::getDomain() that casts to
/// the right type and passes through null values.
inline const DeepenInst *getDomain(const SDetachInst *DI) {
  if (DI) return cast_or_null<DeepenInst>(DI->getDomain());
  else return nullptr;
}
inline DeepenInst *getDomain(SDetachInst *DI) {
  if (DI) return cast_or_null<DeepenInst>(DI->getDomain());
  else return nullptr;
}

/// Populate Result with all blocks in the dominator subtree rooted at R,
/// ignoring nested dominator subtrees corresponding to nested detached regions.
/// In other words, populate Result with all blocks dominated by R that are
/// in the same task as R, i.e., not part of a detached child task.
/// This works similarly to DominatorTree::getDescendants().
void getNonDetachDescendants(const DominatorTree &DT,
                             BasicBlock *R,
                             SmallVectorImpl<BasicBlock *> &Result);

/// Populate OuterDetaches with detaches in F,
/// ignoring any nested within detached regions.
/// This excludes detaches without a timestamp.
void getOuterDetaches(Function *F,
                      const DominatorTree &DT,
                      SmallVectorImpl<SDetachInst *> &OuterDetaches);
/// Populate OuterDetaches with detach instructions from the the region L in F,
/// ignoring any nested within detached regions contained in L.
/// This excludes detaches without a timestamp.
void getOuterDetaches(Loop &L,
                      const DominatorTree &DT,
                      SmallVectorImpl<SDetachInst *> &OuterDetaches);
/// Populate OuterDetaches with the detach instructions nested within the
/// detached region associated with DI, ignoring further nested inner detaches.
/// This excludes detaches without a timestamp.
void getOuterDetaches(SDetachInst *DI,
                      const DominatorTree &DT,
                      SmallVectorImpl<SDetachInst *> &OuterDetaches);
/// Populate OuterDetaches with the detach instructions dominated by Domain,
/// but not dominated by any matching undeepen,
/// ignoring further nested inner detaches.
/// This excludes detaches without a timestamp.
void getOuterDetaches(const DeepenInst *Domain,
                      const DominatorTree &DT,
                      SmallVectorImpl<const SDetachInst *> &OuterDetaches);

/// Return true if RI is paired with DI, i.e., if RI marks a task-ending
/// boundary for the task that begins with DI->getDetached().
bool areMatching(const SDetachInst *DI,
                 const SReattachInst *RI,
                 const DominatorTree &DT);

/// Populate MatchingReattaches with the sreattaches for which
/// areMatching() returns true.
//TODO(victory)? Replace this with iterator interfaces, since this utility's
// users usually don't need to store this collection, and we could create
// clean const and non-const versions of iterators.
void getMatchingReattaches(
        const SDetachInst *DI,
        const DominatorTree &DT,
        SmallVectorImpl<const SReattachInst *> &MatchingReattaches);
inline void getMatchingReattaches(
        SDetachInst *DI,
        const DominatorTree &DT,
        SmallVectorImpl<SReattachInst *> &MatchingReattaches) {
  SmallVector<const SReattachInst *, 8> MatchingConstReattaches;
  getMatchingReattaches(DI, DT, MatchingConstReattaches);
  transform(MatchingConstReattaches, std::back_inserter(MatchingReattaches),
            [](const SReattachInst *RI) {
              return const_cast<SReattachInst *>(RI);
            });
}

/// If getMatchingReattaches would provide exactly one matching SReattachInst,
/// return it, otherwise return nullptr.
const SReattachInst *getUniqueMatchingSReattach(
        const SDetachInst *DI,
        const DominatorTree &DT);
inline SReattachInst *getUniqueMatchingSReattach(
        SDetachInst *DI,
        const DominatorTree &DT) {
  const SDetachInst *CDI = static_cast<const SDetachInst *>(DI);
  return const_cast<SReattachInst *>(getUniqueMatchingSReattach(CDI, DT));
}

/**
 * Checks if this loop is the form of Swarm loop that LoopExpansion can handle.
 * Right now we check that the loop is in a canonical form:
 * - The loop should be in LoopSimplify form, i.e., it must have a single latch
 *   and dedicated exits.
 * - The header detaches the body, and continues straight to the latch.
 * - All code with side effects is in the detached loop body, and the header
 *   must not access memory.
 * - The loop only has one non-dead-end continuation which is reached only from
 *   a simple conditional branch in the latch.
 * - The loop produces no SSA values used in its proper continuation.
 * - The header detach and loop latch need non-null DebugLocs.
 */
bool isExpandableSwarmLoop(const Loop *L, const DominatorTree &DT);

/// Return true if BI might terminate the latch block of an expandable loop.
///
/// Return false only if BI defintely is not the latch of an expandable loop.
/// Conservatively return true if not sure.
bool isExpandableLoopLatch(const BranchInst *BI);

/// If the loop latch has exactly two predecessors, one of which is the header,
/// and the other terminates with an SReattachInst, return the SReattachInst.
/// Otherwise return nullptr.
const SReattachInst *getUniqueSReattachIntoLatch(const Loop &L);
inline SReattachInst *getUniqueSReattachIntoLatch(Loop &L) {
  const Loop &CL = static_cast<const Loop &>(L);
  return const_cast<SReattachInst *>(getUniqueSReattachIntoLatch(CL));
}

/// Hoist DI and its detached CFG to before MoveBefore.
/// In pseudocode, given code of the form
///
/// ts = ...
/// %movebefore
/// [ ... middle code ... ]
/// detach (ts) {
///   ... code ...
/// }
/// [ ... continuation ... ]
///
/// This function transforms it into
///
/// ts = ...
/// detach (ts) {
///   ... code ...
/// }
/// %movebefore
/// [ ... middle code ... ]
/// [ ... continuation ... ]
///
void hoistDetach(SDetachInst *DI,
                 Instruction *MoveBefore,
                 DominatorTree *DT = nullptr,
                 LoopInfo *LI = nullptr);

/// Return the earliest instruction before which it is safe to hoist the given
/// detach instruction and its detached region. This can reduce the critical
/// path to spawning parallel work, while not changing program order.
///
/// The function does not cross
/// * function boundaries
/// * enclosing detached regions
/// Therefore the current highest possible hoist point is in DI's containing
/// detched region and/or function entry block.
const Instruction *getEarliestHoistPoint(
        const SDetachInst *DI,
        const DominatorTree &DT,
        const LoopInfo &LI);
inline Instruction *getEarliestHoistPoint(
        SDetachInst *DI,
        const DominatorTree &DT,
        const LoopInfo &LI) {
  const SDetachInst *CDI = static_cast<const SDetachInst *>(DI);
  return const_cast<Instruction *>(getEarliestHoistPoint(CDI, DT, LI));
}


/// Set the cache line address of Address as a hint for DI
/// i.e. use static_cast<uint64_t>(Address) / 64
///
/// Address may be a pointer or 64-bit integer
void setCacheLineHintFromAddress(SDetachInst *DI, Value *Address);

/// Insert before InsertBefore a task spawn that targets the superdomain
/// Superdomain, where the task just calls free() on Ptr.
SDetachInst *createSuperdomainDetachedFree(Value *Ptr, DeepenInst *Superdomain,
                                           Instruction *InsertBefore,
                                           DominatorTree *DT = nullptr,
                                           LoopInfo *LI = nullptr);

/// Replace the detach with a branch to the continuation,
/// erasing the detached CFG in the original function.
void eraseDetach(DetachInst *DI, DominatorTree &DT, LoopInfo *LI=nullptr);

} // namespace llvm

#endif
