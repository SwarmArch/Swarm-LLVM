//===- Reductions.cpp - Parallelize reduction loops using the SCC runtime--===//
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
// Implementation of SCCRT-assisted parallel reduction functions and classes
//
//===----------------------------------------------------------------------===//

#include "Reductions.h"

#include "Utils/CFGRegions.h"
#include "Utils/Misc.h"
#include "Utils/SCCRT.h"
#include "Utils/SwarmABI.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationDiagnosticInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;

#define SR_NAME "swarm-reductions"
#define DEBUG_TYPE SR_NAME

namespace {


class ReductionOutputReplacer {
  Loop &L;
  const DominatorTree &DT;
  OptimizationRemarkEmitter &ORE;

  PHINode *const Output; // L's (sole) output phi node
  PHINode *ReductionPHI;
  RecurrenceDescriptor RD;
  bool IsInteger;

  Module *const M;
  LLVMContext &Context;

public:
  ReductionOutputReplacer(Loop &, PHINode *Output,
                          const DominatorTree &,
                          OptimizationRemarkEmitter &);
  bool run();

private:
  bool setup();

  void debugNotParallelizable(const Twine &Reason) const;
  void remarkNotParallelizable(StringRef RemarkName, const Twine &Reason,
                               const Instruction *Inst = nullptr) const;
};


} // end anonymous namespace


// RecurrenceDescriptor methods are inexplicably not const, so RD is not const
static Function *getInitializationFunction(RecurrenceDescriptor &RD) {
  Module *M = RD.getLoopExitInstr()->getModule();
  Function *Ret;
  if (RecurrenceDescriptor::isIntegerRecurrenceKind(RD.getRecurrenceKind())) {
    Ret = RUNTIME_FUNC(__sccrt_reduction_uint64_t_init, M);
  } else if (RD.getRecurrenceType()->isFloatTy()) {
    Ret = RUNTIME_FUNC(__sccrt_reduction_float_init, M);
  } else {
    assert(RD.getRecurrenceType()->isDoubleTy());
    Ret = RUNTIME_FUNC(__sccrt_reduction_double_init, M);
  }
  Ret->setReturnDoesNotAlias();
  return Ret;
}


static Function *getUpdateFunction(RecurrenceDescriptor &RD) {
  Module *M = RD.getLoopExitInstr()->getModule();
  Type *Tp = RD.getRecurrenceType();

  switch (RD.getMinMaxRecurrenceKind()) {
  case RecurrenceDescriptor::MRK_UIntMin:
    return RUNTIME_FUNC(__sccrt_reduction_uint64_t_min, M);
  case RecurrenceDescriptor::MRK_UIntMax:
    return RUNTIME_FUNC(__sccrt_reduction_uint64_t_max, M);
  case RecurrenceDescriptor::MRK_SIntMin:
    return RUNTIME_FUNC(__sccrt_reduction_int64_t_min, M);
  case RecurrenceDescriptor::MRK_SIntMax:
    return RUNTIME_FUNC(__sccrt_reduction_int64_t_max, M);
  case RecurrenceDescriptor::MRK_FloatMin:
    if (Tp->isFloatTy()) {
      return RUNTIME_FUNC(__sccrt_reduction_float_min, M);
    } else {
      assert(Tp->isDoubleTy());
      return RUNTIME_FUNC(__sccrt_reduction_double_min, M);
    }
  case RecurrenceDescriptor::MRK_FloatMax:
    if (Tp->isFloatTy()) {
      return RUNTIME_FUNC(__sccrt_reduction_float_max, M);
    } else {
      assert(Tp->isDoubleTy());
      return RUNTIME_FUNC(__sccrt_reduction_double_max, M);
    }
  default:
    // Try the kinds below
    break;
  }

  switch (RD.getRecurrenceKind()) {
  case RecurrenceDescriptor::RK_IntegerAdd:
    return RUNTIME_FUNC(__sccrt_reduction_uint64_t_plus, M);
  case RecurrenceDescriptor::RK_IntegerMult:
    return RUNTIME_FUNC(__sccrt_reduction_uint64_t_multiplies, M);
  case RecurrenceDescriptor::RK_IntegerOr:
    return RUNTIME_FUNC(__sccrt_reduction_uint64_t_bit_or, M);
  case RecurrenceDescriptor::RK_IntegerAnd:
    return RUNTIME_FUNC(__sccrt_reduction_uint64_t_bit_and, M);
  case RecurrenceDescriptor::RK_IntegerXor:
    return RUNTIME_FUNC(__sccrt_reduction_uint64_t_bit_xor, M);
  case RecurrenceDescriptor::RK_FloatAdd:
    if (Tp->isFloatTy()) {
      return RUNTIME_FUNC(__sccrt_reduction_float_plus, M);
    } else {
      assert(Tp->isDoubleTy());
      return RUNTIME_FUNC(__sccrt_reduction_double_plus, M);
    }
  case RecurrenceDescriptor::RK_FloatMult:
    if (Tp->isFloatTy()) {
      return RUNTIME_FUNC(__sccrt_reduction_float_multiplies, M);
    } else {
      assert(Tp->isDoubleTy());
      return RUNTIME_FUNC(__sccrt_reduction_double_multiplies, M);
    }
  default:
    llvm_unreachable("Unhandled recurrence kind");
    return nullptr;
  }
}


static void getRecurrenceKind(
        StringRef FName,
        RecurrenceDescriptor::RecurrenceKind *RK,
        RecurrenceDescriptor::MinMaxRecurrenceKind *MK) {
  assert(RK && MK);
  if (!FName.startswith("__sccrt_reduction_")
      || FName.endswith("_collapse")
      || FName.endswith("_init")) {
    *RK = RecurrenceDescriptor::RK_NoRecurrence;
    *MK = RecurrenceDescriptor::MRK_Invalid;
  } else if (FName.equals("__sccrt_reduction_uint64_t_min")) {
    *RK = RecurrenceDescriptor::RK_IntegerMinMax;
    *MK = RecurrenceDescriptor::MRK_UIntMin;
  } else if (FName.equals("__sccrt_reduction_int64_t_min")) {
    *RK = RecurrenceDescriptor::RK_IntegerMinMax;
    *MK = RecurrenceDescriptor::MRK_SIntMin;
  } else if (FName.equals("__sccrt_reduction_float_min")) {
    *RK = RecurrenceDescriptor::RK_FloatMinMax;
    *MK = RecurrenceDescriptor::MRK_FloatMin;
  } else if (FName.equals("__sccrt_reduction_double_min")) {
    *RK = RecurrenceDescriptor::RK_FloatMinMax;
    *MK = RecurrenceDescriptor::MRK_FloatMin;
  } else if (FName.equals("__sccrt_reduction_uint64_t_max")) {
    *RK = RecurrenceDescriptor::RK_IntegerMinMax;
    *MK = RecurrenceDescriptor::MRK_UIntMax;
  } else if (FName.equals("__sccrt_reduction_int64_t_max")) {
    *RK = RecurrenceDescriptor::RK_IntegerMinMax;
    *MK = RecurrenceDescriptor::MRK_SIntMax;
  } else if (FName.equals("__sccrt_reduction_float_max")) {
    *RK = RecurrenceDescriptor::RK_FloatMinMax;
    *MK = RecurrenceDescriptor::MRK_FloatMax;
  } else if (FName.equals("__sccrt_reduction_double_max")) {
    *RK = RecurrenceDescriptor::RK_FloatMinMax;
    *MK = RecurrenceDescriptor::MRK_FloatMax;
  } else if (FName.equals("__sccrt_reduction_uint64_t_plus")) {
    *RK = RecurrenceDescriptor::RK_IntegerAdd;
    *MK = RecurrenceDescriptor::MRK_Invalid;
  } else if (FName.equals("__sccrt_reduction_float_plus")) {
    *RK = RecurrenceDescriptor::RK_FloatAdd;
    *MK = RecurrenceDescriptor::MRK_Invalid;
  } else if (FName.equals("__sccrt_reduction_double_plus")) {
    *RK = RecurrenceDescriptor::RK_FloatAdd;
    *MK = RecurrenceDescriptor::MRK_Invalid;
  } else if (FName.equals("__sccrt_reduction_uint64_t_multiplies")) {
    *RK = RecurrenceDescriptor::RK_IntegerMult;
    *MK = RecurrenceDescriptor::MRK_Invalid;
  } else if (FName.equals("__sccrt_reduction_float_multiplies")) {
    *RK = RecurrenceDescriptor::RK_FloatMult;
    *MK = RecurrenceDescriptor::MRK_Invalid;
  } else if (FName.equals("__sccrt_reduction_double_multiplies")) {
    *RK = RecurrenceDescriptor::RK_FloatMult;
    *MK = RecurrenceDescriptor::MRK_Invalid;
  } else if (FName.equals("__sccrt_reduction_uint64_t_bit_or")) {
    *RK = RecurrenceDescriptor::RK_IntegerOr;
    *MK = RecurrenceDescriptor::MRK_Invalid;
  } else if (FName.equals("__sccrt_reduction_uint64_t_bit_and")) {
    *RK = RecurrenceDescriptor::RK_IntegerAnd;
    *MK = RecurrenceDescriptor::MRK_Invalid;
  } else if (FName.equals("__sccrt_reduction_uint64_t_bit_xor")) {
    *RK = RecurrenceDescriptor::RK_IntegerXor;
    *MK = RecurrenceDescriptor::MRK_Invalid;
  } else {
    llvm_unreachable("Unexpected function name");
  }
}


static Function *getCollapseFunction(RecurrenceDescriptor &RD) {
  Module *M = RD.getLoopExitInstr()->getModule();
  Type *Tp = RD.getRecurrenceType();

  switch (RD.getMinMaxRecurrenceKind()) {
  case RecurrenceDescriptor::MRK_UIntMin:
    return RUNTIME_FUNC(__sccrt_reduction_uint64_t_min_collapse, M);
  case RecurrenceDescriptor::MRK_UIntMax:
    return RUNTIME_FUNC(__sccrt_reduction_uint64_t_max_collapse, M);
  case RecurrenceDescriptor::MRK_SIntMin:
    return RUNTIME_FUNC(__sccrt_reduction_int64_t_min_collapse, M);
  case RecurrenceDescriptor::MRK_SIntMax:
    return RUNTIME_FUNC(__sccrt_reduction_int64_t_max_collapse, M);
  case RecurrenceDescriptor::MRK_FloatMin:
    if (Tp->isFloatTy()) {
      return RUNTIME_FUNC(__sccrt_reduction_float_min_collapse, M);
    } else {
      assert(Tp->isDoubleTy());
      return RUNTIME_FUNC(__sccrt_reduction_double_min_collapse, M);
    }
  case RecurrenceDescriptor::MRK_FloatMax:
    if (Tp->isFloatTy()) {
      return RUNTIME_FUNC(__sccrt_reduction_float_max_collapse, M);
    } else {
      assert(Tp->isDoubleTy());
      return RUNTIME_FUNC(__sccrt_reduction_double_max_collapse, M);
    }
  default:
    // Try the kinds below
    break;
  }

  switch (RD.getRecurrenceKind()) {
  case RecurrenceDescriptor::RK_IntegerAdd:
    return RUNTIME_FUNC(__sccrt_reduction_uint64_t_plus_collapse, M);
  case RecurrenceDescriptor::RK_IntegerMult:
    return RUNTIME_FUNC(__sccrt_reduction_uint64_t_multiplies_collapse, M);
  case RecurrenceDescriptor::RK_IntegerOr:
    return RUNTIME_FUNC(__sccrt_reduction_uint64_t_bit_or_collapse, M);
  case RecurrenceDescriptor::RK_IntegerAnd:
    return RUNTIME_FUNC(__sccrt_reduction_uint64_t_bit_and_collapse, M);
  case RecurrenceDescriptor::RK_IntegerXor:
    return RUNTIME_FUNC(__sccrt_reduction_uint64_t_bit_xor_collapse, M);
  case RecurrenceDescriptor::RK_FloatAdd:
    if (Tp->isFloatTy()) {
      return RUNTIME_FUNC(__sccrt_reduction_float_plus_collapse, M);
    } else {
      assert(Tp->isDoubleTy());
      return RUNTIME_FUNC(__sccrt_reduction_double_plus_collapse, M);
    }
  case RecurrenceDescriptor::RK_FloatMult:
    if (Tp->isFloatTy()) {
      return RUNTIME_FUNC(__sccrt_reduction_float_multiplies_collapse, M);
    } else {
      assert(Tp->isDoubleTy());
      return RUNTIME_FUNC(__sccrt_reduction_double_multiplies_collapse, M);
    }
  default:
    llvm_unreachable("Unhandled recurrence kind");
    return nullptr;
  }
}


static Constant *getRecurrenceIdentity(
        RecurrenceDescriptor::RecurrenceKind RK,
        RecurrenceDescriptor::MinMaxRecurrenceKind MK,
        Type *Tp) {
  IntegerType *ITp = dyn_cast<IntegerType>(Tp);
  ConstantInt *CI;
  switch (MK) {
  case RecurrenceDescriptor::MRK_UIntMin:
    CI = cast<ConstantInt>(Constant::getAllOnesValue(Tp));
    assert(CI->isMaxValue(false));
    return CI;
  case RecurrenceDescriptor::MRK_UIntMax:
    CI = cast<ConstantInt>(ConstantInt::getNullValue(Tp));
    assert(CI->isMinValue(false));
    return CI;
  case RecurrenceDescriptor::MRK_SIntMin:
    CI = ConstantInt::getSigned(ITp, ITp->getSignBit() - 1);
    assert(CI->isMaxValue(true));
    return CI;
  case RecurrenceDescriptor::MRK_SIntMax:
    CI = ConstantInt::getSigned(ITp, ITp->getSignBit());
    assert(CI->isMinValue(true));
    return CI;
  case RecurrenceDescriptor::MRK_FloatMin:
    return ConstantFP::getInfinity(Tp);
  case RecurrenceDescriptor::MRK_FloatMax:
    return ConstantFP::getInfinity(Tp, /*Negative=*/ true);
  default:
    // Try the non-min/max kinds
    return RecurrenceDescriptor::getRecurrenceIdentity(RK, Tp);
  }
}


// RecurrenceDescriptor methods are inexplicably not const, so RD is not const
static Constant *getRecurrenceIdentity(
        RecurrenceDescriptor &RD,
        Type *Tp) {
  assert((!isa<IntegerType>(Tp) ||
          Tp->getIntegerBitWidth() >= RD.getRecurrenceType()
                                        ->getIntegerBitWidth())
         && "The provided type should be a superset of the recurrence type");
  return getRecurrenceIdentity(RD.getRecurrenceKind(),
                               RD.getMinMaxRecurrenceKind(),
                               Tp);
}


ReductionOutputReplacer::ReductionOutputReplacer(
        Loop &L,
        PHINode *Output,
        const DominatorTree &DT,
        OptimizationRemarkEmitter &ORE)
    : L(L), DT(DT), ORE(ORE)
    , Output(Output)
    , ReductionPHI(nullptr)
    , IsInteger(false)
    , M(Output->getModule())
    , Context(M->getContext())
{
  SmallVector<BasicBlock *, 4> Exits;
  L.getExitBlocks(Exits);
  assert(is_contained(Exits, Output->getParent())
         && "Output is not in any of this loop's exit blocks");
}


bool ReductionOutputReplacer::run() {
  if (!setup()) return false;

  // TODO(mcj) Make a copy of the serial loop and offer that alternative path
  // when the loop limit is dynamically observed to be short (and won't spawn
  // work). After all, parallel reductions add non-trivial overheads.

  ORE.emit((RD.getLoopExitInstr()->getDebugLoc()
            ? OptimizationRemark(SR_NAME, "ParallelReduction",
                                 RD.getLoopExitInstr())
            : OptimizationRemark(SR_NAME, "ParallelReduction",
                                 L.getStartLoc(), L.getHeader()))
           << "replacing loop reduction with calls to __sccrt_reduction*.");

  DEBUG(dbgs() << "  reduction phi " << *ReductionPHI
               << "\n  loop exit instr " << *RD.getLoopExitInstr()
               << "\n  which has recurrence kind " << RD.getRecurrenceKind()
               << "\n  and min/max recurrence " << RD.getMinMaxRecurrenceKind()
               << "\n  recurrence type " << *RD.getRecurrenceType()
               << "\n  is signed " << RD.isSigned()
               << "\n");

  // The integer-based __sccrt_reduction functions operate on 64-bit integers.
  // Because of two's complement operations, integer addition, multiplication,
  // and bitwise operations can receive sign- or zero-extended values, as we'll
  // always truncate them to the right results. For example if we add 8-bit
  // numbers
  //   0x04 + 0x80 = 0x84
  // this is equivalent to adding sign-extended 64-bit numbers, and truncating
  // to the 8 LSBs,
  //   0x0000000000000004 + 0xFFFFFFFFFFFFFF80 = 0xFFFFFFFFFFFFFF84
  // and it's also equivalent to adding zero-extended 64-bit numbers, and
  // truncating to the 8 LSBs:
  //   0x0000000000000004 + 0x0000000000000080 = 0x0000000000000084
  //
  // But for signed min/max we must sign-extend

  const bool IsSignedExt = (
          RD.getMinMaxRecurrenceKind() == RecurrenceDescriptor::MRK_SIntMin ||
          RD.getMinMaxRecurrenceKind() == RecurrenceDescriptor::MRK_SIntMax
          );

  // Allocate and serially initialize an sccrt Reducer object
  // TODO(mcj) we may want/need to initialize the thread-private memory
  // in parallel tasks but that introduces annoying timestamp subtleties.
  // These would require tinkering with either the loop's
  // CanonicalIV-to-timestamp mapping (currently identity) or changing the
  // spawn site timestamps
  CallInst *Reducer = nullptr;
  IRBuilder<> B(L.getLoopPreheader()->getTerminator());
  {
    Value *StartValue = !IsInteger
            ? RD.getRecurrenceStartValue().getValPtr()
            : B.CreateIntCast(RD.getRecurrenceStartValue(),
                              Type::getInt64Ty(Context),
                              IsSignedExt,
                              "start_value");
    Value *Identity = !IsInteger
            ? getRecurrenceIdentity(RD, RD.getRecurrenceType())
            : getRecurrenceIdentity(RD, Type::getInt64Ty(Context));

    Function *IF = getInitializationFunction(RD);
    Reducer = B.CreateCall(IF, {StartValue, Identity}, "reducer");
  }

  // Replace the Output PHI node with a call to
  // sccrt's serial reduction collapse
  // TODO(mcj) similarly, we may want to collapse the Reducer in parallel, but
  // that requires outlining the loop continuation, and passing it to the
  // collapse function.
  {
    B.SetInsertPoint(&*Output->getParent()->getFirstInsertionPt());
    Function *CF = getCollapseFunction(RD);
    CallInst *CollapseCall = B.CreateCall(CF, {Reducer}, "reduction_output");
    if (!CollapseCall->getDebugLoc())
      CollapseCall->setDebugLoc(getSafeDebugLoc(L));

    // Possibly truncate the output the collapse function
    Value *Collapsed = !IsInteger
            ? CollapseCall
            : B.CreateIntCast(CollapseCall,
                              Output->getType(),
                              RD.isSigned(),
                              "collapsed");
    Output->replaceAllUsesWith(Collapsed);
    Output->eraseFromParent();
    assert(RD.getLoopExitInstr()->hasOneUse() &&
           (RD.getLoopExitInstr()->user_back() == ReductionPHI) &&
           "Since Output was replaced with a collapse call, "
           "ReductionPHI should be the only user of the LoopExitInstr");
  }

  // Replace the ReductionPHI with the identity for the given recurrence so
  // that the LoopExitInstr will contain an update we can offer to __sccrt
  {
    // Due to possible zext/sext, the ReductionPHI identity might be a
    // different type than the identity we use to initialize the Reducer above
    Constant *Identity = getRecurrenceIdentity(RD, ReductionPHI->getType());
    DEBUG(dbgs() << "  replacing reduction phi with identity "
                 << *Identity << "\n");
    ReductionPHI->replaceAllUsesWith(Identity);
    ReductionPHI->eraseFromParent();
    ReductionPHI = nullptr;
  }

  // Update the Reducer with whatever value the LoopExitInstr holds
  {
    Instruction *LoopExitInstr = RD.getLoopExitInstr();
    Instruction *InsertBefore = isa<PHINode>(LoopExitInstr)
                              ? LoopExitInstr->getParent()->getFirstNonPHI()
                              : LoopExitInstr->getNextNode();
    assert(none_of(L.getSubLoops(), [InsertBefore](const Loop *SubLoop) {
                    return SubLoop->contains(InsertBefore); }) &&
           "Ensure getAnyReductionUpdateCall() looks here when appropriate.");
    B.SetInsertPoint(InsertBefore);
    Value *ReductionUpdate = !IsInteger
            ? RD.getLoopExitInstr()
            : B.CreateIntCast(RD.getLoopExitInstr(),
                              Type::getInt64Ty(Context),
                              IsSignedExt,
                              "reduction_update");
    Value *Timestamp = swarm_abi::createGetTimestampInst(&*B.GetInsertPoint());
    Function *UF = getUpdateFunction(RD);
    UF->setOnlyAccessesArgMemory();
    CallInst *CI = B.CreateCall(UF, {Timestamp, Reducer, ReductionUpdate});
    DEBUG(dbgs() << "  added call to " << CI->getCalledFunction()->getName()
                 << "\n  with update arg " << *ReductionUpdate << "\n");
  }

  DEBUG(assertVerifyFunction(*L.getHeader()->getParent(),
                             "Allocated __sccrt reducer object, "
                             "replaced loop output phi with call to __sccrt, "
                             "replaced reduction phi with identity, and "
                             "passed loop exit instruction to __sccrt update",
                             &DT));
  return true;
}


bool ReductionOutputReplacer::setup() {
  // Find the reduction phi in the loop header whose def-use chain feeds into
  // the given loop Output phi, if any exists.
  bool HasAnyReductionPHIs = false;
  for (PHINode &PN : L.getHeader()->phis()) {
    if (RecurrenceDescriptor::isReductionPHI(&PN, &L, RD)) {
      HasAnyReductionPHIs = true;
      if (is_contained(Output->incoming_values(), RD.getLoopExitInstr())) {
        ReductionPHI = &PN;
        break;
      }
    }
  }
  if (!HasAnyReductionPHIs) {
    debugNotParallelizable("because the header has no reduction phis");
    return false;
  }
  if (!ReductionPHI) {
    debugNotParallelizable("because no reduction phi matched the given output");
    return false;
  }

  if (const Instruction *Unsafe = RD.getUnsafeAlgebraInst()) {
    remarkNotParallelizable("UnsafeAlgebraInst",
                            "because the recurrence has unsafe algebra "
                            "and no flag cleared its parallelization",
                            Unsafe);
    DEBUG(dbgs() << "  Unsafe " << *Unsafe << "\n");
    return false;
  }
  if (RD.isSigned()) {
    remarkNotParallelizable("RedSigned",
                            "because we do not yet know what signed means");
    return false;
  }

  // We can exit early if such a loop exists
  //victory: These are practically guaranteed by LoopSimplify
  assert(L.getLoopPreheader() && "reduction loop lacks dedicated preheader");
  assert(L.getLoopLatch() && "reduction loop has multiple backedges");
  assert(ReductionPHI->getNumIncomingValues() == 2);

  // LoopExitInstr is the value produced after the reduction is evaluated up to
  // the end of a loop iteration. E.g. the value of the partial sum after an
  // iteration's contribution has been added to it. It provides a value downward
  // to the loop output phi and upward to the next iteration's reduction phi.
  Instruction *LoopExitInstr = RD.getLoopExitInstr();
  SmallPtrSet<User *, 2> LoopExitInstrUsers(LoopExitInstr->user_begin(),
                                            LoopExitInstr->user_end());
  assert(LoopExitInstrUsers.count(ReductionPHI));
  assert(LoopExitInstrUsers.count(Output));
  assert(LoopExitInstrUsers.size() == 2);
  if (Output->hasConstantValue() != LoopExitInstr) {
    remarkNotParallelizable("RedNonConstantOutput",
                            "because the loop output phi does not "
                            "unconditionally use the value produced by the "
                            "reduction (aka the loop exit instruction)",
                            LoopExitInstr);
    DEBUG(dbgs() << "  Output phi " << *Output << "\n");
    return false;
  }

  IsInteger =
        RecurrenceDescriptor::isIntegerRecurrenceKind(RD.getRecurrenceKind());
  assert(!IsInteger ||
         RD.getRecurrenceStartValue()->getType()->getIntegerBitWidth() <= 64);
  assert(!IsInteger || LoopExitInstr->getType()->getIntegerBitWidth() <= 64);
  assert(!IsInteger || Output->getType()->getIntegerBitWidth() <= 64);

  return true;
}


void ReductionOutputReplacer::debugNotParallelizable(
        const Twine &Reason) const {
  DEBUG(dbgs() << "The output of " << L << " cannot be replaced "
               << "with a parallel reduction "
               << Reason << ".\n");
}


void ReductionOutputReplacer::remarkNotParallelizable(
        StringRef RemarkName,
        const Twine &Reason,
        const Instruction *Inst) const {
  std::string Msg;
  raw_string_ostream OS(Msg);
  OS << "loop output cannot be replaced with a parallel reduction "
     << Reason << ".\n";
  OS.flush();
  ORE.emit(((Inst && Inst->getDebugLoc())
            ? OptimizationRemark(SR_NAME, RemarkName, Inst)
            : OptimizationRemark(SR_NAME, RemarkName,
                                 L.getStartLoc(), L.getHeader()))
           << Msg);
}


bool llvm::replaceLoopOutputWithReductionCalls(
        Loop &L,
        PHINode *Output,
        const DominatorTree &DT,
        OptimizationRemarkEmitter &ORE) {
  return ReductionOutputReplacer(L, Output, DT, ORE).run();
}


/// If there is any reduction update call associated with L,
/// return it and set *RetRK and *RetMK appropriately.
/// Otherwise, return null and set *RetRK to RK_NoRecurance and
/// *RetMK to MRK_Invalid.
static CallInst *getAnyReductionUpdateCall(const Loop &L,
        RecurrenceDescriptor::RecurrenceKind *RetRK = nullptr,
        RecurrenceDescriptor::MinMaxRecurrenceKind *RetMK = nullptr) {
  if (RetRK) *RetRK = RecurrenceDescriptor::RK_NoRecurrence;
  if (RetMK) *RetMK = RecurrenceDescriptor::MRK_Invalid;

  for (BasicBlock *BB : L.blocks()) {
    if (any_of(L.getSubLoops(), [BB](const Loop *SubLoop) {
            return SubLoop->contains(BB); })) {
      // Don't look for update calls within inner loops
      continue;
    }
    for (Instruction &I : *BB)
      if (CallInst *CI = dyn_cast<CallInst>(&I))
        if (Function *CF = CI->getCalledFunction()) {
          RecurrenceDescriptor::RecurrenceKind RK;
          RecurrenceDescriptor::MinMaxRecurrenceKind MK;
          getRecurrenceKind(CF->getName(), &RK, &MK);
          if (RK != RecurrenceDescriptor::RK_NoRecurrence) {
            DEBUG(dbgs() << "Reduction update call found: " << *CI
                         << "\n in: " << CI->getParent()->getName() << '\n');
            // We must be careful to ignore reduction update calls that
            // belonged to inner loops but have been moved out to this loop by
            // prolog/epilog generation in LoopCoarsen. For example, consider
            // the original loop nest tree:
            //    A
            //    |
            //    B
            // That is, an inner loop B nested within loop A. LoopCoarsen can
            // generate multiple copies of B's body, and put them into a more
            // complex loop nest tree such as:
            //       A
            //    /  |  \
            // B.pro B B.epi
            //       |
            //    B.inner
            // Where all three of B.pro, B.inner, and B.epi might have
            // reduction update calls moved out to their parent loops.
            // We must to avoid considering update calls from B.pro or B.epi
            // as associated with loop A.
            if (!L.isLoopInvariant(CI->getArgOperand(1))) {
              // Argument 1 (2nd argument) is the Reducer object, which is
              // always set up outside of the loop to which it is associated.
              DEBUG(dbgs() << " Ignoring reduction call because it should be "
                              "associated with a nested inner loop, although "
                              "it has been moved out to this loop.\n");
              continue;
            }
            if (RetRK) *RetRK = RK;
            if (RetMK) *RetMK = MK;
            return CI;
          }
        }
  }
  return nullptr;
}


bool llvm::hasReductionCalls(const Loop &L) {
  return !!getAnyReductionUpdateCall(L);
}


static void moveReductionCallAfterLoop(
        Loop &L,
        CallInst *ReductionUpdateCI,
        RecurrenceDescriptor::RecurrenceKind RK,
        RecurrenceDescriptor::MinMaxRecurrenceKind MK) {
  assert(ReductionUpdateCI);
  DEBUG(dbgs() << "  pushing "
               << ReductionUpdateCI->getCalledFunction()->getName()
               << " call after loop\n  "
               << L
               << "  and implanting recurrence of kind "
               << RK << "," << MK << "\n");

  assert(none_of(L.blocks(),
                  [] (BasicBlock *BB) {
                    TerminatorInst *TI = BB->getTerminator();
                    return isa<SDetachInst>(TI) || isa<SReattachInst>(TI);
                  }) &&
         "Unsafe to move an __sccrt_reduction update from a loop "
         "with detaches/reattaches");

  // The third argument to the __sccrt_reduction function is the update value
  Value *ReductionUpdate = ReductionUpdateCI->getArgOperand(2);
  Type *ReductionType = ReductionUpdate->getType();
  Constant *Identity = getRecurrenceIdentity(RK, MK, ReductionType);

  // Set up the reduction phi in the header
  IRBuilder<> Builder(L.getHeader()->getFirstNonPHI());
  PHINode *ReductionPHI = Builder.CreatePHI(ReductionType, 2, "coarse_red_phi");
  ReductionPHI->addIncoming(Identity, L.getLoopPreheader());

  // Create the associative/commutative operation to reduce the update values
  auto Op = static_cast<Instruction::BinaryOps>(
          RecurrenceDescriptor::getRecurrenceBinOp(RK));
  Builder.SetInsertPoint(ReductionUpdateCI);
  Value *ReductionOp = ((RK == RecurrenceDescriptor::RK_IntegerMinMax ||
                         RK == RecurrenceDescriptor::RK_FloatMinMax)
          ? RecurrenceDescriptor::createMinMaxOp(Builder, MK, ReductionPHI,
                                                 ReductionUpdate)
          : Builder.CreateBinOp(Op, ReductionPHI, ReductionUpdate,
                                "coarse_red_op"));
  ReductionPHI->addIncoming(ReductionOp, L.getLoopLatch());

  // Add a loop output phi that consumes the reduction op, move the update call
  // instruction to the exit block, and make the call consume the output value
  BasicBlock *Exit = getUniqueNonDeadendExitBlock(L);
  BasicBlock *Latch = L.getLoopLatch();
  assert(Exit->getUniquePredecessor() == Latch &&
         "This function assumes the caller ensured "
         "there is one way to get to the loop exit");
  Builder.SetInsertPoint(Exit->getFirstNonPHI());
  PHINode *OutputPHI = Builder.CreatePHI(ReductionType, 1, "coarse_red_output");
  OutputPHI->addIncoming(ReductionOp, Latch);
  ReductionUpdateCI->moveBefore(&*Exit->getFirstInsertionPt());
  ReductionUpdateCI->setArgOperand(2, OutputPHI);

  // Verify that we set up a proper reduction
  RecurrenceDescriptor RD;
  bool Success = RecurrenceDescriptor::isReductionPHI(ReductionPHI, &L, RD);
  assert(Success && "Failed to set up a reduction when moving the call");
  assert(RD.getRecurrenceKind() == RK);
  assert(RD.getMinMaxRecurrenceKind() == MK);
  assert(RD.getLoopExitInstr() == ReductionOp);
  assert(RD.getRecurrenceType() == ReductionType);
}


void llvm::moveReductionCallsAfterLoop(Loop &L) {
  RecurrenceDescriptor::RecurrenceKind RK;
  RecurrenceDescriptor::MinMaxRecurrenceKind MK;
  while (CallInst *UpdateCall = getAnyReductionUpdateCall(L, &RK, &MK))
    moveReductionCallAfterLoop(L, UpdateCall, RK, MK);
}
