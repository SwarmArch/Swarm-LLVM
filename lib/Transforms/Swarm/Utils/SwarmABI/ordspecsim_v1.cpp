//===- ordspecsim_v1.cpp - Simulator ABI interface ------------------------===//
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
// Generating x86_64 inline assembly for the Magic Ops supported by ordspecsim.
//
//===----------------------------------------------------------------------===//

#include "Impl.h"

#include "Utils/Misc.h"

#include "llvm/CodeGen/ValueTypes.h"  // For EVT
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"

#include "swarm_runtime/include/swarm/hooks.h"
#define FROM_PLS_API
#include "swarm_runtime/include/swarm/impl/hwtasks.h" // For swarm::enqueueMagicOp()
#undef FROM_PLS_API

using namespace llvm;

#define DEBUG_TYPE "swarmabi"


CallInst *llvm::oss_v1::createGetTimestampInst(Instruction *InsertBefore,
                                               bool isSuper) {
  assert(InsertBefore);
  IntegerType *Int64Ty = Type::getInt64Ty(InsertBefore->getContext());
  InlineAsm *IA = InlineAsm::get(FunctionType::get(Int64Ty, Int64Ty, false),
                                 "xchg %rcx, %rcx;", "={rcx},{rcx}",
                                 false /* hasSideEffects */);

  auto Opcode = ConstantInt::get(Int64Ty, isSuper ? MAGIC_OP_GET_TIMESTAMP_SUPER
                                                  : MAGIC_OP_GET_TIMESTAMP);
  CallInst *CI = CallInst::Create(IA, Opcode, "timestamp", InsertBefore);
  CI->setDebugLoc(InsertBefore->getDebugLoc());
  return CI;
}

CallInst *llvm::oss_v1::createDeepenInst(Instruction *InsertBefore) {
  assert(InsertBefore);
  LLVMContext &Context = InsertBefore->getContext();
  IntegerType *Int64Ty = Type::getInt64Ty(Context);
  InlineAsm *IA = InlineAsm::get(
      FunctionType::get(Type::getVoidTy(Context), {Int64Ty, Int64Ty}, false),
      "xchg %rcx, %rcx;", "{rcx},{rdi}", true /* hasSideEffects */);

  auto Opcode = ConstantInt::get(Int64Ty, MAGIC_OP_DEEPEN);
  auto MaxTS = ConstantInt::get(Int64Ty, UINT64_MAX);
  CallInst *CI = CallInst::Create(IA, {Opcode, MaxTS}, "", InsertBefore);
  CI->setDebugLoc(InsertBefore->getDebugLoc());
  return CI;
}

CallInst *llvm::oss_v1::createUndeepenInst(Instruction *InsertBefore) {
  assert(InsertBefore);
  LLVMContext &Context = InsertBefore->getContext();
  IntegerType *Int64Ty = Type::getInt64Ty(Context);
  InlineAsm *IA = InlineAsm::get(
      FunctionType::get(Type::getVoidTy(Context), {Int64Ty}, false),
      "xchg %rcx, %rcx;", "{rcx}", true /* hasSideEffects */);

  auto Opcode = ConstantInt::get(Int64Ty, MAGIC_OP_UNDEEPEN);
  CallInst *CI = CallInst::Create(IA, Opcode, "", InsertBefore);
  CI->setDebugLoc(InsertBefore->getDebugLoc());
  return CI;
}

void llvm::oss_v1::optimizeTaskFunction(Function *TaskFn) {
  // Use a dequeue instruction instead of a return
  ReturnInst* RI = getUniqueReturnInst(*TaskFn);
  assert(RI && "Swarm task should have a single return at this point");
  LLVMContext &Context = RI->getContext();
  InlineAsm *IA =
      InlineAsm::get(FunctionType::get(Type::getVoidTy(Context), {}, false),
                     "xchg %rdx, %rdx;", "", true /* hasSideEffects */);

  CallInst *CI = CallInst::Create(IA, None, "", RI);
  CI->setDebugLoc(RI->getDebugLoc());
  // Dequeue magic ops go straight to the dequeue loop, never return
  // victory: The noreturn attribute makes LLVM generate functions that just
  // end with an "xchg %rdx, %rdx" followed by padding and sometimes non-code
  // data past the end of the function.  Pin/our simulator isn't smart enough
  // to recognize the "xchg %rdx, %rdx" as marking the end of a region of valid
  // instructions, leading to an attempt to decode the data past the end of the
  // task function as instructions, causing spurious simulator warnings.  So,
  // for now, don't tell LLVM the "xchg %rdx, %rdx" never returns, and generate
  // a normal function epilog and return instruction at the end of each task
  // function, even though the epilogue and return will never execute.
  //CI->setDoesNotReturn();
  CI->setDoesNotThrow();
  assert(isDequeueInst(CI));

  // Do codegen with zero callee-save registers.
  TaskFn->setCallingConv(CallingConv::X86_64_Swarm);
}

CallInst *llvm::oss_v1::createHeartbeatInst(Instruction *InsertBefore) {
  assert(InsertBefore);
  LLVMContext &Context = InsertBefore->getContext();
  IntegerType *Int64Ty = Type::getInt64Ty(Context);
  InlineAsm *IA = InlineAsm::get(
      FunctionType::get(Type::getVoidTy(Context), {Int64Ty}, false),
      "xchg %rcx, %rcx;", "{rcx}", true /* hasSideEffects */);

  auto Opcode = ConstantInt::get(Int64Ty, MAGIC_OP_HEARTBEAT);
  CallInst *CI = CallInst::Create(IA, Opcode, "", InsertBefore);
  CI->setDebugLoc(InsertBefore->getDebugLoc());
  return CI;
}


CallInst *llvm::oss_v1::createSwarmEnqueue(Function *TaskFn,
                                           ArrayRef<Value *> Args,
                                           SDetachInst *DI) {
  assert(DI->isSubdomain() + DI->isSuperdomain() <= 1);
  assert(DI->hasHint() + DI->isNoHint() + DI->isSameHint() == 1);

  LLVMContext &Context = TaskFn->getContext();
  IntegerType *Int64Ty = Type::getInt64Ty(Context);

  bool hasHint = DI->hasHint();

  SmallVector<Type *, 8> ParamTy;
  {
    ParamTy.push_back(Int64Ty);  // Magic Op
    ParamTy.push_back(Int64Ty);  // Timestamp
    for (const Value *V : Args) {
      Type *T = V->getType();
      assert(isa<PointerType>(T) || isa<IntegerType>(T) &&
             "Arguments should already be cast to a integer or pointer type");
      assert(EVT::getEVT(T).isSimple() &&
             "InlineAsm operands must be simple value types");
      ParamTy.push_back(T);
    }
    ParamTy.push_back(TaskFn->getType());
    if (hasHint)
      ParamTy.push_back(Int64Ty);
  }

  std::stringstream constraints;
  {
    constraints << "{rcx},{rdi},";  // Magic Op and Timestamp
    const constexpr char* registers[] =
        {"{rsi}","{rdx}","{r8}","{r9}","{r10}","{r11}","{r12}"};
    constexpr unsigned maxRegisters = sizeof(registers)/sizeof(registers[0]);
    static_assert(maxRegisters == SIM_MAX_ENQUEUE_REGS + 2, "");
    assert(Args.size() <= SIM_MAX_ENQUEUE_REGS);
    unsigned i = 0;
    while (i < Args.size())
      constraints << registers[i++] << ',';
    constraints << registers[i++];  // Function Pointer:
    if (hasHint)
      constraints << ',' << registers[i++];
  }

  InlineAsm *IA = InlineAsm::get(
      FunctionType::get(Type::getVoidTy(Context), ParamTy, false),
      "xchg %rcx, %rcx;", constraints.str(), true /* hasSideEffects */);

  IRBuilder<> Builder(DI);

  SmallVector<Value *, 8> Inputs;
  {
    Value *MagicOp = nullptr;
    auto domainFlags = EnqFlags(
            (DI->isSubdomain() ? SUBDOMAIN : 0) |
            (DI->isSuperdomain() ? SUPERDOMAIN|PARENTDOMAIN : 0));
    if (DI->hasSameHintCondition()) {
      auto shFlags = EnqFlags(domainFlags | SAMEHINT);
      auto nhFlags = EnqFlags(domainFlags | NOHINT);
      uint64_t shMagicOp = swarm::enqueueMagicOp(Args.size(), shFlags);
      uint64_t nhMagicOp = swarm::enqueueMagicOp(Args.size(), nhFlags);
      MagicOp = Builder.CreateSelect(
              DI->getSameHintCondition(),
              ConstantInt::get(Int64Ty, shMagicOp),
              ConstantInt::get(Int64Ty, nhMagicOp),
              "conditional_samehint_magicop");
    } else {
      auto hintFlags = EnqFlags(
              (DI->isNoHint() ? NOHINT : 0) |
              (DI->isSameHint() ? SAMEHINT : 0));
      auto flags = EnqFlags(domainFlags | hintFlags);
      uint64_t magicOp = swarm::enqueueMagicOp(Args.size(), flags);
      MagicOp = ConstantInt::get(Int64Ty, magicOp);
    }
    Inputs.push_back(MagicOp);
    assert(!DI->isRelativeTimestamp());
    Inputs.push_back(DI->getTimestamp());

    for (Value *Arg : Args)
      Inputs.push_back(Arg);

    Inputs.push_back(TaskFn);
    if (hasHint)
      Inputs.push_back(DI->getHint());
  }
  auto Call = Builder.CreateCall(IA, Inputs);
  Call->setDebugLoc(DI->getDebugLoc());
  return Call;
}


static bool isEnqueueMagicOp(const ConstantInt *CI) {
  return CI->getZExtValue() >= MAGIC_OP_TASK_ENQUEUE_BEGIN &&
         CI->getZExtValue() < MAGIC_OP_TASK_ENQUEUE_END;
}

static bool isEnqueueMagicOp(const SelectInst *SI) {
  const ConstantInt *TV = dyn_cast<ConstantInt>(SI->getTrueValue());
  const ConstantInt *FV = dyn_cast<ConstantInt>(SI->getFalseValue());
  return TV && FV && isEnqueueMagicOp(TV) && isEnqueueMagicOp(FV) &&
         (TV->getZExtValue() & EnqFlags::SAMEHINT) &&
         (FV->getZExtValue() & EnqFlags::NOHINT);
}

bool llvm::oss_v1::isSwarmEnqueueInst(const CallInst *CI) {
  if (CI->getNumArgOperands() < 1)
    return false;
  if (auto *IA = dyn_cast<InlineAsm>(CI->getCalledValue())) {
    if (IA->getAsmString() != "xchg %rcx, %rcx;")
      return false;
    if (auto *MO = dyn_cast<ConstantInt>(CI->getArgOperand(0)))
      return isEnqueueMagicOp(MO);
    if (auto *SI = dyn_cast<SelectInst>(CI->getArgOperand(0)))
      return isEnqueueMagicOp(SI);
    if (auto *ConstExpr = dyn_cast<ConstantExpr>(CI->getArgOperand(0))) {
      // TODO(victory): Investigate this case: how is this happening?
      // Why is there a select whose operands are all static constants,
      // but it hasn't itself been constant-folded to a ConstantInt?
      Instruction *ConstExprAsInst = ConstExpr->getAsInstruction();
      auto SI = dyn_cast<SelectInst>(ConstExprAsInst);
      bool ret = SI && isEnqueueMagicOp(SI);
      ConstExprAsInst->deleteValue();
      return ret;
    }
  }
  return false;
}

const Constant *llvm::oss_v1::getEnqueueFunction(const CallInst *EnqueueI) {
  // isSwarmEnqueueInst guarantees that the MagicOp is either a ConstantInt or a
  // SelectInst. For the latter, the enqueue uses conditional SAMEHINT.
  auto *MagicOp = dyn_cast<ConstantInt>(EnqueueI->getArgOperand(0));
  bool hasHint = !(!MagicOp ||
                   (MagicOp->getZExtValue() & EnqFlags(SAMEHINT | NOHINT)));
  return cast<Constant>(hasHint
          ? EnqueueI->getArgOperand(EnqueueI->getNumArgOperands() - 2)
          : EnqueueI->getArgOperand(EnqueueI->getNumArgOperands() - 1));
}

bool llvm::oss_v1::isGetTimestampInst(const CallInst *CI) {
  if (CI->getNumArgOperands() != 1)
    return false;
  if (auto Opcode = dyn_cast<ConstantInt>(CI->getArgOperand(0)))
    if (auto IA = dyn_cast<InlineAsm>(CI->getCalledValue()))
      return (Opcode->getZExtValue() == MAGIC_OP_GET_TIMESTAMP ||
              Opcode->getZExtValue() == MAGIC_OP_GET_TIMESTAMP_SUPER) &&
             IA->getAsmString() == "xchg %rcx, %rcx;";
  return false;
}

bool llvm::oss_v1::isDeepenInst(const CallInst *CI) {
  if (CI->getNumArgOperands() != 2)
    return false;
  if (auto Opcode = dyn_cast<ConstantInt>(CI->getArgOperand(0)))
    if (auto IA = dyn_cast<InlineAsm>(CI->getCalledValue()))
      return Opcode->getZExtValue() == MAGIC_OP_DEEPEN &&
             IA->getAsmString() == "xchg %rcx, %rcx;";
  return false;
}

bool llvm::oss_v1::isUndeepenInst(const CallInst *CI) {
  if (CI->getNumArgOperands() != 1)
    return false;
  if (auto Opcode = dyn_cast<ConstantInt>(CI->getArgOperand(0)))
    if (auto IA = dyn_cast<InlineAsm>(CI->getCalledValue()))
      return Opcode->getZExtValue() == MAGIC_OP_UNDEEPEN &&
             IA->getAsmString() == "xchg %rcx, %rcx;";
  return false;
}

bool llvm::oss_v1::isDequeueInst(const CallInst *CI) {
  if (CI->getNumArgOperands())
    return false;
  if (CI->mayThrow())
    return false;
  if (auto IA = dyn_cast<InlineAsm>(CI->getCalledValue()))
    return IA->getAsmString() == "xchg %rdx, %rdx;";
  return false;
}

bool llvm::oss_v1::isHeartbeatInst(const CallInst *CI) {
  if (CI->getNumArgOperands() != 1)
    return false;
  if (auto Opcode = dyn_cast<ConstantInt>(CI->getArgOperand(0)))
    if (auto IA = dyn_cast<InlineAsm>(CI->getCalledValue()))
      return Opcode->getZExtValue() == MAGIC_OP_HEARTBEAT &&
             IA->getAsmString() == "xchg %rcx, %rcx;";
  return false;
}
