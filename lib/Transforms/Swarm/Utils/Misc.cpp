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


#include "Misc.h"

#include "Flags.h"
#include "Tasks.h"

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Verifier.h"

using namespace llvm;

#define DEBUG_TYPE "misc"


const ReturnInst *llvm::getUniqueReturnInst(const Function &F) {
  // We're relying on UnifyFunctionExitNodes (a.k.a. -mergereturn) to have
  // produced a single returning block.
  const ReturnInst *Ret = nullptr;
  for (const BasicBlock &BB : F) {
    const TerminatorInst *TI = BB.getTerminator();
    if (const ReturnInst *RI = dyn_cast<ReturnInst>(TI)) {
      assert(!Ret && "Multiple return instructions found");
      Ret = RI;
    }
  }
  return Ret;
}


bool llvm::errorIfAssertSwarmified(const Function &F) {
  if (F.hasFnAttribute(SwarmAttr::AssertSwarmified)) {
    F.getContext().diagnose(DiagnosticInfoUnsupported(
            F,
            "Function has bad __attribute__((assertswarmified))",
            F.getSubprogram()));
    return true;
  }
  return false;
}


void llvm::assertVerifyFunction(const Function &F, const Twine &msg,
                                const DominatorTree *const DT,
                                const LoopInfo *const LI) {
  DEBUG(dbgs() << "Running verification " << msg << '\n');

  if (!hasValidSwarmFlags(F)) {
    dbgs() << F.getName() << "() has some incongruous Swarm attributes:\n";
    F.getAttributes().getFnAttributes().dump();
    llvm_unreachable("Function has incongruous Swarm attributes");
  }

  if (DT) {
    DominatorTree OtherDT;
    OtherDT.recalculate(const_cast<Function&>(F));
    if (DT->compare(OtherDT)) {
      dbgs() << msg << ":\n";
      dbgs() << "DominatorTree is not up to date!\nCurrent:\n";
      DT->print(dbgs());
      dbgs() << "\nRecomputed actual:\n";
      OtherDT.print(dbgs());
      DEBUG(F.viewCFG());
      llvm_unreachable("DominatorTree verification failed");
    }

    for (const BasicBlock &BB : F) {
      if (const auto *InnerDI = dyn_cast<SDetachInst>(BB.getTerminator())) {
        assert(InnerDI->hasTimestamp() || !getDomain(InnerDI));
        if (InnerDI->hasTimestamp()) {
          if (const SDetachInst *OuterDI = getEnclosingTask(InnerDI, *DT)) {
            assert(OuterDI->hasTimestamp());
            const DeepenInst *OuterDomain = getDomain(OuterDI);
            const DeepenInst *InnerDomain = getDomain(InnerDI);
            if (InnerDI->isSuperdomain()) {
              assert(!InnerDomain || DT->dominates(InnerDomain, OuterDI));
              if (!InnerDomain && !OuterDomain) {
                // Who knows?  Is there anything we should assert here?
              } else if (InnerDomain == OuterDomain) {
                assert(getPreceedingDeepen(InnerDI, *DT, OuterDI->getParent()) &&
                       "Must be a superdomain spawn after a deepen");
              } else if (OuterDomain) {
                assert(InnerDomain == OuterDomain->getSuperdomain(*DT));
              }
              continue;
            }
            if (OuterDomain && !InnerDomain &&
                !InnerDI->getMetadata(SwarmFlag::TempNullDomain)) {
              dbgs() << msg << ":\n";
              dbgs() << "Outer detach:" << *OuterDI << "\n";
              dbgs() << "Inner detach:" << *InnerDI << "\n\n";
              dbgs() << F << "\n\n";
              DEBUG(F.viewCFG());
              llvm_unreachable("Inner detach not associated with deepen");
            }
            if (InnerDomain && InnerDomain != OuterDomain &&
                !OuterDI->getMetadata(SwarmFlag::TempNullDomain) &&
                !DT->dominates(OuterDI, InnerDomain)) {
              assert(DT->dominates(InnerDomain, OuterDI));
              dbgs() << msg << ":\n";
              dbgs() << "Outer detach:" << *OuterDI << "\n";
              dbgs() << "Inner detach:" << *InnerDI << "\n\n";
              dbgs() << F << "\n\n";
              DEBUG(F.viewCFG());
              llvm_unreachable("Inner detach associated with outer deepen");
            }
          }
        }
      }
      for (const Instruction &I : BB) {
        if (const auto *Domain = dyn_cast<DeepenInst>(&I)) {
          SmallVector<const SDetachInst *, 8> DeepenedDetaches;
          getOuterDetaches(Domain, *DT, DeepenedDetaches);
          for (const SDetachInst *DI : DeepenedDetaches) {
            if (DI->getMetadata(SwarmFlag::TempNullDomain)) {
              assert(!DI->getDomain());
            } else if (DI->isSuperdomain()) {
              if (DI->getDomain() != Domain->getSuperdomain(*DT)) {
                dbgs() << msg << ":\n";
                dbgs() << "Detach not associated with superdomain after deepen:\n";
                dbgs() << *Domain << '\n' << *DI << "\n\n" << F << "\n\n";
                DEBUG(F.viewCFG());
                llvm_unreachable("Superdomain detach and deepen mismatch");
              }
            } else if (DI->getDomain() != Domain) {
              dbgs() << msg << ":\n";
              dbgs() << "Detach not associated with preceeding deepen:\n";
              dbgs() << *Domain << '\n' << *DI << "\n\n" << F << "\n\n";
              DEBUG(F.viewCFG());
              llvm_unreachable("Detach and deepen mismatch");
            }
          }
        }
      }
    }
  }

  if (LI) {
    assert(DT && "DominatorTree needed to verify LoopInfo");
    //TODO(victory): If this verification fails, we should print out msg?
    LI->verify(*DT);
  }

  if (verifyFunction(F, &dbgs())) {
    dbgs() << msg << ":\n" << F;
    DEBUG(F.viewCFG());
    llvm_unreachable("Function verification failed");
  }
}


Instruction *llvm::createDummyValue(Type *T,
                                    const Twine &Name,
                                    Instruction *InsertBefore) {
  std::string TypeName;
  raw_string_ostream SO(TypeName);
  T->print(SO, false, /*NoDetails=*/true);
  SO.flush();

  Constant *DummyFunc = InsertBefore->getModule()->getOrInsertFunction(
      "__dummy_value_" + TypeName, T);

  return CallInst::Create(DummyFunc, Name, InsertBefore);
}


bool llvm::isSafeToSinkOrHoist(const Instruction *I) {
  // This function's implementation copies from LICM's canSinkOrHoistInst.

  if (const LoadInst *LI = dyn_cast<LoadInst>(I)) {
    // Loads have extra constraints we have to verify before we can hoist them.

    // Don't hoist volatile/atomic loads!
    if (!LI->isUnordered()) return false;

    // Loads from constant memory are always safe to move, even if they end up
    // in the same alias set as something that ends up being modified.
    //if (AA->pointsToConstantMemory(LI->getOperand(0)))
    //  return true;
    if (LI->getMetadata(LLVMContext::MD_invariant_load))
      return true;

    return false;
  } else if (const CallInst *CI = dyn_cast<CallInst>(I)) {
    //victory: It's not clear to me if sinking or hoisting these will make
    // DWARF do a worse job tracking where source local variables are stored,
    // but we shouldn't care unless we're running SCC code through a debugger.
    if (isa<DbgInfoIntrinsic>(I))
      return true;

    // Don't sink calls which can throw.
    if (CI->mayThrow())
      return false;
    return false;
  }

  // Besides loads and calls handled above,
  // only these instructions are hoistable/sinkable.
  if (!isa<BinaryOperator>(I) && !isa<CastInst>(I) && !isa<SelectInst>(I) &&
      !isa<GetElementPtrInst>(I) && !isa<CmpInst>(I) &&
      !isa<InsertElementInst>(I) && !isa<ExtractElementInst>(I) &&
      !isa<ShuffleVectorInst>(I) && !isa<ExtractValueInst>(I) &&
      !isa<InsertValueInst>(I))
    return false;

  // It's safe to move these instructions if they cannot trap.
  // There's no point passing more arguments to isSafeToSpeculativelyExecute()
  // as we already separately handled loads.
  return isSafeToSpeculativelyExecute(I);
}


bool llvm::isInvariantInPath(const Value *Ptr,
                             const Instruction *Begin,
                             const Instruction *End) {
  assert(Begin->getNextNode());
  const BasicBlock *BeginBlock = Begin->getParent();
  const BasicBlock *EndBlock = End->getParent();

  //TODO(victory): It turns out treating all stores as if they might store to
  // Ptr is enough to handle the case I was trying to solve, but what we really
  // want here is to query alias analysis to find out whether any given
  // instruction that might write to a pointer that aliases with Ptr.
  auto mayWriteToPtr = [](const Instruction &I) -> bool {
    // Treat the following as not writing memory:
    // * SCCRT reduction calls, since the memory they write will not alias with
    //   any IR-visible loads or stores.
    if (const auto *CI = dyn_cast<CallInst>(&I))
      if (const Function *F = CI->getCalledFunction())
        if (F->getName().startswith("__sccrt_reduction"))
          return false;

    if (I.mayWriteToMemory()) {
      DEBUG(dbgs() << "  Culprit that may write to memory " << I << "\n");
      return true;
    } else {
      return false;
    }
  };

  // Simple case first: if everything is in the same block:
  if (BeginBlock == EndBlock) {
    return std::none_of(Begin->getNextNode()->getIterator(),
                        End->getIterator(),
                        mayWriteToPtr);
  }

  // If we're looking across different blocks, we need to be careful.

  if (std::any_of(Begin->getNextNode()->getIterator(),
                  BeginBlock->end(),
                  mayWriteToPtr)) {
    DEBUG(dbgs() << "  begin preceeds intervening store?\n");
    return false;
  }
  if (std::any_of(EndBlock->begin(),
                  End->getIterator(),
                  mayWriteToPtr)) {
    DEBUG(dbgs() << "  end follows intervening store?\n");
    return false;
  }

  // Backwards DFS from End to cover all control-flow paths from Begin to End.
  SmallPtrSet<const BasicBlock *, 8> Visited;
  Visited.insert(BeginBlock);
  std::function<bool(const BasicBlock *)> DFSHelper =
          [&Visited, mayWriteToPtr, &DFSHelper](const BasicBlock *BB) {
    if (!Visited.insert(BB).second) return true;
    if (any_of(*BB, mayWriteToPtr)) {
      DEBUG(dbgs() << "  intervening store in intervening block?\n");
      return false;
    }
    return all_of(predecessors(BB), [&DFSHelper](const BasicBlock *Pred) {
      return isa<SReattachInst>(Pred->getTerminator()) || DFSHelper(Pred);
    });
  };
  return DFSHelper(EndBlock);
}


Value *llvm::copyComputation(Value *Expr,
                             const ValueToValueMapTy &BaseMap,
                             IRBuilder<> &Builder,
                             const BasicBlock *PhiPred) {
  assert(Expr);

  if (BaseMap.count(Expr))
    return BaseMap.lookup(Expr);

  if (isa<Constant>(Expr))
    return Expr;
  if (isa<Argument>(Expr))
    return Expr;

  if (auto I = dyn_cast<Instruction>(Expr)) {
    BasicBlock *BB = I->getParent();
    assert(BB && "Instruction not in basic block");

    if (auto PN = dyn_cast<PHINode>(I)) {
      assert(PhiPred && BB == PhiPred->getSingleSuccessor() &&
             "Improper phi node during copying.");
      return copyComputation(PN->getIncomingValueForBlock(PhiPred),
                             BaseMap, Builder, PhiPred);
    }

    DEBUG(dbgs() << "Copying instruction: " << *I << '\n');

    if (auto *LI = dyn_cast<LoadInst>(I)) {
      assert(LI->getMetadata(SwarmFlag::Closure));
      Value *SubExprCopy = copyComputation(LI->getPointerOperand(),
                                           BaseMap, Builder, PhiPred);
      return Builder.CreateLoad(SubExprCopy, "CopiedCompute");
    }

    assert(isSafeToSpeculativelyExecute(I, nullptr, nullptr) &&
           "Expr may trap");

    if (auto CI = dyn_cast<CastInst>(I)) {
      Value *SubExprCopy = copyComputation(CI->getOperand(0),
                                           BaseMap, Builder, PhiPred);
      return Builder.CreateCast(CI->getOpcode(),
                                SubExprCopy,
                                CI->getDestTy(),
                                "CopiedCompute");
    } else if (auto BO = dyn_cast<BinaryOperator>(I)) {
      Value *SubExpr0Copy = copyComputation(BO->getOperand(0),
                                            BaseMap, Builder, PhiPred);
      Value *SubExpr1Copy = copyComputation(BO->getOperand(1),
                                            BaseMap, Builder, PhiPred);
      return Builder.CreateBinOp(BO->getOpcode(),
                                 SubExpr0Copy,
                                 SubExpr1Copy,
                                 "CopiedCompute");
    } else if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
      Value *PtrSubExprCopy = copyComputation(GEP->getPointerOperand(),
                                              BaseMap, Builder, PhiPred);
      SmallVector<Value *, 4> IndexSubExprCopies;
      for (Value *Idx : GEP->indices())
        IndexSubExprCopies.push_back(copyComputation(Idx,
                                                     BaseMap, Builder, PhiPred));
      return Builder.CreateGEP(PtrSubExprCopy, IndexSubExprCopies,
                               "CopiedCompute");
    }
    // TODO(victory): Handle some of the following instruction types:
    // dyn_cast<SelectInst>(I)
    // dyn_cast<CmpInst>(I)
    // dyn_cast<InsertElementInst>(I)
    // dyn_cast<ExtractElementInst>(I)
    // dyn_cast<ShuffleVectorInst>(I)
    // dyn_cast<ExtractValueInst>(I)
    // dyn_cast<InsertValueInst>(I)

    dbgs() << "\nUnhandled instruction to copy: " << *I << '\n';
    dbgs() << "In: " << *BB->getParent();
    llvm_unreachable("Unhandled instruction to copy");
  }

  dbgs() << "\nUnhandled expression to copy: " << *Expr << '\n';
  llvm_unreachable("Unhandled expression to copy");
}


DebugLoc llvm::getSafeDebugLoc(const Loop &L) {
  // Try the loop ID.
  if (const MDNode *LoopID = L.getLoopID()) {
    for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i)
      if (const DILocation *DL = dyn_cast<DILocation>(LoopID->getOperand(i))) {
        if (const DILocalScope *Scope = DL->getInlinedAtScope()) {
          if (const DISubprogram *SP = Scope->getSubprogram())
            if (!SP->describes(L.getHeader()->getParent()))
              continue;
        }
        return DebugLoc(DL);
      }
  }

  // Try the pre-header.
  const BasicBlock *Preheader = L.getLoopPreheader();
  if (Preheader)
    if (DebugLoc DL = Preheader->getTerminator()->getDebugLoc())
      return DL;

  // If we have no pre-header or there are no instructions with debug
  // info in it, try the header.
  const BasicBlock *Header = L.getHeader();
  if (Header)
    if (DebugLoc DL = Header->getTerminator()->getDebugLoc())
      return DL;

  // No luck with the preheader nor with the header? Try the latch.
  const BasicBlock *Latch = L.getLoopLatch();
  if (Latch)
    if (DebugLoc DL = Latch->getTerminator()->getDebugLoc())
      return DL;

  // Try an exhaustive search?
  for (const BasicBlock *BB : L.blocks())
    for (const Instruction &I : *BB)
      if (DebugLoc DL = I.getDebugLoc())
        return DL;
  if (Preheader)
    for (const Instruction &I : *Preheader)
      if (DebugLoc DL = I.getDebugLoc())
        return DL;

  // Try the predecessor?
  if (const BasicBlock *PHeadBB = L.getLoopPredecessor())
    if (DebugLoc DL = PHeadBB->getTerminator()->getDebugLoc())
      return DL;

  return DebugLoc();
}


bool llvm::canStrengthenIV(PHINode *PN, ScalarEvolution &SE) {
  DEBUG(dbgs() << "Can we strengthen? " << *PN << "\n");
  if (!SE.isSCEVable(PN->getType())) {
    DEBUG(dbgs() << " Scalar evolution doesn't understand types such as "
                 << *PN << "\n");
    return false;
  }
  const SCEV *S = SE.getSCEV(PN);
  DEBUG(dbgs() << " SCEV " << *S << "\n"
               << "  has type " << *S->getType() << "\n");
  if (isa<SCEVCouldNotCompute>(S)) {
    DEBUG(dbgs() << " Could not compute scalar evolution of " << *PN << "\n");
    return false;
  } else if (isa<SCEVUnknown>(S)) {
    DEBUG(dbgs() << " Do not know scalar evolution of " << *PN << "\n");
    return false;
  } else if (S->getType() != PN->getType()
             && !(S->getType()->isPointerTy()
                  && PN->getType()->isPointerTy())) {
    // TODO(victory): Understand this case better:
    // can it be handled with bitcasts?
    DEBUG(dbgs() << " Scalar evolution produced an expression of type "
                 << *S->getType()
                 << " for a PHI node of type "
                 << *PN->getType() << "\n");
    return false;
  }
  DEBUG(dbgs() << " Yes!" << "\n");
  return true;
}


static Value *createStrengthenedIV(
        PHINode &PN,
        Instruction *IP,
        ScalarEvolution &SE,
        SCEVExpander &Exp) {
  DEBUG(dbgs() << "Strengthen " << PN << '\n');
  const SCEV *S = SE.getSCEV(&PN);
  Value *NewIV = Exp.expandCodeFor(S, S->getType(), IP);
  assert(NewIV->getType() == S->getType());
  DEBUG(dbgs() << " New IV " << *NewIV << '\n');
  if (NewIV->getType() != PN.getType()) {
    assert(NewIV->getType()->isPointerTy() && PN.getType()->isPointerTy());
    if (auto I = dyn_cast<Instruction>(NewIV)) {
      NewIV = CastInst::CreatePointerCast(NewIV, PN.getType(),
                                          "cast",
                                          I->getNextNode());
    } else {
      Constant *C = cast<Constant>(NewIV);
      NewIV = CastInst::CreatePointerCast(C, PN.getType(), "cast", IP);
    }
    DEBUG(dbgs() << " bitcast New IV " << *NewIV << '\n');
  }
  return NewIV;
}


// Remove the IVs (other than CanonicalIV), if possible, and replace them with
// their stronger forms.
static void strengthenIVs(PHINode *CanonicalIV, BasicBlock *Header,
                          ScalarEvolution &SE, SCEVExpander &Exp) {
  SmallVector<PHINode*, 8> IVsToStrengthen;
  for (PHINode &PN : Header->phis()) {
    if (&PN == CanonicalIV) continue;
    if (canStrengthenIV(&PN, SE)) IVsToStrengthen.push_back(&PN);
  }
  for (PHINode *PN : IVsToStrengthen) {
    Instruction *IP = &*Header->getFirstInsertionPt();
    Value *NewIV = createStrengthenedIV(*PN, IP, SE, Exp);
    PN->replaceAllUsesWith(NewIV);
  }
  for (PHINode *PN : IVsToStrengthen) {
    PN->eraseFromParent();
  }
}


static PHINode *canonicalizeIVs(
        Loop &L,
        unsigned MinBitWidth,
        const DominatorTree &DT,
        ScalarEvolution &SE,
        bool AllOrNothing) {
  BasicBlock *Header = L.getHeader();
  assert(Header);

  // Sharing an Exp with the caller leads to hard-to-track
  // assertion failures and sadness
  SCEVExpander Exp(SE,
                   Header->getParent()
                         ->getParent()
                         ->getDataLayout(),
                   "civs");

  SmallVector<WeakTrackingVH, 16> DeadInsts;
  Exp.replaceCongruentIVs(&L, &DT, DeadInsts);
  for (WeakTrackingVH V : DeadInsts) {
    DEBUG(dbgs() << "Erasing dead inst " << *V << "\n");
    cast<Instruction>(V)->eraseFromParent();
  }

  PHINode *CanonicalIV = L.getCanonicalInductionVariable();
  // NOTE: SCEVExpander::getOrInsertCanonicalInductionVariable is fragile:
  // If a preexisting canonical induction variable has different width than the
  // desired bitwidth, it may 1.) simply generate a trunc or *ext instruction
  // to produce the value of the desired width, and 2.) immediately crash when
  // it attempts to run cast<PHINode>() on the non-phi instruction.
  if (!CanonicalIV) {
    CanonicalIV = Exp.getOrInsertCanonicalInductionVariable(&L,
          IntegerType::get(Header->getContext(), MinBitWidth));
  }
  assert(CanonicalIV->getType()->getIntegerBitWidth() >= MinBitWidth);

  if (AllOrNothing) {
    // mcj is not fond of bool arguments, but it is hidden in this static
    // function, and duplicating much of the functionality here does not seem
    // worthwhile.
    for (PHINode &PN : Header->phis()) {
      if (&PN == CanonicalIV) continue;
      if (!canStrengthenIV(&PN, SE)) return nullptr;
    }
  }

  DEBUG(dbgs() << "Header before canonicalizing IVs:" << *Header);

  strengthenIVs(CanonicalIV, Header, SE, Exp);
  // The strengthening may have added new PHINodes, and may have changed the
  // CanonicalIV, so strengthen again to see if the new nodes are removable.
  CanonicalIV = L.getCanonicalInductionVariable();
  assert(CanonicalIV);
  assert(CanonicalIV->getType()->getIntegerBitWidth() >= MinBitWidth);
  Exp.clear();
  strengthenIVs(CanonicalIV, Header, SE, Exp);

  DEBUG(dbgs() << "Header after canonicalizing IVs:" << *Header);
  return CanonicalIV;
}


PHINode *llvm::canonicalizeIVs(
        Loop &L,
        unsigned BW,
        const DominatorTree &DT,
        ScalarEvolution &SE) {
  return ::canonicalizeIVs(L, BW, DT, SE, false);
}


PHINode *llvm::canonicalizeIVsAllOrNothing(
        Loop &L,
        unsigned BW,
        const DominatorTree &DT,
        ScalarEvolution &SE) {
  PHINode *CanonicalIV = ::canonicalizeIVs(L, BW, DT, SE, true);
  BasicBlock *Header = L.getHeader();
  assert((!CanonicalIV ||
          (CanonicalIV == &Header->front() &&
           !isa<PHINode>(CanonicalIV->getNextNode())
          )
         ) && "CanonicalIV should be the only phi in Header");
  return CanonicalIV;
}


CallInst *llvm::createPrintString(StringRef Str,
                                  const Twine &Name,
                                  Instruction *InsertBefore) {
  IRBuilder<> B(InsertBefore);
  Value *GlobalStr = B.CreateGlobalStringPtr(Str, Name);

  // The following code is copied from BuildLibCalls's emitPutS().
  // We avoid using emitPutS() because it wants to consult a TargetLibraryInfo,
  // but we avoid asking our callers for a TargetLibraryInfo to make it as
  // easy as possible to call this utility to debug error conditions.
  Module *M = B.GetInsertBlock()->getModule();
  Value *PutS =
      M->getOrInsertFunction("puts", B.getInt32Ty(), B.getInt8PtrTy());
  //inferLibFuncAttributes(*M->getFunction("puts"), *TLI);
  CallInst *CI = B.CreateCall(PutS, GlobalStr, "puts");
  if (const Function *F = dyn_cast<Function>(PutS->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}
