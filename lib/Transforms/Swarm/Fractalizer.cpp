//===- Fractalizer.cpp - Split code into tasks and domains ----------------===//
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
// Break up the IR into fine-grain tasks in Fractal domains.
//
//===----------------------------------------------------------------------===//

#include "CreateParallelizableCopy.h"
#include "LoopIterDetacher.h"
#include "Utils/CFGRegions.h"
#include "Utils/Flags.h"
#include "Utils/Misc.h"
#include "Utils/SCCRT.h"
#include "Utils/Tasks.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/SwarmAA.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Swarm.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;

#define PASS_NAME "fractalization"
#define DEBUG_TYPE PASS_NAME

STATISTIC(FunctionsFullyFractalized,
          "Number of functions fractalized");
STATISTIC(FunctionsIncompletelyFractalized,
          "Number of functions not fully fractalized");
STATISTIC(FunctionsNotFractalized,
          "Number of functions without parallelization opportunities");

static cl::opt<unsigned> EagerOutlineThreshold("swarm-eageroutlinethreshold",
        cl::init(150),
        cl::desc("Minimum number of basic blocks duplicated that will "
                 "trigger eager task outlining during fractalization"));

static cl::opt<bool> DelayCallCont("swarm-delaycallcont", cl::init(false),
        cl::desc("Pass the continuation of a detached function call "
                 "whose return value is unused to the task, delaying the "
                 "continuation's enqueue"));

static cl::opt<bool> DisableIndirectSpawn("swarm-disableindirectspawn", cl::init(false),
        cl::desc("Serialize indirect function calls"));

static cl::opt<uint64_t> LongMemoryOperationThreshold(
        "swarm-longmemopthreshold",
        cl::init(16),
        cl::desc("Minimum number of cache lines for a mem{set,cpy,move} "
                 "that is deemed long enough to spawn"));


// Forward declarations.

namespace {


enum class SpawnSiteStatus {
  NoneFound,
  AllDetached,
  FailedDetaching,
  FailedTopoSort,
};

class Fractalizer {
private:
  Function &F;

  // Used only for assertions and debug output.
  const unsigned EarlyChainLoopLevel;

  AssumptionCache &AC;
  DominatorTree &DT;
  LoopInfo &LI;
  TargetLibraryInfo &TLI;
  TargetTransformInfo &TTI;
  OptimizationRemarkEmitter &ORE;

  Module &M;
  LLVMContext &Context;
  ReturnInst *const Return;
  IntegerType *const StaticTimestampTy;

public:
  Fractalizer(Function &F,
              AssumptionCache &AC, DominatorTree &DT, LoopInfo &LI,
              TargetLibraryInfo &TLI, TargetTransformInfo &TTI,
              OptimizationRemarkEmitter &ORE,
              unsigned EarlyChainLoopLevel = 0)
      : F(F), EarlyChainLoopLevel(EarlyChainLoopLevel),
        AC(AC), DT(DT), LI(LI), TLI(TLI), TTI(TTI), ORE(ORE),
        M(*F.getParent()), Context(F.getContext()),
        Return(getUniqueReturnInst(F)),
        StaticTimestampTy(Type::getInt32Ty(Context)) {
    LibFunc LF; (void) LF;
  }

  SpawnSiteStatus fractalize();

private:
  SpawnSiteStatus processWithinLoop(Loop *L, bool MustCreateDomain = false);

  BasicBlock *getEndOfCanonicalLoopBody(Loop *L) const;
  BasicBlock *sanitizeEndOfCanonicalLoopBody(Loop *L);

  bool shouldSpawn(const Loop *) const;

  SpawnSiteStatus processLoop(Loop *L,
                              BasicBlock *ContinuationEnd,
                              unsigned timestamp,
                              DeepenInst *Domain,
                              bool OnlySpawnsite);
  bool processCall(CallInst *CI,
                   BasicBlock *ContinuationEnd,
                   unsigned timestamp,
                   DeepenInst *Domain);
  void processStore(StoreInst *SI, BasicBlock *ContinuationEnd,
                    unsigned timestamp, DeepenInst *Domain);
  unsigned processTaskAnnotation(SDetachInst *DI, BasicBlock *ContinuationEnd,
                                 unsigned timestamp, DeepenInst *Domain);
  void spawnCallAndPassContinuation(CallInst *CI,
                                    BasicBlock *ContinuationEnd,
                                    unsigned timestamp,
                                    DeepenInst *Domain);
  void spawnCallAndParallelContinuation(CallInst *CI,
                                        BasicBlock *ContinuationEnd,
                                        unsigned timestamp,
                                        DeepenInst *Domain);
  CallInst *replaceWithParallelCall(CallInst *SCI,
                                    Value *ContClosure = nullptr);
  static CallInst *replaceIntrinsicWithSubstitute(CallInst *CI);
  bool shouldSpawn(const CallInst *) const;
  bool cannotSpawn(const CallInst *) const;
  static bool shouldSpawnParallelContinuation(const CallInst *);
  static bool shouldSpawn(const StoreInst *);

  void detachContinuationTask(Instruction *ContStart,
                              BasicBlock *ContEnd,
                              Value *Timestamp,
                              DeepenInst *Domain,
                              DetachKind Kind,
                              const Twine &Name = Twine(),
                              SDetachInst **rDI = nullptr,
                              SReattachInst **rRI = nullptr);
};


enum class SpawnType { Loop, Call, Annotation, Store };
struct SpawnSite {
  unsigned timestamp;
  union {
    Loop *L;
    CallInst *CI;
    SDetachInst *DI;
    StoreInst *SI;
  };
  SpawnType Type;

  SpawnSite(unsigned ts, Loop *L_) :
      timestamp(ts), L(L_), Type(SpawnType::Loop) {}
  SpawnSite(unsigned ts, CallInst *C) :
      timestamp(ts), CI(C), Type(SpawnType::Call) {}
  SpawnSite(unsigned ts, SDetachInst *D) :
      timestamp(ts), DI(D), Type(SpawnType::Annotation) {}
  SpawnSite(unsigned ts, StoreInst *S) :
      timestamp(ts), SI(S), Type(SpawnType::Store) {}
};


} // anonymous namespace

static BasicBlock *getCallBlockForPassedContinuation(Function &F);
static void retargetToSuperdomain(SDetachInst *DI,
                                  DeepenInst *EnclosingDomain,
                                  DominatorTree &DT,
                                  LoopInfo *LI = nullptr);

// End forward declarations.


raw_ostream &operator<<(raw_ostream &OS, SpawnSiteStatus s) {
  switch(s) {
  case SpawnSiteStatus::NoneFound:
    OS << "no spawn sites found";
    break;
  case SpawnSiteStatus::AllDetached:
    OS << "all spawn sites detached";
    break;
  case SpawnSiteStatus::FailedDetaching:
    OS << "failed detaching spawn sites";
    break;
  case SpawnSiteStatus::FailedTopoSort:
    OS << "failed topological sort";
    break;
  }
  return OS;
}


raw_ostream &operator<<(raw_ostream &OS, const SpawnSite &SS) {
  switch(SS.Type) {
  case SpawnType::Call:
    OS << "CallSpawn @ " << SS.timestamp << ": " << *SS.CI;
    break;
  case SpawnType::Loop:
    OS << "LoopSpawn @ " << SS.timestamp << ": " << *SS.L;
    break;
  case SpawnType::Annotation:
    OS << "Annotation @ " << SS.timestamp << ": " << *SS.DI;
    break;
  case SpawnType::Store:
    OS << "StoreSpawn @ " << SS.timestamp << ": " << *SS.SI;
    break;
  }
  return OS;
}


//TODO: Move this to SwarmUtils if it proves useful.
/// A wrapper for LLVM's SplitBlockAndInsertIfThenElse()
/// that can update a DominatorTree and a LoopInfo.
/// \returns the terminators of the newly created Then and Else blocks.
static std::pair<TerminatorInst *, TerminatorInst *> splitBlockAndInsertIfThenElse(
        Value *Cond, Instruction *SplitBefore,
        DominatorTree *DT = nullptr, LoopInfo *LI = nullptr) {
  BasicBlock *PrevBlock = SplitBefore->getParent();

  TerminatorInst *ThenTerm, *ElseTerm;
  SplitBlockAndInsertIfThenElse(Cond, SplitBefore,
                                &ThenTerm, &ElseTerm);

  if (DT) {
    DomTreeNode *OldNode = DT->getNode(PrevBlock);
    assert(OldNode);
    SmallVector<DomTreeNode *, 4> Children(OldNode->begin(), OldNode->end());

    DomTreeNode *NewNode = DT->addNewBlock(SplitBefore->getParent(), PrevBlock);

    for (DomTreeNode *Child : Children)
      DT->changeImmediateDominator(Child, NewNode);

    DT->addNewBlock(ThenTerm->getParent(), PrevBlock);
    DT->addNewBlock(ElseTerm->getParent(), PrevBlock);
  }

  if (LI) {
    if (Loop *L = LI->getLoopFor(PrevBlock)) {
      L->addBasicBlockToLoop(ThenTerm->getParent(), *LI);
      L->addBasicBlockToLoop(ElseTerm->getParent(), *LI);
      L->addBasicBlockToLoop(SplitBefore->getParent(), *LI);
    }
  }

  return std::make_pair(ThenTerm, ElseTerm);
}


// TODO(victory): Dedupe with the copy in LoopIterDetacher.cpp
static void finishFractalization(Function &F) {
  assert(all_of(F, [](const BasicBlock &BB) {
    const auto *DI = dyn_cast<SDetachInst>(BB.getTerminator());
    return !DI || DI->hasTimestamp();
  }));
  assert(hasValidSwarmFlags(F));
  StringRef From = SwarmFlag::Parallelizing;
  assert(F.hasFnAttribute(From));
  F.removeFnAttr(From);
  F.addFnAttr(SwarmFlag::Parallelized);
  // Make sure not to panic if we re-encounter some outlined portion of F
  F.removeFnAttr(SwarmAttr::AssertSwarmified);
  assert(hasValidSwarmFlags(F));
}


/// Return false if used for something other than a hint
static bool isOnlyUsedForHint(const Instruction *I) {
  if (I->mayHaveSideEffects()) return false;
  if (isa<TerminatorInst>(I)) return false;
  if (isa<PHINode>(I)) return false;
  return all_of(I->uses(), [](const Use &U) {
    if (const auto *DI = dyn_cast<SDetachInst>(U.getUser()))
      //FIXME(victory): Ugh, depending on the hint being in position 1 for now
      return U.getOperandNo() == 1;
    else
      return isOnlyUsedForHint(cast<Instruction>(U.getUser()));
  });
}


// A version of detachTask that handles potentially non-dominated detached
// regions and reattach edges by possibly duplicating some blocks and
// possibly creating a new dedicated reattach destination.
void Fractalizer::detachContinuationTask(Instruction *ContStart,
                                         BasicBlock *ContEnd,
                                         Value *Timestamp,
                                         DeepenInst *Domain,
                                         DetachKind Kind,
                                         const Twine &Name,
                                         SDetachInst **rDI,
                                         SReattachInst **rRI) {
  BasicBlock *ContStartBB = ContStart->getParent();
  assert(ContStartBB != ContEnd);

  // Detaching this region is pointless if all memory accesses within this
  // region are already detached.  For example, if the continuation of this
  // call was a previously detached call, then they can be detached (spawned)
  // "in parallel" from the parent task.  To determine if this is the case, we
  // check if no instructions in the continuation access memory between the
  // start of the continuation and the first detach along each control flow
  // path in the continutaion.  All memory accesses after any detaches were the
  // responsibility of those later spawnsites to have already detached.
  SmallPtrSet<const Instruction *, 32> Visited;
  std::function<const Instruction *(const Instruction *)>
          getMemAccessBeforeDetach =
      [ContEnd, &Visited, &getMemAccessBeforeDetach](const Instruction *I)
              -> const Instruction * {
        if (!Visited.insert(I).second) return nullptr;
        if (isa<SDetachInst>(I)) return nullptr;
        if (I->mayReadOrWriteMemory() && !isOnlyUsedForHint(I))
          return I;
        if (const auto *TI = dyn_cast<TerminatorInst>(I)) {
          assert(!isa<SReattachInst>(TI));
          assert(isa<BranchInst>(TI) || isa<SwitchInst>(TI)
                 || isa<IndirectBrInst>(TI) || isa<UnreachableInst>(TI));
          // TODO(victory): LLVM version 8 has successors(Instruction *)
          // removing the need for this getParent() call.
          for (const BasicBlock *Succ : successors(TI->getParent())) {
            if (Succ == ContEnd) continue;
            if (const auto *Ret = getMemAccessBeforeDetach(&Succ->front()))
              return Ret;
          }
          return nullptr;
        }
        return getMemAccessBeforeDetach(I->getNextNode());
      };
  if (const Instruction *MemoryAccess = getMemAccessBeforeDetach(ContStart)) {
    DEBUG(dbgs() << "Continuation contains memory accesses such as:\n  "
                 << *MemoryAccess << '\n');
  } else {
    DEBUG(dbgs() << "Continuation lacks non-detached memory accesses.\n"
                    "Not bothering to detach continuation.\n");
    if (rDI)
      *rDI = nullptr;
    if (rRI)
      *rRI = nullptr;
    return;
  }

  DEBUG(dbgs() << "Cloning basic blocks to prepare for detaching continuation\n");
  DEBUG(dbgs() << " starting at detach point " << *ContStart << '\n');
  DEBUG(dbgs() << " and cloning up to but not including "
               << ContEnd->getName() << '\n');
  ContStartBB = SplitBlock(ContStartBB, ContStart, &DT, &LI);
  SmallSetVector<BasicBlock *, 8> ReachableBBs;
  unsigned CopiedBlocks;
  BasicBlock *ReattachingBlock = makeReachableDominated(ContStartBB,
                                                        ContEnd,
                                                        DT, LI,
                                                        ReachableBBs,
                                                        &CopiedBlocks);
  assert(ReattachingBlock);
  DEBUG(dbgs() << CopiedBlocks
               << " blocks copied to make continuation dominated\n");
  // We are finally ready for detaching

  SDetachInst *DI = nullptr;
  if (CopiedBlocks < EagerOutlineThreshold) {
    DEBUG(dbgs() << "Detaching copied region\n");
    detachTask(ContStart,
               ReattachingBlock->getTerminator(),
               Timestamp,
               Domain,
               Kind,
               Name,
               &DT,
               &LI,
               &DI,
               rRI);
  } else {
    // To avoid an exponential blowup in code size,
    // sometimes eagerly outline the task code
    ORE.emit(OptimizationRemark(PASS_NAME, "EagerOutlining", ContStart)
             << "Eagerly outlining large task in " << F.getName() << "().\n");
    SetVector<Value *> Inputs;
    findInputsNoOutputs(ReachableBBs, Inputs);
    Function *ContFunc = outline(Inputs, ReachableBBs, ContStartBB, ".eager");
    ContFunc->setCallingConv(CallingConv::C);
    finishFractalization(*ContFunc);
    assertVerifyFunction(*ContFunc, "Eagerly outlined continuation task");
    DEBUG(dbgs() << "Eagerly outlined task " << ContFunc->getName() << "()\n");

    auto Call = CallInst::Create(ContFunc, Inputs.getArrayRef(), "", ContStart);
    Call->setDebugLoc(ContStart->getDebugLoc());

    SplitBlock(Call->getParent(), ContStart, &DT, &LI);
    auto AfterCall = cast<BranchInst>(Call->getNextNode());
    AfterCall->setSuccessor(0, ContEnd);
    eraseDominatorSubtree(ContStart->getParent(), DT, &LI);
    detachTask(Call, AfterCall, Timestamp, Domain, Kind, Name,
               &DT, &LI, &DI, rRI);
  }
  assert(DI->getDebugLoc() || !F.getSubprogram());

  if (rDI)
    *rDI = DI;
}


BasicBlock *Fractalizer::getEndOfCanonicalLoopBody(Loop *L) const {
  if (L) {
    SReattachInst *Reattach = getUniqueSReattachIntoLatch(*L);
    assert(Reattach);
    return Reattach->getParent();
  } else {
    return Return->getParent();
  }
}

BasicBlock *Fractalizer::sanitizeEndOfCanonicalLoopBody(Loop *L) {
  BasicBlock *BodyEnd = getEndOfCanonicalLoopBody(L);
  if (BodyEnd->size() != 1) {
    BodyEnd = SplitBlock(BodyEnd, BodyEnd->getTerminator(), &DT, &LI);
    BodyEnd->setName((L ? L->getHeader()->getName() : F.getName()) + ".end");
  }
  return BodyEnd;
}


// For now this includes all exception-handling code
static bool hasUnexpectedExceptionHandling(const BasicBlock &BB) {
  const TerminatorInst *TI = BB.getTerminator();
  return BB.isEHPad()
          || isa<CatchReturnInst>(TI)
          || isa<CatchSwitchInst>(TI)
          || isa<CleanupReturnInst>(TI)
          || isa<InvokeInst>(TI)
          || isa<ResumeInst>(TI)
          ;
}


static bool isUnspawnableInstruction(const Instruction *I) {
  // victory: We were previously scared of inline assembly that likely signified
  // swarm::enqueue() calls. These days, those don't seem like a real issue,
  // but our inability to insert simple zsim_heartbeat() calls into serial code
  // without destroying autoparallelization is a real issue.
  //if (const auto CI = dyn_cast<CallInst>(I))
  //  if (const auto IA = dyn_cast<InlineAsm>(CI->getCalledValue()))
  //    if (IA->hasSideEffects())
  //      return true;

  // TODO(mcj) Add exception-handling instructions here
  return false;
}


static const Instruction *getAnyUnspawnableInstruction(const Loop *L) {
  for (const BasicBlock *BB : L->blocks())
    for (const Instruction &I : *BB)
      if (isUnspawnableInstruction(&I))
        return &I;
  return nullptr;
}


static bool isRuntimeSpawnCall(const CallInst *CI) {
  StringRef FName = CI->getCalledValue()->getName();
  return (FName.startswith("_ZN3pls") || FName.startswith("_ZN5swarm")) &&
         !(CI->getCalledFunction() &&
           CI->getCalledFunction()->hasFnAttribute(SwarmAttr::NoSwarmify));
}


// If L is null, fractalize the entire function.
// Otherwise, assume L is a canonicalized loop, fractalize within its body.
//
// processWithinLoop and processLoop use mutual recursion to walk loop nests
// in pre-order.
SpawnSiteStatus Fractalizer::processWithinLoop(Loop *L, bool MustCreateDomain) {
  const DiagnosticLocation &DLoc = L ? DiagnosticLocation(L->getStartLoc())
                                     : DiagnosticLocation(F.getSubprogram());
  const BasicBlock *const CodeRegion = L ? L->getHeader() : &F.getEntryBlock();
  const unsigned LoopLevel = EarlyChainLoopLevel + (L ? L->getLoopDepth() : 0U);

  DEBUG(dbgs() << "Processing region starting at "
               << DLoc.getFilename() << ':'
               << DLoc.getLine() << ':' << DLoc.getColumn() << '\n');
  DEBUG(dbgs() << " at loop nesting level " << LoopLevel << "...\n");

  BasicBlock *const EntryBlock = (L)
          ? cast<SDetachInst>(L->getHeader()->getTerminator())->getDetached()
          : &F.getEntryBlock();
  DEBUG(dbgs() << "Entry block is: " << EntryBlock->getName() << '\n');

  BasicBlock *const EndBlock = sanitizeEndOfCanonicalLoopBody(L);
  DEBUG(dbgs() << "End block is: " << EndBlock->getName() << '\n');

  SpawnSiteStatus Ret = SpawnSiteStatus::NoneFound;

  //Get a topological order on basic blocks and subloops.
  SmallVector<BasicBlock *, 8> SortedBBs;
  topologicalSort(EntryBlock, EndBlock, LI, SortedBBs);
  if (SortedBBs.empty()) {  // Topological sort failure.
    ORE.emit(DiagnosticInfoOptimizationFailure(PASS_NAME, "TopologicalFailure",
                                               DLoc, CodeRegion)
             << "Topological sort failed due to irreducible control flow.");
    Ret = SpawnSiteStatus::FailedTopoSort;
  }

  // Use the topological order to assign timestamps.
  // The subdomain creator notionally occupies timestamp 0,
  // spawnsites are assigned odd timestamps,
  // and their continuations may run at even timestamps.
  SmallVector<SpawnSite, 8> SpawnSites;
  unsigned Timestamp = 1;
  for (auto BB : SortedBBs) {
    Loop *Loop = LI.getLoopFor(BB);
    if (Loop != L) {
      assert(LI.isLoopHeader(BB) && Loop->getHeader() == BB
             && Loop->getParentLoop() == L);
      if (shouldSpawn(Loop)) {
        SpawnSites.push_back(SpawnSite(Timestamp, Loop));
        Timestamp += 2;
        if (const Instruction *I = getAnyUnspawnableInstruction(Loop)) {
          DEBUG(dbgs() << "Found unspawnable instruction: " << *I << '\n');
          ORE.emit(OptimizationRemark(PASS_NAME, "LoopDropSpawnSites", I)
                   << "Dropping spawn sites that precede loop with "
                      "an instruction we don't feel safe copying\n");
          SpawnSites.clear();
          Ret = SpawnSiteStatus::FailedDetaching;
        }
      } else {
        ORE.emit(OptimizationRemarkAnalysis(PASS_NAME, "ShouldNotSpawnLoop",
                                            Loop->getStartLoc(), Loop->getHeader())
                 << "Ignoring loop that is not worth spawning. (Level "
                 << ore::NV("LoopLevel", LoopLevel) << ")");
      }
    } else {
      assert(!LI.isLoopHeader(BB));
      for (Instruction &I : *BB) {
        if (auto CI = dyn_cast<CallInst>(&I)) {
          DEBUG({
            if (!isa<DbgInfoIntrinsic>(CI))
              dbgs() << "Deciding whether to spawn: " << *CI << '\n';
          });
          if (shouldSpawn(CI)) {
            DEBUG(dbgs() << " should spawn.\n");
            SpawnSites.push_back(SpawnSite(Timestamp, CI));
            Timestamp += 2;
          }

          if (isRuntimeSpawnCall(CI)) {
            DEBUG(dbgs() << " must create domain for call: " << *CI << "\n");
            MustCreateDomain = true;
          }
        } else if (auto SI = dyn_cast<StoreInst>(&I)) {
          if (shouldSpawn(SI)) {
            // We will later ignore store spawns that do not precede a spawn
            // site with potential parallelism (like a loop or call)
            SpawnSites.push_back(SpawnSite(Timestamp, SI));
            Timestamp += 2;
          }
        } else if (auto DI = dyn_cast<SDetachInst>(&I)) {
          DEBUG(dbgs() << "Encountered existing detach: " << *DI << "\n");
          assert((getDetachKind(DI) == DetachKind::CallPassedCont ||
                  getDetachKind(DI) == DetachKind::SubsumedCont ||
                  getDetachKind(DI) == DetachKind::RetargetSuperdomain ||
                  getDetachKind(DI) == DetachKind::EarlyChainIter) ==
                 DI->hasTimestamp());
          if (!DI->hasTimestamp()) {
            DEBUG(dbgs() << "Will spawn manually annotated task region: "
                         << *DI << "\n");
            assert(getDetachKind(DI) == DetachKind::Unknown);
            SpawnSites.push_back(SpawnSite(Timestamp, DI));
            SmallVector<BasicBlock *, 8> SpawnedBlocks;
            DT.getDescendants(DI->getDetached(), SpawnedBlocks);
            Timestamp +=
                2 * (1 + count_if(SpawnedBlocks, [](const BasicBlock *BB) {
                       auto *DI = dyn_cast<SDetachInst>(BB->getTerminator());
                       assert(!(DI && DI->hasTimestamp()) &&
                              "Nested detaches should be manual annotations");
                       return !!DI;
                     }));
          }
        }
        if (isUnspawnableInstruction(&I) && !SpawnSites.empty()) {
          DEBUG(dbgs() << "Found unspawnable instruction: " << I << '\n');
          ORE.emit(OptimizationRemarkMissed(PASS_NAME, "DropSpawnSites", &I)
                   << "Dropping spawn sites that precede an instruction "
                      "we don't feel safe copying\n");
          SpawnSites.clear();
          Ret = SpawnSiteStatus::FailedDetaching;
        }
      }
    }
  }

  if (all_of(SpawnSites,
             [](const SpawnSite &SS) { return SS.Type == SpawnType::Store; })) {
    // Since we will ignore all these spawn sites, no point creating a domain.
    SpawnSites.clear();
  }

  if (SpawnSites.empty() && !MustCreateDomain) {
    DEBUG(dbgs() << "No fractalization opportunity found.\n");
  } else {
    // Create a domain for the tasks with static timestamps.
    DeepenInst *Deepen =
            DeepenInst::Create("bodydomain", &*EntryBlock->getFirstInsertionPt());
    assert(EndBlock->size() == 1);
    auto *Undeepen = UndeepenInst::Create(Deepen, EndBlock->getTerminator());
    DEBUG(dbgs() << "Created domain: " << *EntryBlock);
    // Since we just deepened, we must modify any existing
    // detaches to compensate, sending them back to the superdomain.
    SmallVector<SDetachInst *, 8> PreexistingDetaches;
    if (L) { // Inside a canonicalized loop
      DEBUG(dbgs() << " Retargeting subsumed continuation spawns to superdomain.\n");
      getOuterDetaches(cast<SDetachInst>(L->getHeader()->getTerminator()),
                       DT,
                       PreexistingDetaches);
      assert(all_of(PreexistingDetaches, [](const SDetachInst *DI) {
        return DI->isSuperdomain(); }));
    } else {
      getOuterDetaches(&F, DT, PreexistingDetaches);
#ifndef NDEBUG
      if (!!EarlyChainLoopLevel) {
        // Two types of detaches exist: the single detached recursive call,
        // and the continuation detach(s) that are already superdomain
        // detaches, but need to be wrapped now to go a level up again.
        SDetachInst *RecurDetach =
            *findUnique(PreexistingDetaches,
                        [](SDetachInst *DI) { return !DI->isSuperdomain(); });
        assert(getDetachKind(RecurDetach) == DetachKind::EarlyChainIter);
        assert(cast<CallInst>(RecurDetach->getDetached()->front())
                   .getCalledFunction() == &F);
        assert(all_of(PreexistingDetaches, [RecurDetach](SDetachInst *DI) {
          return DI == RecurDetach || DI->isSuperdomain();
        }));
        DEBUG(dbgs() << " Retargeting spawns of early chain's next iteration"
                        " and continuation(s) back up to the superdomain.\n");
      } else {
        // We are just beginning to fractalize an entire original function.
        if (const BasicBlock *ContCallBlock =
                getCallBlockForPassedContinuation(F)) {
          DEBUG(dbgs() << " Retargeting passed continuation to superdomain.\n");
          assert(PreexistingDetaches.size() == 1);
          const SDetachInst *ExistingDetach = PreexistingDetaches[0];
          assert(ExistingDetach->getDetached() == ContCallBlock &&
                 "Continuation spawn created by CPC is only spawn");
          assert(!ExistingDetach->isSuperdomain());
          assert(getDetachKind(ExistingDetach) == DetachKind::CallPassedCont);
        } else {
          DEBUG(dbgs() << " Function has no continuation spawn.\n");
          assert(PreexistingDetaches.empty());
        }
      }
#endif // NDEBUG
    }
    for (SDetachInst *DI : PreexistingDetaches) {
      retargetToSuperdomain(DI, Deepen->getSuperdomain(DT), DT, &LI);
      // TODO(victory): Relax this condition to allow loop nests with subsumed
      // continuations to use only one domain level per loop level.
      MustCreateDomain = true;
    }
    DEBUG(assertVerifyFunction(F,
            "After retargeting existing detaches to superdomain", &DT, &LI));

    if (Ret == SpawnSiteStatus::NoneFound)
      Ret = SpawnSiteStatus::AllDetached;

    // Finally, use the timestamps and detach the code.
    bool HighPotentialParallelism = false;
    for (SpawnSite &SS : reverse(SpawnSites)) {
      switch (SS.Type) {
      case SpawnType::Loop:
        switch (processLoop(SS.L, EndBlock, SS.timestamp, Deepen,
                            (SpawnSites.size() == 1) && !MustCreateDomain)) {
        case SpawnSiteStatus::AllDetached:
          HighPotentialParallelism = true;
          break;
        case SpawnSiteStatus::NoneFound:
          break;
        case SpawnSiteStatus::FailedDetaching:
          // Although there's something left unparallelized in this loop,
          // we may have partially parallelized it.
          HighPotentialParallelism = true;
          Ret = SpawnSiteStatus::FailedDetaching;
          break;
        case SpawnSiteStatus::FailedTopoSort:
          llvm_unreachable("processLoop() returns FailedDetaching only");
        }
        break;
      case SpawnType::Call:
        if (processCall(SS.CI, EndBlock, SS.timestamp, Deepen)) {
          HighPotentialParallelism = true;
        } else {
          Ret = SpawnSiteStatus::FailedDetaching;
        }
        break;
      case SpawnType::Annotation:
        processTaskAnnotation(SS.DI, EndBlock, SS.timestamp, Deepen);
        break;
      case SpawnType::Store:
        // Store spawns are worth the overhead when it enables us to decouple a
        // store from the spawn of subsequent high-potential-parallelism tasks.
        // It is a terrible shame to kill the parallelism of an expanded
        // executing loop due to a dumb WAR/WAW conflict on a store that
        // precedes the loop (e.g. see 456.hmmer/fast_algorithms.c).
        if (HighPotentialParallelism)
          processStore(SS.SI, EndBlock, SS.timestamp, Deepen);
        else
          DEBUG({dbgs() << "ignoring store: " << *SS.SI << "\n  from ";
            if (SS.SI->getDebugLoc())
              SS.SI->getDebugLoc()->print(dbgs());
            else
              dbgs() << "unknown location";
            dbgs() << "\n";
          });
        break;
      }
      assert(EndBlock == getEndOfCanonicalLoopBody(L));
      assert(isa<ReattachInst>(EndBlock->getTerminator())
             || isa<ReturnInst>(EndBlock->getTerminator()));
      assert(EndBlock->size() == 1 ||
             (EndBlock->size() == 2 && &EndBlock->front() == Undeepen));
    }
  }

  DEBUG(dbgs() << "Finished processing region at loop nesting level "
               << LoopLevel << ".\n");
  DEBUG(dbgs() << "  Region status: " << Ret << '\n');
  return Ret;
}


// Attempt to spawn a fractalized version of L and and its continuation,
// including fractalizing within the body of L.
//
// processWithinLoop and processLoop use mutual recursion to walk loop nests
// in pre-order.
SpawnSiteStatus Fractalizer::processLoop(Loop *L,
                                         BasicBlock *const ContinuationEnd,
                                         unsigned timestamp, DeepenInst *Domain,
                                         bool OnlySpawnsite) {
  const unsigned LoopLevel = EarlyChainLoopLevel + L->getLoopDepth();
  DEBUG(dbgs() << "Processing loop: " << *L);
  DEBUG(dbgs() << "  starting at: ");
  DEBUG(L->getStartLoc().print(dbgs()));
  DEBUG(dbgs() << "\n  Loop nesting level is " << LoopLevel << '\n');
  ORE.emit(OptimizationRemark(PASS_NAME, "ShouldNotSpawnLoop",
                              L->getStartLoc(), L->getHeader())
           << "Processing loop... (Level "
           << ore::NV("LoopLevel", LoopLevel) << ")");

  SmallVector<BasicBlock *, 8> NonDetachedBlocks;
  getNonDetachDescendants(DT, L->getHeader(), NonDetachedBlocks);
  bool ParallelizeAtAllCosts = false;
  for (BasicBlock *BB : NonDetachedBlocks) {
    if (!L->contains(BB)) continue;
    Loop *BBLoop = LI.getLoopFor(BB);
    if (BBLoop == L) {
      if (any_of(*BB, [this](const Instruction &I) {
            const auto *CI = dyn_cast<CallInst>(&I);
            return CI && (shouldSpawn(CI) || isRuntimeSpawnCall(CI));
          }))
        ParallelizeAtAllCosts = true;

      if (auto *DI = dyn_cast<SDetachInst>(BB->getTerminator())) {
        assert(DI->hasTimestamp() ==
               (getDetachKind(DI) == DetachKind::CallPassedCont ||
                getDetachKind(DI) == DetachKind::RetargetSuperdomain));
        assert(DI->hasTimestamp() !=
               (getDetachKind(DI) == DetachKind::Unknown));
        if (!DI->hasTimestamp())  // this is a manual task annotation
          ParallelizeAtAllCosts = true;
      }
    } else {
      assert(L->contains(BBLoop));
      if (shouldSpawn(BBLoop))
        ParallelizeAtAllCosts = true;
    }
  }
  DEBUG(if (ParallelizeAtAllCosts)
    dbgs() << "Must fractalize loop to spawn internal spawn sites.\n");

  // detachLoopIterations() blows away the loop if it generates an early chain,
  // so preserve some information
  DebugLoc StartLoc = L->getStartLoc();
  std::string HeaderName = L->getHeader()->getName().str();
  BasicBlock *Preheader = L->getLoopPreheader();

  SpawnSiteStatus Ret;

  // First, attempt to detach each iteration of the loop
  BasicBlock *ExitBlock =
      detachLoopIterations(*L, Domain, ContinuationEnd, ParallelizeAtAllCosts,
                           AC, DT, LI, TLI, TTI, ORE);

  Constant *Timestamp = ConstantInt::get(StaticTimestampTy, timestamp);
  Constant *TimestampP1 = ConstantInt::get(StaticTimestampTy, timestamp + 1);
  if (!ExitBlock) {  // Iterations of loop are not detached as tasks
    assert(!L->isInvalid());
    assert(L->getLoopPreheader() == Preheader);
    ExitBlock = getUniqueNonDeadendExitBlock(*L);
    if (ExitBlock
        && !isa<PHINode>(ExitBlock->front())
        && ExitBlock != ContinuationEnd
        && L->getLoopLatch() == ExitBlock->getUniquePredecessor()
        && L->isLCSSAForm(DT)) {
      // The loop has a unique exit block with no loop outputs,
      // and the loop continuation is not trivial
      // (i.e. the exit does not simply return/reattach).
      // It's innocuous to spawn in parallel with its continuation.
      // FIXME(mcj) there are other ways to test for a "trivial" continuation,
      // but the best solution IMO is to always detach the loop here, then
      // another pass is responsible to remove unnecessary detaches.
      Instruction *const TaskStart = Preheader->getTerminator();
      Instruction *ContinuationStart = &ExitBlock->front();

      DEBUG(dbgs() << "Spawning serialized loop with timestamp "
                   << *Timestamp << ".\n");
      detachTask(TaskStart, ContinuationStart, Timestamp, Domain,
                 DetachKind::SerialLoop, HeaderName + ".serialloop", &DT, &LI);

      DEBUG(dbgs() << "Spawning serialized loop continuation with timestamp "
                   << *TimestampP1 << ".\n");
      if (!ContinuationStart->getDebugLoc())
        ContinuationStart->setDebugLoc(TaskStart->getDebugLoc());
      detachContinuationTask(ContinuationStart, ContinuationEnd,
                             TimestampP1, Domain,
                             DetachKind::LoopIndependentCont,
                             HeaderName + ".serialloopcont");

      DEBUG(assertVerifyFunction(F,
              "After detaching serial loop & continuation", &DT, &LI));
    }

    if (ParallelizeAtAllCosts) {
      ORE.emit(DiagnosticInfoOptimizationFailure(PASS_NAME, "FailWithinLoop",
                                                 StartLoc, L->getHeader())
               << "Loop with eligible function spawn sites "
                  "was not parallelized");
      Ret = SpawnSiteStatus::FailedDetaching;
    } else {
      Ret = SpawnSiteStatus::NoneFound;
    }
  } else if (!L->isInvalid()) {
    // Loop was canonicalized
    assert(L->getLoopPreheader() == Preheader);
    assert(L->getLoopLatch() == ExitBlock->getSinglePredecessor());
    assert(isExpandableSwarmLoop(L, DT));
    auto *InnerDI = cast<SDetachInst>(L->getHeader()->getTerminator());
    assert(getDetachKind(InnerDI) == DetachKind::UnexpandedIter);
    bool MustSpawnLatch = any_of(*L->getLoopLatch(), [](const Instruction &I) {
      return I.mayReadOrWriteMemory();
    });
    if (MustSpawnLatch)
      InnerDI->setMetadata(SwarmFlag::MustSpawnLatch,
                           MDNode::get(F.getContext(), {}));

    // Spawn the continuation.
    DEBUG(dbgs() << "Spawning loop continuation with timestamp " << *TimestampP1 << ".\n");
    Instruction *ContinuationStart = &ExitBlock->front();
    if (!ContinuationStart->getDebugLoc())
      ContinuationStart->setDebugLoc(Preheader->getTerminator()->getDebugLoc());
    SDetachInst *ContDI;
    detachContinuationTask(ContinuationStart, ContinuationEnd, TimestampP1, Domain,
                           DetachKind::LoopIndependentCont,
                           HeaderName + ".loopcont", &ContDI);
    DEBUG(assertVerifyFunction(F,
            "After spawning loop continuation", &DT, &LI));

    // Create a domain for loop index-based dynamic timestamps.
    DeepenInst *Deepen =
            DeepenInst::Create("loopdomain", Preheader->getTerminator());
    // The only detach in a canonical loop is the detach of the loop body,
    // which we want to send into the new domain we are creating.
    SmallVector<SDetachInst *, 8> DetachesWithinLoop;
    getOuterDetaches(*L, DT, DetachesWithinLoop);
    assert(DetachesWithinLoop.size() == 1);
    assert(DetachesWithinLoop[0] == InnerDI);
    assert(InnerDI->getMetadata(SwarmFlag::TempNullDomain));
    InnerDI->setMetadata(SwarmFlag::TempNullDomain, nullptr);
    InnerDI->setDomain(Deepen);
    // Nested within the loop body detach, subsumed continuation detaches must
    // be retargeted to the original domain.
    DetachesWithinLoop.clear();
    getOuterDetaches(InnerDI, DT, DetachesWithinLoop);
    for (SDetachInst *DI : DetachesWithinLoop) {
      assert(getDetachKind(DI) == DetachKind::SubsumedCont ||
             getDetachKind(DI) == DetachKind::RetargetSuperdomain);
      retargetToSuperdomain(DI, Domain, DT, &LI);
    }
    // Continuation should remain in the original domain
    assert(ExitBlock != ContinuationEnd
           && "LoopIterDetacher's sanitizeContinuation() ensures "
              "ExitBlock is a safe place to put this undeepen");
    UndeepenInst *Undeepen =
            UndeepenInst::Create(Deepen, &ExitBlock->front());
    assert(!ContDI || DT.dominates(Undeepen, ContDI));
    if (OnlySpawnsite && !ContDI && DetachesWithinLoop.empty()) {
      DEBUG(dbgs()
            << "This loop is the only thing in its enclosing function/loop "
               "body.\nWe can avoid an extra nested domain level\n");
      assert(Domain->hasOneUse());
      auto *OuterUndeepen = cast<UndeepenInst>(Domain->user_back());
      OuterUndeepen->eraseFromParent();
      Domain->eraseFromParent();
    } else {  // This is the general case
      // Spawn the entire loop with its continuation.
      // This is the dedicated domain-creator task.
      DEBUG(dbgs() << "Spawning loop with timestamp " << *Timestamp << ".\n");
      Instruction *WillBeReattach =
          ContDI ? ContDI->getContinue()->getTerminator() : ContinuationStart;
      detachTask(Deepen, WillBeReattach, Timestamp, Domain,
                 DetachKind::LoopStart, HeaderName + ".loop", &DT, &LI);
    }
    DEBUG(assertVerifyFunction(F, "After setting up loop domain", &DT, &LI));

    // Finally, spawn things within the detached loop body.
    SpawnSiteStatus InnerStatus = processWithinLoop(L, MustSpawnLatch);
    assert(!(ParallelizeAtAllCosts && (InnerStatus == SpawnSiteStatus::NoneFound)));
    switch(InnerStatus) {
    case SpawnSiteStatus::FailedDetaching:
    case SpawnSiteStatus::FailedTopoSort:
      ORE.emit(DiagnosticInfoOptimizationFailure(PASS_NAME, "FailWithinLoop",
                                                 StartLoc, L->getHeader())
               << "Parallelization failure occured within body of loop.");
      Ret = SpawnSiteStatus::FailedDetaching;
      break;
    case SpawnSiteStatus::NoneFound:
      DEBUG(dbgs() << "Tagging loop as coarsenable.\n");
      InnerDI->setMetadata(SwarmFlag::Coarsenable,
                           MDNode::get(F.getContext(), {}));
      LLVM_FALLTHROUGH;
    case SpawnSiteStatus::AllDetached:
      Ret = SpawnSiteStatus::AllDetached;
    }
  } else {
    // Loop was outlined and transformed to an early recursive chain
    assert(L->isInvalid());
    L = nullptr;
    assert(ExitBlock == ContinuationEnd &&
           "The continuation should be spawned within RecursiveLoop");
    auto DI = cast<SDetachInst>(Preheader->getTerminator());
    BasicBlock *DetachedTopCall = DI->getDetached();
    assert(DetachedTopCall->size() == 2);
    auto *RI = cast<SReattachInst>(DetachedTopCall->getTerminator());
    assert(areMatching(DI, RI, DT)); (void)RI;
    Function *RecursiveLoop =
            cast<CallInst>(DetachedTopCall->front()).getCalledFunction();
    assert(RecursiveLoop);
    DominatorTree NewDT(*RecursiveLoop);

    // Create a domain for loop index-based dynamic timestamps.
    auto *WillBeReattach =
            cast<BranchInst>(DI->getContinue()->getTerminator());
    assert(WillBeReattach->isUnconditional() &&
           WillBeReattach->getSuccessor(0) == ContinuationEnd);
    DeepenInst *Deepen =
            DeepenInst::Create("loopdomain", Preheader->getTerminator());
    assert(DI->getMetadata(SwarmFlag::TempNullDomain));
    DI->setMetadata(SwarmFlag::TempNullDomain, nullptr);
    DI->setDomain(Deepen);
    UndeepenInst::Create(Deepen, WillBeReattach);
    // Spawn a dedicated domain-creator task that contains the Deepen.
    DEBUG(dbgs() << "Spawning loop with timestamp " << *Timestamp << ".\n");
    detachTask(Deepen, WillBeReattach, Timestamp, Domain,
               DetachKind::LoopStart, HeaderName + ".loop",
               &DT, &LI);

    // Compensate for deepening by sending any subsumed continuations back to
    // the original domain, although the recursive spawns of loop iterations
    // should stay in the subdomain.
    auto *RecurCall = cast<CallInst>(
            *findUnique(RecursiveLoop->users(),
                        [RecursiveLoop](User *U) {
                          auto *CI = cast<CallInst>(U);
                          assert(CI->getCalledFunction() == RecursiveLoop);
                          return CI->getFunction() == RecursiveLoop;
                        }));
    BasicBlock *RecurDetacher = RecurCall->getParent()->getSinglePredecessor();
    assert(RecurDetacher && "Recursive call should be spawned");
    auto *RecurDetach = cast<SDetachInst>(RecurDetacher->getTerminator());
    SmallVector<SDetachInst *, 8> OuterDetaches;
    getOuterDetaches(RecursiveLoop, NewDT, OuterDetaches);
    assert(is_contained(OuterDetaches, RecurDetach));
    for (SDetachInst *DI : OuterDetaches) {
      if (DI == RecurDetach) continue;
      assert(getDetachKind(DI) == DetachKind::SubsumedCont ||
             getDetachKind(DI) == DetachKind::RetargetSuperdomain);
      retargetToSuperdomain(DI, nullptr, NewDT);
    }
    DEBUG(assertVerifyFunction(*RecursiveLoop,
            "After sending chain's continuations to superdomain", &NewDT));

    // Finally, spawn things within the outlined loop body.
    //TODO(victory): After making this pass a module pass, figure out how to
    // get analyses from the pass manager to avoid redundant computation.
    AssumptionCache NewAC(*RecursiveLoop);
    LoopInfo NewLI(NewDT);
    Fractalizer F(*RecursiveLoop, NewAC, NewDT, NewLI, TLI, TTI, ORE, LoopLevel);
    SpawnSiteStatus InnerStatus = F.processWithinLoop(nullptr);
    assert(!(ParallelizeAtAllCosts && (InnerStatus == SpawnSiteStatus::NoneFound)));
    switch(InnerStatus) {
    case SpawnSiteStatus::FailedDetaching:
    case SpawnSiteStatus::FailedTopoSort:
      ORE.emit(DiagnosticInfoOptimizationFailure(PASS_NAME, "FailWithinLoop",
                                                 StartLoc, &RecursiveLoop->getEntryBlock())
               << "Parallelization failure occured within body of loop.");
      Ret = SpawnSiteStatus::FailedDetaching;
      break;
    case SpawnSiteStatus::AllDetached:
    case SpawnSiteStatus::NoneFound:
      Ret = SpawnSiteStatus::AllDetached;
      break;
    }
    finishFractalization(*RecursiveLoop);
  }

  DEBUG(dbgs() << "Finished processing loop at nesting level "
               << LoopLevel << ".\n");
  DEBUG(dbgs() << "  Loop status: " << Ret << '\n');
  return Ret;
}


static void ensureDebugLoc(StoreInst *SI) {
  if (SI->getDebugLoc()) return;

  DebugLoc Loc = SI->getNextNode()->getDebugLoc();
  for (auto *I = &SI->getParent()->front(); !Loc && I; I = I->getNextNode())
    Loc = I->getDebugLoc();
  if (!Loc) {
    Instruction *Ptr = dyn_cast<Instruction>(SI->getPointerOperand());
    if (Ptr) Loc = Ptr->getDebugLoc();
  }
  if (!Loc) {
    Instruction *Ptr = dyn_cast<Instruction>(SI->getPointerOperand()
                                               ->stripPointerCasts());
    if (Ptr) Loc = Ptr->getDebugLoc();
  }
  if (!Loc) {
    Instruction *Value = dyn_cast<Instruction>(SI->getValueOperand());
    if (Value) Loc = Value->getDebugLoc();
  }
  if (!Loc) {
    Instruction *Value = dyn_cast<Instruction>(SI->getValueOperand()
                                                 ->stripPointerCasts());
    if (Value) Loc = Value->getDebugLoc();
  }

  // Final fallback: use the starting line of the function
  if (!Loc)
    if (auto SP = SI->getFunction()->getSubprogram())
      Loc = DebugLoc::get(SP->getScopeLine(), 0, SP);

  if (!SI->getDebugLoc()) SI->setDebugLoc(Loc);
}


void Fractalizer::processStore(StoreInst *const SI,
                               BasicBlock *const ContinuationEnd,
                               const unsigned timestamp,
                               DeepenInst *Domain) {
  assert(SI->user_empty());

  ensureDebugLoc(SI);

  SmallString<256> TaskName = SI->getParent()->getName();
  TaskName += "-storeOf-";
  TaskName += SI->getValueOperand()->getName();
  TaskName += "-To-";
  TaskName += SI->getPointerOperand()->getName();

  ConstantInt* Timestamp = ConstantInt::get(StaticTimestampTy, timestamp);
  DEBUG({
    dbgs() << "Spawning store with timestamp " << *Timestamp
           << "\n  " << *SI << "\n  from ";
    if (SI->getDebugLoc()) SI->getDebugLoc()->print(dbgs());
    else dbgs() << "unknown location";
    dbgs() << '\n';
  });
  
  SmallPtrSet<const Instruction*, 16> Visited;
  Visited.insert(SI);

  Instruction *TaskStart = SI;
  // search upward from the store for a load in the same basic block
  Instruction *PrevI = SI->getPrevNode();
  while (PrevI) {
    // give up if value is used after store
    if (!all_of(PrevI->users(), [&Visited](User *Usr) {
          return Visited.count(cast<Instruction>(Usr));
        })) break;

    if (PrevI->mayReadOrWriteMemory()) {
      if (LoadInst *LI = dyn_cast<LoadInst>(PrevI)) {
        if (LI->getPointerOperand() == SI->getPointerOperand()) {
          DEBUG(dbgs() << "Expanding store task to include preceding load: " << *LI << "\n");
          TaskStart = PrevI;
          if (!TaskStart->getDebugLoc()) {
            TaskStart->setDebugLoc(SI->getDebugLoc());
          }
        }
      }
      break;
    } else if (isa<CallInst>(PrevI) || PrevI == SI->getParent()->getFirstNonPHI()) {
      break;
    }
    Visited.insert(PrevI);
    PrevI = PrevI->getPrevNode();
  }
  Instruction *ContinuationStart = SI->getNextNode();
  detachTask(TaskStart, ContinuationStart, Timestamp, Domain,
             DetachKind::Write, TaskName,
             &DT, &LI);

  Timestamp = ConstantInt::get(StaticTimestampTy, timestamp + 1);
  DEBUG(dbgs() << "Spawning store continuation with timestamp "
               << *Timestamp << ".\n");
  if (!ContinuationStart->getDebugLoc())
    ContinuationStart->setDebugLoc(TaskStart->getDebugLoc());
  detachContinuationTask(ContinuationStart, ContinuationEnd, Timestamp, Domain,
                         DetachKind::WriteCont, TaskName + ".cont");
  DEBUG(assertVerifyFunction(F, "After detaching store", &DT, &LI));
}


unsigned Fractalizer::processTaskAnnotation(SDetachInst *const DI,
                                            BasicBlock *const ContinuationEnd,
                                            const unsigned timestamp,
                                            DeepenInst *Domain) {
  DEBUG(dbgs() << "Processing preexisting detach: " << *DI << "\n  from ");
  DEBUG(DI->getDebugLoc()->print(dbgs()));
  DEBUG(dbgs() << '\n');

  SmallVector<SReattachInst *, 8> MatchingReattaches;
  getMatchingReattaches(DI, DT, MatchingReattaches);
  //FIXME(victory): Do we need to support multiple reattaches here?
  assert(MatchingReattaches.size() == 1);
  SReattachInst *TaskEnd = MatchingReattaches[0];

  ConstantInt *Timestamp = ConstantInt::get(StaticTimestampTy, timestamp);
  DEBUG(dbgs() << "Spawning with timestamp " << *Timestamp << ".\n");
  assert(!DI->hasTimestamp());
  DI->setTimestamp(Timestamp);
  assert(!DI->getDomain());
  DI->setDomain(Domain);
  setDetachKind(DI, DetachKind::Annotation);

  unsigned NestedSpawns = 0;
  // Ensure dedicated reattach block
  SplitBlock(TaskEnd->getParent(), TaskEnd, &DT, &LI);
  SmallVector<BasicBlock *, 8> SpawnedBlocks;
  topologicalSort(DI->getDetached(), TaskEnd->getParent(), LI, SpawnedBlocks);
  assert(!SpawnedBlocks.empty());
  assert(SpawnedBlocks[0] == DI->getDetached());
  assert(is_contained(SpawnedBlocks, TaskEnd->getParent()));
  for (BasicBlock *BB : SpawnedBlocks) {
    assert(DT.dominates(DI->getDetached(), BB));
    if (auto *DI = dyn_cast<SDetachInst>(BB->getTerminator()))
      NestedSpawns += processTaskAnnotation(
          DI, TaskEnd->getParent(), timestamp + 1 + 2*NestedSpawns, Domain);
  }

  Timestamp =
      ConstantInt::get(StaticTimestampTy, timestamp + 1 + 2*NestedSpawns);
  BasicBlock *Continue = TaskEnd->getDetachContinue();
  if (Continue != ContinuationEnd) {
    DEBUG(dbgs() << "Spawning detach continuation " << Continue->getName()
                 << " with timestamp " << *Timestamp << ".\n");
    detachContinuationTask(Continue->getFirstNonPHI(), ContinuationEnd,
                           Timestamp, Domain, DetachKind::AnnotationCont);
  }

  DEBUG(assertVerifyFunction(F, "After detaching task annotated by programmer",
                             &DT, &LI));

  return 1 + NestedSpawns;
}

// Attempt to spawn a parallel version of CI and and its continuation,
// possibly involving continuation-passing.
// \returns true on success.
bool Fractalizer::processCall(CallInst *const CI,
                              BasicBlock *const ContinuationEnd,
                              const unsigned timestamp,
                              DeepenInst *Domain) {
  if (!CI->getDebugLoc()) {
    assert(isa<IntrinsicInst>(CI));
    CI->setDebugLoc(CI->getNextNode()->getDebugLoc()
                    ? CI->getNextNode()->getDebugLoc()
                    : CI->getParent()->getTerminator()->getDebugLoc());
  }

  DEBUG({
    dbgs() << "Processing call: " << *CI << "\n  from ";
    if (CI->getDebugLoc()) CI->getDebugLoc()->print(dbgs());
    else dbgs() << "unknown location";
    dbgs() << '\n';
  });

  assert(shouldSpawn(CI));

  if (cannotSpawn(CI))
    return false;

  if (shouldSpawnParallelContinuation(CI))
    spawnCallAndParallelContinuation(CI, ContinuationEnd, timestamp, Domain);
  else
    spawnCallAndPassContinuation(CI, ContinuationEnd, timestamp, Domain);

  return true;
}

void Fractalizer::spawnCallAndParallelContinuation(
        CallInst *CI,
        BasicBlock *const ContinuationEnd,
        const unsigned timestamp,
        DeepenInst *Domain) {
  assert(CI->user_empty() &&
         "Use of return values requires continuation passing");
  Instruction *TaskStart;
  Instruction *ContinuationStart = CI->getNextNode();
  SmallString<256> TaskName = CI->getParent()->getName();
  if (Function *Callee = CI->getCalledFunction()) {
    TaskName += "-callTo-";
    TaskName += Callee->getName();

    CI = replaceWithParallelCall(CI);
    DEBUG(dbgs() << " replaced with parallel call: " << *CI << '\n');

    TaskStart = CI;
  } else {
    Value *SerialCalleePtr = CI->getCalledValue();
    TaskName += "-indirectCallTo-";
    TaskName += SerialCalleePtr->getName();

    // In C-like pseudocode, given a call of the form
    //    SerialCalleePtr(args...);
    // We are replacing it with the following:
    //    if (SerialCalleePtr & 0xF) {
    //      ParFuncPtr = *(SerialCalleePtr - 1);
    //      ParFuncPtr(args...);
    //    } else { SerialCalleePtr(args...); }
    Instruction *HasParVersion = createHasParVersion(SerialCalleePtr, CI);
    TerminatorInst *ThenTerm, *ElseTerm;
    std::tie(ThenTerm, ElseTerm) = splitBlockAndInsertIfThenElse(
        HasParVersion, CI, &DT, &LI);
    Value *ParFuncPtr = createGetParFuncPtr(SerialCalleePtr, ThenTerm);
    CallInst *ParCI = createParallelCall(ParFuncPtr, CI, ThenTerm);
    CI->moveBefore(ElseTerm);
    DEBUG(dbgs() << " created conditional indirect parallel call: "
                 << *ParCI << '\n');

    TaskStart = HasParVersion;
  }

  DEBUG(dbgs() << " detaching continuation in parallel.\n");

  ConstantInt* Timestamp = ConstantInt::get(StaticTimestampTy, timestamp);
  DEBUG(dbgs() << "Spawning call with timestamp " << *Timestamp << ".\n");
  detachTask(TaskStart, ContinuationStart, Timestamp, Domain,
             DetachKind::Call, TaskName,
             &DT, &LI);

  Timestamp = ConstantInt::get(StaticTimestampTy, timestamp + 1);
  DEBUG(dbgs() << "Spawning call continuation with timestamp "
               << *Timestamp << ".\n");
  if (!ContinuationStart->getDebugLoc())
    ContinuationStart->setDebugLoc(TaskStart->getDebugLoc());
  detachContinuationTask(ContinuationStart, ContinuationEnd, Timestamp, Domain,
                         DetachKind::CallIndependentCont,
                         TaskName + ".cont");

  DEBUG(assertVerifyFunction(F, "After spawning parallel call and continuation",
                             &DT, &LI));
}

void Fractalizer::spawnCallAndPassContinuation(CallInst *CI,
                                               BasicBlock *const ContinuationEnd,
                                               const unsigned timestamp,
                                               DeepenInst *Domain) {
  DEBUG(dbgs() << "Attempting to pass continuation to parallel call.\n");

  // Gather the continuation blocks to outline.
  BasicBlock *ContinuationStart = SplitBlock(CI->getParent(),
                                             CI->getNextNode(),
                                             &DT, &LI);
  BranchInst *BranchAfterCI = cast<BranchInst>(CI->getNextNode());
  SmallSetVector<BasicBlock *, 8> ContinuationBBs;
  BasicBlock *NewContinuationEnd = makeReachableDominated(ContinuationStart,
                                                          ContinuationEnd,
                                                          DT, LI,
                                                          ContinuationBBs);
  assert(NewContinuationEnd);
  assert(all_of(CI->users(), [&ContinuationBBs](User *U) {
    return ContinuationBBs.count(cast<Instruction>(U)->getParent());
  }) && "Return value used other than by instruction in continuation?");

  //TODO(victory): Change shrinkInputs() API to look more like our outline()
  // utility, allowing us to reduce the boilerplate code here.
  {
    SmallVector<BasicBlock *, 32> Blocks;
    Blocks.push_back(ContinuationStart);
    for (BasicBlock *BB : ContinuationBBs)
      if (BB != ContinuationStart)
        Blocks.push_back(BB);
    shrinkInputs(Blocks, {CI}, TTI, &ORE);
  }

  // Collect the inputs that must be captured for the continuation
  SetVector<Value *> Captures;
  // Reserve enough space for a pointer at the start of the closure struct.
  // This is where we will store the function pointer.
  Captures.insert(ConstantPointerNull::get((Type::getInt8PtrTy(Context))));
  findInputsNoOutputs(ContinuationBBs, Captures);
  Captures.remove(CI);

  Value *SerialCalleePtr = CI->getCalledValue();
  SmallString<256> TaskName;
  TaskName += CI->getParent()->getName();
  if (CI->getCalledFunction())
    TaskName += "-callWithContTo-";
  else
    TaskName += "-indirectCallWithContTo-";
  TaskName += SerialCalleePtr->getName();

  // First element is a placeholder for the function pointer
  const unsigned Skip = 1;
  Instruction *Closure = createClosure(
          Captures.getArrayRef(), CI,
          "continuation_closure", Skip);
  unpackClosure(
          Closure, Captures.getArrayRef(),
          [this, ContinuationStart] (Instruction *Usr) -> Instruction * {
            if (DT.dominates(ContinuationStart, Usr->getParent()))
              return &*ContinuationStart->getFirstInsertionPt();
            else
              return nullptr;
          },
          Skip);

  // Free the closure at the end of the continuation, inspired by dsm's SwarmABI
  // commit 1f49128c180ee188. The memory would not be freed until the
  // continuation task commits anyway.
  Instruction *Free =
      CallInst::CreateFree(Closure, NewContinuationEnd->getTerminator());
  addSwarmMemArgsForceAliasMetadata(cast<CallInst>(Free));

  // Outline the continuation
  SetVector<Value *> Arguments;
  if (!CI->getType()->isVoidTy())
    Arguments.insert(CI);
  Arguments.insert(Closure);
  Function *ContinuationFunc =
      outline(Arguments, ContinuationBBs, ContinuationStart, ".cps");
  DEBUG(dbgs() << "Outlined continuation to pass of type "
               << *ContinuationFunc->getFunctionType() << "\n");
  ContinuationFunc->setName(F.getName() + "_" + TaskName + ".contfunc");
  ContinuationFunc->setCallingConv(CallingConv::C);
  finishFractalization(*ContinuationFunc);
  assertVerifyFunction(*ContinuationFunc, "Outlined continuation");

  // Write the function pointer for the continuation into the first field
  // of the closure
  IRBuilder<> B(Closure->getNextNode());
  StoreInst *Store = B.CreateStore(ContinuationFunc, B.CreatePointerCast(
          Closure, PointerType::getUnqual(ContinuationFunc->getType())));
  Store->setMetadata(SwarmFlag::Closure, MDNode::get(Context, {}));

  // Now here's where we're going to really change the code's behavior and try
  // to do continuation passing instead of serially executing the continuation
  // after the call.
  // Here are some assertions to remind you of the CFG structure we've set up
  // so far, to help you understand what we're doing:
  assert(BranchAfterCI == CI->getNextNode());
  assert(BranchAfterCI->isUnconditional());
  assert(BranchAfterCI->getSuccessor(0) == ContinuationStart);
  assert(NewContinuationEnd->getSingleSuccessor() == ContinuationEnd);
  CallInst *ParCI;
  if (CI->getCalledFunction()) {
    // Unlink and delete the ContinuationBBs.
    BranchAfterCI->setSuccessor(0, ContinuationEnd);
    if (DT.dominates(ContinuationStart, ContinuationEnd))
      DT.changeImmediateDominator(ContinuationEnd, CI->getParent());
    assert(DT.dominates(ContinuationStart, NewContinuationEnd));
    assert(!DT.dominates(ContinuationStart, ContinuationEnd));
    eraseDominatorSubtree(ContinuationStart, DT, &LI);
    DEBUG(assertVerifyFunction(F, "Outlined and deleted continuation", &DT, &LI));

    // Pass continuation to a parallel version of CI.
    ParCI = replaceWithParallelCall(CI, Closure);
    DEBUG(dbgs() << " replaced with parallel call: " << *ParCI << '\n');
  } else {
    // In pseudocode, given something of the form
    //    ret = SerialCalleePtr(args...);
    //    continuation(ret);
    // We are replacing it with the following:
    //    if (SerialCalleePtr & 0xF) {
    //      ParFuncPtr = *(SerialCalleePtr - 1);
    //      ParFuncPtr(args..., continuation);
    //    } else {
    //      ret = SerialCalleePtr(args...);
    //      continuation(ret);
    //    }
    Instruction *HasParVersion = createHasParVersion(SerialCalleePtr, CI);
    TerminatorInst *ThenTerm =
        SplitBlockAndInsertIfThen(HasParVersion, CI, true, nullptr, &DT, &LI);
    Value *ParFuncPtr = createGetParFuncPtr(SerialCalleePtr, ThenTerm);
    ParCI = createParallelCall(ParFuncPtr, CI, Closure, ThenTerm);
    ReplaceInstWithInst(ThenTerm, BranchInst::Create(ContinuationEnd));
    if (DT.dominates(ContinuationStart, ContinuationEnd))
      DT.changeImmediateDominator(ContinuationEnd, HasParVersion->getParent());
    DEBUG(dbgs() << " created conditional indirect parallel call: "
                 << *ParCI << '\n');
  }

  // Detach everything with the desired timestamp.
  auto Timestamp = ConstantInt::get(StaticTimestampTy, timestamp);
  DEBUG(dbgs() << "Spawning call with passed continuation with timestamp "
               << *Timestamp << ".\n");
  detachTask(ParCI, ParCI->getNextNode(), Timestamp, Domain, DetachKind::Call, TaskName, &DT, &LI);

  DEBUG(assertVerifyFunction(F, "Spawned call with passed continuation", &DT, &LI));
}


/// Find the parallel version of SCI's callee, assuming SCI is not an indirect
/// call, and insert a call to the parallel version in place of SCI.
/// \return the new (replacement) CallInst.
CallInst *Fractalizer::replaceWithParallelCall(CallInst *SCI,
                                               Value *ContClosure) {
  Function *SF = SCI->getCalledFunction();
  assert(SF);

  if (SF->isIntrinsic()) {
    assert(!ContClosure &&
           "We currently don't handle continuation parameters for intrinsics");
    // Use a manually-parallelized version of an intrinsic, if any,
    // otherwise call the original intrinsic.
    CallInst *PCI = replaceIntrinsicWithSubstitute(SCI);
    if (PCI) return PCI;
    else return SCI;
  }

  LibFunc Func;
  if (TLI.getLibFunc(*SF, Func) && Func == LibFunc_calloc) {
    assert(ContClosure);
    assert(SCI->getNumArgOperands() == 2);
    Function *CallocFn = RUNTIME_FUNC(__sccrt_calloc, SCI->getModule());
    IRBuilder<> Builder(SCI);
    Value *Cont = Builder.CreatePointerCast(
        ContClosure, CallocFn->getFunctionType()->getParamType(2), "cont");
    CallInst *PCI = Builder.CreateCall(
        CallocFn, {SCI->getArgOperand(0), SCI->getArgOperand(1), Cont});
    SCI->eraseFromParent();
    return PCI;
  }

  Function *PF = getOrInsertParallelVersion(SF, SCI->getAttributes());
  assert(PF->getCallingConv() == SCI->getCallingConv());

  CallInst *PCI = createParallelCall(PF, SCI, ContClosure, SCI);
  if (PCI != SCI) SCI->eraseFromParent();
  return PCI;
}


/// \return null if no replacement was made
CallInst* Fractalizer::replaceIntrinsicWithSubstitute(CallInst *const CI) {
  const Function *Callee = CI->getCalledFunction();
  assert(Callee && Callee->isIntrinsic());

  LLVMContext &Context = Callee->getContext();
  IRBuilder<> Builder(CI);
  CallInst *NewCI = nullptr;

  if (const MemSetInst *MSI = dyn_cast<MemSetInst>(CI)) {
    Value *Val = MSI->getValue();
    assert(Val->getType()->isIntegerTy(8) &&
           "LLVM LangRef says the value is 8 bits");
    assert(!MSI->isVolatile() && "Volatile memset not supported");

    Value *Length = Builder.CreateIntCast(MSI->getLength(),
                                          Type::getInt64Ty(Context),
                                          /* isSigned */ false);
    Function *NewFn = RUNTIME_FUNC(__sccrt_memset, CI->getModule());
    NewCI = CallInst::Create(NewFn, {MSI->getRawDest(), Val, Length});
    ReplaceInstWithInst(CI, NewCI);
  } else if (const MemCpyInst *MCI = dyn_cast<MemCpyInst>(CI)) {
    assert(!MCI->isVolatile() && "Volatile memcpy not supported");
    Value *Dest = MCI->getRawDest();
    Value *Source = MCI->getRawSource();
    Value *Length = Builder.CreateIntCast(MCI->getLength(),
                                          Type::getInt64Ty(Context),
                                          /* isSigned */ false);
    Function *NewFn = RUNTIME_FUNC(__sccrt_memcpy, CI->getModule());
    NewCI = CallInst::Create(NewFn, {Dest, Source, Length});
    ReplaceInstWithInst(CI, NewCI);
  }
  return NewCI;
}


bool Fractalizer::shouldSpawnParallelContinuation(const CallInst *CI) {
  if (CI->hasStructRetAttr()) {
    // TODO(mcj) it's probably wise to pass a continuation. However, the
    // returned struct may not have any users in the caller, but I suspect that
    // CI->user_empty() doesn't work properly with sret, since the function
    // returns void.

    // TODO(mcj) for an sret CallInst in a loop, I'm concerned that all
    // iterations (unnecessarily) share the same pointer. Fix this in a later
    // PR.

    // FIXME(dsm): Just no. This is no way to perform this check. It's failing
    // the moment I pack and unpack the pointer. If for some reason this is
    // important to keep, attach metadata to the pointer and add the code to
    // carry it across closures. Otherwise, delete this code.
#if 0
    const auto *SRetSource = CI->getArgOperand(0)->stripPointerCasts();
    const auto *SRetSourceCI = dyn_cast<CallInst>(SRetSource);
    assert(((SRetSourceCI &&
             SRetSourceCI->getCalledFunction()->getName() == "malloc")
            || isa<Argument>(SRetSource)
           ) &&
           "An sret operand pointer is not the progeny of a malloc. "
           "Originally the pointer came from an AllocaInst, but "
           "Fractalizer converted it to a malloc CallInst");
#endif
  }

  if (!CI->user_empty()) {
    DEBUG(dbgs() << " call's return value is used by continuation.\n");
    // We *need* to pass the continuation in this case.
    return false;
  }
  DEBUG(dbgs() << " call has no users of return value.\n");

  const Function *CF = CI->getCalledFunction();
  if (CF && CF->isIntrinsic()) {
    // For now, we do not support passing continuations to intrinsics
    //TODO(victory): it may not make a lot of sense to spawn the
    // continuation of a memset in parallel with the memset. Maybe handle
    // this someday.
    return true;
  }

  // Now, we have the choice to either spawn the continuation in parallel
  // or pass it to be spawned later which may reduce overspeculation.
  // Either will be correct, and we should choose based on performance.
  //TODO: When static analysis finds a definitive dependence between
  // the callee and the continuation, return false
  return !DelayCallCont;
}


// Return true if it would be desirable to spawn a possibly parallel version of
// the call CI.
bool Fractalizer::shouldSpawn(const CallInst *CI) const {
  if (CI->isInlineAsm()) {
    DEBUG(dbgs() << " ignored due to inline assembly.\n");
    return false;
  }

  if (!CI->mayReadOrWriteMemory()) {
    DEBUG({
      if (!isa<DbgInfoIntrinsic>(CI))
        dbgs() << " ignored as the call does not access memory.\n";
    });
    return false;
  }

  if (!CI->mayWriteToMemory()) {
    // We see this condition most commonly in standard library calls.
    // It is very rare for this conditition to be taken for calls to
    // application code: If a call is known to be read-only, it is probably
    // because its definition is supplied within the current translation unit.
    // Additionally, most read-only functions are small.  Therefore, most calls
    // known to be read-only are inlined and will never be considered as a
    // spawn site. As long as we don't have reason to believe spawning some
    // read-only functions is important for performance, I choose not to spawn
    // them, just because it's a convenient way to permit bailing out of making
    // early chains.
    DEBUG(dbgs() << " ignored as the call only reads memory.\n");
    return false;
  }

  // N.B. we do not treat varargs calls with no fixed parameters as variadic.
  // Such calls most commonly occur when compiling C code, as calls to external
  // functions with an old (K&R) style declarations.  LLVM/Clang considers such
  // calls as varargs, since it lacks the callees' prototypes, although the
  // callees cannot actually be variadic. See comments in cannotSpawn().
  if (CI->getFunctionType()->isVarArg() &&
      CI->getFunctionType()->getNumParams()) {
    // The callee was declared with a prototype that specified at least one
    // fixed parameter plus variadic parameters.  CPC generates a warning
    // when it fails to make a parallel copies of any such callees, so let's
    // emit a remark but avoid considering this further as a spawnsite,
    // so we do not generate additional noisy warnings.
    ORE.emit(OptimizationRemarkMissed(PASS_NAME, "SpawnVarArgs", CI)
             << "Function call not spawned due to variadic arguments");
    return false;
  }

  const Function *Callee = CI->getCalledFunction();

  if (!Callee) {
    if (DisableIndirectSpawn) {
      DEBUG(dbgs() << " ignored due to -swarm-disableindirectspawn.\n");
      return false;
    }
    DEBUG(dbgs() << " This indirect call should be spawned.\n");
    return true;
  }

  assert(!Callee->isVarArg() ||
         (Callee->arg_size() == 0 && CI->getNumArgOperands() == 0) &&
         "Callee must not be a true variadic function");

  // Because this only gets called in non-detached blocks
  // (topoSort() filters out detached blocks),
  // It should be impossible to re-encounter parallelized calls.
  assert(!CI->getMetadata(SwarmFlag::ParallelCall));
  assert(Callee->isIntrinsic() || !Callee->getName().contains("."));

  if (Callee->hasFnAttribute(SwarmAttr::NoSwarmify)) {
    DEBUG(dbgs() << " ignored due to __attribute__((noswarmify))\n");
    return false;
  }

  LibFunc Func;
  if (TLI.getLibFunc(*Callee, Func)) {
    switch (Func) {
    case LibFunc_calloc:
      DEBUG(dbgs() << " call to calloc should be spawned using SCCRT\n");
      return true;
    case LibFunc_malloc:
    case LibFunc_free:
    case LibFunc_realloc:
    case LibFunc_posix_memalign:
    case LibFunc_memalign:
    case LibFunc_strdup:
      DEBUG(dbgs() << " ignored as it is provided by our allocator libraries\n");
      return false;
    //case ???:
      //TODO: What standard library calls are good to spawn?
      //return true;
    default:
      // Other standard library calls include I/O syscalls
      // and math library stuff which probably shouldn't be spawned?
      DEBUG(dbgs() << " ignored as it is a standard library call\n");
      return false;
      break;
    }
  }

  StringRef CalleeName = Callee->getName();

  if (Callee->isIntrinsic()) {
    switch (Callee->getIntrinsicID()) {
    case Intrinsic::memcpy:
    case Intrinsic::memmove:
    case Intrinsic::memset: {
      const MemIntrinsic *MI = cast<MemIntrinsic>(CI);
      ConstantInt *Length = dyn_cast<ConstantInt>(MI->getLength());
      if (Length && Length->getZExtValue() <
                        LongMemoryOperationThreshold * SwarmCacheLineSize) {
        // Hopefully this will be lowered into clever assembly.
        DEBUG(dbgs() << " ignored call with short constant length ("
                     << Length->getZExtValue()
                     << " bytes)\n");
        return false;
      }
      break;
    }
    case Intrinsic::not_intrinsic:
      llvm_unreachable("Unknown intrinsic");
    default:
      DEBUG(dbgs() << " ignored call to intrinsic " << CalleeName << '\n');
      return false;
    }
  }

  if (CalleeName.startswith("__sccrt_")) {
    DEBUG(dbgs() << " ignored call to SCC runtime's " << CalleeName << '\n');
    return false;
  }

  return true;
}


// Return true if Fractalizer is incapable of safely spawning a
// possibly parallel version of CI.
bool Fractalizer::cannotSpawn(const CallInst *CI) const {
  // In C, (unlike in C++) functions with no fixed (named) parameters cannot
  // actually be variadic.  So, assuming the application was written in a sane,
  // portable, and standards-compliant way, we can infer that a function call
  // with no fixed parameters isn't actually variadic. See:
  // https://bugs.llvm.org/show_bug.cgi?id=35385#c24
  if (CI->getFunctionType()->isVarArg()) {
    assert(CI->getFunctionType()->getNumParams() == 0 &&
           "Guaranteed by shouldSpawn()");
    // TODO(victory): Checking the file extension is really sketchy.
    // LLVM/Clang version 7 has function attributes
    // that we can check: https://reviews.llvm.org/D48443
    bool CPlusPlus =
        !StringRef(F.getParent()->getSourceFileName()).endswith(".c");
    if (CPlusPlus) {
      ORE.emit(
          DiagnosticInfoOptimizationFailure(PASS_NAME, "CrazyVariadicCall",
                                            CI->getDebugLoc(), CI->getParent())
          << "Call of C++ variadic function with no named arguments?");
      // It'd be kinda insane to have a variadic function that had no named
      // arguments used to determine whether there are any variadic arguments
      // to be accessed with the stdarg.h macros.
      llvm_unreachable("Maybe crazy variadic functions don't exist?");
      //return true;
    } else {
      // When compiling a C program, this warning indicates you should fix the
      // source code, by specifying the function prototype in its declaration.
      ORE.emit(
          DiagnosticInfoOptimizationFailure(PASS_NAME, "PrototypelessCallSpawn",
                                            CI->getDebugLoc(), CI->getParent())
          << "C function called without a declared prototype; "
             "we will assume the function requires no arguments.");
      assert(!CI->getNumArgOperands());
    }
  }

  const AttributeList AL = CI->getAttributes();
  for (unsigned i = 0; i < CI->getNumArgOperands(); ++i) {
    for (const Attribute &Attr : AL.getParamAttributes(i)) {
      if (Attr.isStringAttribute()) {
        llvm_unreachable("parameter has string attribute?");
      } else {
        switch (Attr.getKindAsEnum()) {
        // These guarantee attributes are harmless.
        case Attribute::Alignment:
        case Attribute::NoAlias:
        case Attribute::Dereferenceable:
        case Attribute::DereferenceableOrNull:
        case Attribute::NoCapture:
        case Attribute::NonNull:
        case Attribute::ReadNone:
        case Attribute::ReadOnly:
        case Attribute::WriteOnly:
        // These ABI-impacting attributes are handled in CPC's
        // createParallelCall() or getOrInsertParallelVersion()
        // by copyABIImpactingArgAttributes() or removeGuaranteeAttributes().
        case Attribute::ByVal:
        case Attribute::Returned:
        case Attribute::SExt:
        case Attribute::ZExt:
        case Attribute::StructRet:
          continue;
        case Attribute::InAlloca:
        case Attribute::InReg:
        case Attribute::Nest:
        case Attribute::SwiftError:
        case Attribute::SwiftSelf:
          ORE.emit(DiagnosticInfoOptimizationFailure(PASS_NAME,
                       "SpawnArgAttr", CI->getDebugLoc(), CI->getParent())
                   << "Function call not spawned as arg " << std::to_string(i)
                   << " is passed with attribute " << Attr.getAsString());
          return true;
        default:
          // victory: the list above documents all parameter attributes
          // https://llvm.org/docs/LangRef.html#parameter-attributes
          llvm_unreachable("Unrecognized argument attribute");
        }
      }
    }
  }
  for (const Attribute &Attr : AL.getRetAttributes()) {
    if (Attr.isStringAttribute()) {
      llvm_unreachable("return has string attribute?");
    } else {
      switch (Attr.getKindAsEnum()) {
      // These attributes should be safe to drop.
      case Attribute::Alignment:
      case Attribute::NoAlias:
      case Attribute::Dereferenceable:
      case Attribute::DereferenceableOrNull:
      case Attribute::NonNull:
      // Extension is handled (hopefully) in CPC's createContinuationSpawn()
      case Attribute::SExt:
      case Attribute::ZExt:
        continue;
      default:
        ORE.emit(DiagnosticInfoOptimizationFailure(PASS_NAME,
                     "SpawnReturnAttr", CI->getDebugLoc(), CI->getParent())
                 << "Function call not spawned as return value has attribute "
                 << Attr.getAsString());
        return true;
      }
    }
  }

  return false;
}


bool Fractalizer::shouldSpawn(const Loop *L) const {
  Optional<const MDOperand *> SwarmifyEnable =
      findStringMetadataForLoop(L, "llvm.loop.swarmify.enable");
  if (SwarmifyEnable
      && mdconst::extract<ConstantInt>(**SwarmifyEnable)->isZero()) {
    DEBUG(dbgs() << "Ignoring loop with swarmify(disable) pragma:\n  "
                 << *L->getStartLoc() << '\n');
    return false;
  }

  if (findStringMetadataForLoop(L, SwarmFlag::LoopUnprofitable)) {
    DEBUG(dbgs() << "Ignoring loop deemed unprofitable.\n");
    return false;
  }

  return true;
}


/// Spawn a store to
/// 1) enable store hint serialization
/// 2) avoid a WAR dependence unduly aborting the
///    *potential* parallelism latent in a subsequent loop or call
bool Fractalizer::shouldSpawn(const StoreInst *SI) {
  if (SI->getMetadata(SwarmFlag::Closure)) {
    DEBUG(dbgs() << "Will not spawn store to Swarm closure " << *SI << "\n");
    return false;
  }
  if (SI->getMetadata(SwarmFlag::DoneFlag)) {
    DEBUG(dbgs() << "Will not spawn store to Swarm done flag " << *SI << "\n");
    return false;
  }

  assert(!isa<AllocaInst>(GetUnderlyingObject(
      SI->getPointerOperand(), SI->getModule()->getDataLayout())));

  // TODO(mcj) Identify other stores to statically known private addresses
  // Note, that is a vague idea. Private to what? Task boundaries? We don't yet
  // know them. Welp we changed most AllocaInsts to heap allocations.
  bool WeCanProveThisIsATaskPrivateAddress = false;
  if (WeCanProveThisIsATaskPrivateAddress)
    return false;
  return true;
}


/// Find and return the block in F created by CPC that calls the continuation
/// based on the continuation parameter.
//TODO(victory): Maybe this function should be moved to CPC?
static BasicBlock *getCallBlockForPassedContinuation(Function &F) {
  if (F.arg_empty())
    return nullptr;
  Argument *ContParam = std::prev(F.arg_end());
  if (!F.getAttributes().hasParamAttr(ContParam->getArgNo(),
                                      SwarmFlag::Continuation))
    return nullptr;

  // The continuation should only be used in two places: the check to see if
  // it is null, and the the single block that contains the continuation call.
  ICmpInst *CheckForNull = nullptr;
  BasicBlock *CallContBlock = nullptr;
  for (User *U : ContParam->users()) {
    if (auto Usr = dyn_cast<ICmpInst>(U)) {
      assert(Usr->isEquality());
      Value *Op0 = Usr->getOperand(0), *Op1 = Usr->getOperand(1);
      assert(Op0 == ContParam || Op1 == ContParam);
      assert(isa<ConstantPointerNull>(Op0) || isa<ConstantPointerNull>(Op1));
      assert(!CheckForNull && "Found multiple checks for continuation ?= null");
      CheckForNull = Usr;
    } else {
      auto UsrBlock = cast<Instruction>(U)->getParent();
      assert(!CallContBlock || CallContBlock == UsrBlock);
      CallContBlock = UsrBlock;
    }
  }
  assert(CheckForNull);
  assert(CallContBlock);
  assert(CheckForNull->getParent() != CallContBlock);

  // Verify that CallContBlock contains the continuation call.
  auto I = std::find_if(CallContBlock->begin(), CallContBlock->end(),
                        [](const Instruction &I) { return isa<CallInst>(I); });
  assert(std::find_if(std::next(I), CallContBlock->end(),
                      [](const Instruction &I) { return isa<CallInst>(I); })
         == CallContBlock->end() &&
         "Continuation-calling block contains multiple calls");
  auto CI = cast<CallInst>(&*I);
  DEBUG(dbgs() << "Found call to continuation:\n" << *CI << '\n');
  assert(all_of(*CallContBlock, [CI](const Instruction &I) {
    return &I == CI || !I.mayWriteToMemory();
  }) && "Continuation-calling block has other instructions with side effects");
  const DataLayout &DL = F.getParent()->getDataLayout(); (void)DL;
  assert(GetUnderlyingObject(cast<LoadInst>(
                 GetUnderlyingObject(CI->getCalledValue(), DL)
                                           )->getPointerOperand(), DL)
         == ContParam
         && "Continuation call isn't properly calling continuation closure?");

  return CallContBlock;
}


// Retarget a detach to go one level higher in the domain heirarchy.
// Intended to compensate if a swarm::deepen() call was placed before the detach.
// If DT and LI are provided, they will be updated.
static void retargetToSuperdomain(SDetachInst *DI,
                                  DeepenInst *EnclosingDomain,
                                  DominatorTree &DT,
                                  LoopInfo *LI) {
  assert(DI->hasTimestamp());
  assert(!DI->isSubdomain() &&
         "Autoparallelization uses old-style enqueue flags");
  assert(!EnclosingDomain || DT.dominates(EnclosingDomain, DI));
  if (DI->isSuperdomain()) {
    DEBUG(dbgs() << "  Spawning existing superdomain detach to superdomain:\n"
                 << "    " << *DI << '\n');
    assert((EnclosingDomain &&
            DI->getDomain() == EnclosingDomain->getSuperdomain(DT)) ||
           !DI->getDomain());
    BasicBlock *ContinueBlock = DI->getContinue();
    if (!DT.dominates(DI, ContinueBlock)) {
      // Split off a new continue block that is dedicated to this detach.
      SmallVector<SReattachInst *, 1> Reattaches;
      getMatchingReattaches(DI, DT, Reattaches);
      SmallVector<BasicBlock *, 2> ContinuePreds({DI->getParent()});
      for (SReattachInst *RI : Reattaches)
        ContinuePreds.push_back(RI->getParent());
      assert(all_of(ContinuePreds, [ContinueBlock](BasicBlock *BB) {
        return is_contained(predecessors(ContinueBlock), BB); }));
      ContinueBlock = SplitBlockPredecessors(ContinueBlock, ContinuePreds,
                                             ".super.re", &DT, LI);
      assert(ContinueBlock);
    }
    SDetachInst *NewDI;
    detachTask(DI, &ContinueBlock->front(),
               ConstantInt::getTrue(DI->getContext()) /*TS doesn't matter*/,
               EnclosingDomain,
               DetachKind::RetargetSuperdomain,
               DI->getParent()->getName() + ".super", &DT, LI, &NewDI);
    NewDI->setSuperdomain(true);
    NewDI->setRelativeTimestamp(true);
  } else {
    DI->setSuperdomain(true);
    DEBUG(dbgs() << "  Retargeted simple detach to superdomain:\n  ");
    DEBUG(dbgs() << *DI << '\n');
    assert(DI->getDomain() == EnclosingDomain);
  }
}


SpawnSiteStatus Fractalizer::fractalize() {
  StringRef FName = F.getName();
  DEBUG(dbgs() << "Fractalizing function " << FName << "() ...\n");

  DEBUG(assertVerifyFunction(F, "Before fractalization", &DT, &LI));
  assert(Return && "F should have a unique return instruction");
  assert(!Return->getReturnValue() && "F should return void");
  assert(!F.callsFunctionThatReturnsTwice());
  assert(none_of(F, hasUnexpectedExceptionHandling));

  SpawnSiteStatus Ret = processWithinLoop(nullptr);

  assertVerifyFunction(F, "After fractalization", &DT, &LI);
  switch(Ret) {
  case SpawnSiteStatus::AllDetached:
    FunctionsFullyFractalized++;
    DEBUG(dbgs() << "\n\nFully fractalized function "
                 << FName << "()\n\n");
    break;
  case SpawnSiteStatus::FailedDetaching:
  case SpawnSiteStatus::FailedTopoSort:
    Ret = SpawnSiteStatus::FailedDetaching;
    FunctionsIncompletelyFractalized++;
    DEBUG(dbgs() << "\n\nFailed to fully fractalize function "
                 << FName << "()\n\n");
    break;
  case SpawnSiteStatus::NoneFound:
    FunctionsNotFractalized++;
    DEBUG(dbgs() << "\n\nFound no opportunity to fractalize function "
                 << FName << "()\n\n");
    break;
  }
  //DEBUG(dbgs() << F);

  return Ret;
}




namespace {
struct Fractalization : public FunctionPass {
  /// Pass identification, replacement for typeid
  static char ID;

  explicit Fractalization() : FunctionPass(ID) {
    initializeFractalizationPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    if (!F.hasFnAttribute(SwarmFlag::Parallelizable))
      return false;

    assert(hasValidSwarmFlags(F));
    F.removeFnAttr(SwarmFlag::Parallelizable);
    F.addFnAttr(SwarmFlag::Parallelizing);
    assert(hasValidSwarmFlags(F));

    auto &ORE = getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();
    auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
    auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);

    SpawnSiteStatus S = Fractalizer(F, AC, DT, LI, TLI, TTI, ORE).fractalize();

    assert((S != SpawnSiteStatus::FailedDetaching || !errorIfAssertSwarmified(F))
           && "Failed full fractalization when programmer asserts success");

    finishFractalization(F);

    return true;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredID(LoopSimplifyID);
    AU.addRequiredID(LCSSAID);
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
  }
};
} // anonymous namespace

char Fractalization::ID = 0;
//static RegisterPass<Fractalization> X(PASS_NAME, "Split code into tasks and domains", false, false);
/*
INITIALIZE_PASS(Fractalization, PASS_NAME,
                "Split code into tasks and domains",
                false, false)
*/
INITIALIZE_PASS_BEGIN(Fractalization, PASS_NAME,
                      "Split code into tasks and domains",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LCSSAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_END(Fractalization, PASS_NAME,
                    "Split code into tasks and domains",
                    false, false)

namespace llvm {
Pass *createFractalizationPass() {
  return new Fractalization();
}
}
