//===-- Swarm.cpp ---------------------------------------------------------===//
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
// This file implements common infrastructure for libLLVMSwarmOpts.a, which
// implements several transformations over the Tapir/LLVM intermediate
// representation, including the C bindings for that library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Swarm.h"
#include "llvm-c/Initialization.h"
#include "llvm-c/Transforms/Swarm.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/IR/LegacyPassManager.h"

using namespace llvm;


cl::opt<unsigned> llvm::SwarmRegistersTransferred(
    "swarm-registerstransferred", cl::init(5),
    cl::desc(
        "Number of 64-bit registers that can be transfered as task inputs"));

cl::opt<uint64_t>
    llvm::SwarmCacheLineSize("swarm-cachelinesize", cl::init(64),
                             cl::desc("Size of cache line in bytes."));


/// initializeSwarmOpts - Initialize all passes linked into the
/// SwarmOpts library.
void llvm::initializeSwarmOpts(PassRegistry &Registry) {
  initializeBundlingPass(Registry);
  initializeCreateParallelizableCopyPass(Registry);
  initializeFractalizationPass(Registry);
  initializeInsertHeartbeatsPass(Registry);
  initializeLoopCoarsenPass(Registry);
  initializeLoopExpansionPass(Registry);
  initializeLowerTapirToSwarmPass(Registry);
  initializeSpatialHintsPass(Registry);
}

void LLVMInitializeSwarmOpts(LLVMPassRegistryRef R) {
  initializeSwarmOpts(*unwrap(R));
}


void LLVMAddBundlingPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createBundlingPass());
}

void LLVMAddCreateParallelizableCopyPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createCreateParallelizableCopyPass());
}

void LLVMAddFractalizationPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createFractalizationPass());
}

void LLVMAddInsertHeartbeatsPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createInsertHeartbeatsPass());
}

void LLVMAddLoopCoarsenPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLoopCoarsenPass());
}

void LLVMAddLoopExpansionPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLoopExpansionPass());
}

void LLVMAddLowerTapirToSwarmPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLowerTapirToSwarmPass());
}

void LLVMAddSpatialHintsPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createSpatialHintsPass());
}
