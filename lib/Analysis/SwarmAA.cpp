//===-- SwarmAA.cpp - Utilities for Swarm Alias Analysis -------*- C++ -*--===//
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
// This file contains utility methods for Swarm Alias Analysis. It isn't part
// of SwarmUtils because it needs to be used by code within lib/Analysis, which
// rarely includes anything from lib/Transforms, and when it does, that code
// should not add other deps from lib/Transforms (o/w lld fails to build).
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/SwarmAA.h"

#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "swarm-aa"

void llvm::addSwarmMemArgsMetadata(LoadInst *LI) {
  // We hijack noalias metadata to encode SwarmMemArgs
  // This doesn't match the typical usage of noalias and alias.scope metadata,
  // so it requires some changes to AA to handle. However, it doesn't change
  // the semantics of the original analysis (as long as we don't touch
  // alias.scope metadata, extra noalias metadata doesn't matter). We could use
  // a different metadata tag and write a full AA pass that uses it. This
  // would be cleaner but there's a fair amount of code the would need to
  // change to pass a new AA metadata tag (see AAMDNodes).
  if (isSwarmMemArgs(LI->getMetadata(LLVMContext::MD_noalias)))
    return; // avoid tagging multiple times

  MDBuilder MDB(LI->getContext());
  MDNode *Domain = MDB.createAliasScopeDomain("SwarmMemArgsDomain");
  MDNode *Scope = MDB.createAliasScope("SwarmMemArgs", Domain);
  SmallVector<Metadata *, 1> NoAliases;
  NoAliases.push_back(Scope);
  LI->setMetadata(
      LLVMContext::MD_noalias,
      MDNode::concatenate(LI->getMetadata(LLVMContext::MD_noalias),
                          MDNode::get(LI->getContext(), NoAliases)));
}

void llvm::addSwarmMemArgsForceAliasMetadata(CallInst *CI) {
  // We hijack noalias metadata to encode SwarmMemArgsForceAlias.
  if (isSwarmMemArgsForceAlias(CI->getMetadata(LLVMContext::MD_noalias)))
    return; // avoid tagging multiple times

  MDBuilder MDB(CI->getContext());
  MDNode *Domain = MDB.createAliasScopeDomain("SwarmMemArgsForceAliasDomain");
  MDNode *Scope = MDB.createAliasScope("SwarmMemArgsForceAlias", Domain);
  SmallVector<Metadata *, 1> NoAliases;
  NoAliases.push_back(Scope);
  CI->setMetadata(
      LLVMContext::MD_noalias,
      MDNode::concatenate(CI->getMetadata(LLVMContext::MD_noalias),
                          MDNode::get(CI->getContext(), NoAliases)));
}

static inline bool hasDomainWithName(const MDNode *NoAlias, const char *Name) {
  // Code based on ScopedNoAliasAA
  if (!NoAlias) {
    DEBUG(dbgs() << "No matches found for " << Name << " (empty metadata)\n");
    return false;
  }
  auto getDomain = [](const MDNode *Node) -> const MDNode * {
    if (Node->getNumOperands() < 2)
      return nullptr;
    return dyn_cast_or_null<MDNode>(Node->getOperand(1));
  };
  for (const MDOperand &MDOp : NoAlias->operands())
    if (const MDNode *NAMD = dyn_cast<MDNode>(MDOp))
      if (const MDNode *Domain = getDomain(NAMD))
        for (const MDOperand &MDDOp : Domain->operands())
          if (const MDString *DomStr = dyn_cast<MDString>(MDDOp)) {
            if (DomStr->getString() == Name) {
              DEBUG(dbgs() << "Found matching domain " << Name << "\n");
              return true;
            } else {
              DEBUG(dbgs() << "Found non-matching domain "
                           << DomStr->getString() << " !=" << Name << "\n");
            }
          }
  DEBUG(dbgs() << "No matches found for " << Name << "\n");
  return false;
}

bool llvm::isSwarmMemArgs(const MDNode *NoAlias) {
    return hasDomainWithName(NoAlias, "SwarmMemArgsDomain");
}

bool llvm::isSwarmMemArgsForceAlias(const MDNode *NoAlias) {
    return hasDomainWithName(NoAlias, "SwarmMemArgsForceAliasDomain");
}
