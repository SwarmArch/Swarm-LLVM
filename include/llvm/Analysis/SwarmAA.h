//===-- SwarmAA.h - Utilities for Swarm Alias Analysis ---------*- C++ -*--===//
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

#ifndef LLVM_ANALYSIS_SWARMAA_H
#define LLVM_ANALYSIS_SWARMAA_H

namespace llvm {

class CallInst;
class LoadInst;
class MDNode;

/// Tag this load as accessing a Swarm memory arguments structure, which
/// enables aggressive alias analysis: these loads do NOT alias with any other
/// loads, storres, or calls. The exception is the call to free the MemArgs
/// structure, which must be tagged with addSwarmMemArgsForceAliasMetadata().
/// This alias analysis is safe because MemArgs are constant whenever read
/// (they are stored to only before a task boundary, across which we never hoist
/// a load). Therefore, other stores or function calls cannot modify them.
void addSwarmMemArgsMetadata(LoadInst *LI);

/// Tag this call as aliasing with SwarmMemArgs loads. Ally only to free()s of
/// SwarmMemArgs data.
void addSwarmMemArgsForceAliasMetadata(CallInst *CI);

/// Returns true if MD_noalias metadata denotes this is a SwarmMemArgs load
bool isSwarmMemArgs(const MDNode *NoAlias);

/// Returns true if MD_alias_scope metadata denotes this is a
/// SwarmMemArgsForceAlias call
bool isSwarmMemArgsForceAlias(const MDNode *NoAlias);

} // namespace llvm

#endif
