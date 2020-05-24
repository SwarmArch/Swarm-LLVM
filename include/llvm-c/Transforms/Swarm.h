/*===---------------------------Swarm.h ------------------------- -*- C -*-===*\
|*===----------- Swarm Transformation Library C Interface -----------------===*|
|*                                                                            *|
|*                       The SCC Parallelizing Compiler                       *|
|*                                                                            *|
|*          Copyright (c) 2020 Massachusetts Institute of Technology          *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to libLLVMSwarmOpts.a, which          *|
|* implements various Swarm transformations of the LLVM IR.                   *|
|*                                                                            *|
|* Many exotic languages can interoperate with C code but have a harder time  *|
|* with C++ due to name mangling. So in addition to C, this interface enables *|
|* tools written in such languages.                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_TRANSFORMS_SWARM_H
#define LLVM_C_TRANSFORMS_SWARM_H

#include "llvm-c/Types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup LLVMCTransformsSwarm Swarm transformations
 * @ingroup LLVMCTransforms
 *
 * @{
 */

/** See llvm::create*Pass functions. */
void LLVMAddBundlingPass(LLVMPassManagerRef PM);
void LLVMAddCreateParallelizableCopyPass(LLVMPassManagerRef PM);
void LLVMAddFractalizationPass(LLVMPassManagerRef PM);
void LLVMAddInsertHeartbeatsPass(LLVMPassManagerRef PM);
void LLVMAddLoopCoarsenPass(LLVMPassManagerRef PM);
void LLVMAddLoopExpansionPass(LLVMPassManagerRef PM);
void LLVMAddLowerTapirToSwarmPass(LLVMPassManagerRef PM);
void LLVMAddProfitabilityPass(LLVMPassManagerRef PM);
void LLVMAddSpatialHintsPass(LLVMPassManagerRef PM);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif /* defined(__cplusplus) */

#endif
