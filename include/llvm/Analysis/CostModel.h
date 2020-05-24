//===-- CostModel.h - Basic instruction cost model ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the external interface of the basic cost model.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_COSTMODEL_H
#define LLVM_ANALYSIS_COSTMODEL_H

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Instructions.h"

namespace llvm{

unsigned getInstrCost(const Instruction *I, const TargetTransformInfo *TTI);

}

#endif
