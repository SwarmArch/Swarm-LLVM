//===- SCCRT.h - Compiler runtime interface -------------------------------===//
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
// Utilities for generating calls to libsccrt, SCC's runtime library.
//
//===----------------------------------------------------------------------===//

#ifndef SCCRT_H_
#define SCCRT_H_

#include "swarm_runtime/include/scc/rt.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/TypeBuilder.h"

// [victory] TypeBuilder was a feature in upstream LLVM, which is doing wonders
//           for us here so we'll just have to keep a copy of it around for SCC.
//           It was removed from upstream for the LLVM 8.0 release:
//           https://reviews.llvm.org/D56573
//           You can see Tapir moving away from TypeBuilder as a result here:
//           https://github.com/wsmoses/Tapir-LLVM/commit/14df85e4ca19b01083bd99ed8dbea47fd4b85892#diff-c28f34ee61c1819df521c7ba06215730

// A bug in GCC prior to version 7 requires us to open a namespace here. See:
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=56480
// https://stackoverflow.com/q/25594644/12178985
namespace llvm {

// TypeBuilder.h provides template specializations for function types with
// 0 to 5 arguments, but our sccrt_serial enqueue functions have 7 arguments.
template<typename R, typename A1, typename A2, typename A3, typename A4,
         typename A5, typename A6, typename A7, bool cross>
class TypeBuilder<R(A1, A2, A3, A4, A5, A6, A7), cross> {
public:
  static FunctionType *get(LLVMContext &Context) {
    Type *params[] = {
      TypeBuilder<A1, cross>::get(Context),
      TypeBuilder<A2, cross>::get(Context),
      TypeBuilder<A3, cross>::get(Context),
      TypeBuilder<A4, cross>::get(Context),
      TypeBuilder<A5, cross>::get(Context),
      TypeBuilder<A6, cross>::get(Context),
      TypeBuilder<A7, cross>::get(Context),
    };
    return FunctionType::get(TypeBuilder<R, cross>::get(Context),
                             params, false);
  }
};

} // namespace llvm


#define RUNTIME_FUNC(name, M) \
  llvm::cast<llvm::Function>((M)->getOrInsertFunction( \
            #name, \
            llvm::TypeBuilder<decltype(name), false>::get((M)->getContext())))

#endif
