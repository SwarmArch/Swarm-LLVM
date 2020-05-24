##===- env.sh - Set up environment to use SCC -----------------------------===##
#
#                       The SCC Parallelizing Compiler
#
#          Copyright (c) 2020 Massachusetts Institute of Technology
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
#
# This script is intended to be sourced in order to set the PATH as well as
# some other environment variables that may aid in using the compiler.
#
##===----------------------------------------------------------------------===##

# Change this line to indicate where you put the build directory.
BUILD_DIR_ROOT="/scratch/${USER}"

if [[ $1 == "--debug" || $1 == "-d" ]]; then
  echo "Setting up to use DEBUG build."
  DIR="${BUILD_DIR_ROOT}/llvm-swarm-debug"
elif [[ $1 == "--release" || $1 == "-r" ]]; then
  echo "Setting up to use RELEASE build."
  DIR="${BUILD_DIR_ROOT}/llvm-swarm-release"
else
  echo "Setting up to use DEV build."
  DIR="${BUILD_DIR_ROOT}/llvm-swarm-build"
fi
export LLVM_DIR="${DIR}/lib/cmake/llvm"
export PATH="${DIR}/bin:${PATH}"
export LIBRARY_PATH="${DIR}/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${DIR}/lib:${LD_LIBRARY_PATH}"

#export PATH="/data/sanchez/tools/cmake-3.5.2-Linux-x86_64/bin:${PATH}"
