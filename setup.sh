#!/usr/bin/env bash
##===- setup.sh - Set up build of SCC -------------------------------------===##
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
# This script invokes CMake to set up a build directory.
#
# Before running this script, modify the variable GCC_INSTALL_DIR below to
# point to where GCC is installed that will be used by SCC, and please modify
# BUILD_DIR_ROOT to point to a place where you would like SCC to be built.
# Finally, please ensure you have installed CMake version 3.4.3 or newer.
#
# This script will automatically set up your build to use Ninja and LLD if they
# are installed and available on your PATH.  This substantially speeds up
# incremental rebuilds.  If Ninja isn't available, we'll fall back on using
# Make.  If LLD isn't available, we'll fall back on using ld.gold or ld.bfd.
#
# To learn more about using the LLVM build system and it's options, see:
# https://llvm.org/docs/GettingStarted.html
# https://llvm.org/docs/CMake.html
#
##===----------------------------------------------------------------------===##

# The default value of "/usr" will use your system's default gcc and g++ located in /usr/bin/,
# You may change this to point to another place where you have installed GCC 5.1 or newer.
GCC_INSTALL_DIR="/usr"
# For example, on the Sanchez cluster machines, you could uncomment this line:
#GCC_INSTALL_DIR="/data/sanchez/tools/gcc-6.2"

# Change this line to choose where to put the build directory.
BUILD_DIR_ROOT="/scratch/${USER}"
# WARNING: Build directory contents will exceed 20 GB, and possibly much more
#          depending on build configuration.

set -e

if [[ $1 == "--docs" ]]; then
  echo "Setting up DOCS build."
  BUILD_DIR_NAME=llvm-swarm-docs
elif [[ $1 == "--debug" || $1 == "-d" ]]; then
  echo "Setting up DEBUG build (disabling optimization)."
  BUILD_DIR_NAME=llvm-swarm-debug
  CMAKE_BUILD_TYPE=Debug
elif [[ $1 == "--release" || $1 == "-r" ]]; then
  echo "Setting up RELEASE build (disabling assertions and debug info)."
  BUILD_DIR_NAME=llvm-swarm-release
  CMAKE_BUILD_TYPE=Release
else
  echo "Setting up DEVELOPMENT build."
  BUILD_DIR_NAME=llvm-swarm-build
  CMAKE_BUILD_TYPE=RelWithDebInfo
fi

echo "Using GCC version $("${GCC_INSTALL_DIR}/bin/gcc" -dumpversion)"

# See http://www.ostricher.com/2014/10/the-right-way-to-get-the-directory-of-a-bash-script/
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Compiling from: ${THIS_DIR}"

BUILD_DIR=${BUILD_DIR_ROOT}/${BUILD_DIR_NAME}
echo "Setting up build in: ${BUILD_DIR}"

mkdir -p $BUILD_DIR
cd $BUILD_DIR
if [ -n "$(ls -A)" ]; then
  echo "WARNING! $(pwd) is non-empty."
  read -p "Continue attempt to set up build? (y/n)" -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit
  fi
fi

GENERATOR_NAME="Unix Makefiles"
BUILD="make"
# Prefer to use Ninja, which is faster than Make
if command -v ninja >/dev/null 2>&1; then
  NINJA_VERSION=$(ninja --version 2>&1 | grep -o "[\.0-9]*")
  echo "Found Ninja: $NINJA_VERSION"
  if [[ ! $NINJA_VERSION < "1.3.0" ]]; then
    GENERATOR_NAME="Ninja"
    BUILD="ninja"
  fi
fi
echo "Using generator: ${GENERATOR_NAME}."

CC="${GCC_INSTALL_DIR}/bin/gcc"
CXX="${GCC_INSTALL_DIR}/bin/g++"
if [ -d /data/sanchez/tools/llvm-6.0.1-x86_64-linux-gnu-ubuntu-14.04 ]; then
  source /data/sanchez/tools/llvm-6.0.1-x86_64-linux-gnu-ubuntu-14.04/env.sh
  CC="clang"
  CXX="clang++"
fi

# Pick the fastest available GNU-like linker
LINKER="bfd"
if command -v ld.gold >/dev/null 2>&1; then
  LINKER="gold"
fi
if command -v ld.lld >/dev/null 2>&1; then
  CC_VERSION="$(${CC} -dumpversion)"
  if [[ $CXX == *"clang++"* ]]; then
    # Clang has supported -fuse-ld=lld since version 3.8
    if [[ ! $CC_VERSION < "3.8.0" ]]; then
      LINKER="lld"
    fi
  else
    # GCC didn't support -fuse-ld=lld until version 9
    if [[ ! $CC_VERSION < "9.1.0" ]]; then
      LINKER="lld"
    fi
  fi
fi
echo "Using linker: ${LINKER}"


CMAKE="cmake"
CMAKE_INSTALL_DIR="/data/sanchez/tools/cmake-3.5.2-Linux-x86_64/"
if [ -d ${CMAKE_INSTALL_DIR} ]; then
  CMAKE="${CMAKE_INSTALL_DIR}/bin/cmake"
fi


if [[ $1 == "--docs" ]]; then
  ${CMAKE} \
        ${THIS_DIR} \
        -G "${GENERATOR_NAME}" \
        -DLLVM_BUILD_TOOLS=Off \
        -DCLANG_BUILD_TOOLS=Off \
        -DLLVM_BUILD_DOCS=On \
        -DLLVM_ENABLE_DOXYGEN=On \
        -DLLVM_INSTALL_DOXYGEN_HTML_DIR=/no/such/path/

  echo "Setup to generate docs from source code complete."
  echo "To generate docs from source code, go to ${BUILD_DIR} and run the command:"
  echo "  ${BUILD} doxygen-llvm"
  exit 0
fi


${CMAKE} \
        ${THIS_DIR} \
        -G "${GENERATOR_NAME}" \
        -DCMAKE_INSTALL_PREFIX=/no/such/path/ \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -DCMAKE_C_COMPILER=${CC} \
        -DCMAKE_CXX_COMPILER=${CXX} \
        -DCMAKE_CXX_LINK_FLAGS="-Wl,-rpath=${GCC_INSTALL_DIR}/lib64 -L${GCC_INSTALL_DIR}/lib64" \
        -DLLVM_ENABLE_ASSERTIONS=On \
        -DLLVM_TARGETS_TO_BUILD=X86 \
        -DLLVM_USE_LINKER=${LINKER} \
        -DGCC_INSTALL_PREFIX=${GCC_INSTALL_DIR}/ \
        -DPACKAGE_VENDOR:STRING=SCC \

date >| time_of_build_setup

echo "Build setup complete."
echo "To build, go to ${BUILD_DIR} and run the command:"
echo "  ${BUILD}"
echo "You may wish to pass the -j option to ${BUILD} to specify"
echo "how many jobs to run concurrently."
