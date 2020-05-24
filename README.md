SCC: The Swarm Compiler Collection
==================================

The SCC project builds compilers for parallelizing sequential code
for execution on [the Swarm architecture](http://swarm.csail.mit.edu/).
SCC was initially developed as a fork of the
[Tapir/LLVM](https://github.com/wsmoses/Tapir-LLVM) compiler framework.
SCC includes the standard LLVM/Clang tools (`clang`, `clang++`, `opt`, etc.)
that perform ordinary compilation of C/C++ programs to run natively on x86,
but with additional capabilities that can be invoked with SCC-specific command
line flags and C/C++ language extensions, to produce binaries that use Swarm
hardware features.

The design of SCC is described in the following paper:

Victor A. Ying, Mark C. Jeffrey, and Daniel Sanchez.
"T4: Compiling Sequential Code for Effective Speculative Parallelization in Hardware."
Proceedings of the 47th Annual International Symposium on Computer Architecture (ISCA), June 2020.

SCC is open-source software.  You are free to modify and distribute it
under the terms of the license agreement found in LICENSE.TXT.
If you use SCC in your research, we ask that you cite the T4 paper in your
publications and that you send a citation of your work to swarm@csail.mit.edu.

# Getting started

Please see the file `setup.sh` to see how to build SCC.  After building SCC,
please see `env.sh` to set up your environment to use SCC.  After you have set
up your environment appropriately, check the output of
```
clang --version
```
to confirm you are set up to run SCC.

See [the README in `lib/Transforms/Swarm`](lib/Transforms/Swarm/README.mdown)
to find the implementation of SCC's new compiler passes that transform code
to run efficiently in parallel on Swarm.
