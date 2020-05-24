add_llvm_library(LLVMSwarmOpts
  Bundling.cpp
  CreateParallelizableCopy.cpp
  Fractalizer.cpp
  InsertHeartbeats.cpp
  LoopCoarsen.cpp
  LoopExpansion.cpp
  LoopIterDetacher.cpp
  LowerToSwarm.cpp
  Profitability.cpp
  SpatialHints.cpp
  Swarm.cpp
  Utils/CFGRegions.cpp
  Utils/InstructionCost.cpp
  Utils/Misc.cpp
  Utils/Reductions.cpp
  Utils/Tasks.cpp
  Utils/SwarmABI/SwarmABI.cpp
  Utils/SwarmABI/ordspecsim_v1.cpp
  Utils/SwarmABI/sccrt_serial.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  )

add_dependencies(LLVMSwarmOpts intrinsics_gen)
