add_mlir_dialect_library(GraphForgeKernel
  KernelDialect.cpp
  KernelOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/graphforge/Dialect/Kernel

  DEPENDS
  MLIRKernelOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
