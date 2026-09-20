add_mlir_dialect_library(TigaKernel
  KernelDialect.cpp
  KernelOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tiga/Dialect/Kernel

  DEPENDS
  MLIRKernelOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
