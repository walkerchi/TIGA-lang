add_mlir_dialect_library(GraphForgeTensor
  TensorDialect.cpp
  TensorOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/graphforge/Dialect/Tensor

  DEPENDS
  MLIRTensorOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)
