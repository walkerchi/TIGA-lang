add_mlir_dialect_library(GraphForgeControl
  ControlDialect.cpp
  ControlOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/graphforge/Dialect/Control

  DEPENDS
  MLIRControlOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)
