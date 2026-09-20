add_mlir_dialect_library(TigaControl
  ControlDialect.cpp
  ControlOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tiga/Dialect/Control

  DEPENDS
  MLIRControlOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)
