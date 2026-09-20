add_mlir_library(TigaTargetTriton
  Translate.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tiga/Target/Triton

  LINK_LIBS PUBLIC
  TigaControl
  TigaDomain
  TigaKernel
  TigaStorage
  TigaTask
  TigaTensor
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRVectorDialect
  MLIRIR
  MLIRTranslateLib
)
