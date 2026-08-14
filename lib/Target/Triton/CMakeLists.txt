add_mlir_library(GraphForgeTargetTriton
  Translate.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/graphforge/Target/Triton

  LINK_LIBS PUBLIC
  GraphForgeControl
  GraphForgeDomain
  GraphForgeKernel
  GraphForgeStorage
  GraphForgeTask
  GraphForgeTensor
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRVectorDialect
  MLIRIR
  MLIRTranslateLib
)
