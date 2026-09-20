add_mlir_tool(gf-opt gf-opt.cpp)

set_target_properties(gf-opt PROPERTIES
  INSTALL_RPATH "${TIGA_TOOL_RPATH}")

target_link_libraries(gf-opt PRIVATE
  TigaControl
  TigaDomain
  TigaIter
  TigaKernel
  TigaStorage
  TigaTask
  TigaTensor
  TigaTransforms
  MLIRArithDialect
  MLIRComplexDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIROptLib
)

install(
  TARGETS gf-opt
  RUNTIME DESTINATION tiga/bin
  COMPONENT TigaWheel)
