find_package(Python3 REQUIRED COMPONENTS Development.Module)

Python3_add_library(_tiga_compiler MODULE compiler_module.cpp WITH_SOABI)
target_link_libraries(_tiga_compiler PRIVATE
  TigaControl
  TigaDomain
  TigaIter
  TigaKernel
  TigaStorage
  TigaTask
  TigaTensor
  TigaTransforms
  MLIRArithDialect
  MLIRArithToLLVM
  MLIRBuiltinToLLVMIRTranslation
  MLIRComplexDialect
  MLIRComplexToLLVM
  MLIRControlFlowToLLVM
  MLIRConvertToLLVMPass
  MLIRExecutionEngine
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRIndexToLLVM
  MLIRMathDialect
  MLIRMathToLLVM
  MLIRMemRefDialect
  MLIRMemRefToLLVM
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRToLLVMIRTranslationRegistration
  MLIRTransforms
  MLIRVectorDialect
  MLIRVectorToLLVMPass
)
target_compile_definitions(_tiga_compiler PRIVATE PY_SSIZE_T_CLEAN)
set_target_properties(_tiga_compiler PROPERTIES
  PREFIX ""
  INSTALL_RPATH "${TIGA_EXTENSION_RPATH}")

install(
  TARGETS _tiga_compiler
  LIBRARY DESTINATION tiga
  COMPONENT TigaWheel)
