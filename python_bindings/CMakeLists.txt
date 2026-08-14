find_package(Python3 REQUIRED COMPONENTS Development.Module)

Python3_add_library(_graphforge_compiler MODULE compiler_module.cpp WITH_SOABI)
target_link_libraries(_graphforge_compiler PRIVATE
  GraphForgeDomain
  GraphForgeIter
  GraphForgeKernel
  GraphForgeStorage
  GraphForgeTask
  GraphForgeTensor
  GraphForgeTransforms
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
target_compile_definitions(_graphforge_compiler PRIVATE PY_SSIZE_T_CLEAN)
set_target_properties(_graphforge_compiler PROPERTIES
  PREFIX ""
  INSTALL_RPATH "${GRAPHFORGE_EXTENSION_RPATH}")

install(
  TARGETS _graphforge_compiler
  LIBRARY DESTINATION graphforge
  COMPONENT GraphForgeWheel)
