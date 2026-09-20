add_mlir_library(TigaTransforms
  FusionAnalysis.cpp
  FusionPasses.cpp
  LowerDomainToIter.cpp
  LowerIterToKernel.cpp
  SelectKernelSchedule.cpp
  PlanSplitRows.cpp
  PlanDegreeBuckets.cpp
  DecomposeDegreeWorklists.cpp
  PlanDistributedTasks.cpp
  TensorVJP.cpp
  PlanTensorCheckpoints.cpp
  LowerTensorToCPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tiga/Transforms

  DEPENDS
  TigaPassesIncGen

  LINK_LIBS PUBLIC
  TigaControl
  TigaDomain
  TigaIter
  TigaKernel
  TigaStorage
  TigaTask
  TigaTensor
  MLIRArithDialect
  MLIRComplexDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
)
