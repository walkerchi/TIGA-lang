add_mlir_library(GraphForgeTransforms
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
  ${PROJECT_SOURCE_DIR}/include/graphforge/Transforms

  DEPENDS
  GraphForgePassesIncGen

  LINK_LIBS PUBLIC
  GraphForgeControl
  GraphForgeDomain
  GraphForgeIter
  GraphForgeKernel
  GraphForgeStorage
  GraphForgeTask
  GraphForgeTensor
  MLIRArithDialect
  MLIRComplexDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
)
