add_mlir_dialect_library(GraphForgeTask
  TaskDialect.cpp
  TaskOps.cpp
  TaskTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/graphforge/Dialect/Task

  DEPENDS
  MLIRTaskOpsIncGen
  GraphForgeTaskTypesIncGen

  LINK_LIBS PUBLIC
  GraphForgeDomain
  GraphForgeStorage
  MLIRIR
)
