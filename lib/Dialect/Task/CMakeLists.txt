add_mlir_dialect_library(TigaTask
  TaskDialect.cpp
  TaskOps.cpp
  TaskTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tiga/Dialect/Task

  DEPENDS
  MLIRTaskOpsIncGen
  TigaTaskTypesIncGen

  LINK_LIBS PUBLIC
  TigaDomain
  TigaStorage
  MLIRIR
)
