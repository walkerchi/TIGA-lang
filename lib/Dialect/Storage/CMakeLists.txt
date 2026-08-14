add_mlir_dialect_library(GraphForgeStorage
  StorageDialect.cpp
  StorageOps.cpp
  StorageTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/graphforge/Dialect/Storage

  DEPENDS
  MLIRStorageOpsIncGen
  GraphForgeStorageTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
