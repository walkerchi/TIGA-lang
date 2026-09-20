add_mlir_dialect_library(TigaStorage
  StorageDialect.cpp
  StorageOps.cpp
  StorageTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tiga/Dialect/Storage

  DEPENDS
  MLIRStorageOpsIncGen
  TigaStorageTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
