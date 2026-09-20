add_mlir_dialect_library(TigaIter
  IterDialect.cpp
  IterOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tiga/Dialect/Iter

  DEPENDS
  MLIRIterOpsIncGen

  LINK_LIBS PUBLIC
  TigaDomain
  MLIRIR
)
