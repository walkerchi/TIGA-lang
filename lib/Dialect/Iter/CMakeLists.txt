add_mlir_dialect_library(GraphForgeIter
  IterDialect.cpp
  IterOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/graphforge/Dialect/Iter

  DEPENDS
  MLIRIterOpsIncGen

  LINK_LIBS PUBLIC
  GraphForgeDomain
  MLIRIR
)
