add_mlir_dialect_library(GraphForgeDomain
  DomainDialect.cpp
  DomainOps.cpp
  DomainTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/graphforge/Dialect/Domain

  DEPENDS
  MLIRDomainOpsIncGen
  GraphForgeDomainTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)
