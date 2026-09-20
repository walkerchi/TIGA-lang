add_mlir_dialect_library(TigaDomain
  DomainDialect.cpp
  DomainOps.cpp
  DomainTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tiga/Dialect/Domain

  DEPENDS
  MLIRDomainOpsIncGen
  TigaDomainTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)
