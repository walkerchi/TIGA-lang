#include "graphforge/Dialect/Domain/DomainDialect.h"

#include "graphforge/Dialect/Domain/DomainOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::graphforge;

void GraphForgeDomainDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "graphforge/Dialect/Domain/DomainTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "graphforge/Dialect/Domain/DomainOps.cpp.inc"
      >();
}
