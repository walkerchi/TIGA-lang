#include "tiga/Dialect/Domain/DomainDialect.h"

#include "tiga/Dialect/Domain/DomainOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::tiga;

void TigaDomainDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "tiga/Dialect/Domain/DomainTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "tiga/Dialect/Domain/DomainOps.cpp.inc"
      >();
}
