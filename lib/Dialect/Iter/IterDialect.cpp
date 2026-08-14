#include "graphforge/Dialect/Iter/IterDialect.h"

#include "graphforge/Dialect/Iter/IterOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::graphforge::iter;

void GraphForgeIterDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "graphforge/Dialect/Iter/IterOps.cpp.inc"
      >();
}
