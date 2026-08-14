#include "graphforge/Dialect/Task/TaskDialect.h"

#include "graphforge/Dialect/Task/TaskOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::graphforge::task;

void GraphForgeTaskDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "graphforge/Dialect/Task/TaskTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "graphforge/Dialect/Task/TaskOps.cpp.inc"
      >();
}
