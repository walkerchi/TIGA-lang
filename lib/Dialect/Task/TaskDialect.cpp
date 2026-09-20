#include "tiga/Dialect/Task/TaskDialect.h"

#include "tiga/Dialect/Task/TaskOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::tiga::task;

void TigaTaskDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "tiga/Dialect/Task/TaskTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "tiga/Dialect/Task/TaskOps.cpp.inc"
      >();
}
