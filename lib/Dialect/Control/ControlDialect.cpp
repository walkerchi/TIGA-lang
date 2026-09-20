#include "tiga/Dialect/Control/ControlDialect.h"

#include "tiga/Dialect/Control/ControlOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::tiga::control;

void TigaControlDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tiga/Dialect/Control/ControlOps.cpp.inc"
      >();
}
