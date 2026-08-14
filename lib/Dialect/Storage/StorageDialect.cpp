#include "graphforge/Dialect/Storage/StorageDialect.h"

#include "graphforge/Dialect/Storage/StorageOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::graphforge::storage;

void GraphForgeStorageDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "graphforge/Dialect/Storage/StorageTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "graphforge/Dialect/Storage/StorageOps.cpp.inc"
      >();
}
