#include "tiga/Dialect/Storage/StorageDialect.h"

#include "tiga/Dialect/Storage/StorageOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::tiga::storage;

void TigaStorageDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "tiga/Dialect/Storage/StorageTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "tiga/Dialect/Storage/StorageOps.cpp.inc"
      >();
}
