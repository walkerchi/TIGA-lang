#ifndef TIGA_DIALECT_STORAGE_STORAGEDIALECT_H
#define TIGA_DIALECT_STORAGE_STORAGEDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "tiga/Dialect/Storage/StorageOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "tiga/Dialect/Storage/StorageTypes.h.inc"

#define GET_OP_CLASSES
#include "tiga/Dialect/Storage/StorageOps.h.inc"

#endif
