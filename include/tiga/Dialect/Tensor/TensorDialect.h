#ifndef TIGA_DIALECT_TENSOR_TENSORDIALECT_H
#define TIGA_DIALECT_TENSOR_TENSORDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "tiga/Dialect/Tensor/TensorOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "tiga/Dialect/Tensor/TensorOps.h.inc"

#endif
