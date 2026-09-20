#ifndef TIGA_DIALECT_TASK_TASKDIALECT_H
#define TIGA_DIALECT_TASK_TASKDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"

#include "tiga/Dialect/Domain/DomainDialect.h"
#include "tiga/Dialect/Storage/StorageDialect.h"
#include "tiga/Dialect/Task/TaskOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "tiga/Dialect/Task/TaskTypes.h.inc"

#define GET_OP_CLASSES
#include "tiga/Dialect/Task/TaskOps.h.inc"

#endif
