#include "graphforge/Dialect/Control/ControlDialect.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::graphforge::control;

LogicalResult RepeatOp::verify() {
  if (getInputs().empty())
    return emitOpError("requires an initial loop-carried tensor");
  if (getInputs().front().getType() != getResult().getType())
    return emitOpError("initial and result tensor types must match");
  if (!llvm::hasSingleElement(getBody()))
    return emitOpError("requires exactly one body block");
  Block &block = getBody().front();
  if (block.getNumArguments() != getInputs().size())
    return emitOpError("requires one body argument per input");
  for (auto [argument, input] : llvm::zip(block.getArguments(), getInputs()))
    if (argument.getType() != input.getType())
      return emitOpError("body argument types must match input types");
  auto yield = dyn_cast<ControlYieldOp>(block.getTerminator());
  if (!yield)
    return emitOpError("body must terminate with gf_control.yield");
  if (yield.getValue().getType() != getResult().getType())
    return emitOpError("yielded tensor type must match the loop result");
  return success();
}

LogicalResult ControlYieldOp::verify() {
  auto repeat = dyn_cast_or_null<RepeatOp>((*this)->getParentOp());
  if (!repeat)
    return emitOpError("must terminate a gf_control.repeat body");
  if (getValue().getType() != repeat.getResult().getType())
    return emitOpError("value type must match the repeat result");
  return success();
}

#define GET_OP_CLASSES
#include "graphforge/Dialect/Control/ControlOps.cpp.inc"
