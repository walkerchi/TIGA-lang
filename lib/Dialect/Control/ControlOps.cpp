#include "graphforge/Dialect/Control/ControlDialect.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::graphforge::control;

LogicalResult RepeatOp::verify() {
  int64_t carried = getNumCarried();
  if (carried <= 0 || static_cast<size_t>(carried) > getInputs().size())
    return emitOpError("num_carried must select a non-empty input prefix");
  if (getResults().size() != static_cast<size_t>(carried))
    return emitOpError("requires one result per loop-carried input");
  for (int64_t index = 0; index < carried; ++index)
    if (getInputs()[index].getType() != getResults()[index].getType())
      return emitOpError("carried input and result tensor types must match");
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
  if (yield.getValues().size() != static_cast<size_t>(carried))
    return emitOpError("body must yield one tensor per carried input");
  for (int64_t index = 0; index < carried; ++index)
    if (yield.getValues()[index].getType() != getResults()[index].getType())
      return emitOpError("yielded tensor types must match loop results");
  return success();
}

LogicalResult ControlYieldOp::verify() {
  auto repeat = dyn_cast_or_null<RepeatOp>((*this)->getParentOp());
  if (!repeat)
    return emitOpError("must terminate a gf_control.repeat body");
  if (getValues().size() != repeat.getResults().size())
    return emitOpError("must yield one value per repeat result");
  for (auto [value, result] : llvm::zip(getValues(), repeat.getResults()))
    if (value.getType() != result.getType())
      return emitOpError("value types must match the repeat results");
  return success();
}

#define GET_OP_CLASSES
#include "graphforge/Dialect/Control/ControlOps.cpp.inc"
