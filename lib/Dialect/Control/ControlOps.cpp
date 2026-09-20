#include "tiga/Dialect/Control/ControlDialect.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::tiga::control;

namespace {

LogicalResult verifyControlState(Operation *operation, ValueRange inputs,
                                 ValueRange results, int64_t carried,
                                 Region &region, StringRef regionName) {
  if (carried <= 0 || static_cast<size_t>(carried) > inputs.size())
    return operation->emitOpError(
        "num_carried must select a non-empty input prefix");
  if (results.size() != static_cast<size_t>(carried))
    return operation->emitOpError(
        "requires one result per loop-carried input");
  for (int64_t index = 0; index < carried; ++index)
    if (inputs[index].getType() != results[index].getType())
      return operation->emitOpError(
          "carried input and result tensor types must match");
  if (!llvm::hasSingleElement(region))
    return operation->emitOpError() << "requires exactly one " << regionName
                                    << " block";
  Block &block = region.front();
  if (block.getNumArguments() != inputs.size())
    return operation->emitOpError() << "requires one " << regionName
                                    << " argument per input";
  for (auto [argument, input] : llvm::zip(block.getArguments(), inputs))
    if (argument.getType() != input.getType())
      return operation->emitOpError() << regionName
                                      << " argument types must match input types";
  return success();
}

} // namespace

LogicalResult RepeatOp::verify() {
  int64_t carried = getNumCarried();
  if (failed(verifyControlState(*this, getInputs(), getResults(), carried,
                                getBody(), "body")))
    return failure();
  Block &block = getBody().front();
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

LogicalResult WhileOp::verify() {
  int64_t carried = getNumCarried();
  if (failed(verifyControlState(*this, getInputs(), getResults(), carried,
                                getCondition(), "condition")) ||
      failed(verifyControlState(*this, getInputs(), getResults(), carried,
                                getBody(), "body")))
    return failure();
  auto condition = dyn_cast<ConditionOp>(
      getCondition().front().getTerminator());
  if (!condition)
    return emitOpError("condition must terminate with gf_control.condition");
  auto conditionType = condition.getValue().getType();
  if (conditionType.getRank() != 0 ||
      !conditionType.getElementType().isInteger(1))
    return emitOpError("condition must yield a rank-zero tensor<i1>");
  auto yield = dyn_cast<ControlYieldOp>(getBody().front().getTerminator());
  if (!yield)
    return emitOpError("body must terminate with gf_control.yield");
  if (yield.getValues().size() != static_cast<size_t>(carried))
    return emitOpError("body must yield one tensor per carried input");
  for (int64_t index = 0; index < carried; ++index)
    if (yield.getValues()[index].getType() != getResults()[index].getType())
      return emitOpError("yielded tensor types must match loop results");
  return success();
}

LogicalResult ConditionOp::verify() {
  auto bounded = dyn_cast_or_null<WhileOp>((*this)->getParentOp());
  if (!bounded || getOperation() !=
                      bounded.getCondition().front().getTerminator())
    return emitOpError("must terminate a gf_control.while condition region");
  auto type = getValue().getType();
  if (type.getRank() != 0 || !type.getElementType().isInteger(1))
    return emitOpError("requires a rank-zero tensor<i1> value");
  return success();
}

LogicalResult ControlYieldOp::verify() {
  Operation *parent = (*this)->getParentOp();
  ValueRange results;
  Region *body = nullptr;
  if (auto repeat = dyn_cast_or_null<RepeatOp>(parent)) {
    results = repeat.getResults();
    body = &repeat.getBody();
  } else if (auto bounded = dyn_cast_or_null<WhileOp>(parent)) {
    results = bounded.getResults();
    body = &bounded.getBody();
  } else {
    return emitOpError("must terminate a gf_control loop body");
  }
  if (getOperation() != body->front().getTerminator())
    return emitOpError("must be the loop body terminator");
  if (getValues().size() != results.size())
    return emitOpError("must yield one value per loop result");
  for (auto [value, result] : llvm::zip(getValues(), results))
    if (value.getType() != result.getType())
      return emitOpError("value types must match the loop results");
  return success();
}

#define GET_OP_CLASSES
#include "tiga/Dialect/Control/ControlOps.cpp.inc"
