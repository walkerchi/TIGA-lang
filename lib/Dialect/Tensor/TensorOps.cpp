#include "graphforge/Dialect/Tensor/TensorDialect.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::graphforge::tensor;

namespace {

static bool compatibleElementTypes(RankedTensorType lhs, RankedTensorType rhs,
                                   RankedTensorType result) {
  return lhs.getElementType() == rhs.getElementType() &&
         lhs.getElementType() == result.getElementType();
}

static SmallVector<int64_t> broadcastShape(ArrayRef<int64_t> lhs,
                                           ArrayRef<int64_t> rhs) {
  int64_t rank = std::max(lhs.size(), rhs.size());
  SmallVector<int64_t> shape(rank, 1);
  for (int64_t index = 0; index < rank; ++index) {
    int64_t lhsIndex = index - (rank - lhs.size());
    int64_t rhsIndex = index - (rank - rhs.size());
    int64_t left = lhsIndex < 0 ? 1 : lhs[lhsIndex];
    int64_t right = rhsIndex < 0 ? 1 : rhs[rhsIndex];
    if (left == ShapedType::kDynamic || right == ShapedType::kDynamic)
      shape[index] = ShapedType::kDynamic;
    else if (left == right || left == 1)
      shape[index] = right;
    else if (right == 1)
      shape[index] = left;
    else
      return {};
  }
  return shape;
}

static LogicalResult verifyBinary(Operation *operation, Value lhsValue,
                                  Value rhsValue, Value resultValue) {
  auto lhs = cast<RankedTensorType>(lhsValue.getType());
  auto rhs = cast<RankedTensorType>(rhsValue.getType());
  auto result = cast<RankedTensorType>(resultValue.getType());
  if (!compatibleElementTypes(lhs, rhs, result))
    return operation->emitOpError("requires identical element types");
  SmallVector<int64_t> expected = broadcastShape(lhs.getShape(), rhs.getShape());
  if (expected.empty() && std::max(lhs.getRank(), rhs.getRank()) != 0)
    return operation->emitOpError("operands are not broadcast-compatible");
  if (ArrayRef<int64_t>(expected) != result.getShape())
    return operation->emitOpError("result shape does not match broadcasting");
  return success();
}

static LogicalResult verifySameElement(Operation *operation,
                                       RankedTensorType input,
                                       RankedTensorType result) {
  if (input.getElementType() != result.getElementType())
    return operation->emitOpError("must preserve the element type");
  return success();
}

static std::optional<int64_t> staticNumElements(RankedTensorType type) {
  if (!type.hasStaticShape())
    return std::nullopt;
  return type.getNumElements();
}

} // namespace

LogicalResult InputOp::verify() {
  auto buffer = getBuffer().getType();
  auto result = getResult().getType();
  if (buffer != result)
    return emitOpError("must preserve the ABI tensor type");
  if (getOffsetAttr().getInt() < 0)
    return emitOpError("requires a non-negative storage offset");
  if (static_cast<int64_t>(getStrides().size()) != result.getRank())
    return emitOpError("requires one stride per logical dimension");
  if (llvm::any_of(getStrides(), [](int64_t stride) { return stride < 0; }))
    return emitOpError("does not support negative strides yet");
  return success();
}

LogicalResult AddOp::verify() {
  return verifyBinary(*this, getLhs(), getRhs(), getResult());
}

LogicalResult MulOp::verify() {
  return verifyBinary(*this, getLhs(), getRhs(), getResult());
}

LogicalResult MatmulOp::verify() {
  auto lhs = getLhs().getType();
  auto rhs = getRhs().getType();
  auto result = getResult().getType();
  if (lhs.getRank() != 2 || rhs.getRank() != 2 || result.getRank() != 2)
    return emitOpError("requires rank-two operands and result");
  if (!compatibleElementTypes(lhs, rhs, result))
    return emitOpError("requires identical element types");
  if (!isa<FloatType, ComplexType>(lhs.getElementType()))
    return emitOpError("requires floating-point or complex elements");
  if (lhs.getDimSize(1) != rhs.getDimSize(0))
    return emitOpError("contraction dimensions must match");
  if (result.getDimSize(0) != lhs.getDimSize(0) ||
      result.getDimSize(1) != rhs.getDimSize(1))
    return emitOpError("result shape must be [lhs rows, rhs columns]");
  return success();
}

LogicalResult DivOp::verify() {
  auto element = getLhs().getType().getElementType();
  if (!isa<FloatType, ComplexType>(element))
    return emitOpError("requires a floating-point or complex element type");
  return verifyBinary(*this, getLhs(), getRhs(), getResult());
}

LogicalResult NegOp::verify() {
  if (getInput().getType() != getResult().getType())
    return emitOpError("must preserve shape and element type");
  auto element = getInput().getType().getElementType();
  if (!isa<IntegerType, FloatType, ComplexType>(element))
    return emitOpError("requires an arithmetic element type");
  return success();
}

LogicalResult ExpOp::verify() {
  if (getInput().getType() != getResult().getType())
    return emitOpError("must preserve shape and element type");
  if (!isa<FloatType>(getInput().getType().getElementType()))
    return emitOpError("requires a floating-point element type");
  return success();
}

LogicalResult SqrtOp::verify() {
  if (getInput().getType() != getResult().getType())
    return emitOpError("must preserve shape and element type");
  if (!isa<FloatType>(getInput().getType().getElementType()))
    return emitOpError("requires a floating-point element type");
  return success();
}

LogicalResult CSREuclideanDistanceSumVJPOp::verify() {
  auto positions = getPositions().getType();
  auto sourceValue = getSourceValue().getType();
  auto destination = getDestination().getType();
  auto sourceIndex = getSourceIndex().getType();
  auto upstream = getUpstream().getType();
  auto lattice = getLattice().getType();
  auto inverse = getInverseLattice().getType();
  auto result = getResult().getType();
  int64_t dimensions = getDimensionsAttr().getInt();
  if (dimensions != 2 && dimensions != 3)
    return emitOpError("currently supports dimensions=2 or dimensions=3");
  if (positions.getRank() != 2 || positions.getDimSize(1) != dimensions ||
      !positions.getElementType().isF32())
    return emitOpError("requires FP32 positions shaped [N,D]");
  int64_t rows = positions.getDimSize(0);
  if (sourceValue.getRank() != 1 || sourceValue.getDimSize(0) != rows ||
      sourceValue.getElementType() != positions.getElementType() ||
      upstream.getRank() != 1 || upstream.getDimSize(0) != rows ||
      upstream.getElementType() != positions.getElementType())
    return emitOpError("requires FP32 source/upstream values shaped [N]");
  if (destination.getRank() != 1 || sourceIndex.getRank() != 1 ||
      destination.getShape() != sourceIndex.getShape() ||
      !destination.getElementType().isInteger(64) ||
      sourceIndex.getElementType() != destination.getElementType())
    return emitOpError("requires matching i64 destination/source edge indices");
  for (RankedTensorType matrix : {lattice, inverse}) {
    if (matrix.getRank() != 2 || matrix.getDimSize(0) != dimensions ||
        matrix.getDimSize(1) != dimensions ||
        matrix.getElementType() != positions.getElementType())
      return emitOpError("requires FP32 lattice matrices shaped [D,D]");
  }
  if (result.getRank() != 2 || result.getDimSize(0) != rows ||
      result.getDimSize(1) != dimensions + 1 ||
      result.getElementType() != positions.getElementType())
    return emitOpError("packed result must be FP32 [N,D+1]");
  return success();
}

LogicalResult ConjOp::verify() {
  auto input = getInput().getType();
  auto result = getResult().getType();
  if (!isa<ComplexType>(input.getElementType()))
    return emitOpError("requires a complex element type");
  if (input != result)
    return emitOpError("must preserve shape and element type");
  return success();
}

LogicalResult ReshapeOp::verify() {
  auto input = getInput().getType();
  auto result = getResult().getType();
  if (failed(verifySameElement(*this, input, result)))
    return failure();
  if (getShape() != result.getShape())
    return emitOpError("shape attribute must equal the result shape");
  if (staticNumElements(input) != staticNumElements(result))
    return emitOpError("cannot change the number of elements");
  return success();
}

LogicalResult PermuteOp::verify() {
  auto input = getInput().getType();
  auto result = getResult().getType();
  if (failed(verifySameElement(*this, input, result)))
    return failure();
  ArrayRef<int64_t> axes = getAxes();
  if (static_cast<int64_t>(axes.size()) != input.getRank() ||
      result.getRank() != input.getRank())
    return emitOpError("requires one axis per input dimension");
  llvm::SmallDenseSet<int64_t> seen;
  SmallVector<int64_t> expected;
  for (int64_t axis : axes) {
    if (axis < 0 || axis >= input.getRank() || !seen.insert(axis).second)
      return emitOpError("axes must be a permutation of input dimensions");
    expected.push_back(input.getDimSize(axis));
  }
  if (ArrayRef<int64_t>(expected) != result.getShape())
    return emitOpError("result shape does not match the permutation");
  return success();
}

LogicalResult BroadcastOp::verify() {
  auto input = getInput().getType();
  auto result = getResult().getType();
  if (failed(verifySameElement(*this, input, result)))
    return failure();
  if (getShape() != result.getShape())
    return emitOpError("shape attribute must equal the result shape");
  if (input.getRank() > result.getRank())
    return emitOpError("cannot broadcast to a lower rank");
  int64_t padding = result.getRank() - input.getRank();
  for (int64_t axis = 0; axis < input.getRank(); ++axis) {
    int64_t source = input.getDimSize(axis);
    int64_t target = result.getDimSize(padding + axis);
    if (source != ShapedType::kDynamic && target != ShapedType::kDynamic &&
        source != 1 && source != target)
      return emitOpError("has a non-singleton dimension that cannot expand");
  }
  return success();
}

LogicalResult CheckpointOp::verify() {
  if (getInput().getType() != getResult().getType())
    return emitOpError("must preserve shape and element type");
  return success();
}

LogicalResult CheckpointCandidateOp::verify() {
  if (getInput().getType() != getResult().getType())
    return emitOpError("must preserve shape and element type");
  return success();
}

LogicalResult ReduceSumOp::verify() {
  auto input = getInput().getType();
  auto result = getResult().getType();
  if (failed(verifySameElement(*this, input, result)))
    return failure();
  llvm::SmallDenseSet<int64_t> reduced;
  for (int64_t axis : getAxes())
    if (axis < 0 || axis >= input.getRank() || !reduced.insert(axis).second)
      return emitOpError("axes must be unique input dimensions");
  SmallVector<int64_t> expected;
  for (int64_t axis = 0; axis < input.getRank(); ++axis) {
    if (reduced.contains(axis)) {
      if (getKeepDims())
        expected.push_back(1);
    } else {
      expected.push_back(input.getDimSize(axis));
    }
  }
  if (ArrayRef<int64_t>(expected) != result.getShape())
    return emitOpError("result shape does not match axes/keep_dims");
  return success();
}

LogicalResult CumsumOp::verify() {
  auto input = getInput().getType();
  auto result = getResult().getType();
  if (input != result)
    return emitOpError("must preserve shape and element type");
  if (input.getRank() == 0)
    return emitOpError("requires a non-scalar tensor");
  int64_t axis = getAxisAttr().getInt();
  if (axis < 0 || axis >= input.getRank())
    return emitOpError("axis must name an input dimension");
  if (!isa<IntegerType, FloatType, ComplexType>(input.getElementType()))
    return emitOpError("requires an arithmetic element type");
  return success();
}

LogicalResult GatherOp::verify() {
  auto input = getInput().getType();
  auto index = getIndex().getType();
  auto result = getResult().getType();
  if (input.getRank() < 1 || index.getRank() != 1 || result.getRank() < 1)
    return emitOpError("requires ranked entity input/result and rank-one index");
  if (!isa<IntegerType>(index.getElementType()))
    return emitOpError("requires an integer index");
  if (input.getElementType() != result.getElementType())
    return emitOpError("must preserve the input element type");
  SmallVector<int64_t> expected{index.getDimSize(0)};
  expected.append(input.getShape().begin() + 1, input.getShape().end());
  if (ArrayRef<int64_t>(expected) != result.getShape())
    return emitOpError("result shape must replace the leading extent by index size");
  return success();
}

LogicalResult SegmentSumOp::verify() {
  auto input = getInput().getType();
  auto index = getIndex().getType();
  auto result = getResult().getType();
  if (input.getRank() < 1 || index.getRank() != 1 || result.getRank() < 1)
    return emitOpError("requires ranked message/result and rank-one index");
  if (!isa<IntegerType>(index.getElementType()))
    return emitOpError("requires an integer index");
  if (input.getDimSize(0) != index.getDimSize(0))
    return emitOpError("message and index leading extents must match");
  if (input.getElementType() != result.getElementType())
    return emitOpError("must preserve the message element type");
  SmallVector<int64_t> expected{static_cast<int64_t>(getNumSegments())};
  expected.append(input.getShape().begin() + 1, input.getShape().end());
  if (ArrayRef<int64_t>(expected) != result.getShape())
    return emitOpError("result shape must begin with num_segments");
  return success();
}

LogicalResult CSRExpandRowsOp::verify() {
  auto input = getInput().getType();
  auto rowPtr = getRowPtr().getType();
  auto result = getResult().getType();
  if (input.getRank() < 1 || rowPtr.getRank() != 1 || result.getRank() < 1)
    return emitOpError("requires ranked node/result and rank-one row_ptr");
  if (!isa<IntegerType>(rowPtr.getElementType()))
    return emitOpError("requires an integer row_ptr");
  if (rowPtr.getDimSize(0) != input.getDimSize(0) + 1)
    return emitOpError("row_ptr extent must equal node rows plus one");
  if (input.getElementType() != result.getElementType())
    return emitOpError("must preserve the input element type");
  SmallVector<int64_t> expected{static_cast<int64_t>(getNumEdges())};
  expected.append(input.getShape().begin() + 1, input.getShape().end());
  if (ArrayRef<int64_t>(expected) != result.getShape())
    return emitOpError("result shape must begin with num_edges");
  return success();
}

LogicalResult CSRSegmentSumOp::verify() {
  auto input = getInput().getType();
  auto rowPtr = getRowPtr().getType();
  auto result = getResult().getType();
  if (input.getRank() < 1 || rowPtr.getRank() != 1 || result.getRank() < 1)
    return emitOpError("requires ranked message/result and rank-one row_ptr");
  if (!isa<IntegerType>(rowPtr.getElementType()))
    return emitOpError("requires an integer row_ptr");
  if (rowPtr.getDimSize(0) != static_cast<int64_t>(getNumRows()) + 1)
    return emitOpError("row_ptr extent must equal num_rows plus one");
  if (input.getElementType() != result.getElementType())
    return emitOpError("must preserve the message element type");
  SmallVector<int64_t> expected{static_cast<int64_t>(getNumRows())};
  expected.append(input.getShape().begin() + 1, input.getShape().end());
  if (ArrayRef<int64_t>(expected) != result.getShape())
    return emitOpError("result shape must begin with num_rows");
  return success();
}

static LogicalResult verifyCSRProductShape(
    Operation *operation, RankedTensorType input, RankedTensorType rowPtr,
    RankedTensorType destination, RankedTensorType result, int64_t numRows,
    int64_t maxDegree) {
  if (input.getRank() < 1 || rowPtr.getRank() != 1 ||
      destination.getRank() != 1 || result.getRank() < 1)
    return operation->emitOpError(
        "requires ranked message/result and rank-one topology tensors");
  if (!isa<FloatType>(input.getElementType()))
    return operation->emitOpError("requires floating-point messages");
  if (!isa<IntegerType>(rowPtr.getElementType()) ||
      !isa<IntegerType>(destination.getElementType()))
    return operation->emitOpError("requires integer topology tensors");
  if (rowPtr.getDimSize(0) != numRows + 1)
    return operation->emitOpError(
        "row_ptr extent must equal num_rows plus one");
  if (destination.getDimSize(0) != input.getDimSize(0))
    return operation->emitOpError(
        "destination extent must equal the message edge extent");
  if (maxDegree < 0)
    return operation->emitOpError("max_degree must be non-negative");
  if (input.getElementType() != result.getElementType())
    return operation->emitOpError("must preserve the message element type");
  SmallVector<int64_t> expected{numRows};
  expected.append(input.getShape().begin() + 1, input.getShape().end());
  if (ArrayRef<int64_t>(expected) != result.getShape())
    return operation->emitOpError("result shape must begin with num_rows");
  return success();
}

LogicalResult CSRSegmentProductOp::verify() {
  return verifyCSRProductShape(
      *this, getInput().getType(), getRowPtr().getType(),
      getDestination().getType(), getResult().getType(), getNumRows(),
      getMaxDegree());
}

LogicalResult CSRSegmentProductVJPOp::verify() {
  auto input = getInput().getType();
  auto upstream = getUpstream().getType();
  auto result = getResult().getType();
  SmallVector<int64_t> upstreamShape{static_cast<int64_t>(getNumRows())};
  upstreamShape.append(input.getShape().begin() + 1, input.getShape().end());
  if (upstream.getShape() != ArrayRef<int64_t>(upstreamShape) ||
      upstream.getElementType() != input.getElementType())
    return emitOpError(
        "upstream shape must be [num_rows, *message_features]");
  if (result != input)
    return emitOpError("result type must equal the message input type");
  return verifyCSRProductShape(
      *this, input, getRowPtr().getType(), getDestination().getType(),
      upstream, getNumRows(), getMaxDegree());
}

LogicalResult CSRSegmentMaxStopGradientOp::verify() {
  auto input = getInput().getType();
  auto rowPtr = getRowPtr().getType();
  auto result = getResult().getType();
  if (input.getRank() < 1 || rowPtr.getRank() != 1 || result.getRank() < 1)
    return emitOpError("requires ranked message/result and rank-one row_ptr");
  if (!isa<IntegerType>(rowPtr.getElementType()))
    return emitOpError("requires an integer row_ptr");
  if (!isa<FloatType>(input.getElementType()))
    return emitOpError("requires floating-point messages");
  if (rowPtr.getDimSize(0) != static_cast<int64_t>(getNumRows()) + 1)
    return emitOpError("row_ptr extent must equal num_rows plus one");
  if (input.getElementType() != result.getElementType())
    return emitOpError("must preserve the message element type");
  SmallVector<int64_t> expected{static_cast<int64_t>(getNumRows())};
  expected.append(input.getShape().begin() + 1, input.getShape().end());
  if (ArrayRef<int64_t>(expected) != result.getShape())
    return emitOpError("result shape must begin with num_rows");
  return success();
}

LogicalResult GradOp::verify() {
  if (getOutput().getType() != getCotangent().getType())
    return emitOpError("cotangent type must equal the differentiated output type");
  if (getWrt().getType() != getResult().getType())
    return emitOpError("result type must equal the differentiated input type");
  return success();
}

#define GET_OP_CLASSES
#include "graphforge/Dialect/Tensor/TensorOps.cpp.inc"
