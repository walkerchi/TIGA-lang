#include "graphforge/Dialect/Tensor/TensorDialect.h"
#include "graphforge/Transforms/Passes.h"

#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::graphforge {

#define GEN_PASS_DEF_GFLOWERTENSORTOCPU
#include "graphforge/Transforms/Passes.h.inc"

namespace {
namespace gft = mlir::graphforge::tensor;

static int64_t elementCount(RankedTensorType type) {
  return type.hasStaticShape() ? type.getNumElements() : -1;
}

static bool hasContiguousLayout(gft::InputOp input) {
  auto type = cast<RankedTensorType>(input.getResult().getType());
  if (input.getOffsetAttr().getInt() != 0 ||
      input.getStrides().size() != static_cast<size_t>(type.getRank()))
    return false;
  int64_t expected = 1;
  for (int64_t axis = type.getRank() - 1; axis >= 0; --axis) {
    if (input.getStrides()[axis] != expected) return false;
    expected *= type.getDimSize(axis);
  }
  return true;
}

static bool isVectorizablePointwise(Value value, int64_t elements,
                                    llvm::DenseSet<Value> &visited) {
  if (!visited.insert(value).second) return true;
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || elementCount(type) != elements ||
      !isa<FloatType>(type.getElementType()))
    return false;
  Operation *operation = value.getDefiningOp();
  if (auto input = dyn_cast_or_null<gft::InputOp>(operation))
    return hasContiguousLayout(input);
  if (auto add = dyn_cast_or_null<gft::AddOp>(operation))
    return isVectorizablePointwise(add.getLhs(), elements, visited) &&
           isVectorizablePointwise(add.getRhs(), elements, visited);
  if (auto mul = dyn_cast_or_null<gft::MulOp>(operation))
    return isVectorizablePointwise(mul.getLhs(), elements, visited) &&
           isVectorizablePointwise(mul.getRhs(), elements, visited);
  if (auto divide = dyn_cast_or_null<gft::DivOp>(operation))
    return isVectorizablePointwise(divide.getLhs(), elements, visited) &&
           isVectorizablePointwise(divide.getRhs(), elements, visited);
  if (auto neg = dyn_cast_or_null<gft::NegOp>(operation))
    return isVectorizablePointwise(neg.getInput(), elements, visited);
  if (auto exponential = dyn_cast_or_null<gft::ExpOp>(operation))
    return isVectorizablePointwise(exponential.getInput(), elements, visited);
  if (auto root = dyn_cast_or_null<gft::SqrtOp>(operation))
    return isVectorizablePointwise(root.getInput(), elements, visited);
  if (auto reshape = dyn_cast_or_null<gft::ReshapeOp>(operation))
    return isVectorizablePointwise(reshape.getInput(), elements, visited);
  if (auto checkpoint = dyn_cast_or_null<gft::CheckpointOp>(operation))
    return isVectorizablePointwise(checkpoint.getInput(), elements, visited);
  if (auto candidate =
          dyn_cast_or_null<gft::CheckpointCandidateOp>(operation))
    return isVectorizablePointwise(candidate.getInput(), elements, visited);
  return false;
}

class VectorEmitter {
public:
  VectorEmitter(func::FuncOp source, func::FuncOp target, OpBuilder &builder,
                VectorType type)
      : source(source), target(target), builder(builder), type(type),
        location(source.getLoc()) {}

  FailureOr<Value> emit(Value value, Value linear) {
    Operation *operation = value.getDefiningOp();
    if (auto input = dyn_cast_or_null<gft::InputOp>(operation)) {
      auto argument = dyn_cast<BlockArgument>(input.getBuffer());
      if (!argument || argument.getOwner() != &source.front())
        return input.emitError("vector CPU lowering requires ABI inputs");
      return Value(builder.create<vector::LoadOp>(
          location, type, target.getArgument(argument.getArgNumber()),
          ValueRange{linear}));
    }
    if (auto reshape = dyn_cast_or_null<gft::ReshapeOp>(operation))
      return emit(reshape.getInput(), linear);
    if (auto checkpoint = dyn_cast_or_null<gft::CheckpointOp>(operation))
      return emit(checkpoint.getInput(), linear);
    if (auto candidate =
            dyn_cast_or_null<gft::CheckpointCandidateOp>(operation))
      return emit(candidate.getInput(), linear);
    if (auto neg = dyn_cast_or_null<gft::NegOp>(operation)) {
      FailureOr<Value> operand = emit(neg.getInput(), linear);
      if (failed(operand)) return failure();
      return Value(builder.create<arith::NegFOp>(location, *operand));
    }
    if (auto exponential = dyn_cast_or_null<gft::ExpOp>(operation)) {
      FailureOr<Value> operand = emit(exponential.getInput(), linear);
      if (failed(operand)) return failure();
      return Value(builder.create<math::ExpOp>(location, *operand));
    }
    if (auto root = dyn_cast_or_null<gft::SqrtOp>(operation)) {
      FailureOr<Value> operand = emit(root.getInput(), linear);
      if (failed(operand)) return failure();
      return Value(builder.create<math::SqrtOp>(location, *operand));
    }
    Value lhs, rhs;
    enum class Kind { Add, Mul, Div } kind;
    if (auto add = dyn_cast_or_null<gft::AddOp>(operation)) {
      lhs = add.getLhs(); rhs = add.getRhs(); kind = Kind::Add;
    } else if (auto mul = dyn_cast_or_null<gft::MulOp>(operation)) {
      lhs = mul.getLhs(); rhs = mul.getRhs(); kind = Kind::Mul;
    } else if (auto divide = dyn_cast_or_null<gft::DivOp>(operation)) {
      lhs = divide.getLhs(); rhs = divide.getRhs(); kind = Kind::Div;
    } else {
      if (operation)
        operation->emitError("unsupported producer in vector CPU lowering");
      return failure();
    }
    FailureOr<Value> left = emit(lhs, linear);
    FailureOr<Value> right = emit(rhs, linear);
    if (failed(left) || failed(right)) return failure();
    if (kind == Kind::Add)
      return Value(builder.create<arith::AddFOp>(location, *left, *right));
    if (kind == Kind::Mul)
      return Value(builder.create<arith::MulFOp>(location, *left, *right));
    return Value(builder.create<arith::DivFOp>(location, *left, *right));
  }

private:
  func::FuncOp source;
  func::FuncOp target;
  OpBuilder &builder;
  VectorType type;
  Location location;
};

class ScalarEmitter {
public:
  ScalarEmitter(func::FuncOp source, func::FuncOp target, OpBuilder &builder)
      : source(source), target(target), builder(builder), location(source.getLoc()) {}

  FailureOr<Value> emit(Value value, ArrayRef<Value> coordinates) {
    Operation *operation = value.getDefiningOp();
    if (auto input = dyn_cast_or_null<gft::InputOp>(operation))
      return emitInput(input, coordinates);
    if (auto add = dyn_cast_or_null<gft::AddOp>(operation))
      return emitBinary(add.getLhs(), add.getRhs(), add.getResult(),
                        coordinates, BinaryKind::Add);
    if (auto mul = dyn_cast_or_null<gft::MulOp>(operation))
      return emitBinary(mul.getLhs(), mul.getRhs(), mul.getResult(),
                        coordinates, BinaryKind::Mul);
    if (auto matmul = dyn_cast_or_null<gft::MatmulOp>(operation))
      return emitMatmul(matmul, coordinates);
    if (auto divide = dyn_cast_or_null<gft::DivOp>(operation))
      return emitBinary(divide.getLhs(), divide.getRhs(), divide.getResult(),
                        coordinates, BinaryKind::Div);
    if (auto neg = dyn_cast_or_null<gft::NegOp>(operation)) {
      FailureOr<Value> operand = emit(neg.getInput(), coordinates);
      if (failed(operand)) return failure();
      if (isa<ComplexType>(operand->getType())) {
        auto type = cast<ComplexType>(operand->getType());
        Value real = builder.create<complex::ReOp>(location, *operand);
        Value imaginary = builder.create<complex::ImOp>(location, *operand);
        real = builder.create<arith::NegFOp>(location, real);
        imaginary = builder.create<arith::NegFOp>(location, imaginary);
        return Value(builder.create<complex::CreateOp>(
            location, type, real, imaginary));
      }
      if (isa<FloatType>(operand->getType()))
        return Value(builder.create<arith::NegFOp>(location, *operand));
      return Value(builder.create<arith::SubIOp>(
          location, zero(operand->getType()), *operand));
    }
    if (auto exponential = dyn_cast_or_null<gft::ExpOp>(operation)) {
      FailureOr<Value> operand = emit(exponential.getInput(), coordinates);
      if (failed(operand)) return failure();
      return Value(builder.create<math::ExpOp>(location, *operand));
    }
    if (auto root = dyn_cast_or_null<gft::SqrtOp>(operation)) {
      FailureOr<Value> operand = emit(root.getInput(), coordinates);
      if (failed(operand)) return failure();
      return Value(builder.create<math::SqrtOp>(location, *operand));
    }
    if (auto reshape = dyn_cast_or_null<gft::ReshapeOp>(operation)) {
      auto outputType = cast<RankedTensorType>(reshape.getResult().getType());
      auto inputType = cast<RankedTensorType>(reshape.getInput().getType());
      return emit(reshape.getInput(),
                  delinearize(linearize(coordinates, outputType.getShape()),
                              inputType.getShape()));
    }
    if (auto broadcast = dyn_cast_or_null<gft::BroadcastOp>(operation)) {
      auto inputType = cast<RankedTensorType>(broadcast.getInput().getType());
      auto outputType = cast<RankedTensorType>(broadcast.getResult().getType());
      return emit(broadcast.getInput(),
                  broadcastCoordinates(inputType.getShape(),
                                       outputType.getShape(), coordinates));
    }
    if (auto checkpoint = dyn_cast_or_null<gft::CheckpointOp>(operation))
      return emit(checkpoint.getInput(), coordinates);
    if (auto candidate =
            dyn_cast_or_null<gft::CheckpointCandidateOp>(operation))
      return emit(candidate.getInput(), coordinates);
    if (auto permute = dyn_cast_or_null<gft::PermuteOp>(operation)) {
      SmallVector<Value> projected(coordinates.size());
      for (auto [outputAxis, inputAxis] : llvm::enumerate(permute.getAxes()))
        projected[inputAxis] = coordinates[outputAxis];
      return emit(permute.getInput(), projected);
    }
    if (auto conjugate = dyn_cast_or_null<gft::ConjOp>(operation)) {
      FailureOr<Value> operand = emit(conjugate.getInput(), coordinates);
      if (failed(operand)) return failure();
      auto type = cast<ComplexType>(operand->getType());
      Value real = builder.create<complex::ReOp>(location, *operand);
      Value imaginary = builder.create<complex::ImOp>(location, *operand);
      imaginary = builder.create<arith::NegFOp>(location, imaginary);
      return Value(
          builder.create<complex::CreateOp>(location, type, real, imaginary));
    }
    if (auto reduction = dyn_cast_or_null<gft::ReduceSumOp>(operation))
      return emitReduction(reduction, coordinates);
    if (auto scan = dyn_cast_or_null<gft::CumsumOp>(operation))
      return emitCumsum(scan, coordinates);
    if (auto gather = dyn_cast_or_null<gft::GatherOp>(operation))
      return emitGather(gather, coordinates);
    if (auto scatter = dyn_cast_or_null<gft::ScatterRowsOp>(operation))
      return emitScatterRows(scatter, coordinates);
    if (auto segment = dyn_cast_or_null<gft::SegmentSumOp>(operation))
      return emitSegmentSum(segment, coordinates);
    if (auto expand = dyn_cast_or_null<gft::CSRExpandRowsOp>(operation))
      return emitCSRExpandRows(expand, coordinates);
    if (auto segment = dyn_cast_or_null<gft::CSRSegmentSumOp>(operation))
      return emitCSRSegmentSum(segment, coordinates);
    if (auto product = dyn_cast_or_null<gft::CSRSegmentProductOp>(operation))
      return emitCSRSegmentProduct(product, coordinates);
    if (auto vjp = dyn_cast_or_null<gft::CSRSegmentProductVJPOp>(operation))
      return emitCSRSegmentProductVJP(vjp, coordinates);
    if (auto segment =
            dyn_cast_or_null<gft::CSRSegmentMaxStopGradientOp>(operation))
      return emitCSRSegmentMax(segment, coordinates);
    if (operation)
      operation->emitError("unsupported gf_tensor producer in CPU lowering");
    return failure();
  }

private:
  enum class BinaryKind { Add, Mul, Div };

  Value constantIndex(int64_t value) {
    return builder.create<arith::ConstantIndexOp>(location, value);
  }

  Value linearize(ArrayRef<Value> coordinates, ArrayRef<int64_t> shape) {
    Value result = constantIndex(0);
    int64_t stride = 1;
    for (int64_t axis = shape.size() - 1; axis >= 0; --axis) {
      Value term = coordinates[axis];
      if (stride != 1)
        term = builder.create<arith::MulIOp>(location, term, constantIndex(stride));
      result = builder.create<arith::AddIOp>(location, result, term);
      stride *= shape[axis];
    }
    return result;
  }

  SmallVector<Value> delinearize(Value linear, ArrayRef<int64_t> shape) {
    SmallVector<Value> coordinates(shape.size());
    int64_t stride = 1;
    for (int64_t axis = shape.size() - 1; axis >= 0; --axis) {
      Value coordinate = linear;
      if (stride != 1)
        coordinate = builder.create<arith::DivUIOp>(location, coordinate,
                                                     constantIndex(stride));
      if (axis != 0)
        coordinate = builder.create<arith::RemUIOp>(
            location, coordinate, constantIndex(shape[axis]));
      coordinates[axis] = coordinate;
      stride *= shape[axis];
    }
    return coordinates;
  }

  SmallVector<Value> broadcastCoordinates(ArrayRef<int64_t> inputShape,
                                          ArrayRef<int64_t> outputShape,
                                          ArrayRef<Value> outputCoordinates) {
    SmallVector<Value> result;
    int64_t padding = outputShape.size() - inputShape.size();
    for (auto [axis, extent] : llvm::enumerate(inputShape))
      result.push_back(extent == 1 ? constantIndex(0)
                                   : outputCoordinates[padding + axis]);
    return result;
  }

  FailureOr<Value> emitInput(gft::InputOp input,
                             ArrayRef<Value> coordinates) {
    auto argument = dyn_cast<BlockArgument>(input.getBuffer());
    if (!argument || argument.getOwner() != &source.front())
      return input.emitError("CPU lowering requires function ABI inputs");
    Value linear = constantIndex(input.getOffsetAttr().getInt());
    for (auto [coordinate, stride] : llvm::zip(coordinates, input.getStrides())) {
      if (stride == 0) continue;
      Value term = coordinate;
      if (stride != 1)
        term = builder.create<arith::MulIOp>(location, term, constantIndex(stride));
      linear = builder.create<arith::AddIOp>(location, linear, term);
    }
    return Value(builder.create<memref::LoadOp>(
        location, target.getArgument(argument.getArgNumber()),
        ValueRange{linear}));
  }

  FailureOr<Value> emitBinary(Value lhs, Value rhs, Value result,
                              ArrayRef<Value> coordinates, BinaryKind kind) {
    auto resultType = cast<RankedTensorType>(result.getType());
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto rhsType = cast<RankedTensorType>(rhs.getType());
    FailureOr<Value> left = emit(
        lhs, broadcastCoordinates(lhsType.getShape(), resultType.getShape(),
                                  coordinates));
    FailureOr<Value> right = emit(
        rhs, broadcastCoordinates(rhsType.getShape(), resultType.getShape(),
                                  coordinates));
    if (failed(left) || failed(right)) return failure();
    Type type = left->getType();
    if (isa<ComplexType>(type)) {
      if (kind == BinaryKind::Add)
        return Value(builder.create<complex::AddOp>(location, *left, *right));
      if (kind == BinaryKind::Mul)
        return Value(builder.create<complex::MulOp>(location, *left, *right));
      return Value(builder.create<complex::DivOp>(location, *left, *right));
    }
    if (isa<FloatType>(type)) {
      if (kind == BinaryKind::Add)
        return Value(builder.create<arith::AddFOp>(location, *left, *right));
      if (kind == BinaryKind::Mul)
        return Value(builder.create<arith::MulFOp>(location, *left, *right));
      return Value(builder.create<arith::DivFOp>(location, *left, *right));
    }
    if (kind == BinaryKind::Add)
      return Value(builder.create<arith::AddIOp>(location, *left, *right));
    if (kind == BinaryKind::Mul)
      return Value(builder.create<arith::MulIOp>(location, *left, *right));
    return Value(builder.create<arith::DivSIOp>(location, *left, *right));
  }

  Value zero(Type type) {
    if (auto floating = dyn_cast<FloatType>(type))
      return builder.create<arith::ConstantOp>(
          location, builder.getFloatAttr(floating, 0.0));
    if (auto integer = dyn_cast<IntegerType>(type))
      return builder.create<arith::ConstantOp>(
          location, builder.getIntegerAttr(integer, 0));
    auto complexType = cast<ComplexType>(type);
    Value scalar = builder.create<arith::ConstantOp>(
        location, builder.getFloatAttr(complexType.getElementType(), 0.0));
    return builder.create<complex::CreateOp>(location, complexType, scalar, scalar);
  }

  Value add(Value left, Value right) {
    if (isa<ComplexType>(left.getType()))
      return builder.create<complex::AddOp>(location, left, right);
    if (isa<FloatType>(left.getType()))
      return builder.create<arith::AddFOp>(location, left, right);
    return builder.create<arith::AddIOp>(location, left, right);
  }

  Value multiply(Value left, Value right) {
    if (isa<ComplexType>(left.getType()))
      return builder.create<complex::MulOp>(location, left, right);
    if (isa<FloatType>(left.getType()))
      return builder.create<arith::MulFOp>(location, left, right);
    return builder.create<arith::MulIOp>(location, left, right);
  }

  Value one(Type type) {
    if (auto floating = dyn_cast<FloatType>(type))
      return builder.create<arith::ConstantOp>(
          location, builder.getFloatAttr(floating, 1.0));
    if (auto integer = dyn_cast<IntegerType>(type))
      return builder.create<arith::ConstantOp>(
          location, builder.getIntegerAttr(integer, 1));
    auto complexType = cast<ComplexType>(type);
    Value real = builder.create<arith::ConstantOp>(
        location, builder.getFloatAttr(complexType.getElementType(), 1.0));
    Value imaginary = builder.create<arith::ConstantOp>(
        location, builder.getFloatAttr(complexType.getElementType(), 0.0));
    return builder.create<complex::CreateOp>(
        location, complexType, real, imaginary);
  }

  Value negativeInfinity(Type type) {
    auto floating = cast<FloatType>(type);
    return builder.create<arith::ConstantOp>(
        location, builder.getFloatAttr(
                      floating, APFloat::getInf(floating.getFloatSemantics(),
                                                /*negative=*/true)));
  }

  FailureOr<Value> emitReduction(gft::ReduceSumOp reduction,
                                 ArrayRef<Value> outputCoordinates) {
    auto inputType = cast<RankedTensorType>(reduction.getInput().getType());
    llvm::SmallDenseSet<int64_t> reduced;
    SmallVector<int64_t> reducedShape;
    for (int64_t axis : reduction.getAxes()) {
      reduced.insert(axis);
      reducedShape.push_back(inputType.getDimSize(axis));
    }
    int64_t count = 1;
    for (int64_t extent : reducedShape) count *= extent;
    auto loop = builder.create<scf::ForOp>(
        location, constantIndex(0), constantIndex(count), constantIndex(1),
        ValueRange{zero(inputType.getElementType())});
    OpBuilder::InsertionGuard guard(builder);
    if (loop.getBody()->empty()) {
      builder.setInsertionPointToEnd(loop.getBody());
      builder.create<scf::YieldOp>(location, loop.getRegionIterArgs());
    }
    Operation *oldTerminator = loop.getBody()->getTerminator();
    builder.setInsertionPoint(oldTerminator);
    SmallVector<Value> reducedCoordinates =
        delinearize(loop.getInductionVar(), reducedShape);
    SmallVector<Value> inputCoordinates;
    int64_t reducedPosition = 0, outputPosition = 0;
    for (int64_t axis = 0; axis < inputType.getRank(); ++axis) {
      if (reduced.contains(axis)) {
        inputCoordinates.push_back(reducedCoordinates[reducedPosition++]);
        if (reduction.getKeepDims()) ++outputPosition;
      } else {
        inputCoordinates.push_back(outputCoordinates[outputPosition++]);
      }
    }
    FailureOr<Value> item = emit(reduction.getInput(), inputCoordinates);
    if (failed(item)) return failure();
    Value next = add(loop.getRegionIterArgs()[0], *item);
    oldTerminator->setOperands(ValueRange{next});
    return loop.getResult(0);
  }

  FailureOr<Value> emitCumsum(gft::CumsumOp scan,
                              ArrayRef<Value> outputCoordinates) {
    auto inputType = cast<RankedTensorType>(scan.getInput().getType());
    int64_t axis = scan.getAxisAttr().getInt();
    Value coordinate = outputCoordinates[axis];
    Value lower = scan.getReverse() ? coordinate : constantIndex(0);
    Value upper = scan.getReverse()
                      ? constantIndex(inputType.getDimSize(axis))
                      : builder.create<arith::AddIOp>(
                            location, coordinate, constantIndex(1));
    auto loop = builder.create<scf::ForOp>(
        location, lower, upper, constantIndex(1),
        ValueRange{zero(inputType.getElementType())});
    OpBuilder::InsertionGuard guard(builder);
    if (loop.getBody()->empty()) {
      builder.setInsertionPointToEnd(loop.getBody());
      builder.create<scf::YieldOp>(location, loop.getRegionIterArgs());
    }
    Operation *oldTerminator = loop.getBody()->getTerminator();
    builder.setInsertionPoint(oldTerminator);
    SmallVector<Value> inputCoordinates(outputCoordinates);
    inputCoordinates[axis] = loop.getInductionVar();
    FailureOr<Value> item = emit(scan.getInput(), inputCoordinates);
    if (failed(item)) return failure();
    Value next = add(loop.getRegionIterArgs()[0], *item);
    oldTerminator->setOperands(ValueRange{next});
    return loop.getResult(0);
  }

  FailureOr<Value> emitMatmul(gft::MatmulOp matmul,
                              ArrayRef<Value> outputCoordinates) {
    auto lhsType = cast<RankedTensorType>(matmul.getLhs().getType());
    auto loop = builder.create<scf::ForOp>(
        location, constantIndex(0), constantIndex(lhsType.getDimSize(1)),
        constantIndex(1), ValueRange{zero(lhsType.getElementType())});
    OpBuilder::InsertionGuard guard(builder);
    if (loop.getBody()->empty()) {
      builder.setInsertionPointToEnd(loop.getBody());
      builder.create<scf::YieldOp>(location, loop.getRegionIterArgs());
    }
    Operation *oldTerminator = loop.getBody()->getTerminator();
    builder.setInsertionPoint(oldTerminator);
    SmallVector<Value> lhsCoordinates{
        outputCoordinates[0], loop.getInductionVar()};
    SmallVector<Value> rhsCoordinates{
        loop.getInductionVar(), outputCoordinates[1]};
    FailureOr<Value> lhs = emit(matmul.getLhs(), lhsCoordinates);
    FailureOr<Value> rhs = emit(matmul.getRhs(), rhsCoordinates);
    if (failed(lhs) || failed(rhs)) return failure();
    Value product;
    if (isa<ComplexType>(lhs->getType()))
      product = builder.create<complex::MulOp>(location, *lhs, *rhs);
    else
      product = builder.create<arith::MulFOp>(location, *lhs, *rhs);
    Value next = add(loop.getRegionIterArgs()[0], product);
    oldTerminator->setOperands(ValueRange{next});
    return loop.getResult(0);
  }

  FailureOr<Value> emitGather(gft::GatherOp gather,
                              ArrayRef<Value> outputCoordinates) {
    FailureOr<Value> row = emit(gather.getIndex(), outputCoordinates.take_front(1));
    if (failed(row)) return failure();
    Value sourceRow = builder.create<arith::IndexCastOp>(
        location, builder.getIndexType(), *row);
    SmallVector<Value> inputCoordinates{sourceRow};
    inputCoordinates.append(outputCoordinates.begin() + 1,
                            outputCoordinates.end());
    return emit(gather.getInput(), inputCoordinates);
  }

  FailureOr<Value> emitScatterRows(gft::ScatterRowsOp scatter,
                                   ArrayRef<Value> outputCoordinates) {
    SmallVector<Value> inverseCoordinates{outputCoordinates.front()};
    FailureOr<Value> rawSource = emit(
        scatter.getInverse(), inverseCoordinates);
    if (failed(rawSource)) return failure();
    Value source = builder.create<arith::IndexCastOp>(
        location, builder.getIndexType(), *rawSource);
    Value active = builder.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::sge, source, constantIndex(0));
    Value safeSource = builder.create<arith::SelectOp>(
        location, active, source, constantIndex(0));
    SmallVector<Value> inputCoordinates{safeSource};
    inputCoordinates.append(outputCoordinates.begin() + 1,
                            outputCoordinates.end());
    auto elementType = cast<RankedTensorType>(
        scatter.getInput().getType()).getElementType();
    auto conditional = builder.create<scf::IfOp>(
        location, TypeRange{elementType}, active, true);
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&conditional.getThenRegion().front());
      FailureOr<Value> item = emit(scatter.getInput(), inputCoordinates);
      if (failed(item)) return failure();
      builder.create<scf::YieldOp>(location, *item);
      builder.setInsertionPointToStart(&conditional.getElseRegion().front());
      builder.create<scf::YieldOp>(location, zero(elementType));
    }
    return conditional.getResult(0);
  }

  FailureOr<Value> emitSegmentSum(gft::SegmentSumOp segment,
                                  ArrayRef<Value> outputCoordinates) {
    auto inputType = cast<RankedTensorType>(segment.getInput().getType());
    auto indexType = cast<RankedTensorType>(segment.getIndex().getType());
    Value destination = builder.create<arith::IndexCastOp>(
        location, indexType.getElementType(), outputCoordinates.front());
    auto loop = builder.create<scf::ForOp>(
        location, constantIndex(0), constantIndex(inputType.getDimSize(0)),
        constantIndex(1), ValueRange{zero(inputType.getElementType())});
    OpBuilder::InsertionGuard guard(builder);
    if (loop.getBody()->empty()) {
      builder.setInsertionPointToEnd(loop.getBody());
      builder.create<scf::YieldOp>(location, loop.getRegionIterArgs());
    }
    Operation *oldTerminator = loop.getBody()->getTerminator();
    builder.setInsertionPoint(oldTerminator);
    SmallVector<Value> indexCoordinates{loop.getInductionVar()};
    FailureOr<Value> index = emit(segment.getIndex(), indexCoordinates);
    SmallVector<Value> inputCoordinates{loop.getInductionVar()};
    inputCoordinates.append(outputCoordinates.begin() + 1,
                            outputCoordinates.end());
    FailureOr<Value> item = emit(segment.getInput(), inputCoordinates);
    if (failed(index) || failed(item)) return failure();
    Value selected = builder.create<arith::SelectOp>(
        location,
        builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq,
                                      *index, destination),
        *item, zero(inputType.getElementType()));
    Value next = add(loop.getRegionIterArgs()[0], selected);
    oldTerminator->setOperands(ValueRange{next});
    return loop.getResult(0);
  }

  FailureOr<Value> emitCSRExpandRows(gft::CSRExpandRowsOp expand,
                                     ArrayRef<Value> outputCoordinates) {
    auto inputType = cast<RankedTensorType>(expand.getInput().getType());
    Value sourceRow = constantIndex(0);
    for (int64_t row = 0; row < inputType.getDimSize(0); ++row) {
      SmallVector<Value> beginCoordinate{constantIndex(row)};
      SmallVector<Value> pointerCoordinate{constantIndex(row + 1)};
      FailureOr<Value> begin = emit(expand.getRowPtr(), beginCoordinate);
      FailureOr<Value> end = emit(expand.getRowPtr(), pointerCoordinate);
      if (failed(begin) || failed(end)) return failure();
      Value beginIndex = builder.create<arith::IndexCastOp>(
          location, builder.getIndexType(), *begin);
      Value endIndex = builder.create<arith::IndexCastOp>(
          location, builder.getIndexType(), *end);
      Value afterBegin = builder.create<arith::CmpIOp>(
          location, arith::CmpIPredicate::uge,
          outputCoordinates.front(), beginIndex);
      Value beforeEnd = builder.create<arith::CmpIOp>(
          location, arith::CmpIPredicate::ult,
          outputCoordinates.front(), endIndex);
      Value within = builder.create<arith::AndIOp>(
          location, afterBegin, beforeEnd);
      sourceRow = builder.create<arith::SelectOp>(
          location, within, constantIndex(row), sourceRow);
    }
    SmallVector<Value> inputCoordinates{sourceRow};
    inputCoordinates.append(outputCoordinates.begin() + 1,
                            outputCoordinates.end());
    return emit(expand.getInput(), inputCoordinates);
  }

  FailureOr<Value> emitCSRSegmentSum(gft::CSRSegmentSumOp segment,
                                     ArrayRef<Value> outputCoordinates) {
    SmallVector<Value> beginCoordinate{outputCoordinates.front()};
    Value nextRow = builder.create<arith::AddIOp>(
        location, outputCoordinates.front(), constantIndex(1));
    SmallVector<Value> endCoordinate{nextRow};
    FailureOr<Value> rawBegin = emit(segment.getRowPtr(), beginCoordinate);
    FailureOr<Value> rawEnd = emit(segment.getRowPtr(), endCoordinate);
    if (failed(rawBegin) || failed(rawEnd)) return failure();
    Value begin = builder.create<arith::IndexCastOp>(
        location, builder.getIndexType(), *rawBegin);
    Value end = builder.create<arith::IndexCastOp>(
        location, builder.getIndexType(), *rawEnd);
    auto inputType = cast<RankedTensorType>(segment.getInput().getType());
    auto loop = builder.create<scf::ForOp>(
        location, begin, end, constantIndex(1),
        ValueRange{zero(inputType.getElementType())});
    OpBuilder::InsertionGuard guard(builder);
    if (loop.getBody()->empty()) {
      builder.setInsertionPointToEnd(loop.getBody());
      builder.create<scf::YieldOp>(location, loop.getRegionIterArgs());
    }
    Operation *oldTerminator = loop.getBody()->getTerminator();
    builder.setInsertionPoint(oldTerminator);
    SmallVector<Value> inputCoordinates{loop.getInductionVar()};
    inputCoordinates.append(outputCoordinates.begin() + 1,
                            outputCoordinates.end());
    FailureOr<Value> item = emit(segment.getInput(), inputCoordinates);
    if (failed(item)) return failure();
    Value next = add(loop.getRegionIterArgs()[0], *item);
    oldTerminator->setOperands(ValueRange{next});
    return loop.getResult(0);
  }

  FailureOr<Value> emitCSRSegmentProduct(
      gft::CSRSegmentProductOp product,
      ArrayRef<Value> outputCoordinates) {
    SmallVector<Value> beginCoordinate{outputCoordinates.front()};
    Value nextRow = builder.create<arith::AddIOp>(
        location, outputCoordinates.front(), constantIndex(1));
    SmallVector<Value> endCoordinate{nextRow};
    FailureOr<Value> rawBegin = emit(product.getRowPtr(), beginCoordinate);
    FailureOr<Value> rawEnd = emit(product.getRowPtr(), endCoordinate);
    if (failed(rawBegin) || failed(rawEnd)) return failure();
    Value begin = builder.create<arith::IndexCastOp>(
        location, builder.getIndexType(), *rawBegin);
    Value end = builder.create<arith::IndexCastOp>(
        location, builder.getIndexType(), *rawEnd);
    auto inputType = cast<RankedTensorType>(product.getInput().getType());
    auto loop = builder.create<scf::ForOp>(
        location, begin, end, constantIndex(1),
        ValueRange{one(inputType.getElementType())});
    OpBuilder::InsertionGuard guard(builder);
    if (loop.getBody()->empty()) {
      builder.setInsertionPointToEnd(loop.getBody());
      builder.create<scf::YieldOp>(location, loop.getRegionIterArgs());
    }
    Operation *oldTerminator = loop.getBody()->getTerminator();
    builder.setInsertionPoint(oldTerminator);
    SmallVector<Value> inputCoordinates{loop.getInductionVar()};
    inputCoordinates.append(outputCoordinates.begin() + 1,
                            outputCoordinates.end());
    FailureOr<Value> item = emit(product.getInput(), inputCoordinates);
    if (failed(item)) return failure();
    oldTerminator->setOperands(
        ValueRange{multiply(loop.getRegionIterArgs()[0], *item)});
    return loop.getResult(0);
  }

  FailureOr<Value> emitCSRSegmentProductVJP(
      gft::CSRSegmentProductVJPOp vjp,
      ArrayRef<Value> outputCoordinates) {
    SmallVector<Value> edgeCoordinate{outputCoordinates.front()};
    FailureOr<Value> rawRow = emit(vjp.getDestination(), edgeCoordinate);
    if (failed(rawRow)) return failure();
    Value row = builder.create<arith::IndexCastOp>(
        location, builder.getIndexType(), *rawRow);
    SmallVector<Value> beginCoordinate{row};
    Value nextRow = builder.create<arith::AddIOp>(
        location, row, constantIndex(1));
    SmallVector<Value> endCoordinate{nextRow};
    FailureOr<Value> rawBegin = emit(vjp.getRowPtr(), beginCoordinate);
    FailureOr<Value> rawEnd = emit(vjp.getRowPtr(), endCoordinate);
    if (failed(rawBegin) || failed(rawEnd)) return failure();
    Value begin = builder.create<arith::IndexCastOp>(
        location, builder.getIndexType(), *rawBegin);
    Value end = builder.create<arith::IndexCastOp>(
        location, builder.getIndexType(), *rawEnd);
    auto inputType = cast<RankedTensorType>(vjp.getInput().getType());
    auto loop = builder.create<scf::ForOp>(
        location, begin, end, constantIndex(1),
        ValueRange{one(inputType.getElementType())});
    {
      OpBuilder::InsertionGuard guard(builder);
      if (loop.getBody()->empty()) {
        builder.setInsertionPointToEnd(loop.getBody());
        builder.create<scf::YieldOp>(location, loop.getRegionIterArgs());
      }
      Operation *oldTerminator = loop.getBody()->getTerminator();
      builder.setInsertionPoint(oldTerminator);
      SmallVector<Value> inputCoordinates{loop.getInductionVar()};
      inputCoordinates.append(outputCoordinates.begin() + 1,
                              outputCoordinates.end());
      FailureOr<Value> item = emit(vjp.getInput(), inputCoordinates);
      if (failed(item)) return failure();
      Value isSelf = builder.create<arith::CmpIOp>(
          location, arith::CmpIPredicate::eq, loop.getInductionVar(),
          outputCoordinates.front());
      Value factor = builder.create<arith::SelectOp>(
          location, isSelf, one(inputType.getElementType()), *item);
      oldTerminator->setOperands(
          ValueRange{multiply(loop.getRegionIterArgs()[0], factor)});
    }
    builder.setInsertionPointAfter(loop);
    Value excludedProduct = loop.getResult(0);
    SmallVector<Value> upstreamCoordinates{row};
    upstreamCoordinates.append(outputCoordinates.begin() + 1,
                               outputCoordinates.end());
    FailureOr<Value> upstream = emit(vjp.getUpstream(), upstreamCoordinates);
    if (failed(upstream)) return failure();
    return multiply(excludedProduct, *upstream);
  }

  FailureOr<Value> emitCSRSegmentMax(
      gft::CSRSegmentMaxStopGradientOp segment,
      ArrayRef<Value> outputCoordinates) {
    SmallVector<Value> beginCoordinate{outputCoordinates.front()};
    Value nextRow = builder.create<arith::AddIOp>(
        location, outputCoordinates.front(), constantIndex(1));
    SmallVector<Value> endCoordinate{nextRow};
    FailureOr<Value> rawBegin = emit(segment.getRowPtr(), beginCoordinate);
    FailureOr<Value> rawEnd = emit(segment.getRowPtr(), endCoordinate);
    if (failed(rawBegin) || failed(rawEnd)) return failure();
    Value begin = builder.create<arith::IndexCastOp>(
        location, builder.getIndexType(), *rawBegin);
    Value end = builder.create<arith::IndexCastOp>(
        location, builder.getIndexType(), *rawEnd);
    auto inputType = cast<RankedTensorType>(segment.getInput().getType());
    auto loop = builder.create<scf::ForOp>(
        location, begin, end, constantIndex(1),
        ValueRange{negativeInfinity(inputType.getElementType())});
    OpBuilder::InsertionGuard guard(builder);
    if (loop.getBody()->empty()) {
      builder.setInsertionPointToEnd(loop.getBody());
      builder.create<scf::YieldOp>(location, loop.getRegionIterArgs());
    }
    Operation *oldTerminator = loop.getBody()->getTerminator();
    builder.setInsertionPoint(oldTerminator);
    SmallVector<Value> inputCoordinates{loop.getInductionVar()};
    inputCoordinates.append(outputCoordinates.begin() + 1,
                            outputCoordinates.end());
    FailureOr<Value> item = emit(segment.getInput(), inputCoordinates);
    if (failed(item)) return failure();
    Value next = builder.create<arith::MaximumFOp>(
        location, loop.getRegionIterArgs()[0], *item);
    oldTerminator->setOperands(ValueRange{next});
    return loop.getResult(0);
  }

  func::FuncOp source;
  func::FuncOp target;
  OpBuilder &builder;
  Location location;
};

class LowerTensorToCPUPass
    : public impl::GFLowerTensorToCPUBase<LowerTensorToCPUPass> {
public:
  using impl::GFLowerTensorToCPUBase<
      LowerTensorToCPUPass>::GFLowerTensorToCPUBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, complex::ComplexDialect,
                    func::FuncDialect, memref::MemRefDialect,
                    math::MathDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() final {
    SmallVector<func::FuncOp> functions;
    getOperation().walk([&](func::FuncOp function) {
      if (!function.getOps<gft::InputOp>().empty()) functions.push_back(function);
    });
    for (func::FuncOp source : functions) {
      if (source.empty() || !llvm::hasSingleElement(source.getBody())) {
        source.emitError("CPU lowering requires one-block functions");
        return signalPassFailure();
      }
      auto returnOp = dyn_cast<func::ReturnOp>(source.front().getTerminator());
      if (!returnOp || returnOp.getNumOperands() != 1) {
        source.emitError("CPU lowering requires one Tensor result");
        return signalPassFailure();
      }
      auto outputType = dyn_cast<RankedTensorType>(returnOp.getOperand(0).getType());
      if (!outputType || elementCount(outputType) < 0) {
        source.emitError("CPU lowering requires a static ranked result");
        return signalPassFailure();
      }
      SmallVector<Type> storageTypes;
      for (Type type : source.getArgumentTypes()) {
        auto tensor = dyn_cast<RankedTensorType>(type);
        if (!tensor || !tensor.hasStaticShape()) {
          source.emitError("CPU lowering requires static ranked inputs");
          return signalPassFailure();
        }
        storageTypes.push_back(MemRefType::get(
            {ShapedType::kDynamic}, tensor.getElementType()));
      }
      size_t outputArgument = storageTypes.size();
      storageTypes.push_back(MemRefType::get(
          {ShapedType::kDynamic}, outputType.getElementType()));
      storageTypes.push_back(IndexType::get(source.getContext()));
      storageTypes.push_back(IndexType::get(source.getContext()));
      std::string name = source.getName().str();
      source.setName(name + "__gf_semantic");
      OpBuilder builder(source);
      builder.setInsertionPoint(source);
      auto target = builder.create<func::FuncOp>(
          source.getLoc(), name,
          builder.getFunctionType(storageTypes, TypeRange{}));
      target->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
      Block *entry = target.addEntryBlock();
      builder.setInsertionPointToStart(entry);
      auto constantIndex = [&](int64_t value) {
        return Value(builder.create<arith::ConstantIndexOp>(source.getLoc(), value));
      };
      int64_t elements = elementCount(outputType);
      int64_t stride = 1;
      SmallVector<int64_t> strides(outputType.getRank());
      for (int64_t axis = outputType.getRank() - 1; axis >= 0; --axis) {
        strides[axis] = stride;
        stride *= outputType.getDimSize(axis);
      }

      auto emitScalarRange = [&](Value lower, Value upper) -> LogicalResult {
        auto loop = builder.create<scf::ForOp>(
            source.getLoc(), lower, upper,
            constantIndex(1));
        if (loop.getBody()->empty()) {
          builder.setInsertionPointToEnd(loop.getBody());
          builder.create<scf::YieldOp>(source.getLoc());
        }
        builder.setInsertionPoint(loop.getBody()->getTerminator());
        SmallVector<Value> coordinates(outputType.getRank());
        for (int64_t axis = 0; axis < outputType.getRank(); ++axis) {
          Value coordinate = loop.getInductionVar();
          if (strides[axis] != 1)
            coordinate = builder.create<arith::DivUIOp>(
                source.getLoc(), coordinate, constantIndex(strides[axis]));
          if (axis != 0)
            coordinate = builder.create<arith::RemUIOp>(
                source.getLoc(), coordinate,
                constantIndex(outputType.getDimSize(axis)));
          coordinates[axis] = coordinate;
        }
        ScalarEmitter emitter(source, target, builder);
        FailureOr<Value> scalar =
            emitter.emit(returnOp.getOperand(0), coordinates);
        if (failed(scalar)) return failure();
        builder.create<memref::StoreOp>(
            source.getLoc(), *scalar,
            target.getArgument(outputArgument),
            ValueRange{loop.getInductionVar()});
        builder.setInsertionPointAfter(loop);
        return success();
      };

      auto floating = dyn_cast<FloatType>(outputType.getElementType());
      // Use one 512-bit semantic vector. LLVM legally splits it on narrower
      // hosts and retains a single AVX-512 operation on the current Zen 5
      // target, while the scalar cleanup keeps arbitrary extents correct.
      int64_t vectorWidth = floating ? 512 / floating.getWidth() : 1;
      llvm::DenseSet<Value> visited;
      bool vectorize = vectorWidth > 1 && elements >= vectorWidth &&
          isVectorizablePointwise(returnOp.getOperand(0), elements, visited);
      Value rangeBegin = target.getArgument(outputArgument + 1);
      Value rangeEnd = target.getArgument(outputArgument + 2);
      Value vectorEnd = rangeBegin;
      if (vectorize) {
        Value range = builder.create<arith::SubIOp>(
            source.getLoc(), rangeEnd, rangeBegin);
        Value chunks = builder.create<arith::DivUIOp>(
            source.getLoc(), range, constantIndex(vectorWidth));
        Value vectorSpan = builder.create<arith::MulIOp>(
            source.getLoc(), chunks, constantIndex(vectorWidth));
        vectorEnd = builder.create<arith::AddIOp>(
            source.getLoc(), rangeBegin, vectorSpan);
        auto vectorType = VectorType::get(
            {vectorWidth}, outputType.getElementType());
        auto loop = builder.create<scf::ForOp>(
            source.getLoc(), rangeBegin, vectorEnd,
            constantIndex(vectorWidth));
        if (loop.getBody()->empty()) {
          builder.setInsertionPointToEnd(loop.getBody());
          builder.create<scf::YieldOp>(source.getLoc());
        }
        builder.setInsertionPoint(loop.getBody()->getTerminator());
        VectorEmitter emitter(source, target, builder, vectorType);
        FailureOr<Value> packed =
            emitter.emit(returnOp.getOperand(0), loop.getInductionVar());
        if (failed(packed)) return signalPassFailure();
        builder.create<vector::StoreOp>(
            source.getLoc(), *packed,
            target.getArgument(outputArgument),
            ValueRange{loop.getInductionVar()});
        builder.setInsertionPointAfter(loop);
        target->setAttr("graphforge.cpu.vector_width",
                        builder.getI64IntegerAttr(vectorWidth));
      }
      if (failed(emitScalarRange(vectorEnd, rangeEnd)))
        return signalPassFailure();
      builder.create<func::ReturnOp>(source.getLoc());
      source.erase();
    }
  }
};

} // namespace
} // namespace mlir::graphforge
