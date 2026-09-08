#include "graphforge/Dialect/Control/ControlDialect.h"
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
namespace gfc = mlir::graphforge::control;
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
  ScalarEmitter(func::FuncOp source, func::FuncOp target, OpBuilder &builder,
                DenseMap<Value, Value> tensorBuffers = {},
                DenseMap<Value, Value> tensorAliases = {})
      : source(source), target(target), builder(builder),
        tensorBuffers(std::move(tensorBuffers)),
        tensorAliases(std::move(tensorAliases)), location(source.getLoc()) {}

  FailureOr<Value> emit(Value value, ArrayRef<Value> coordinates) {
    // Loop lowering materializes shared tensor SSA and every scalar SSA into
    // scratch storage once per iteration.  These entries are not restricted
    // to region block arguments: reductions and dependent scalar algebra are
    // ordinary operation results.  Check the complete value map before
    // recursively following producers, otherwise a consumer expands a saved
    // reduction again inside its element loop and turns O(n) solver steps
    // into O(n^2) work.
    auto saved = tensorBuffers.find(value);
    if (saved != tensorBuffers.end()) {
      auto type = cast<RankedTensorType>(value.getType());
      return Value(builder.create<memref::LoadOp>(
          location, saved->second,
          ValueRange{linearize(coordinates, type.getShape())}));
    }
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      auto alias = tensorAliases.find(argument);
      if (alias != tensorAliases.end())
        return emit(alias->second, coordinates);
      return failure();
    }
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
    if (auto compare = dyn_cast_or_null<gft::CompareOp>(operation))
      return emitCompare(compare, coordinates);
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

  FailureOr<Value> emitCompare(gft::CompareOp compare,
                               ArrayRef<Value> coordinates) {
    auto resultType = compare.getResult().getType();
    auto lhsType = compare.getLhs().getType();
    auto rhsType = compare.getRhs().getType();
    FailureOr<Value> left = emit(
        compare.getLhs(), broadcastCoordinates(
            lhsType.getShape(), resultType.getShape(), coordinates));
    FailureOr<Value> right = emit(
        compare.getRhs(), broadcastCoordinates(
            rhsType.getShape(), resultType.getShape(), coordinates));
    if (failed(left) || failed(right)) return failure();
    StringRef predicate = compare.getPredicate();
    if (isa<FloatType>(left->getType())) {
      arith::CmpFPredicate mapped = arith::CmpFPredicate::OEQ;
      if (predicate == "ne") mapped = arith::CmpFPredicate::ONE;
      else if (predicate == "lt") mapped = arith::CmpFPredicate::OLT;
      else if (predicate == "le") mapped = arith::CmpFPredicate::OLE;
      else if (predicate == "gt") mapped = arith::CmpFPredicate::OGT;
      else if (predicate == "ge") mapped = arith::CmpFPredicate::OGE;
      return Value(builder.create<arith::CmpFOp>(
          location, mapped, *left, *right));
    }
    arith::CmpIPredicate mapped = arith::CmpIPredicate::eq;
    if (predicate == "ne") mapped = arith::CmpIPredicate::ne;
    else if (predicate == "lt") mapped = arith::CmpIPredicate::slt;
    else if (predicate == "le") mapped = arith::CmpIPredicate::sle;
    else if (predicate == "gt") mapped = arith::CmpIPredicate::sgt;
    else if (predicate == "ge") mapped = arith::CmpIPredicate::sge;
    return Value(builder.create<arith::CmpIOp>(
        location, mapped, *left, *right));
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
  DenseMap<Value, Value> tensorBuffers;
  DenseMap<Value, Value> tensorAliases;
  Location location;
};

struct LoopTemporary {
  Value semanticValue;
  Value storage;
  RankedTensorType type;
  bool heapAllocated;
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

      if (auto repeat = dyn_cast_or_null<gfc::RepeatOp>(
              returnOp.getOperand(0).getDefiningOp())) {
        if (!llvm::hasSingleElement(repeat.getBody())) {
          repeat.emitError("CPU lowering requires one repeat body block");
          return signalPassFailure();
        }
        auto yield = dyn_cast<gfc::ControlYieldOp>(
            repeat.getBody().front().getTerminator());
        if (!yield) {
          repeat.emitError("CPU lowering requires gf_control.yield");
          return signalPassFailure();
        }
        target->setAttr("tiga.cpu.serial_control", builder.getUnitAttr());
        int64_t carried = repeat.getNumCarried();
        Block &semanticBody = repeat.getBody().front();
        target->setAttr("tiga.cpu.loop_buffers",
                        builder.getI64IntegerAttr(2 * carried));
        SmallVector<RankedTensorType> carriedTypes;
        SmallVector<Value> firstBuffers;
        SmallVector<Value> secondBuffers;
        for (int64_t index = 0; index < carried; ++index) {
          auto type = cast<RankedTensorType>(repeat.getInputs()[index].getType());
          carriedTypes.push_back(type);
          auto scratchType = MemRefType::get(
              {ShapedType::kDynamic}, type.getElementType());
          Value extent = constantIndex(elementCount(type));
          firstBuffers.push_back(builder.create<memref::AllocOp>(
              source.getLoc(), scratchType, ValueRange{extent}));
          secondBuffers.push_back(builder.create<memref::AllocOp>(
              source.getLoc(), scratchType, ValueRange{extent}));
        }
        // Materialize scalar SSA and shared vector SSA once per iteration, in
        // block order. Scalars include reductions and their dependent algebra;
        // shared vectors include values such as CG's A(p) and next residual,
        // which would otherwise be recursively recomputed by each consumer.
        // Single-use vector values remain fused into their consumer.
        SmallVector<LoopTemporary> temporaries;
        SmallVector<Value> tensorTemporaryBuffers;
        int64_t scalarTemporaryCount = 0;
        int64_t tensorTemporaryCount = 0;
        for (Operation &operation : semanticBody.without_terminator()) {
          for (Value result : operation.getResults()) {
            auto type = dyn_cast<RankedTensorType>(result.getType());
            if (!type || (type.getRank() != 0 && result.hasOneUse())) continue;
            Value storage;
            bool heapAllocated = type.getRank() != 0;
            if (heapAllocated) {
              auto scratchType = MemRefType::get(
                  {ShapedType::kDynamic}, type.getElementType());
              storage = builder.create<memref::AllocOp>(
                  source.getLoc(), scratchType,
                  ValueRange{constantIndex(elementCount(type))});
              tensorTemporaryBuffers.push_back(storage);
              ++tensorTemporaryCount;
            } else {
              auto scratchType = MemRefType::get({1}, type.getElementType());
              storage = builder.create<memref::AllocaOp>(
                  source.getLoc(), scratchType);
              ++scalarTemporaryCount;
            }
            temporaries.push_back({result, storage, type, heapAllocated});
          }
        }
        target->setAttr("tiga.cpu.loop_scalar_temporaries",
                        builder.getI64IntegerAttr(scalarTemporaryCount));
        target->setAttr("tiga.cpu.loop_tensor_temporaries",
                        builder.getI64IntegerAttr(tensorTemporaryCount));

        auto coordinatesFor = [&](Value linear, RankedTensorType type) {
          int64_t localStride = 1;
          SmallVector<int64_t> localStrides(type.getRank());
          for (int64_t axis = type.getRank() - 1; axis >= 0; --axis) {
            localStrides[axis] = localStride;
            localStride *= type.getDimSize(axis);
          }
          SmallVector<Value> coordinates(type.getRank());
          for (int64_t axis = 0; axis < type.getRank(); ++axis) {
            Value coordinate = linear;
            if (localStrides[axis] != 1)
              coordinate = builder.create<arith::DivUIOp>(
                  source.getLoc(), coordinate,
                  constantIndex(localStrides[axis]));
            if (axis != 0)
              coordinate = builder.create<arith::RemUIOp>(
                  source.getLoc(), coordinate,
                  constantIndex(type.getDimSize(axis)));
            coordinates[axis] = coordinate;
          }
          return coordinates;
        };

        for (int64_t index = 0; index < carried; ++index) {
          Value extent = constantIndex(elementCount(carriedTypes[index]));
          auto initialize = builder.create<scf::ForOp>(
              source.getLoc(), constantIndex(0), extent, constantIndex(1));
          builder.setInsertionPoint(initialize.getBody()->getTerminator());
          ScalarEmitter emitter(source, target, builder);
          FailureOr<Value> item = emitter.emit(
              repeat.getInputs()[index], coordinatesFor(
                  initialize.getInductionVar(), carriedTypes[index]));
          if (failed(item)) return signalPassFailure();
          builder.create<memref::StoreOp>(
              source.getLoc(), *item, firstBuffers[index],
              ValueRange{initialize.getInductionVar()});
          builder.setInsertionPointAfter(initialize);
        }

        SmallVector<Value> loopBuffers(firstBuffers);
        llvm::append_range(loopBuffers, secondBuffers);
        auto iterations = builder.create<scf::ForOp>(
            source.getLoc(), constantIndex(0),
            constantIndex(repeat.getIterations()), constantIndex(1),
            loopBuffers);
        if (iterations.getBody()->empty()) {
          builder.setInsertionPointToEnd(iterations.getBody());
          builder.create<scf::YieldOp>(
              source.getLoc(), iterations.getRegionIterArgs());
        }
        Operation *iterationYield = iterations.getBody()->getTerminator();
        builder.setInsertionPoint(iterationYield);
        DenseMap<Value, Value> buffers;
        DenseMap<Value, Value> aliases;
        for (int64_t index = 0; index < carried; ++index)
          buffers[semanticBody.getArgument(index)] =
              iterations.getRegionIterArgs()[index];
        for (unsigned index = carried;
             index < semanticBody.getNumArguments(); ++index)
          aliases[semanticBody.getArgument(index)] = repeat.getInputs()[index];
        for (const LoopTemporary &temporary : temporaries) {
          if (temporary.type.getRank() == 0) {
            ScalarEmitter emitter(source, target, builder, buffers, aliases);
            FailureOr<Value> scalar = emitter.emit(
                temporary.semanticValue, {});
            if (failed(scalar)) return signalPassFailure();
            builder.create<memref::StoreOp>(
                source.getLoc(), *scalar, temporary.storage,
                ValueRange{constantIndex(0)});
          } else {
            Value extent = constantIndex(elementCount(temporary.type));
            auto compute = builder.create<scf::ForOp>(
                source.getLoc(), constantIndex(0), extent, constantIndex(1));
            builder.setInsertionPoint(compute.getBody()->getTerminator());
            ScalarEmitter emitter(source, target, builder, buffers, aliases);
            FailureOr<Value> item = emitter.emit(
                temporary.semanticValue,
                coordinatesFor(compute.getInductionVar(), temporary.type));
            if (failed(item)) return signalPassFailure();
            builder.create<memref::StoreOp>(
                source.getLoc(), *item, temporary.storage,
                ValueRange{compute.getInductionVar()});
            builder.setInsertionPoint(iterationYield);
          }
          buffers[temporary.semanticValue] = temporary.storage;
        }
        for (int64_t index = 0; index < carried; ++index) {
          Value extent = constantIndex(elementCount(carriedTypes[index]));
          auto compute = builder.create<scf::ForOp>(
              source.getLoc(), constantIndex(0), extent, constantIndex(1));
          builder.setInsertionPoint(compute.getBody()->getTerminator());
          ScalarEmitter emitter(source, target, builder, buffers, aliases);
          FailureOr<Value> item = emitter.emit(
              yield.getValues()[index], coordinatesFor(
                  compute.getInductionVar(), carriedTypes[index]));
          if (failed(item)) return signalPassFailure();
          builder.create<memref::StoreOp>(
              source.getLoc(), *item,
              iterations.getRegionIterArgs()[carried + index],
              ValueRange{compute.getInductionVar()});
          builder.setInsertionPoint(iterationYield);
        }
        SmallVector<Value> nextIteration;
        for (int64_t index = 0; index < carried; ++index)
          nextIteration.push_back(
              iterations.getRegionIterArgs()[carried + index]);
        for (int64_t index = 0; index < carried; ++index)
          nextIteration.push_back(iterations.getRegionIterArgs()[index]);
        iterationYield->setOperands(nextIteration);
        builder.setInsertionPointAfter(iterations);

        auto returnedResult = cast<OpResult>(returnOp.getOperand(0));
        unsigned selectedResult = returnedResult.getResultNumber();
        Value finalState = iterations.getResult(selectedResult);
        auto copy = builder.create<scf::ForOp>(
            source.getLoc(), target.getArgument(outputArgument + 1),
            target.getArgument(outputArgument + 2), constantIndex(1));
        builder.setInsertionPoint(copy.getBody()->getTerminator());
        Value finalItem = builder.create<memref::LoadOp>(
            source.getLoc(), finalState, ValueRange{copy.getInductionVar()});
        builder.create<memref::StoreOp>(
            source.getLoc(), finalItem, target.getArgument(outputArgument),
            ValueRange{copy.getInductionVar()});
        builder.setInsertionPointAfter(copy);
        for (Value buffer : firstBuffers)
          builder.create<memref::DeallocOp>(source.getLoc(), buffer);
        for (Value buffer : secondBuffers)
          builder.create<memref::DeallocOp>(source.getLoc(), buffer);
        for (Value buffer : tensorTemporaryBuffers)
          builder.create<memref::DeallocOp>(source.getLoc(), buffer);
        builder.create<func::ReturnOp>(source.getLoc());
        source.erase();
        continue;
      }

      if (auto bounded = dyn_cast_or_null<gfc::WhileOp>(
              returnOp.getOperand(0).getDefiningOp())) {
        if (!llvm::hasSingleElement(bounded.getCondition()) ||
            !llvm::hasSingleElement(bounded.getBody())) {
          bounded.emitError(
              "CPU lowering requires one condition and one body block");
          return signalPassFailure();
        }
        auto condition = dyn_cast<gfc::ConditionOp>(
            bounded.getCondition().front().getTerminator());
        auto yield = dyn_cast<gfc::ControlYieldOp>(
            bounded.getBody().front().getTerminator());
        if (!condition || !yield) {
          bounded.emitError("CPU lowering requires control terminators");
          return signalPassFailure();
        }
        target->setAttr("tiga.cpu.serial_control", builder.getUnitAttr());
        target->setAttr("tiga.cpu.bounded_while", builder.getUnitAttr());
        target->setAttr(
            "tiga.cpu.max_iterations",
            builder.getI64IntegerAttr(bounded.getMaxIterations()));
        int64_t carried = bounded.getNumCarried();
        Block &semanticCondition = bounded.getCondition().front();
        Block &semanticBody = bounded.getBody().front();
        target->setAttr("tiga.cpu.loop_buffers",
                        builder.getI64IntegerAttr(2 * carried));
        SmallVector<RankedTensorType> carriedTypes;
        SmallVector<Value> firstBuffers;
        SmallVector<Value> secondBuffers;
        for (int64_t index = 0; index < carried; ++index) {
          auto type = cast<RankedTensorType>(bounded.getInputs()[index].getType());
          carriedTypes.push_back(type);
          auto scratchType = MemRefType::get(
              {ShapedType::kDynamic}, type.getElementType());
          Value extent = constantIndex(elementCount(type));
          firstBuffers.push_back(builder.create<memref::AllocOp>(
              source.getLoc(), scratchType, ValueRange{extent}));
          secondBuffers.push_back(builder.create<memref::AllocOp>(
              source.getLoc(), scratchType, ValueRange{extent}));
        }

        SmallVector<LoopTemporary> temporaries;
        SmallVector<Value> tensorTemporaryBuffers;
        int64_t scalarTemporaryCount = 0;
        int64_t tensorTemporaryCount = 0;
        for (Operation &operation : semanticBody.without_terminator()) {
          for (Value result : operation.getResults()) {
            auto type = dyn_cast<RankedTensorType>(result.getType());
            if (!type || (type.getRank() != 0 && result.hasOneUse())) continue;
            Value storage;
            bool heapAllocated = type.getRank() != 0;
            if (heapAllocated) {
              auto scratchType = MemRefType::get(
                  {ShapedType::kDynamic}, type.getElementType());
              storage = builder.create<memref::AllocOp>(
                  source.getLoc(), scratchType,
                  ValueRange{constantIndex(elementCount(type))});
              tensorTemporaryBuffers.push_back(storage);
              ++tensorTemporaryCount;
            } else {
              auto scratchType = MemRefType::get({1}, type.getElementType());
              storage = builder.create<memref::AllocaOp>(
                  source.getLoc(), scratchType);
              ++scalarTemporaryCount;
            }
            temporaries.push_back({result, storage, type, heapAllocated});
          }
        }
        target->setAttr(
            "tiga.cpu.loop_scalar_temporaries",
            builder.getI64IntegerAttr(scalarTemporaryCount));
        target->setAttr(
            "tiga.cpu.loop_tensor_temporaries",
            builder.getI64IntegerAttr(tensorTemporaryCount));

        auto coordinatesFor = [&](OpBuilder &nested, Value linear,
                                  RankedTensorType type) {
          auto localConstant = [&](int64_t value) {
            return Value(nested.create<arith::ConstantIndexOp>(
                source.getLoc(), value));
          };
          int64_t localStride = 1;
          SmallVector<int64_t> localStrides(type.getRank());
          for (int64_t axis = type.getRank() - 1; axis >= 0; --axis) {
            localStrides[axis] = localStride;
            localStride *= type.getDimSize(axis);
          }
          SmallVector<Value> coordinates(type.getRank());
          for (int64_t axis = 0; axis < type.getRank(); ++axis) {
            Value coordinate = linear;
            if (localStrides[axis] != 1)
              coordinate = nested.create<arith::DivUIOp>(
                  source.getLoc(), coordinate,
                  localConstant(localStrides[axis]));
            if (axis != 0)
              coordinate = nested.create<arith::RemUIOp>(
                  source.getLoc(), coordinate,
                  localConstant(type.getDimSize(axis)));
            coordinates[axis] = coordinate;
          }
          return coordinates;
        };

        for (int64_t index = 0; index < carried; ++index) {
          Value extent = constantIndex(elementCount(carriedTypes[index]));
          auto initialize = builder.create<scf::ForOp>(
              source.getLoc(), constantIndex(0), extent, constantIndex(1));
          builder.setInsertionPoint(initialize.getBody()->getTerminator());
          ScalarEmitter emitter(source, target, builder);
          FailureOr<Value> item = emitter.emit(
              bounded.getInputs()[index], coordinatesFor(
                  builder, initialize.getInductionVar(), carriedTypes[index]));
          if (failed(item)) return signalPassFailure();
          builder.create<memref::StoreOp>(
              source.getLoc(), *item, firstBuffers[index],
              ValueRange{initialize.getInductionVar()});
          builder.setInsertionPointAfter(initialize);
        }

        SmallVector<Value> loopInputs{constantIndex(0)};
        llvm::append_range(loopInputs, firstBuffers);
        llvm::append_range(loopInputs, secondBuffers);
        SmallVector<Type> loopTypes;
        for (Value input : loopInputs) loopTypes.push_back(input.getType());
        bool loweringFailed = false;
        auto loop = builder.create<scf::WhileOp>(
            source.getLoc(), TypeRange(loopTypes), ValueRange(loopInputs),
            [&](OpBuilder &nested, Location location, ValueRange arguments) {
              DenseMap<Value, Value> buffers;
              DenseMap<Value, Value> aliases;
              for (int64_t index = 0; index < carried; ++index)
                buffers[semanticCondition.getArgument(index)] =
                    arguments[1 + index];
              for (unsigned index = carried;
                   index < semanticCondition.getNumArguments(); ++index)
                aliases[semanticCondition.getArgument(index)] =
                    bounded.getInputs()[index];
              ScalarEmitter emitter(
                  source, target, nested, buffers, aliases);
              FailureOr<Value> predicate = emitter.emit(
                  condition.getValue(), {});
              if (failed(predicate)) {
                loweringFailed = true;
                nested.create<scf::ConditionOp>(
                    location,
                    nested.create<arith::ConstantIntOp>(location, 0, 1),
                    arguments);
                return;
              }
              Value limit = nested.create<arith::ConstantIndexOp>(
                  location, bounded.getMaxIterations());
              Value underLimit = nested.create<arith::CmpIOp>(
                  location, arith::CmpIPredicate::ult, arguments.front(), limit);
              Value active = nested.create<arith::AndIOp>(
                  location, underLimit, *predicate);
              nested.create<scf::ConditionOp>(location, active, arguments);
            },
            [&](OpBuilder &nested, Location location, ValueRange arguments) {
              DenseMap<Value, Value> buffers;
              DenseMap<Value, Value> aliases;
              for (int64_t index = 0; index < carried; ++index)
                buffers[semanticBody.getArgument(index)] =
                    arguments[1 + index];
              for (unsigned index = carried;
                   index < semanticBody.getNumArguments(); ++index)
                aliases[semanticBody.getArgument(index)] =
                    bounded.getInputs()[index];
              for (const LoopTemporary &temporary : temporaries) {
                if (temporary.type.getRank() == 0) {
                  ScalarEmitter emitter(
                      source, target, nested, buffers, aliases);
                  FailureOr<Value> scalar = emitter.emit(
                      temporary.semanticValue, {});
                  if (failed(scalar)) {
                    loweringFailed = true;
                    continue;
                  }
                  nested.create<memref::StoreOp>(
                      location, *scalar, temporary.storage,
                      ValueRange{nested.create<arith::ConstantIndexOp>(
                          location, 0)});
                } else {
                  Value extent = nested.create<arith::ConstantIndexOp>(
                      location, elementCount(temporary.type));
                  auto compute = nested.create<scf::ForOp>(
                      location,
                      nested.create<arith::ConstantIndexOp>(location, 0),
                      extent,
                      nested.create<arith::ConstantIndexOp>(location, 1));
                  nested.setInsertionPoint(compute.getBody()->getTerminator());
                  ScalarEmitter emitter(
                      source, target, nested, buffers, aliases);
                  FailureOr<Value> item = emitter.emit(
                      temporary.semanticValue, coordinatesFor(
                          nested, compute.getInductionVar(), temporary.type));
                  if (failed(item)) {
                    loweringFailed = true;
                  } else {
                    nested.create<memref::StoreOp>(
                        location, *item, temporary.storage,
                        ValueRange{compute.getInductionVar()});
                  }
                  nested.setInsertionPointAfter(compute);
                }
                buffers[temporary.semanticValue] = temporary.storage;
              }
              for (int64_t index = 0; index < carried; ++index) {
                Value extent = nested.create<arith::ConstantIndexOp>(
                    location, elementCount(carriedTypes[index]));
                auto compute = nested.create<scf::ForOp>(
                    location,
                    nested.create<arith::ConstantIndexOp>(location, 0),
                    extent,
                    nested.create<arith::ConstantIndexOp>(location, 1));
                nested.setInsertionPoint(compute.getBody()->getTerminator());
                ScalarEmitter emitter(
                    source, target, nested, buffers, aliases);
                FailureOr<Value> item = emitter.emit(
                    yield.getValues()[index], coordinatesFor(
                        nested, compute.getInductionVar(), carriedTypes[index]));
                if (failed(item)) {
                  loweringFailed = true;
                } else {
                  nested.create<memref::StoreOp>(
                      location, *item, arguments[1 + carried + index],
                      ValueRange{compute.getInductionVar()});
                }
                nested.setInsertionPointAfter(compute);
              }
              SmallVector<Value> next;
              next.push_back(nested.create<arith::AddIOp>(
                  location, arguments.front(),
                  nested.create<arith::ConstantIndexOp>(location, 1)));
              for (int64_t index = 0; index < carried; ++index)
                next.push_back(arguments[1 + carried + index]);
              for (int64_t index = 0; index < carried; ++index)
                next.push_back(arguments[1 + index]);
              nested.create<scf::YieldOp>(location, next);
            });
        if (loweringFailed) return signalPassFailure();
        builder.setInsertionPointAfter(loop);

        auto returnedResult = cast<OpResult>(returnOp.getOperand(0));
        unsigned selectedResult = returnedResult.getResultNumber();
        Value finalState = loop.getResult(1 + selectedResult);
        auto copy = builder.create<scf::ForOp>(
            source.getLoc(), target.getArgument(outputArgument + 1),
            target.getArgument(outputArgument + 2), constantIndex(1));
        builder.setInsertionPoint(copy.getBody()->getTerminator());
        Value finalItem = builder.create<memref::LoadOp>(
            source.getLoc(), finalState, ValueRange{copy.getInductionVar()});
        builder.create<memref::StoreOp>(
            source.getLoc(), finalItem, target.getArgument(outputArgument),
            ValueRange{copy.getInductionVar()});
        builder.setInsertionPointAfter(copy);
        for (Value buffer : firstBuffers)
          builder.create<memref::DeallocOp>(source.getLoc(), buffer);
        for (Value buffer : secondBuffers)
          builder.create<memref::DeallocOp>(source.getLoc(), buffer);
        for (Value buffer : tensorTemporaryBuffers)
          builder.create<memref::DeallocOp>(source.getLoc(), buffer);
        builder.create<func::ReturnOp>(source.getLoc());
        source.erase();
        continue;
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
        target->setAttr("tiga.cpu.vector_width",
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
