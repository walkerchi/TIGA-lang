#include "graphforge/Target/Triton/Translate.h"

#include "graphforge/Dialect/Control/ControlDialect.h"
#include "graphforge/Dialect/Domain/DomainDialect.h"
#include "graphforge/Dialect/Kernel/KernelDialect.h"
#include "graphforge/Dialect/Storage/StorageDialect.h"
#include "graphforge/Dialect/Task/TaskDialect.h"
#include "graphforge/Dialect/Tensor/TensorDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include <limits>
#include <cmath>
#include <cctype>
#include <string>

namespace mlir::graphforge {
namespace {

static LogicalResult reject(Operation *operation, const Twine &reason) {
  operation->emitError() << "cannot lower GraphForge Kernel op to TTIR: "
                         << reason;
  return failure();
}

static StringRef reducerKind(Operation *operation, Attribute attribute) {
  if (auto legacy = dyn_cast<StringAttr>(attribute))
    return legacy.getValue();
  auto reference = dyn_cast<FlatSymbolRefAttr>(attribute);
  if (!reference)
    return {};
  auto definition =
      SymbolTable::lookupNearestSymbolFrom<ReducerOp>(operation, reference);
  if (!definition)
    return {};

  auto typeAt = [](ArrayAttr types, size_t index) -> Type {
    if (index >= types.size()) return {};
    auto attribute = dyn_cast<TypeAttr>(types[index]);
    return attribute ? attribute.getValue() : Type();
  };
  // Recognize the stable (max, denominator, numerator) tuple algebra from its
  // typed regions.  The frontend's descriptive `kind` is not an optimization
  // key: user reducers with the same algebra receive the same schedule.
  if (definition.getMessageTypes().size() == 2 &&
      definition.getStateTypes().size() == 3 &&
      definition.getResultTypes().size() == 1) {
    Type f32 = Float32Type::get(operation->getContext());
    auto messageVector = dyn_cast<VectorType>(
        typeAt(definition.getMessageTypes(), 1));
    auto stateVector = dyn_cast<VectorType>(
        typeAt(definition.getStateTypes(), 2));
    auto resultVector = dyn_cast<VectorType>(
        typeAt(definition.getResultTypes(), 0));
    Block &combine = definition.getCombine().front();
    Block &finalize = definition.getFinalize().front();
    auto combineYield = cast<ReducerYieldOp>(combine.getTerminator());
    auto finalizeYield = cast<ReducerYieldOp>(finalize.getTerminator());
    bool typesMatch =
        typeAt(definition.getMessageTypes(), 0) == f32 && messageVector &&
        messageVector.getElementType().isF16() &&
        typeAt(definition.getStateTypes(), 0) == f32 &&
        typeAt(definition.getStateTypes(), 1) == f32 && stateVector &&
        stateVector.getElementType().isF32() && resultVector == messageVector &&
        stateVector.getShape() == messageVector.getShape();
    bool combineShape = combineYield.getValues().size() == 3 &&
        combineYield.getValues()[0].getDefiningOp<arith::MaximumFOp>() &&
        combineYield.getValues()[1].getDefiningOp<arith::AddFOp>() &&
        combineYield.getValues()[2].getDefiningOp<arith::AddFOp>();
    auto truncation = finalizeYield.getValues().size() == 1
                          ? finalizeYield.getValues()[0]
                                .getDefiningOp<arith::TruncFOp>()
                          : arith::TruncFOp();
    bool finalizeShape = truncation &&
        truncation.getIn().getDefiningOp<arith::DivFOp>();
    if (typesMatch && combineShape && finalizeShape)
      return "streaming";
  }
  if (definition.getKind() != "algebraic")
    return definition.getKind();

  // Optimized schedules are selected from reducer algebra, not Python class
  // names. Recognize the canonical scalar additive monoid structurally.
  if (definition.getMessageTypes().size() != 1 ||
      definition.getStateTypes().size() != 1 ||
      definition.getResultTypes().size() != 1)
    return definition.getKind();
  Block &identity = definition.getIdentity().front();
  Block &lift = definition.getLift().front();
  Block &combine = definition.getCombine().front();
  Block &finalize = definition.getFinalize().front();
  auto identityYield = cast<ReducerYieldOp>(identity.getTerminator());
  auto liftYield = cast<ReducerYieldOp>(lift.getTerminator());
  auto combineYield = cast<ReducerYieldOp>(combine.getTerminator());
  auto finalizeYield = cast<ReducerYieldOp>(finalize.getTerminator());
  Value identityValue = identityYield.getValues()[0];
  if (auto broadcast = identityValue.getDefiningOp<vector::BroadcastOp>())
    identityValue = broadcast.getSource();
  auto zero = identityValue.getDefiningOp<arith::ConstantOp>();
  auto zeroValue = zero ? dyn_cast<FloatAttr>(zero.getValue()) : FloatAttr();
  auto addition =
      combineYield.getValues()[0].getDefiningOp<arith::AddFOp>();
  bool combinesArguments = addition &&
      ((addition.getLhs() == combine.getArgument(0) &&
        addition.getRhs() == combine.getArgument(1)) ||
       (addition.getLhs() == combine.getArgument(1) &&
        addition.getRhs() == combine.getArgument(0)));
  if (zeroValue && zeroValue.getValue().isZero() &&
      liftYield.getValues()[0] == lift.getArgument(0) &&
      combinesArguments &&
      finalizeYield.getValues()[0] == finalize.getArgument(0))
    return "sum";
  return definition.getKind();
}

/// Serialize target-independent scalar algebra into the provider TTIR module.
/// This is deliberately an IR-to-IR emitter: it keys on operations and types,
/// never on Python reducer or workload names.  Text exists only at the pinned
/// GraphForge MLIR -> provider MLIR process boundary.
class ScalarAlgebraEmitter {
public:
  ScalarAlgebraEmitter(Operation *owner, llvm::raw_ostream &output,
                       std::string valuePrefix = "%gf_v")
      : owner(owner), output(output), valuePrefix(std::move(valuePrefix)) {}

  FailureOr<SmallVector<std::string>> emit(
      Region &region, ArrayRef<std::string> arguments, StringRef indent) {
    if (!llvm::hasSingleElement(region))
      return fail("scalar algebra region must contain one block");
    Block &block = region.front();
    if (block.getNumArguments() != arguments.size())
      return fail("scalar algebra region argument count is inconsistent");
    llvm::DenseMap<Value, std::string> values;
    for (auto [argument, name] : llvm::zip(block.getArguments(), arguments))
      values.try_emplace(argument, name);

    for (Operation &operation : block.without_terminator()) {
      if (operation.getNumResults() != 1 ||
          !operation.getResult(0).getType().isF32())
        return fail("generic dense TTIR currently supports scalar f32 algebra");
      std::string result = nextValue();
      auto operand = [&](unsigned index) -> FailureOr<std::string> {
        auto found = values.find(operation.getOperand(index));
        if (found == values.end())
          return failValue(
              "scalar algebra references an unavailable SSA value");
        return found->second;
      };
      output << indent << result << " = ";
      if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
        auto value = dyn_cast<FloatAttr>(constant.getValue());
        if (!value) return fail("only floating constants are supported");
        output << "arith.constant ";
        value.print(output);
        output << "\n";
      } else if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp,
                     arith::DivFOp, arith::MaximumFOp>(operation)) {
        FailureOr<std::string> left = operand(0), right = operand(1);
        if (failed(left) || failed(right)) return failure();
        StringRef mnemonic =
            isa<arith::AddFOp>(operation) ? "arith.addf" :
            isa<arith::SubFOp>(operation) ? "arith.subf" :
            isa<arith::MulFOp>(operation) ? "arith.mulf" :
            isa<arith::DivFOp>(operation) ? "arith.divf" : "arith.maximumf";
        output << mnemonic << " " << *left << ", " << *right << " : f32\n";
      } else if (isa<math::ExpOp>(operation)) {
        FailureOr<std::string> input = operand(0);
        if (failed(input)) return failure();
        output << "math.exp " << *input << " : f32\n";
      } else {
        return fail(Twine("unsupported scalar algebra operation '") +
                    operation.getName().getStringRef() + "'");
      }
      values.try_emplace(operation.getResult(0), std::move(result));
    }

    Operation *terminator = block.getTerminator();
    if (!isa<ReducerYieldOp, kernel::YieldOp>(terminator))
      return fail("scalar algebra region has an unsupported terminator");
    SmallVector<std::string> results;
    for (Value value : terminator->getOperands()) {
      auto found = values.find(value);
      if (found == values.end())
        return fail("scalar algebra yield references an unavailable SSA value");
      results.push_back(found->second);
    }
    return results;
  }

private:
  FailureOr<std::string> failValue(const Twine &message) {
    owner->emitError() << "cannot serialize scalar algebra to TTIR: " << message;
    return failure();
  }

  FailureOr<SmallVector<std::string>> fail(const Twine &message) {
    owner->emitError() << "cannot serialize scalar algebra to TTIR: " << message;
    return failure();
  }

  std::string nextValue() { return valuePrefix + std::to_string(next++); }

  Operation *owner;
  llvm::raw_ostream &output;
  std::string valuePrefix;
  unsigned next = 0;
};

static std::string sanitizeIdentifier(StringRef input, unsigned fallback) {
  std::string result;
  for (char character : input) {
    unsigned char value = static_cast<unsigned char>(character);
    result.push_back(std::isalnum(value) || character == '_' ? character : '_');
  }
  if (result.empty() || std::isdigit(static_cast<unsigned char>(result.front())))
    result = "arg" + std::to_string(fallback) + "_" + result;
  return result;
}

static bool isZeroAdditiveState(ReducerOp reducer) {
  Block &identity = reducer.getIdentity().front();
  Block &combine = reducer.getCombine().front();
  auto identityYield = dyn_cast<ReducerYieldOp>(identity.getTerminator());
  auto combineYield = dyn_cast<ReducerYieldOp>(combine.getTerminator());
  const size_t states = reducer.getStateTypes().size();
  if (!identityYield || !combineYield ||
      identityYield.getValues().size() != states ||
      combineYield.getValues().size() != states ||
      combine.getNumArguments() != 2 * states)
    return false;
  for (size_t index = 0; index < states; ++index) {
    auto constant = identityYield.getValues()[index]
                        .getDefiningOp<arith::ConstantOp>();
    auto value = constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr();
    auto addition = combineYield.getValues()[index]
                        .getDefiningOp<arith::AddFOp>();
    if (!value || !value.getValue().isZero() || !addition)
      return false;
    Value left = combine.getArgument(index);
    Value right = combine.getArgument(states + index);
    if (!((addition.getLhs() == left && addition.getRhs() == right) ||
          (addition.getLhs() == right && addition.getRhs() == left)))
      return false;
  }
  return true;
}

static bool isBinaryOperationOf(Value value, Value left, Value right,
                                OperationName name) {
  Operation *operation = value.getDefiningOp();
  if (!operation || operation->getName() != name ||
      operation->getNumOperands() != 2)
    return false;
  return (operation->getOperand(0) == left &&
          operation->getOperand(1) == right) ||
         (operation->getOperand(0) == right &&
          operation->getOperand(1) == left);
}

static bool isMaximumOf(Value value, Value left, Value right) {
  return isBinaryOperationOf(
      value, left, right,
      OperationName(arith::MaximumFOp::getOperationName(),
                    value.getContext()));
}

static bool isStableScale(Value value, Value score, Value factor,
                          Value leftMaximum, Value rightMaximum) {
  auto multiply = value.getDefiningOp<arith::MulFOp>();
  if (!multiply) return false;
  Value exponential = multiply.getLhs() == factor ? multiply.getRhs() :
                      multiply.getRhs() == factor ? multiply.getLhs() : Value();
  auto exp = exponential ? exponential.getDefiningOp<math::ExpOp>()
                         : math::ExpOp();
  auto subtract = exp ? exp.getOperand().getDefiningOp<arith::SubFOp>()
                      : arith::SubFOp();
  return subtract && subtract.getLhs() == score &&
         isMaximumOf(subtract.getRhs(), leftMaximum, rightMaximum);
}

static bool isStableScaledSum(Value value, Value leftScore, Value leftFactor,
                              Value rightScore, Value rightFactor) {
  auto addition = value.getDefiningOp<arith::AddFOp>();
  if (!addition) return false;
  auto matches = [&](Value left, Value right) {
    return isStableScale(left, leftScore, leftFactor, leftScore, rightScore) &&
           isStableScale(right, rightScore, rightFactor,
                         leftScore, rightScore);
  };
  return matches(addition.getLhs(), addition.getRhs()) ||
         matches(addition.getRhs(), addition.getLhs());
}

/// Prove the canonical stable weighted-mean tuple algebra without consulting
/// reducer names. This makes the two-pass max/exp/sum reassociation available
/// to user-defined reducers while refusing near-miss state machines.
static bool isStableWeightedScalarState(ReducerOp reducer) {
  Type f32 = Float32Type::get(reducer.getContext());
  if (reducer.getMessageTypes().size() != 2 ||
      reducer.getStateTypes().size() != 3 ||
      reducer.getResultTypes().size() != 1 ||
      !llvm::all_of(reducer.getMessageTypes(), [&](Attribute attribute) {
        auto type = dyn_cast<TypeAttr>(attribute);
        return type && type.getValue() == f32;
      }) ||
      !llvm::all_of(reducer.getStateTypes(), [&](Attribute attribute) {
        auto type = dyn_cast<TypeAttr>(attribute);
        return type && type.getValue() == f32;
      }) ||
      cast<TypeAttr>(reducer.getResultTypes()[0]).getValue() != f32)
    return false;

  Block &identity = reducer.getIdentity().front();
  Block &lift = reducer.getLift().front();
  Block &combine = reducer.getCombine().front();
  Block &finalize = reducer.getFinalize().front();
  auto identityYield = dyn_cast<ReducerYieldOp>(identity.getTerminator());
  auto liftYield = dyn_cast<ReducerYieldOp>(lift.getTerminator());
  auto combineYield = dyn_cast<ReducerYieldOp>(combine.getTerminator());
  auto finalizeYield = dyn_cast<ReducerYieldOp>(finalize.getTerminator());
  if (!identityYield || !liftYield || !combineYield || !finalizeYield ||
      identityYield.getValues().size() != 3 ||
      lift.getNumArguments() != 2 || liftYield.getValues().size() != 3 ||
      combine.getNumArguments() != 6 || combineYield.getValues().size() != 3 ||
      finalize.getNumArguments() != 3 ||
      finalizeYield.getValues().size() != 1)
    return false;
  auto constantAt = [&](Value value) -> FloatAttr {
    auto constant = value.getDefiningOp<arith::ConstantOp>();
    return constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr();
  };
  FloatAttr maximumIdentity = constantAt(identityYield.getValues()[0]);
  FloatAttr denominatorIdentity = constantAt(identityYield.getValues()[1]);
  FloatAttr numeratorIdentity = constantAt(identityYield.getValues()[2]);
  FloatAttr liftedOne = constantAt(liftYield.getValues()[1]);
  if (!maximumIdentity || !denominatorIdentity || !numeratorIdentity ||
      !liftedOne || !maximumIdentity.getValue().isNegative() ||
      !denominatorIdentity.getValue().isZero() ||
      !numeratorIdentity.getValue().isZero() ||
      !liftedOne.getValue().isExactlyValue(1.0) ||
      liftYield.getValues()[0] != lift.getArgument(0) ||
      liftYield.getValues()[2] != lift.getArgument(1))
    return false;
  Value leftMaximum = combine.getArgument(0);
  Value leftDenominator = combine.getArgument(1);
  Value leftNumerator = combine.getArgument(2);
  Value rightMaximum = combine.getArgument(3);
  Value rightDenominator = combine.getArgument(4);
  Value rightNumerator = combine.getArgument(5);
  if (!isMaximumOf(combineYield.getValues()[0], leftMaximum, rightMaximum) ||
      !isStableScaledSum(combineYield.getValues()[1],
                         leftMaximum, leftDenominator,
                         rightMaximum, rightDenominator) ||
      !isStableScaledSum(combineYield.getValues()[2],
                         leftMaximum, leftNumerator,
                         rightMaximum, rightNumerator))
    return false;
  auto division = finalizeYield.getValues()[0].getDefiningOp<arith::DivFOp>();
  return division && division.getLhs() == finalize.getArgument(2) &&
         division.getRhs() == finalize.getArgument(1);
}

/// Vectorize scalar arith/math regions over one provider tensor tile. This is
/// operation-driven and intentionally has no reducer/workload name cases.
class TensorAlgebraEmitter {
public:
  TensorAlgebraEmitter(Operation *owner, llvm::raw_ostream &output,
                       std::string tensorType,
                       std::string valuePrefix = "%gf_t")
      : owner(owner), output(output), tensorType(std::move(tensorType)),
        valuePrefix(std::move(valuePrefix)) {}

  FailureOr<SmallVector<std::string>> emit(
      Region &region, ArrayRef<std::string> arguments, StringRef indent) {
    if (!llvm::hasSingleElement(region)) return fail("region must be single-block");
    Block &block = region.front();
    if (block.getNumArguments() != arguments.size())
      return fail("region argument count does not match projected inputs");
    llvm::DenseMap<Value, std::string> values;
    for (auto [argument, name] : llvm::zip(block.getArguments(), arguments))
      values.try_emplace(argument, name);
    auto operand = [&](Operation &operation, unsigned index)
        -> FailureOr<std::string> {
      auto found = values.find(operation.getOperand(index));
      return found == values.end() ? FailureOr<std::string>(failure())
                                   : FailureOr<std::string>(found->second);
    };
    for (Operation &operation : block.without_terminator()) {
      if (operation.getNumResults() != 1 ||
          !operation.getResult(0).getType().isF32())
        return fail("tiled algebra currently supports scalar f32 source IR");
      std::string result = valuePrefix + std::to_string(next++);
      output << indent << result << " = ";
      if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
        auto value = dyn_cast<FloatAttr>(constant.getValue());
        if (!value) return fail("only floating constants can be tiled");
        output << "arith.constant dense<"
               << llvm::format("%.9e", value.getValueAsDouble()) << "> : "
               << tensorType << "\n";
      } else if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp,
                     arith::DivFOp, arith::MaximumFOp>(operation)) {
        auto left = operand(operation, 0), right = operand(operation, 1);
        if (failed(left) || failed(right))
          return fail("operation references an unavailable SSA value");
        StringRef mnemonic =
            isa<arith::AddFOp>(operation) ? "arith.addf" :
            isa<arith::SubFOp>(operation) ? "arith.subf" :
            isa<arith::MulFOp>(operation) ? "arith.mulf" :
            isa<arith::DivFOp>(operation) ? "arith.divf" : "arith.maximumf";
        output << mnemonic << " " << *left << ", " << *right << " : "
               << tensorType << "\n";
      } else if (isa<math::ExpOp>(operation)) {
        auto input = operand(operation, 0);
        if (failed(input)) return fail("exp references unavailable SSA");
        output << "math.exp " << *input << " : " << tensorType << "\n";
      } else {
        return fail(Twine("unsupported tiled algebra operation '") +
                    operation.getName().getStringRef() + "'");
      }
      values.try_emplace(operation.getResult(0), result);
    }
    Operation *terminator = block.getTerminator();
    if (!isa<ReducerYieldOp, kernel::YieldOp>(terminator))
      return fail("region has an unsupported terminator");
    SmallVector<std::string> results;
    for (Value value : terminator->getOperands()) {
      auto found = values.find(value);
      if (found == values.end()) return fail("yield references unavailable SSA");
      results.push_back(found->second);
    }
    return results;
  }

private:
  FailureOr<SmallVector<std::string>> fail(const Twine &message) {
    owner->emitError() << "cannot tile scalar algebra: " << message;
    return failure();
  }
  Operation *owner;
  llvm::raw_ostream &output;
  std::string tensorType;
  std::string valuePrefix;
  unsigned next = 0;
};

static void emitGeneratedRadiusDistanceTTIR(llvm::raw_ostream &output,
                                            int64_t numRows,
                                            int64_t dimensions,
                                            int64_t neighborCells,
                                            double cutoff, bool periodic) {
  (void)numRows;
  constexpr int64_t blockD = 32;
  output << "// graphforge.launch entry=gf_generated_radius_distance_sum "
            "block_rows=1 num_warps=1 "
            "abi=cell_ptr,particle_order,cell_coordinates,extents,strides,"
            "neighbor_offsets,lattice,inverse_lattice,positions,x,out\n"
         << "module {\n"
         << "  tt.func public @gf_generated_radius_distance_sum("
         << "%cell_ptr: !tt.ptr<i64>, %particle_order: !tt.ptr<i64>, "
         << "%cell_coordinates: !tt.ptr<i64>, %extents: !tt.ptr<i64>, "
         << "%strides: !tt.ptr<i64>, %neighbor_offsets: !tt.ptr<i64>, "
         << "%lattice: !tt.ptr<f32>, %inverse_lattice: !tt.ptr<f32>, "
         << "%positions: !tt.ptr<f32>, %x: !tt.ptr<f32>, "
         << "%out: !tt.ptr<f32>) attributes {noinline = false} {\n"
         << "    %c0_i64 = arith.constant 0 : i64\n"
         << "    %c1_i64 = arith.constant 1 : i64\n"
         << "    %c32_i64 = arith.constant 32 : i64\n"
         << "    %dimensions = arith.constant " << dimensions << " : i64\n"
         << "    %neighbor_cells = arith.constant " << neighborCells
         << " : i64\n"
         << "    %zero = arith.constant 0.000000e+00 : f32\n"
         << "    %cutoff_squared = arith.constant " << cutoff * cutoff
         << " : f32\n"
         << "    %zero_i64_v = arith.constant dense<0> : tensor<" << blockD
         << "xi64>\n"
         << "    %zero_f32_v = arith.constant dense<0.000000e+00> : tensor<"
         << blockD << "xf32>\n"
         << "    %dimensions_v = arith.constant dense<" << dimensions
         << "> : tensor<" << blockD << "xi64>\n"
         << "    %cutoff_v = arith.constant dense<" << cutoff * cutoff
         << "> : tensor<" << blockD << "xf32>\n"
         << "    %lane_i32 = tt.make_range {end = " << blockD
         << " : i32, start = 0 : i32} : tensor<" << blockD << "xi32>\n"
         << "    %lane = arith.extsi %lane_i32 : tensor<" << blockD
         << "xi32> to tensor<" << blockD << "xi64>\n"
         << "    %row_i32 = tt.get_program_id x : i32\n"
         << "    %row = arith.extsi %row_i32 : i32 to i64\n"
         << "    %row_v = tt.splat %row : i64 -> tensor<" << blockD
         << "xi64>\n"
         << "    %row_base = arith.muli %row, %dimensions : i64\n";
  for (int64_t axis = 0; axis < dimensions; ++axis) {
    output << "    %axis" << axis << " = arith.constant " << axis
           << " : i64\n"
           << "    %axis_v" << axis << " = arith.constant dense<" << axis
           << "> : tensor<" << blockD << "xi64>\n"
           << "    %row_coord_index" << axis
           << " = arith.addi %row_base, %axis" << axis << " : i64\n"
           << "    %row_coord_ptr" << axis
           << " = tt.addptr %cell_coordinates, %row_coord_index" << axis
           << " : !tt.ptr<i64>, i64\n"
           << "    %coordinate" << axis << " = tt.load %row_coord_ptr"
           << axis << " : !tt.ptr<i64>\n"
           << "    %extent_ptr" << axis << " = tt.addptr %extents, %axis"
           << axis << " : !tt.ptr<i64>, i64\n"
           << "    %extent" << axis << " = tt.load %extent_ptr" << axis
           << " : !tt.ptr<i64>\n"
           << "    %stride_ptr" << axis << " = tt.addptr %strides, %axis"
           << axis << " : !tt.ptr<i64>, i64\n"
           << "    %stride" << axis << " = tt.load %stride_ptr" << axis
           << " : !tt.ptr<i64>\n"
           << "    %destination_ptr" << axis
           << " = tt.addptr %positions, %row_coord_index" << axis
           << " : !tt.ptr<f32>, i64\n"
           << "    %destination" << axis << " = tt.load %destination_ptr"
           << axis << " : !tt.ptr<f32>\n"
           << "    %destination_v" << axis << " = tt.splat %destination"
           << axis << " : f32 -> tensor<" << blockD << "xf32>\n";
  }
  output
         << "    %cell_sum = scf.for %neighbor = %c0_i64 to "
            "%neighbor_cells step %c1_i64 iter_args(%cell_acc = %zero) "
            "-> (f32) : i64 {\n";
  output << "      %cell_key0 = arith.constant 0 : i64\n"
         << "      %valid0 = arith.constant true\n";
  for (int64_t axis = 0; axis < dimensions; ++axis) {
    output << "      %offset_base" << axis
           << " = arith.muli %neighbor, %dimensions : i64\n"
           << "      %offset_index" << axis << " = arith.addi %offset_base"
           << axis << ", %axis" << axis << " : i64\n"
           << "      %offset_ptr" << axis
           << " = tt.addptr %neighbor_offsets, %offset_index" << axis
           << " : !tt.ptr<i64>, i64\n"
           << "      %offset" << axis << " = tt.load %offset_ptr" << axis
           << " : !tt.ptr<i64>\n"
           << "      %neighbor_coord_raw" << axis
           << " = arith.addi %coordinate" << axis << ", %offset" << axis
           << " : i64\n";
    if (periodic) {
      output << "      %neighbor_coord_shifted" << axis
             << " = arith.addi %neighbor_coord_raw" << axis << ", %extent"
             << axis << " : i64\n"
             << "      %neighbor_coord" << axis
             << " = arith.remui %neighbor_coord_shifted" << axis
             << ", %extent" << axis << " : i64\n"
             << "      %valid" << axis + 1 << " = arith.constant true\n";
    } else {
      output << "      %neighbor_coord" << axis
             << " = arith.addi %neighbor_coord_raw" << axis
             << ", %c0_i64 : i64\n"
             << "      %lower_ok" << axis
           << " = arith.cmpi sge, %neighbor_coord" << axis
           << ", %c0_i64 : i64\n"
           << "      %upper_ok" << axis
           << " = arith.cmpi slt, %neighbor_coord" << axis << ", %extent"
           << axis << " : i64\n"
           << "      %axis_valid" << axis << " = arith.andi %lower_ok"
           << axis << ", %upper_ok" << axis << " : i1\n"
           << "      %valid" << axis + 1 << " = arith.andi %valid" << axis
           << ", %axis_valid" << axis << " : i1\n";
    }
    output << "      %key_part" << axis
           << " = arith.muli %neighbor_coord" << axis << ", %stride" << axis
           << " : i64\n"
           << "      %cell_key" << axis + 1 << " = arith.addi %cell_key"
           << axis << ", %key_part" << axis << " : i64\n";
  }
  output << "      %safe_key = arith.select %valid" << dimensions
         << ", %cell_key" << dimensions << ", %c0_i64 : i64\n"
         << "      %start_ptr = tt.addptr %cell_ptr, %safe_key : "
            "!tt.ptr<i64>, i64\n"
         << "      %start_raw = tt.load %start_ptr : !tt.ptr<i64>\n"
         << "      %next_key = arith.addi %safe_key, %c1_i64 : i64\n"
         << "      %end_ptr = tt.addptr %cell_ptr, %next_key : "
            "!tt.ptr<i64>, i64\n"
         << "      %end_raw = tt.load %end_ptr : !tt.ptr<i64>\n"
         << "      %start = arith.select %valid" << dimensions
         << ", %start_raw, %c0_i64 : i64\n"
         << "      %end = arith.select %valid" << dimensions
         << ", %end_raw, %c0_i64 : i64\n"
         << "      %slot_sum = scf.for %tile = %start to %end step %c32_i64 "
            "iter_args(%slot_acc = %cell_acc) -> (f32) : i64 {\n"
         << "        %tile_v = tt.splat %tile : i64 -> tensor<" << blockD
         << "xi64>\n"
         << "        %slots = arith.addi %tile_v, %lane : tensor<" << blockD
         << "xi64>\n"
         << "        %end_v = tt.splat %end : i64 -> tensor<" << blockD
         << "xi64>\n"
         << "        %slot_mask = arith.cmpi slt, %slots, %end_v : tensor<"
         << blockD << "xi64>\n"
         << "        %order_base = tt.splat %particle_order : !tt.ptr<i64> "
            "-> tensor<" << blockD << "x!tt.ptr<i64>>\n"
         << "        %order_ptr = tt.addptr %order_base, %slots : tensor<"
         << blockD << "x!tt.ptr<i64>>, tensor<" << blockD << "xi64>\n"
         << "        %source = tt.load %order_ptr, %slot_mask, %zero_i64_v : "
            "tensor<" << blockD << "x!tt.ptr<i64>>\n"
         << "        %source_base = arith.muli %source, %dimensions_v : "
            "tensor<" << blockD << "xi64>\n"
         << "        %distance0 = arith.constant dense<0.000000e+00> : "
            "tensor<" << blockD << "xf32>\n"
         << "        %positions_base = tt.splat %positions : !tt.ptr<f32> "
            "-> tensor<" << blockD << "x!tt.ptr<f32>>\n";
  for (int64_t axis = 0; axis < dimensions; ++axis) {
    output << "        %source_index" << axis
           << " = arith.addi %source_base, %axis_v" << axis << " : tensor<"
           << blockD << "xi64>\n"
           << "        %source_ptr" << axis
           << " = tt.addptr %positions_base, %source_index" << axis
           << " : tensor<" << blockD << "x!tt.ptr<f32>>, tensor<" << blockD
           << "xi64>\n"
           << "        %source_position" << axis << " = tt.load %source_ptr"
           << axis << ", %slot_mask, %zero_f32_v : tensor<" << blockD
           << "x!tt.ptr<f32>>\n"
           << "        %delta" << axis << " = arith.subf %source_position"
           << axis << ", %destination_v" << axis << " : tensor<" << blockD
           << "xf32>\n";
  }
  if (periodic) {
    for (int64_t axis = 0; axis < dimensions; ++axis) {
      output << "        %fractional" << axis
             << "_0 = arith.constant dense<0.000000e+00> : tensor<"
             << blockD << "xf32>\n";
      for (int64_t component = 0; component < dimensions; ++component) {
        int64_t element = component * dimensions + axis;
        output << "        %inverse_index" << component << "_" << axis
               << " = arith.constant " << element << " : i64\n"
               << "        %inverse_ptr" << component << "_" << axis
               << " = tt.addptr %inverse_lattice, %inverse_index" << component
               << "_" << axis << " : !tt.ptr<f32>, i64\n"
               << "        %inverse_value" << component << "_" << axis
               << " = tt.load %inverse_ptr" << component << "_" << axis
               << " : !tt.ptr<f32>\n"
               << "        %inverse_v" << component << "_" << axis
               << " = tt.splat %inverse_value" << component << "_" << axis
               << " : f32 -> tensor<" << blockD << "xf32>\n"
               << "        %fractional_term" << component << "_" << axis
               << " = arith.mulf %delta" << component << ", %inverse_v"
               << component << "_" << axis << " : tensor<" << blockD
               << "xf32>\n"
               << "        %fractional" << axis << "_" << component + 1
               << " = arith.addf %fractional" << axis << "_" << component
               << ", %fractional_term" << component << "_" << axis
               << " : tensor<" << blockD << "xf32>\n";
      }
      output << "        %half" << axis
             << " = arith.constant dense<5.000000e-01> : tensor<" << blockD
             << "xf32>\n"
             << "        %fractional_shifted" << axis
             << " = arith.addf %fractional" << axis << "_" << dimensions
             << ", %half" << axis << " : tensor<" << blockD << "xf32>\n"
             << "        %fractional_round" << axis
             << " = math.floor %fractional_shifted" << axis << " : tensor<"
             << blockD << "xf32>\n"
             << "        %centered" << axis << " = arith.subf %fractional"
             << axis << "_" << dimensions << ", %fractional_round" << axis
             << " : tensor<" << blockD << "xf32>\n";
    }
    for (int64_t axis = 0; axis < dimensions; ++axis) {
      output << "        %cartesian" << axis
             << "_0 = arith.constant dense<0.000000e+00> : tensor<"
             << blockD << "xf32>\n";
      for (int64_t component = 0; component < dimensions; ++component) {
        int64_t element = component * dimensions + axis;
        output << "        %lattice_index" << component << "_" << axis
               << " = arith.constant " << element << " : i64\n"
               << "        %lattice_ptr" << component << "_" << axis
               << " = tt.addptr %lattice, %lattice_index" << component
               << "_" << axis << " : !tt.ptr<f32>, i64\n"
               << "        %lattice_value" << component << "_" << axis
               << " = tt.load %lattice_ptr" << component << "_" << axis
               << " : !tt.ptr<f32>\n"
               << "        %lattice_v" << component << "_" << axis
               << " = tt.splat %lattice_value" << component << "_" << axis
               << " : f32 -> tensor<" << blockD << "xf32>\n"
               << "        %cartesian_term" << component << "_" << axis
               << " = arith.mulf %centered" << component << ", %lattice_v"
               << component << "_" << axis << " : tensor<" << blockD
               << "xf32>\n"
               << "        %cartesian" << axis << "_" << component + 1
               << " = arith.addf %cartesian" << axis << "_" << component
               << ", %cartesian_term" << component << "_" << axis
               << " : tensor<" << blockD << "xf32>\n";
      }
      output << "        %square" << axis << " = arith.mulf %cartesian"
             << axis << "_" << dimensions << ", %cartesian" << axis << "_"
             << dimensions << " : tensor<" << blockD << "xf32>\n"
             << "        %distance" << axis + 1
             << " = arith.addf %distance" << axis << ", %square" << axis
             << " : tensor<" << blockD << "xf32>\n";
    }
  } else {
    for (int64_t axis = 0; axis < dimensions; ++axis) {
      output << "        %square" << axis << " = arith.mulf %delta" << axis
             << ", %delta" << axis << " : tensor<" << blockD << "xf32>\n"
             << "        %distance" << axis + 1 << " = arith.addf %distance"
             << axis << ", %square" << axis << " : tensor<" << blockD
             << "xf32>\n";
    }
  }
  output << "        %not_self = arith.cmpi ne, %source, %row_v : tensor<"
         << blockD << "xi64>\n"
         << "        %candidate_mask = arith.andi %slot_mask, %not_self : "
            "tensor<" << blockD << "xi1>\n"
         << "        %within = arith.cmpf ole, %distance" << dimensions
         << ", %cutoff_v : tensor<" << blockD << "xf32>\n"
         << "        %accept = arith.andi %candidate_mask, %within : tensor<"
         << blockD << "xi1>\n"
         << "        %x_base = tt.splat %x : !tt.ptr<f32> -> tensor<" << blockD
         << "x!tt.ptr<f32>>\n"
         << "        %x_ptr = tt.addptr %x_base, %source : tensor<" << blockD
         << "x!tt.ptr<f32>>, tensor<" << blockD << "xi64>\n"
         << "        %x_value = tt.load %x_ptr, %accept, %zero_f32_v : tensor<"
         << blockD << "x!tt.ptr<f32>>\n"
         << "        %distance = math.sqrt %distance" << dimensions
         << " : tensor<" << blockD << "xf32>\n"
         << "        %weighted = arith.mulf %distance, %x_value : tensor<"
         << blockD << "xf32>\n"
         << "        %tile_sum = \"tt.reduce\"(%weighted) <{axis = 0 : i32}> "
            "({\n"
         << "        ^bb0(%a: f32, %b: f32):\n"
         << "          %combined = arith.addf %a, %b : f32\n"
         << "          tt.reduce.return %combined : f32\n"
         << "        }) : (tensor<" << blockD << "xf32>) -> f32\n"
         << "        %next = arith.addf %slot_acc, %tile_sum : f32\n"
         << "        scf.yield %next : f32\n"
         << "      }\n"
         << "      scf.yield %slot_sum : f32\n"
         << "    }\n"
         << "    %out_ptr = tt.addptr %out, %row : !tt.ptr<f32>, i64\n"
         << "    tt.store %out_ptr, %cell_sum : !tt.ptr<f32>\n"
         << "    tt.return\n"
         << "  }\n"
         << "}\n";
}

static bool isRankOneTensor(Value value, Type elementType) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getRank() == 1 && type.getElementType() == elementType;
}

static int64_t nextPowerOfTwo(int64_t value) {
  int64_t result = 1;
  while (result < value)
    result *= 2;
  return result;
}

static void emitBoundedRaggedTTIR(llvm::raw_ostream &output, StringRef index,
                                  int64_t numRows, int64_t blockM,
                                  int64_t blockD, int64_t numWarps) {
  output << "// graphforge.launch entry=gf_csr_weighted_sum block_rows="
         << blockM << " num_warps=" << numWarps
         << " abi=row_ptr,col_idx,x,weight,out\n"
         << "module {\n"
         << "  tt.func public @gf_csr_weighted_sum("
         << "%row_ptr: !tt.ptr<" << index
         << "> {tt.divisibility = 16 : i32}, "
         << "%col_idx: !tt.ptr<" << index
         << "> {tt.divisibility = 16 : i32}, "
         << "%x: !tt.ptr<f32> {tt.divisibility = 16 : i32}, "
         << "%weight: !tt.ptr<f32> {tt.divisibility = 16 : i32}, "
         << "%out: !tt.ptr<f32> {tt.divisibility = 16 : i32}) "
            "attributes {noinline = false} {\n"
         << "    %source_zero = arith.constant dense<0> : tensor<" << blockM
         << "x" << blockD << "x" << index << ">\n"
         << "    %row_zero = arith.constant dense<0> : tensor<" << blockM
         << "x" << index << ">\n"
         << "    %value_zero = arith.constant dense<0.000000e+00> : tensor<"
         << blockM << "x" << blockD << "xf32>\n"
         << "    %nrows = arith.constant dense<" << numRows << "> : tensor<"
         << blockM << "xi32>\n"
         << "    %cbm = arith.constant " << blockM << " : i32\n"
         << "    %one = arith.constant dense<1> : tensor<" << blockM
         << "xi32>\n"
         << "    %pid = tt.get_program_id x : i32\n"
         << "    %base = arith.muli %pid, %cbm : i32\n"
         << "    %range_m = tt.make_range {end = " << blockM
         << " : i32, start = 0 : i32} : tensor<" << blockM << "xi32>\n"
         << "    %base_v = tt.splat %base : i32 -> tensor<" << blockM
         << "xi32>\n"
         << "    %rows = arith.addi %base_v, %range_m : tensor<" << blockM
         << "xi32>\n"
         << "    %row_mask = arith.cmpi slt, %rows, %nrows : tensor<" << blockM
         << "xi32>\n"
         << "    %row_ptr_base = tt.splat %row_ptr : !tt.ptr<" << index
         << "> -> tensor<" << blockM << "x!tt.ptr<" << index << ">>\n"
         << "    %start_ptr = tt.addptr %row_ptr_base, %rows : tensor<" << blockM
         << "x!tt.ptr<" << index << ">>, tensor<" << blockM << "xi32>\n"
         << "    %starts = tt.load %start_ptr, %row_mask, %row_zero : tensor<"
         << blockM << "x!tt.ptr<" << index << ">>\n"
         << "    %end_ptr = tt.addptr %start_ptr, %one : tensor<"
         << blockM << "x!tt.ptr<" << index << ">>, tensor<" << blockM
         << "xi32>\n"
         << "    %ends = tt.load %end_ptr, %row_mask, %row_zero : tensor<"
         << blockM << "x!tt.ptr<" << index << ">>\n"
         << "    %range_d = tt.make_range {end = " << blockD
         << " : i32, start = 0 : i32} : tensor<" << blockD << "xi32>\n"
         << "    %starts2 = tt.expand_dims %starts {axis = 1 : i32} : tensor<"
         << blockM << "x" << index << "> -> tensor<" << blockM << "x1x"
         << index << ">\n"
         << "    %neighbors = tt.expand_dims %range_d {axis = 0 : i32} : "
            "tensor<" << blockD << "xi32> -> tensor<1x" << blockD
         << "xi32>\n"
         ;
  if (index == "i64") {
    output << "    %neighbors_index = arith.extsi %neighbors : tensor<1x"
           << blockD << "xi32> to tensor<1x" << blockD << "xi64>\n";
  }
  output
         << "    %starts_b = tt.broadcast %starts2 : tensor<" << blockM
         << "x1x" << index << "> -> tensor<" << blockM << "x" << blockD
         << "x" << index << ">\n"
         << "    %neighbors_b = tt.broadcast %neighbors"
         << (index == "i64" ? "_index" : "") << " : tensor<1x" << blockD
         << "x" << index << "> -> tensor<" << blockM << "x" << blockD << "x"
         << index << ">\n";
  output << "    %edges = arith.addi %starts_b, %neighbors_"
         << "b : tensor<" << blockM
         << "x" << blockD << "x" << index << ">\n"
         << "    %row_mask2 = tt.expand_dims %row_mask {axis = 1 : i32} : "
            "tensor<" << blockM << "xi1> -> tensor<" << blockM
         << "x1xi1>\n"
         << "    %ends2 = tt.expand_dims %ends {axis = 1 : i32} : tensor<"
         << blockM << "x" << index << "> -> tensor<" << blockM << "x1x"
         << index << ">\n"
         << "    %ends_b = tt.broadcast %ends2 : tensor<" << blockM << "x1x"
         << index << "> -> tensor<" << blockM << "x" << blockD << "x"
         << index << ">\n"
         << "    %edge_limit = arith.cmpi slt, %edges, %ends_b : tensor<"
         << blockM << "x" << blockD << "x" << index << ">\n"
         << "    %row_mask_b = tt.broadcast %row_mask2 : tensor<" << blockM
         << "x1xi1> -> tensor<" << blockM << "x" << blockD << "xi1>\n"
         << "    %mask = arith.andi %row_mask_b, %edge_limit : tensor<" << blockM
         << "x" << blockD << "xi1>\n"
         << "    %col_base = tt.splat %col_idx : !tt.ptr<" << index
         << "> -> tensor<" << blockM << "x" << blockD << "x!tt.ptr<" << index
         << ">>\n"
         << "    %col_ptr = tt.addptr %col_base, %edges : tensor<" << blockM
         << "x" << blockD << "x!tt.ptr<" << index << ">>, tensor<" << blockM
         << "x" << blockD << "x" << index << ">\n"
         << "    %src = tt.load %col_ptr, %mask, %source_zero : tensor<" << blockM
         << "x" << blockD << "x!tt.ptr<" << index << ">>\n"
         << "    %w_base = tt.splat %weight : !tt.ptr<f32> -> tensor<" << blockM
         << "x" << blockD << "x!tt.ptr<f32>>\n"
         << "    %w_ptr = tt.addptr %w_base, %edges : tensor<" << blockM << "x"
         << blockD << "x!tt.ptr<f32>>, tensor<" << blockM << "x" << blockD
         << "x" << index << ">\n"
         << "    %w_value = tt.load %w_ptr, %mask, %value_zero : tensor<"
         << blockM << "x" << blockD << "x!tt.ptr<f32>>\n"
         << "    %x_base = tt.splat %x : !tt.ptr<f32> -> tensor<" << blockM
         << "x" << blockD << "x!tt.ptr<f32>>\n"
         << "    %x_ptr = tt.addptr %x_base, %src : tensor<" << blockM << "x"
         << blockD << "x!tt.ptr<f32>>, tensor<" << blockM << "x" << blockD
         << "x" << index << ">\n"
         << "    %x_value = tt.load %x_ptr, %mask, %value_zero : tensor<"
         << blockM << "x" << blockD << "x!tt.ptr<f32>>\n"
         << "    %message = arith.mulf %x_value, %w_value : tensor<" << blockM
         << "x" << blockD << "xf32>\n"
         << "    %sum = \"tt.reduce\"(%message) <{axis = 1 : i32}> ({\n"
         << "    ^bb0(%a: f32, %b: f32):\n"
         << "      %combined = arith.addf %a, %b : f32\n"
         << "      tt.reduce.return %combined : f32\n"
         << "    }) : (tensor<" << blockM << "x" << blockD
         << "xf32>) -> tensor<" << blockM << "xf32>\n"
         << "    %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<" << blockM
         << "x!tt.ptr<f32>>\n"
         << "    %out_ptr = tt.addptr %out_base, %rows : tensor<" << blockM
         << "x!tt.ptr<f32>>, tensor<" << blockM << "xi32>\n"
         << "    tt.store %out_ptr, %sum, %row_mask : tensor<" << blockM
         << "x!tt.ptr<f32>>\n"
         << "    tt.return\n"
         << "  }\n"
         << "}\n";
}

static bool isScalarMultiplyRegion(Region &region, unsigned arguments) {
  if (!llvm::hasSingleElement(region))
    return false;
  Block &block = region.front();
  if (block.getNumArguments() != arguments ||
      llvm::range_size(block.without_terminator()) != 1)
    return false;
  auto yield = dyn_cast<kernel::YieldOp>(block.getTerminator());
  auto multiply = yield && yield.getValues().size() == 1
                      ? yield.getValues().front().getDefiningOp<arith::MulFOp>()
                        : arith::MulFOp();
  return multiply && multiply->getBlock() == &block &&
         llvm::is_contained(block.getArguments(), multiply.getLhs()) &&
         llvm::is_contained(block.getArguments(), multiply.getRhs());
}

static bool isVectorScaleMultiplyRegion(Region &region, int64_t width) {
  if (!llvm::hasSingleElement(region)) return false;
  Block &block = region.front();
  if (block.getNumArguments() != 2) return false;
  auto vectorType = VectorType::get({width}, Float32Type::get(region.getContext()));
  bool argumentsMatch =
      (block.getArgument(0).getType() == vectorType &&
       block.getArgument(1).getType().isF32()) ||
      (block.getArgument(1).getType() == vectorType &&
       block.getArgument(0).getType().isF32());
  if (!argumentsMatch) return false;
  auto yield = dyn_cast<kernel::YieldOp>(block.getTerminator());
  auto multiply = yield && yield.getValues().size() == 1
                      ? yield.getValues().front().getDefiningOp<arith::MulFOp>()
                      : arith::MulFOp();
  if (!multiply || multiply.getType() != vectorType) return false;
  auto broadcast = multiply.getLhs().getDefiningOp<vector::BroadcastOp>();
  Value other = multiply.getRhs();
  if (!broadcast) {
    broadcast = multiply.getRhs().getDefiningOp<vector::BroadcastOp>();
    other = multiply.getLhs();
  }
  return broadcast && broadcast.getType() == vectorType &&
         llvm::is_contained(block.getArguments(), broadcast.getSource()) &&
         llvm::is_contained(block.getArguments(), other);
}

static LogicalResult translateGeneratedRadius(
    kernel::GeneratedLaunchOp launch, llvm::raw_ostream &output) {
  MLIRContext *context = launch.getContext();
  Type f32 = Float32Type::get(context);
  Type i64 = IntegerType::get(context, 64);
  if (launch.getNumResults() != 1 || launch.getNumRegions() != 1 ||
      launch.getInputs().size() != 1)
    return reject(launch, "generated radius bootstrap expects one input, "
                          "one edge region, and one output");
  if (launch.getReducers().size() != 1 ||
      reducerKind(launch, launch.getReducers()[0]) != "sum" ||
      launch.getRegionKinds().size() != 1 ||
      launch.getRegionKinds().front() != 0 ||
      launch.getInputSegmentSizes().size() != 1 ||
      launch.getInputSegmentSizes().front() != 1)
    return reject(launch, "expected one generated distance edge segment with "
                          "a sum reducer");
  if (!isRankOneTensor(launch.getInputs()[0], f32) ||
      !isRankOneTensor(launch.getResult(0), f32))
    return reject(launch, "generated radius bootstrap requires scalar FP32 "
                          "source and output fields");
  auto positions = dyn_cast<RankedTensorType>(launch.getPositions().getType());
  if (!positions || positions.getRank() != 2 ||
      positions.getElementType() != f32 ||
      (positions.getDimSize(1) != ShapedType::kDynamic &&
       positions.getDimSize(1) != launch.getDimensionsAttr().getInt()))
    return reject(launch, "positions must be rank-two FP32 with the declared "
                          "dimension");
  for (Value value : {launch.getCellPtr(), launch.getParticleOrder(),
                      launch.getCellCoordinates(), launch.getExtents(),
                      launch.getStrides(), launch.getNeighborOffsets()}) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || type.getElementType() != i64)
      return reject(launch, "cell-directory indices must be i64 tensors");
  }
  for (Value value : {launch.getLattice(), launch.getInverseLattice()}) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || type.getRank() != 2 || type.getElementType() != f32 ||
        type.getDimSize(0) != launch.getDimensionsAttr().getInt() ||
        type.getDimSize(1) != launch.getDimensionsAttr().getInt())
      return reject(launch, "periodic lattice operands must be FP32 [D,D]");
  }
  auto offsets = dyn_cast<RankedTensorType>(
      launch.getNeighborOffsets().getType());
  if (!offsets || offsets.getRank() != 2 ||
      offsets.getDimSize(0) <= 0 ||
      offsets.getDimSize(1) != launch.getDimensionsAttr().getInt())
    return reject(launch, "neighbor offsets must have static shape [K,D]");
  if (!isScalarMultiplyRegion(launch.getRegions().front(), 2))
    return reject(launch, "edge region must multiply the scalar source by "
                          "the implicit Euclidean distance");
  emitGeneratedRadiusDistanceTTIR(
      output, launch.getNumRowsAttr().getInt(),
      launch.getDimensionsAttr().getInt(),
      offsets.getDimSize(0), launch.getCutoffAttr().getValueAsDouble(),
      launch.getPeriodic());
  return success();
}

static bool isScalarMultiplyRegion(Region &region, unsigned arguments);

static void emitFixedDegreeTTIR(llvm::raw_ostream &output, StringRef index,
                                int64_t numRows, int64_t degree,
                                int64_t blockM, int64_t blockD,
                                int64_t numWarps) {
  output << "// graphforge.launch entry=gf_csr_weighted_sum block_rows="
         << blockM << " num_warps=" << numWarps
         << " abi=row_ptr,col_idx,x,weight,out\n"
         << "module {\n"
         << "  tt.func public @gf_csr_weighted_sum("
         << "%row_ptr: !tt.ptr<" << index << ">, "
         << "%col_idx: !tt.ptr<" << index << ">, "
         << "%x: !tt.ptr<f32>, %weight: !tt.ptr<f32>, "
         << "%out: !tt.ptr<f32>) attributes {noinline = false} {\n"
         << "    %source_zero = arith.constant dense<0> : tensor<" << blockM
         << "x" << blockD << "x" << index << ">\n"
         << "    %value_zero = arith.constant dense<0.000000e+00> : tensor<"
         << blockM << "x" << blockD << "xf32>\n"
         << "    %nrows = arith.constant dense<" << numRows << "> : tensor<"
         << blockM << "xi32>\n"
         << "    %degree = arith.constant dense<" << degree << "> : tensor<"
         << blockM << "xi32>\n"
         << "    %degree2 = arith.constant dense<" << degree << "> : tensor<1x"
         << blockD << "xi32>\n"
         << "    %cbm = arith.constant " << blockM << " : i32\n"
         << "    %pid = tt.get_program_id x : i32\n"
         << "    %base = arith.muli %pid, %cbm : i32\n"
         << "    %range_m = tt.make_range {end = " << blockM
         << " : i32, start = 0 : i32} : tensor<" << blockM << "xi32>\n"
         << "    %base_v = tt.splat %base : i32 -> tensor<" << blockM
         << "xi32>\n"
         << "    %rows = arith.addi %base_v, %range_m : tensor<" << blockM
         << "xi32>\n"
         << "    %range_d = tt.make_range {end = " << blockD
         << " : i32, start = 0 : i32} : tensor<" << blockD << "xi32>\n"
         << "    %edge_base = arith.muli %rows, %degree : tensor<" << blockM
         << "xi32>\n"
         << "    %edge_base2 = tt.expand_dims %edge_base {axis = 1 : i32} : "
            "tensor<"
         << blockM << "xi32> -> tensor<" << blockM << "x1xi32>\n"
         << "    %neighbors = tt.expand_dims %range_d {axis = 0 : i32} : "
            "tensor<"
         << blockD << "xi32> -> tensor<1x" << blockD << "xi32>\n"
         << "    %edge_base_b = tt.broadcast %edge_base2 : tensor<" << blockM
         << "x1xi32> -> tensor<" << blockM << "x" << blockD << "xi32>\n"
         << "    %neighbors_b = tt.broadcast %neighbors : tensor<1x" << blockD
         << "xi32> -> tensor<" << blockM << "x" << blockD << "xi32>\n"
         << "    %edges = arith.addi %edge_base_b, %neighbors_b : tensor<"
         << blockM << "x" << blockD << "xi32>\n"
         << "    %row_mask = arith.cmpi slt, %rows, %nrows : tensor<" << blockM
         << "xi32>\n"
         << "    %degree_mask1 = arith.cmpi slt, %neighbors, %degree2 : "
            "tensor<1x"
         << blockD << "xi32>\n"
         << "    %row_mask2 = tt.expand_dims %row_mask {axis = 1 : i32} : "
            "tensor<"
         << blockM << "xi1> -> tensor<" << blockM << "x1xi1>\n"
         << "    %row_mask_b = tt.broadcast %row_mask2 : tensor<" << blockM
         << "x1xi1> -> tensor<" << blockM << "x" << blockD << "xi1>\n"
         << "    %degree_mask = tt.broadcast %degree_mask1 : tensor<1x"
         << blockD << "xi1> -> tensor<" << blockM << "x" << blockD
         << "xi1>\n"
         << "    %mask = arith.andi %row_mask_b, %degree_mask : tensor<"
         << blockM << "x" << blockD << "xi1>\n"
         << "    %col_base = tt.splat %col_idx : !tt.ptr<" << index
         << "> -> tensor<" << blockM << "x" << blockD << "x!tt.ptr<" << index
         << ">>\n"
         << "    %col_ptr = tt.addptr %col_base, %edges : tensor<" << blockM
         << "x" << blockD << "x!tt.ptr<" << index << ">>, tensor<" << blockM
         << "x" << blockD << "xi32>\n"
         << "    %src = tt.load %col_ptr, %mask, %source_zero : tensor<"
         << blockM << "x" << blockD << "x!tt.ptr<" << index << ">>\n"
         << "    %x_base = tt.splat %x : !tt.ptr<f32> -> tensor<" << blockM
         << "x" << blockD << "x!tt.ptr<f32>>\n"
         << "    %x_ptr = tt.addptr %x_base, %src : tensor<" << blockM << "x"
         << blockD << "x!tt.ptr<f32>>, tensor<" << blockM << "x" << blockD
         << "x" << index << ">\n"
         << "    %x_value = tt.load %x_ptr, %mask, %value_zero : tensor<"
         << blockM << "x" << blockD << "x!tt.ptr<f32>>\n"
         << "    %w_base = tt.splat %weight : !tt.ptr<f32> -> tensor<"
         << blockM << "x" << blockD << "x!tt.ptr<f32>>\n"
         << "    %w_ptr = tt.addptr %w_base, %edges : tensor<" << blockM << "x"
         << blockD << "x!tt.ptr<f32>>, tensor<" << blockM << "x" << blockD
         << "xi32>\n"
         << "    %w_value = tt.load %w_ptr, %mask, %value_zero : tensor<"
         << blockM << "x" << blockD << "x!tt.ptr<f32>>\n"
         << "    %message = arith.mulf %x_value, %w_value : tensor<" << blockM
         << "x" << blockD << "xf32>\n"
         << "    %sum = \"tt.reduce\"(%message) <{axis = 1 : i32}> ({\n"
         << "    ^bb0(%a: f32, %b: f32):\n"
         << "      %c = arith.addf %a, %b : f32\n"
         << "      tt.reduce.return %c : f32\n"
         << "    }) : (tensor<" << blockM << "x" << blockD
         << "xf32>) -> tensor<" << blockM << "xf32>\n"
         << "    %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<" << blockM
         << "x!tt.ptr<f32>>\n"
         << "    %out_ptr = tt.addptr %out_base, %rows : tensor<" << blockM
         << "x!tt.ptr<f32>>, tensor<" << blockM << "xi32>\n"
         << "    tt.store %out_ptr, %sum, %row_mask : tensor<" << blockM
         << "x!tt.ptr<f32>>\n"
         << "    tt.return\n"
         << "  }\n"
         << "}\n";
}

/// Fast path for a fused product of scalar multiply-and-sum applies on a
/// fixed-degree CSR relation.  Selection is based solely on typed region and
/// reducer algebra.  The launch batches rows and shares relation/source loads
/// across every result; duplicated projected operands remain in the stable
/// launch ABI but do not cause duplicate device loads.
static LogicalResult emitFixedDegreeProductMultiplyTTIR(
    kernel::LaunchOp launch, StringRef index, int64_t numRows, int64_t degree,
    int64_t blockM, int64_t blockD, int64_t numWarps,
    llvm::raw_ostream &o) {
  ArrayAttr roles = launch.getInputRolesAttr();
  ArrayAttr names = launch.getInputNamesAttr();
  auto segments = launch.getInputSegmentSizesAttr().asArrayRef();
  if (!roles || roles.size() != launch.getInputs().size() ||
      (names && names.size() != launch.getInputs().size()) ||
      segments.size() != launch.getNumResults() ||
      launch.getNumRegions() != launch.getNumResults() ||
      launch.getReducers().size() != launch.getNumResults())
    return failure();
  for (auto [component, size] : llvm::enumerate(segments)) {
    auto reference = dyn_cast<FlatSymbolRefAttr>(launch.getReducers()[component]);
    auto reducer = reference
        ? SymbolTable::lookupNearestSymbolFrom<ReducerOp>(launch, reference)
        : ReducerOp();
    if (size != 2 ||
        !isScalarMultiplyRegion(launch.getRegions()[component], 2) ||
        !reducer || !isZeroAdditiveState(reducer))
      return failure();
  }
  for (auto [input, roleAttribute] : llvm::zip(launch.getInputs(), roles)) {
    StringRef role = cast<StringAttr>(roleAttribute).getValue();
    if ((role != "src" && role != "edge") ||
        !isRankOneTensor(input, Float32Type::get(launch.getContext())))
      return failure();
  }

  SmallVector<std::string> arguments;
  llvm::StringMap<unsigned> occurrences;
  for (auto [position, attribute] : llvm::enumerate(names ? names : roles)) {
    std::string name = sanitizeIdentifier(
        cast<StringAttr>(attribute).getValue(), position);
    unsigned &occurrence = occurrences[name];
    if (occurrence++) name += "_" + std::to_string(occurrence);
    arguments.push_back(std::move(name));
  }
  std::string tile = "tensor<" + std::to_string(blockM) + "x" +
                     std::to_string(blockD);
  std::string tileI32 = tile + "xi32>";
  std::string tileIndex = tile + "x" + index.str() + ">";
  std::string tileF32 = tile + "xf32>";
  std::string tileI1 = tile + "xi1>";
  std::string rowsI32 = "tensor<" + std::to_string(blockM) + "xi32>";
  std::string rowsI1 = "tensor<" + std::to_string(blockM) + "xi1>";
  std::string rowsF32 = "tensor<" + std::to_string(blockM) + "xf32>";

  o << "// graphforge.launch entry=gf_csr_product_additive_tile block_rows="
    << blockM << " num_warps=" << numWarps
    << " abi=row_ptr,col_idx,";
  for (StringRef name : arguments) o << name << ",";
  for (unsigned result = 0; result < launch.getNumResults(); ++result)
    o << "out" << result
      << (result + 1 == launch.getNumResults() ? "" : ",");
  o << "\nmodule {\n  tt.func public @gf_csr_product_additive_tile("
    << "%row_ptr: !tt.ptr<" << index
    << "> {tt.divisibility = 16 : i32}, %col_idx: !tt.ptr<" << index
    << "> {tt.divisibility = 16 : i32}";
  for (auto [position, input] : llvm::enumerate(launch.getInputs()))
    o << ", %" << arguments[position]
      << ": !tt.ptr<f32> {tt.divisibility = 16 : i32}";
  for (unsigned result = 0; result < launch.getNumResults(); ++result)
    o << ", %out" << result
      << ": !tt.ptr<f32> {tt.divisibility = 16 : i32}";
  o << ") attributes {noinline = false} {\n"
    << "    %gf_source_zero = arith.constant dense<0> : " << tileIndex << "\n"
    << "    %gf_value_zero = arith.constant dense<0.000000e+00> : "
    << tileF32 << "\n"
    << "    %gf_nrows = arith.constant dense<" << numRows << "> : "
    << rowsI32 << "\n"
    << "    %gf_degree = arith.constant dense<" << degree << "> : "
    << rowsI32 << "\n"
    << "    %gf_degree2 = arith.constant dense<" << degree
    << "> : tensor<1x" << blockD << "xi32>\n"
    << "    %gf_cbm = arith.constant " << blockM << " : i32\n"
    << "    %gf_pid = tt.get_program_id x : i32\n"
    << "    %gf_base = arith.muli %gf_pid, %gf_cbm : i32\n"
    << "    %gf_range_m = tt.make_range {end = " << blockM
    << " : i32, start = 0 : i32} : " << rowsI32 << "\n"
    << "    %gf_base_v = tt.splat %gf_base : i32 -> " << rowsI32 << "\n"
    << "    %gf_rows = arith.addi %gf_base_v, %gf_range_m : " << rowsI32
    << "\n"
    << "    %gf_range_d = tt.make_range {end = " << blockD
    << " : i32, start = 0 : i32} : tensor<" << blockD << "xi32>\n"
    << "    %gf_edge_base = arith.muli %gf_rows, %gf_degree : " << rowsI32
    << "\n"
    << "    %gf_edge_base2 = tt.expand_dims %gf_edge_base {axis = 1 : i32} : "
    << rowsI32 << " -> tensor<" << blockM << "x1xi32>\n"
    << "    %gf_neighbors = tt.expand_dims %gf_range_d {axis = 0 : i32} : "
       "tensor<" << blockD << "xi32> -> tensor<1x" << blockD << "xi32>\n"
    << "    %gf_edge_base_b = tt.broadcast %gf_edge_base2 : tensor<" << blockM
    << "x1xi32> -> " << tileI32 << "\n"
    << "    %gf_neighbors_b = tt.broadcast %gf_neighbors : tensor<1x" << blockD
    << "xi32> -> " << tileI32 << "\n"
    << "    %gf_edges = arith.addi %gf_edge_base_b, %gf_neighbors_b : "
    << tileI32 << "\n"
    << "    %gf_row_mask = arith.cmpi slt, %gf_rows, %gf_nrows : " << rowsI32
    << "\n"
    << "    %gf_degree_mask1 = arith.cmpi slt, %gf_neighbors, %gf_degree2 : "
       "tensor<1x" << blockD << "xi32>\n"
    << "    %gf_row_mask2 = tt.expand_dims %gf_row_mask {axis = 1 : i32} : "
    << rowsI1 << " -> tensor<" << blockM << "x1xi1>\n"
    << "    %gf_row_mask_b = tt.broadcast %gf_row_mask2 : tensor<" << blockM
    << "x1xi1> -> " << tileI1 << "\n"
    << "    %gf_degree_mask = tt.broadcast %gf_degree_mask1 : tensor<1x"
    << blockD << "xi1> -> " << tileI1 << "\n"
    << "    %gf_mask = arith.andi %gf_row_mask_b, %gf_degree_mask : " << tileI1
    << "\n"
    << "    %gf_col_base = tt.splat %col_idx : !tt.ptr<" << index
    << "> -> tensor<" << blockM << "x" << blockD << "x!tt.ptr<" << index
    << ">>\n"
    << "    %gf_col_ptr = tt.addptr %gf_col_base, %gf_edges : tensor<"
    << blockM << "x" << blockD << "x!tt.ptr<" << index << ">>, " << tileI32
    << "\n"
    << "    %gf_src = tt.load %gf_col_ptr, %gf_mask, %gf_source_zero : tensor<"
    << blockM << "x" << blockD << "x!tt.ptr<" << index << ">>\n";

  SmallVector<std::string> values;
  for (unsigned position = 0; position < launch.getInputs().size(); ++position) {
    StringRef role = cast<StringAttr>(roles[position]).getValue();
    bool reused = false;
    for (unsigned previous = 0; previous < position; ++previous) {
      if (launch.getInputs()[previous] == launch.getInputs()[position] &&
          cast<StringAttr>(roles[previous]).getValue() == role) {
        values.push_back(values[previous]);
        reused = true;
        break;
      }
    }
    if (reused) continue;
    std::string suffix = std::to_string(position);
    std::string base = "%gf_input_base" + suffix;
    std::string pointer = "%gf_input_ptr" + suffix;
    std::string value = "%gf_input" + suffix;
    o << "    " << base << " = tt.splat %" << arguments[position]
      << " : !tt.ptr<f32> -> tensor<" << blockM << "x" << blockD
      << "x!tt.ptr<f32>>\n"
      << "    " << pointer << " = tt.addptr " << base << ", %gf_"
      << (role == "src" ? "src" : "edges") << " : tensor<" << blockM << "x"
      << blockD << "x!tt.ptr<f32>>, "
      << (role == "src" ? tileIndex : tileI32) << "\n"
      << "    " << value << " = tt.load " << pointer
      << ", %gf_mask, %gf_value_zero : tensor<" << blockM << "x" << blockD
      << "x!tt.ptr<f32>>\n";
    values.push_back(std::move(value));
  }

  int64_t offset = 0;
  for (unsigned component = 0; component < launch.getNumResults(); ++component) {
    o << "    %gf_message" << component << " = arith.mulf " << values[offset]
      << ", " << values[offset + 1] << " : " << tileF32 << "\n"
      << "    %gf_sum" << component << " = \"tt.reduce\"(%gf_message"
      << component << ") <{axis = 1 : i32}> ({\n"
      << "    ^bb0(%gf_a" << component << ": f32, %gf_b" << component
      << ": f32):\n"
      << "      %gf_combined" << component << " = arith.addf %gf_a"
      << component << ", %gf_b" << component << " : f32\n"
      << "      tt.reduce.return %gf_combined" << component << " : f32\n"
      << "    }) : (" << tileF32 << ") -> " << rowsF32 << "\n"
      << "    %gf_out_base" << component << " = tt.splat %out" << component
      << " : !tt.ptr<f32> -> tensor<" << blockM << "x!tt.ptr<f32>>\n"
      << "    %gf_out_ptr" << component << " = tt.addptr %gf_out_base"
      << component << ", %gf_rows : tensor<" << blockM
      << "x!tt.ptr<f32>>, " << rowsI32 << "\n"
      << "    tt.store %gf_out_ptr" << component << ", %gf_sum" << component
      << ", %gf_row_mask : tensor<" << blockM << "x!tt.ptr<f32>>\n";
    offset += 2;
  }
  o << "    tt.return\n  }\n}\n";
  return success();
}

static void emitFixedDegreeVectorTTIR(
    llvm::raw_ostream &o, StringRef index, int64_t numRows, int64_t degree,
    int64_t features, int64_t blockM, int64_t blockD, int64_t numWarps,
    bool ragged = false) {
  StringRef edgeOffsetType = ragged ? index : "i32";
  o << "// graphforge.launch entry=gf_csr_weighted_sum block_rows="
    << blockM << " num_warps=" << numWarps
    << " abi=row_ptr,col_idx,x,weight,out\n"
    << "module {\n"
    << "  tt.func public @gf_csr_weighted_sum("
    << "%row_ptr: !tt.ptr<" << index << ">, "
    << "%col_idx: !tt.ptr<" << index << "> {tt.divisibility = 16 : i32}, "
    << "%x: !tt.ptr<f32> {tt.divisibility = 16 : i32}, "
    << "%weight: !tt.ptr<f32> {tt.divisibility = 16 : i32}, "
    << "%out: !tt.ptr<f32> {tt.divisibility = 16 : i32}) "
       "attributes {noinline = false} {\n"
    << "    %source_zero = arith.constant dense<0> : tensor<" << blockM
    << "x" << blockD << "x" << index << ">\n"
    << "    %edge_zero = arith.constant dense<0.000000e+00> : tensor<"
    << blockM << "x" << blockD << "xf32>\n"
    << "    %value_zero = arith.constant dense<0.000000e+00> : tensor<"
    << blockM << "x" << blockD << "x" << features << "xf32>\n"
    << "    %nrows = arith.constant dense<" << numRows << "> : tensor<"
    << blockM << "x1xi32>\n"
    << "    %feature_stride = arith.constant dense<" << features
    << "> : tensor<" << blockM << "x" << blockD << "x1x" << index
    << ">\n"
    << "    %row_stride = arith.constant dense<" << features << "> : tensor<"
    << blockM << "x1xi32>\n"
    << "    %cbm = arith.constant " << blockM << " : i32\n"
    << "    %pid = tt.get_program_id x : i32\n"
    << "    %row_base = arith.muli %pid, %cbm : i32\n"
    << "    %range_m = tt.make_range {end = " << blockM
    << " : i32, start = 0 : i32} : tensor<" << blockM << "xi32>\n"
    << "    %row_base_v = tt.splat %row_base : i32 -> tensor<" << blockM
    << "xi32>\n"
    << "    %rows1 = arith.addi %row_base_v, %range_m : tensor<" << blockM
    << "xi32>\n"
    << "    %rows = tt.expand_dims %rows1 {axis = 1 : i32} : tensor<"
    << blockM << "xi32> -> tensor<" << blockM << "x1xi32>\n"
    << "    %range_d = tt.make_range {end = " << blockD
    << " : i32, start = 0 : i32} : tensor<" << blockD << "xi32>\n"
    << "    %neighbors = tt.expand_dims %range_d {axis = 0 : i32} : tensor<"
    << blockD << "xi32> -> tensor<1x" << blockD << "xi32>\n"
    << "    %range_f = tt.make_range {end = " << features
    << " : i32, start = 0 : i32} : tensor<" << features << "xi32>\n"
    << "    %features1 = tt.expand_dims %range_f {axis = 0 : i32} : tensor<"
    << features << "xi32> -> tensor<1x" << features << "xi32>\n"
    << "    %row_mask1 = arith.cmpi slt, %rows, %nrows : tensor<" << blockM
    << "x1xi32>\n";
  if (ragged) {
    o << "    %row_zero = arith.constant dense<0> : tensor<" << blockM
      << "x1x" << index << ">\n"
      << "    %one = arith.constant dense<1> : tensor<" << blockM
      << "x1xi32>\n"
      << "    %row_ptr_base = tt.splat %row_ptr : !tt.ptr<" << index
      << "> -> tensor<" << blockM << "x1x!tt.ptr<" << index << ">>\n"
      << "    %start_ptr = tt.addptr %row_ptr_base, %rows : tensor<" << blockM
      << "x1x!tt.ptr<" << index << ">>, tensor<" << blockM << "x1xi32>\n"
      << "    %starts = tt.load %start_ptr, %row_mask1, %row_zero : tensor<"
      << blockM << "x1x!tt.ptr<" << index << ">>\n"
      << "    %end_ptr = tt.addptr %start_ptr, %one : tensor<" << blockM
      << "x1x!tt.ptr<" << index << ">>, tensor<" << blockM << "x1xi32>\n"
      << "    %ends = tt.load %end_ptr, %row_mask1, %row_zero : tensor<"
      << blockM << "x1x!tt.ptr<" << index << ">>\n";
    if (index == "i64")
      o << "    %neighbors_index = arith.extsi %neighbors : tensor<1x"
        << blockD << "xi32> to tensor<1x" << blockD << "xi64>\n";
    o << "    %starts_b = tt.broadcast %starts : tensor<" << blockM << "x1x"
      << index << "> -> tensor<" << blockM << "x" << blockD << "x" << index
      << ">\n"
      << "    %neighbor_b = tt.broadcast %neighbors"
      << (index == "i64" ? "_index" : "") << " : tensor<1x" << blockD << "x"
      << index << "> -> tensor<" << blockM << "x" << blockD << "x" << index
      << ">\n"
      << "    %edges = arith.addi %starts_b, %neighbor_b : tensor<" << blockM
      << "x" << blockD << "x" << index << ">\n"
      << "    %ends_b = tt.broadcast %ends : tensor<" << blockM << "x1x"
      << index << "> -> tensor<" << blockM << "x" << blockD << "x" << index
      << ">\n"
      << "    %edge_limit = arith.cmpi slt, %edges, %ends_b : tensor<"
      << blockM << "x" << blockD << "x" << index << ">\n";
  } else {
    o << "    %degree_row = arith.constant dense<" << degree << "> : tensor<"
      << blockM << "x1xi32>\n"
      << "    %degree_limit = arith.constant dense<" << degree
      << "> : tensor<1x" << blockD << "xi32>\n"
      << "    %edge_row = arith.muli %rows, %degree_row : tensor<" << blockM
      << "x1xi32>\n"
      << "    %edge_row_b = tt.broadcast %edge_row : tensor<" << blockM
      << "x1xi32> -> tensor<" << blockM << "x" << blockD << "xi32>\n"
      << "    %neighbor_b = tt.broadcast %neighbors : tensor<1x" << blockD
      << "xi32> -> tensor<" << blockM << "x" << blockD << "xi32>\n"
      << "    %edges = arith.addi %edge_row_b, %neighbor_b : tensor<" << blockM
      << "x" << blockD << "xi32>\n"
      << "    %degree_mask1 = arith.cmpi slt, %neighbors, %degree_limit : tensor<1x"
      << blockD << "xi32>\n"
      << "    %degree_mask = tt.broadcast %degree_mask1 : tensor<1x"
      << blockD << "xi1> -> tensor<" << blockM << "x" << blockD
      << "xi1>\n";
  }
  o
    << "    %row_mask = tt.broadcast %row_mask1 : tensor<" << blockM
    << "x1xi1> -> tensor<" << blockM << "x" << blockD << "xi1>\n"
    << "    %edge_mask = arith.andi %row_mask, %"
    << (ragged ? "edge_limit" : "degree_mask") << " : tensor<"
    << blockM << "x" << blockD << "xi1>\n"
    << "    %col_base = tt.splat %col_idx : !tt.ptr<" << index
    << "> -> tensor<" << blockM << "x" << blockD << "x!tt.ptr<" << index
    << ">>\n"
    << "    %col_ptr = tt.addptr %col_base, %edges : tensor<" << blockM
    << "x" << blockD << "x!tt.ptr<" << index << ">>, tensor<" << blockM
    << "x" << blockD << "x" << edgeOffsetType << ">\n"
    << "    %src = tt.load %col_ptr, %edge_mask, %source_zero : tensor<"
    << blockM << "x" << blockD << "x!tt.ptr<" << index << ">>\n"
    << "    %weight_base = tt.splat %weight : !tt.ptr<f32> -> tensor<"
    << blockM << "x" << blockD << "x!tt.ptr<f32>>\n"
    << "    %weight_ptr = tt.addptr %weight_base, %edges : tensor<" << blockM
    << "x" << blockD << "x!tt.ptr<f32>>, tensor<" << blockM << "x"
    << blockD << "x" << edgeOffsetType << ">\n"
    << "    %weight_value = tt.load %weight_ptr, %edge_mask, %edge_zero : tensor<"
    << blockM << "x" << blockD << "x!tt.ptr<f32>>\n"
    << "    %src3 = tt.expand_dims %src {axis = 2 : i32} : tensor<" << blockM
    << "x" << blockD << "x" << index << "> -> tensor<" << blockM << "x"
    << blockD << "x1x" << index << ">\n"
    << "    %src_offset = arith.muli %src3, %feature_stride : tensor<"
    << blockM << "x" << blockD << "x1x" << index << ">\n"
    << "    %features2 = tt.expand_dims %features1 {axis = 1 : i32} : tensor<1x"
    << features << "xi32> -> tensor<1x1x" << features << "xi32>\n";
  if (index == "i64")
    o << "    %features_index = arith.extsi %features2 : tensor<1x1x"
      << features << "xi32> to tensor<1x1x" << features << "xi64>\n";
  o << "    %src_offset_b = tt.broadcast %src_offset : tensor<" << blockM
    << "x" << blockD << "x1x" << index << "> -> tensor<" << blockM << "x"
    << blockD << "x" << features << "x" << index << ">\n"
    << "    %features_b = tt.broadcast %features"
    << (index == "i64" ? "_index" : "2") << " : tensor<1x1x" << features
    << "x" << index << "> -> tensor<" << blockM << "x" << blockD << "x"
    << features << "x" << index << ">\n"
    << "    %value_offset = arith.addi %src_offset_b, %features_b : tensor<"
    << blockM << "x" << blockD << "x" << features << "x" << index << ">\n"
    << "    %edge_mask3 = tt.expand_dims %edge_mask {axis = 2 : i32} : tensor<"
    << blockM << "x" << blockD << "xi1> -> tensor<" << blockM << "x"
    << blockD << "x1xi1>\n"
    << "    %value_mask = tt.broadcast %edge_mask3 : tensor<" << blockM << "x"
    << blockD << "x1xi1> -> tensor<" << blockM << "x" << blockD << "x"
    << features << "xi1>\n"
    << "    %x_base = tt.splat %x : !tt.ptr<f32> -> tensor<" << blockM << "x"
    << blockD << "x" << features << "x!tt.ptr<f32>>\n"
    << "    %x_ptr = tt.addptr %x_base, %value_offset : tensor<" << blockM
    << "x" << blockD << "x" << features << "x!tt.ptr<f32>>, tensor<"
    << blockM << "x" << blockD << "x" << features << "x" << index << ">\n"
    << "    %x_value = tt.load %x_ptr, %value_mask, %value_zero : tensor<"
    << blockM << "x" << blockD << "x" << features << "x!tt.ptr<f32>>\n"
    << "    %weight3 = tt.expand_dims %weight_value {axis = 2 : i32} : tensor<"
    << blockM << "x" << blockD << "xf32> -> tensor<" << blockM << "x"
    << blockD << "x1xf32>\n"
    << "    %weight_b = tt.broadcast %weight3 : tensor<" << blockM << "x"
    << blockD << "x1xf32> -> tensor<" << blockM << "x" << blockD << "x"
    << features << "xf32>\n"
    << "    %message = arith.mulf %x_value, %weight_b : tensor<" << blockM
    << "x" << blockD << "x" << features << "xf32>\n"
    << "    %sum = \"tt.reduce\"(%message) <{axis = 1 : i32}> ({\n"
    << "    ^bb0(%a: f32, %b: f32):\n"
    << "      %c = arith.addf %a, %b : f32\n"
    << "      tt.reduce.return %c : f32\n"
    << "    }) : (tensor<" << blockM << "x" << blockD << "x" << features
    << "xf32>) -> tensor<" << blockM << "x" << features << "xf32>\n"
    << "    %out_row = arith.muli %rows, %row_stride : tensor<" << blockM
    << "x1xi32>\n"
    << "    %out_row_b = tt.broadcast %out_row : tensor<" << blockM
    << "x1xi32> -> tensor<" << blockM << "x" << features << "xi32>\n"
    << "    %out_feature_b = tt.broadcast %features1 : tensor<1x" << features
    << "xi32> -> tensor<" << blockM << "x" << features << "xi32>\n"
    << "    %out_offset = arith.addi %out_row_b, %out_feature_b : tensor<"
    << blockM << "x" << features << "xi32>\n"
    << "    %out_mask = tt.broadcast %row_mask1 : tensor<" << blockM
    << "x1xi1> -> tensor<" << blockM << "x" << features << "xi1>\n"
    << "    %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<" << blockM
    << "x" << features << "x!tt.ptr<f32>>\n"
    << "    %out_ptr = tt.addptr %out_base, %out_offset : tensor<" << blockM
    << "x" << features << "x!tt.ptr<f32>>, tensor<" << blockM << "x"
    << features << "xi32>\n"
    << "    tt.store %out_ptr, %sum, %out_mask : tensor<" << blockM << "x"
    << features << "x!tt.ptr<f32>>\n"
    << "    tt.return\n"
    << "  }\n"
    << "}\n";
}

static void emitDenseStreamingTTIR(llvm::raw_ostream &o, int64_t rows,
                                   int64_t lanes, int64_t sourceLanes,
                                   int64_t width,
                                   int64_t blockM, int64_t blockN,
                                   int64_t numWarps,
                                   double blockPruneThreshold,
                                   StringRef boundary) {
  bool approximate = blockPruneThreshold > 0.0;
  bool lowerInclusive = boundary == "lower_inclusive";
  int64_t laneStride = rows * width;
  int64_t laneGroup = lanes / sourceLanes;
  o << "// graphforge.launch entry=gf_dense_streaming_reduce block_rows="
    << blockM << " num_warps=" << numWarps
    << " abi=lhs,rhs,payload,scale,out\n"
    << "// graphforge.reducer block_prune=";
  if (approximate)
    o << llvm::formatv("{0:F8}", blockPruneThreshold);
  else
    o << "none";
  o << " boundary=" << boundary << "\nmodule {\n"
    << "  tt.func public @gf_dense_streaming_reduce("
    << "%lhs: !tt.ptr<f16> {tt.divisibility = 16 : i32}, "
    << "%rhs: !tt.ptr<f16> {tt.divisibility = 16 : i32}, "
    << "%payload: !tt.ptr<f16> {tt.divisibility = 16 : i32}, "
    << "%scale: f32, %out: !tt.ptr<f16> {tt.divisibility = 16 : i32}) "
       "attributes {noinline = false} {\n"
    << "    %zero_q = arith.constant dense<0.000000e+00> : tensor<"
    << blockM << "x" << width << "xf16>\n"
    << "    %zero_k = arith.constant dense<0.000000e+00> : tensor<"
    << blockN << "x" << width << "xf16>\n"
    << "    %zero_acc = arith.constant dense<0.000000e+00> : tensor<"
    << blockM << "x" << width << "xf32>\n"
    << "    %zero_scores = arith.constant dense<0.000000e+00> : tensor<"
    << blockM << "x" << blockN << "xf32>\n"
    << "    %zero_den = arith.constant dense<0.000000e+00> : tensor<"
    << blockM << "xf32>\n"
    << "    %neg_scores = arith.constant dense<0xFF800000> : tensor<"
    << blockM << "x" << blockN << "xf32>\n"
    << "    %neg_max = arith.constant dense<0xFF800000> : tensor<"
    << blockM << "xf32>\n"
    << "    %neg_block_max = arith.constant 0xFF800000 : f32\n"
    << "    %rows_limit = arith.constant dense<" << rows << "> : tensor<"
    << blockM << "xi32>\n"
    << "    %sources_limit = arith.constant dense<" << rows
    << "> : tensor<" << blockN << "xi32>\n"
    << "    %width_m = arith.constant dense<" << width << "> : tensor<"
    << blockM << "x1xi32>\n"
    << "    %width_n = arith.constant dense<" << width << "> : tensor<"
    << blockN << "x1xi32>\n"
    << "    %lane_stride = arith.constant " << laneStride << " : i32\n"
    << "    %block_m = arith.constant " << blockM << " : i32\n"
    << "    %lane_group = arith.constant " << laneGroup << " : i32\n"
    << "    %c0 = arith.constant 0 : i32\n"
    << "    %cend = arith.constant " << rows << " : i32\n"
    << "    %cstep = arith.constant " << blockN << " : i32\n"
    << "    %log2e = arith.constant 1.44269502 : f32\n"
    << "    %pid_m = tt.get_program_id x : i32\n"
    << "    %pid_lane = tt.get_program_id y : i32\n"
    << "    %row_start = arith.muli %pid_m, %block_m : i32\n"
    << "    %row_range = tt.make_range {end = " << blockM
    << " : i32, start = 0 : i32} : tensor<" << blockM << "xi32>\n"
    << "    %row_start_v = tt.splat %row_start : i32 -> tensor<" << blockM
    << "xi32>\n"
    << "    %row = arith.addi %row_start_v, %row_range : tensor<" << blockM
    << "xi32>\n";
  if (rows % blockM == 0)
    o << "    %row_mask = arith.constant dense<true> : tensor<" << blockM
      << "xi1>\n";
  else
    o << "    %row_mask = arith.cmpi slt, %row, %rows_limit : tensor<"
      << blockM << "xi32>\n";
  o
    << "    %feature = tt.make_range {end = " << width
    << " : i32, start = 0 : i32} : tensor<" << width << "xi32>\n"
    << "    %feature_2d = tt.expand_dims %feature {axis = 0 : i32} : tensor<"
    << width << "xi32> -> tensor<1x" << width << "xi32>\n"
    << "    %lane_offset = arith.muli %pid_lane, %lane_stride : i32\n"
    << "    %pid_source_lane = arith.divui %pid_lane, %lane_group : i32\n"
    << "    %source_lane_offset = arith.muli %pid_source_lane, %lane_stride : i32\n"
    << "    %lhs_lane = tt.addptr %lhs, %lane_offset : !tt.ptr<f16>, i32\n"
    << "    %row_2d = tt.expand_dims %row {axis = 1 : i32} : tensor<"
    << blockM << "xi32> -> tensor<" << blockM << "x1xi32>\n"
    << "    %row_offset = arith.muli %row_2d, %width_m : tensor<" << blockM
    << "x1xi32>\n"
    << "    %lhs_base = tt.splat %lhs_lane : !tt.ptr<f16> -> tensor<"
    << blockM << "x1x!tt.ptr<f16>>\n"
    << "    %lhs_row = tt.addptr %lhs_base, %row_offset : tensor<" << blockM
    << "x1x!tt.ptr<f16>>, tensor<" << blockM << "x1xi32>\n"
    << "    %lhs_row_b = tt.broadcast %lhs_row : tensor<" << blockM
    << "x1x!tt.ptr<f16>> -> tensor<" << blockM << "x" << width
    << "x!tt.ptr<f16>>\n"
    << "    %feature_bm = tt.broadcast %feature_2d : tensor<1x" << width
    << "xi32> -> tensor<" << blockM << "x" << width << "xi32>\n"
    << "    %lhs_ptr = tt.addptr %lhs_row_b, %feature_bm : tensor<" << blockM
    << "x" << width << "x!tt.ptr<f16>>, tensor<" << blockM << "x" << width
    << "xi32>\n"
    << "    %row_mask_2d = tt.expand_dims %row_mask {axis = 1 : i32} : tensor<"
    << blockM << "xi1> -> tensor<" << blockM << "x1xi1>\n"
    << "    %q_mask = tt.broadcast %row_mask_2d : tensor<" << blockM
    << "x1xi1> -> tensor<" << blockM << "x" << width << "xi1>\n"
    << "    %q = tt.load %lhs_ptr, %q_mask, %zero_q : tensor<" << blockM
    << "x" << width << "x!tt.ptr<f16>>\n"
    << "    %log2_scale = arith.mulf %scale, %log2e : f32\n";
  if (lowerInclusive) {
    o << "    %causal_unclamped = arith.addi %row_start, %block_m : i32\n"
      << "    %causal_end = arith.minsi %causal_unclamped, %cend : i32\n";
  }
  o << "    %state:" << (approximate ? 4 : 3)
    << " = scf.for %start = %c0 to "
    << (lowerInclusive ? "%causal_end" : "%cend")
    << " step %cstep iter_args(";
  if (approximate)
    o << "%old_block_m = %neg_block_max, ";
  o << "%old_m = %neg_max, %old_l = %zero_den, %old_o = %zero_acc) -> (";
  if (approximate)
    o << "f32, ";
  o << "tensor<" << blockM << "xf32>, tensor<" << blockM
    << "xf32>, tensor<" << blockM << "x" << width << "xf32>) : i32 {\n"
    << "      %source_range = tt.make_range {end = " << blockN
    << " : i32, start = 0 : i32} : tensor<" << blockN << "xi32>\n"
    << "      %start_v = tt.splat %start : i32 -> tensor<" << blockN
    << "xi32>\n"
    << "      %source = arith.addi %start_v, %source_range : tensor<" << blockN
    << "xi32>\n";
  if (rows % blockN == 0)
    o << "      %source_mask = arith.constant dense<true> : tensor<" << blockN
      << "xi1>\n";
  else
    o << "      %source_mask = arith.cmpi slt, %source, %sources_limit : tensor<"
      << blockN << "xi32>\n";
  o
    << "      %source_2d = tt.expand_dims %source {axis = 1 : i32} : tensor<"
    << blockN << "xi32> -> tensor<" << blockN << "x1xi32>\n"
    << "      %source_offset = arith.muli %source_2d, %width_n : tensor<"
    << blockN << "x1xi32>\n"
    << "      %rhs_lane = tt.addptr %rhs, %source_lane_offset : !tt.ptr<f16>, i32\n"
    << "      %rhs_base = tt.splat %rhs_lane : !tt.ptr<f16> -> tensor<"
    << blockN << "x1x!tt.ptr<f16>>\n"
    << "      %rhs_row = tt.addptr %rhs_base, %source_offset : tensor<"
    << blockN << "x1x!tt.ptr<f16>>, tensor<" << blockN << "x1xi32>\n"
    << "      %rhs_row_b = tt.broadcast %rhs_row : tensor<" << blockN
    << "x1x!tt.ptr<f16>> -> tensor<" << blockN << "x" << width
    << "x!tt.ptr<f16>>\n"
    << "      %feature_bn = tt.broadcast %feature_2d : tensor<1x" << width
    << "xi32> -> tensor<" << blockN << "x" << width << "xi32>\n"
    << "      %rhs_ptr = tt.addptr %rhs_row_b, %feature_bn : tensor<" << blockN
    << "x" << width << "x!tt.ptr<f16>>, tensor<" << blockN << "x" << width
    << "xi32>\n"
    << "      %source_mask_2d = tt.expand_dims %source_mask {axis = 1 : i32} "
       ": tensor<"
    << blockN << "xi1> -> tensor<" << blockN << "x1xi1>\n"
    << "      %kv_mask = tt.broadcast %source_mask_2d : tensor<" << blockN
    << "x1xi1> -> tensor<" << blockN << "x" << width << "xi1>\n"
    << "      %k = tt.load %rhs_ptr, %kv_mask, %zero_k : tensor<" << blockN
    << "x" << width << "x!tt.ptr<f16>>\n"
    << "      %kt = tt.trans %k {order = array<i32: 1, 0>} : tensor<" << blockN
    << "x" << width << "xf16> -> tensor<" << width << "x" << blockN
    << "xf16>\n"
    << "      %score0 = tt.dot %q, %kt, %zero_scores, inputPrecision = tf32 : "
       "tensor<"
    << blockM << "x" << width << "xf16> * tensor<" << width << "x" << blockN
    << "xf16> -> tensor<" << blockM << "x" << blockN << "xf32>\n"
    << "      %scale_v = tt.splat %log2_scale : f32 -> tensor<" << blockM
    << "x" << blockN << "xf32>\n"
    << "      %score1 = arith.mulf %score0, %scale_v : tensor<" << blockM
    << "x" << blockN << "xf32>\n"
    << "      %source_mask_row = tt.expand_dims %source_mask {axis = 0 : i32} "
       ": tensor<"
    << blockN << "xi1> -> tensor<1x" << blockN << "xi1>\n"
    << "      %source_mask_b = tt.broadcast %source_mask_row : tensor<1x"
    << blockN << "xi1> -> tensor<" << blockM << "x" << blockN << "xi1>\n";
  if (lowerInclusive) {
    o << "      %source_coord_row = tt.expand_dims %source {axis = 0 : i32} : "
      << "tensor<" << blockN << "xi32> -> tensor<1x" << blockN << "xi32>\n"
      << "      %source_coord_b = tt.broadcast %source_coord_row : tensor<1x"
      << blockN << "xi32> -> tensor<" << blockM << "x" << blockN << "xi32>\n"
      << "      %row_coord_b = tt.broadcast %row_2d : tensor<" << blockM
      << "x1xi32> -> tensor<" << blockM << "x" << blockN << "xi32>\n"
      << "      %boundary_mask = arith.cmpi sle, %source_coord_b, %row_coord_b "
      << ": tensor<" << blockM << "x" << blockN << "xi32>\n"
      << "      %score_mask = arith.andi %source_mask_b, %boundary_mask : "
      << "tensor<" << blockM << "x" << blockN << "xi1>\n";
  } else {
    o << "      %score_mask = arith.andi %source_mask_b, %source_mask_b : "
      << "tensor<" << blockM << "x" << blockN << "xi1>\n";
  }
  o << "      %score = arith.select %score_mask, %score1, %neg_scores : tensor<"
    << blockM << "x" << blockN << "xi1>, tensor<" << blockM << "x" << blockN
    << "xf32>\n"
    << "      %tile_m = \"tt.reduce\"(%score) <{axis = 1 : i32}> ({\n"
    << "      ^bb0(%a: f32, %b: f32):\n"
    << "        %m = arith.maxnumf %a, %b : f32\n"
    << "        tt.reduce.return %m : f32\n"
    << "      }) : (tensor<" << blockM << "x" << blockN
    << "xf32>) -> tensor<" << blockM << "xf32>\n";
  if (approximate) {
    o << "      %tile_block_m = \"tt.reduce\"(%tile_m) <{axis = 0 : i32}> ({\n"
      << "      ^bb0(%a: f32, %b: f32):\n"
      << "        %m = arith.maxnumf %a, %b : f32\n"
      << "        tt.reduce.return %m : f32\n"
      << "      }) : (tensor<" << blockM << "xf32>) -> f32\n"
      << "      %block_diff = arith.subf %tile_block_m, %old_block_m : f32\n"
      << "      %prune_log2 = arith.constant "
      << llvm::formatv("{0:E8}", std::log2(blockPruneThreshold))
      << " : f32\n"
      << "      %skip = arith.cmpf olt, %block_diff, %prune_log2 : f32\n"
      << "      %accepted:4 = scf.if %skip -> (f32, tensor<" << blockM
      << "xf32>, tensor<" << blockM << "xf32>, tensor<" << blockM << "x"
      << width << "xf32>) {\n"
      << "        scf.yield %old_block_m, %old_m, %old_l, %old_o : f32, "
      << "tensor<" << blockM << "xf32>, tensor<" << blockM
      << "xf32>, tensor<" << blockM << "x" << width << "xf32>\n"
      << "      } else {\n"
      << "      %new_block_m = arith.maxnumf %old_block_m, %tile_block_m : f32\n";
  }
  o << "      %new_m = arith.maxnumf %old_m, %tile_m : tensor<" << blockM
    << "xf32>\n"
    << "      %new_m_2d = tt.expand_dims %new_m {axis = 1 : i32} : tensor<"
    << blockM << "xf32> -> tensor<" << blockM << "x1xf32>\n"
    << "      %new_m_b = tt.broadcast %new_m_2d : tensor<" << blockM
    << "x1xf32> -> tensor<" << blockM << "x" << blockN << "xf32>\n"
    << "      %shifted = arith.subf %score, %new_m_b : tensor<" << blockM
    << "x" << blockN << "xf32>\n"
    << "      %weight = math.exp2 %shifted : tensor<" << blockM << "x"
    << blockN << "xf32>\n"
    << "      %delta_m = arith.subf %old_m, %new_m : tensor<" << blockM
    << "xf32>\n"
    << "      %correction = math.exp2 %delta_m : tensor<" << blockM
    << "xf32>\n"
    << "      %old_l_scaled = arith.mulf %old_l, %correction : tensor<"
    << blockM << "xf32>\n"
    << "      %tile_l = \"tt.reduce\"(%weight) <{axis = 1 : i32}> ({\n"
    << "      ^bb0(%a: f32, %b: f32):\n"
    << "        %s = arith.addf %a, %b : f32\n"
    << "        tt.reduce.return %s : f32\n"
    << "      }) : (tensor<" << blockM << "x" << blockN
    << "xf32>) -> tensor<" << blockM << "xf32>\n"
    << "      %new_l = arith.addf %old_l_scaled, %tile_l : tensor<" << blockM
    << "xf32>\n"
    << "      %payload_lane = tt.addptr %payload, %source_lane_offset : !tt.ptr<f16>, "
       "i32\n"
    << "      %payload_base = tt.splat %payload_lane : !tt.ptr<f16> -> tensor<"
    << blockN << "x1x!tt.ptr<f16>>\n"
    << "      %payload_row = tt.addptr %payload_base, %source_offset : tensor<"
    << blockN << "x1x!tt.ptr<f16>>, tensor<" << blockN << "x1xi32>\n"
    << "      %payload_row_b = tt.broadcast %payload_row : tensor<" << blockN
    << "x1x!tt.ptr<f16>> -> tensor<" << blockN << "x" << width
    << "x!tt.ptr<f16>>\n"
    << "      %payload_ptr = tt.addptr %payload_row_b, %feature_bn : tensor<"
    << blockN << "x" << width << "x!tt.ptr<f16>>, tensor<" << blockN << "x"
    << width << "xi32>\n"
    << "      %payload_tile = tt.load %payload_ptr, %kv_mask, %zero_k : tensor<"
    << blockN << "x" << width << "x!tt.ptr<f16>>\n"
    << "      %correction_2d = tt.expand_dims %correction {axis = 1 : i32} : "
       "tensor<"
    << blockM << "xf32> -> tensor<" << blockM << "x1xf32>\n"
    << "      %correction_b = tt.broadcast %correction_2d : tensor<" << blockM
    << "x1xf32> -> tensor<" << blockM << "x" << width << "xf32>\n"
    << "      %old_o_scaled = arith.mulf %old_o, %correction_b : tensor<"
    << blockM << "x" << width << "xf32>\n"
    << "      %weight_f16 = arith.truncf %weight : tensor<" << blockM << "x"
    << blockN << "xf32> to tensor<" << blockM << "x" << blockN << "xf16>\n"
    << "      %new_o = tt.dot %weight_f16, %payload_tile, %old_o_scaled, "
       "inputPrecision = tf32 : tensor<"
    << blockM << "x" << blockN << "xf16> * tensor<" << blockN << "x" << width
    << "xf16> -> tensor<" << blockM << "x" << width << "xf32>\n";
  if (approximate) {
    o << "        scf.yield %new_block_m, %new_m, %new_l, %new_o : f32, "
      << "tensor<" << blockM << "xf32>, tensor<" << blockM
      << "xf32>, tensor<" << blockM << "x" << width << "xf32>\n"
      << "      }\n"
      << "      scf.yield %accepted#0, %accepted#1, %accepted#2, %accepted#3 "
      << ": f32, tensor<" << blockM << "xf32>, tensor<" << blockM
      << "xf32>, tensor<" << blockM << "x" << width << "xf32>\n";
  } else {
    o << "      scf.yield %new_m, %new_l, %new_o : tensor<" << blockM
      << "xf32>, tensor<" << blockM << "xf32>, tensor<" << blockM << "x"
      << width << "xf32>\n";
  }
  int denominatorState = approximate ? 2 : 1;
  int numeratorState = approximate ? 3 : 2;
  o << "    }\n"
    << "    %den_2d = tt.expand_dims %state#" << denominatorState
    << " {axis = 1 : i32} : tensor<"
    << blockM << "xf32> -> tensor<" << blockM << "x1xf32>\n"
    << "    %den_b = tt.broadcast %den_2d : tensor<" << blockM
    << "x1xf32> -> tensor<" << blockM << "x" << width << "xf32>\n"
    << "    %result = arith.divf %state#" << numeratorState
    << ", %den_b : tensor<" << blockM << "x"
    << width << "xf32>\n"
    << "    %result_f16 = arith.truncf %result : tensor<" << blockM << "x"
    << width << "xf32> to tensor<" << blockM << "x" << width << "xf16>\n"
    << "    %out_lane = tt.addptr %out, %lane_offset : !tt.ptr<f16>, i32\n"
    << "    %out_base = tt.splat %out_lane : !tt.ptr<f16> -> tensor<" << blockM
    << "x1x!tt.ptr<f16>>\n"
    << "    %out_row = tt.addptr %out_base, %row_offset : tensor<" << blockM
    << "x1x!tt.ptr<f16>>, tensor<" << blockM << "x1xi32>\n"
    << "    %out_row_b = tt.broadcast %out_row : tensor<" << blockM
    << "x1x!tt.ptr<f16>> -> tensor<" << blockM << "x" << width
    << "x!tt.ptr<f16>>\n"
    << "    %out_ptr = tt.addptr %out_row_b, %feature_bm : tensor<" << blockM
    << "x" << width << "x!tt.ptr<f16>>, tensor<" << blockM << "x" << width
    << "xi32>\n"
    << "    tt.store %out_ptr, %result_f16, %q_mask : tensor<" << blockM << "x"
    << width << "x!tt.ptr<f16>>\n"
    << "    tt.return\n"
    << "  }\n"
    << "}\n";
  (void)lanes;
}

static LogicalResult translateDenseScalarAlgebra(
    kernel::DenseLaunchOp launch, ReducerOp reducer,
    llvm::raw_ostream &output) {
  MLIRContext *context = launch.getContext();
  Type f32 = Float32Type::get(context);
  auto allF32 = [&](ArrayAttr attributes) {
    return llvm::all_of(attributes, [&](Attribute attribute) {
      auto type = dyn_cast<TypeAttr>(attribute);
      return type && type.getValue() == f32;
    });
  };
  bool hasNode = launch.getNumRegions() == 2;
  if (launch.getReducers().size() != 1 || launch.getNumResults() != 1 ||
      (launch.getNumRegions() != 1 && !hasNode) ||
      launch.getRegionKinds().size() != launch.getNumRegions() ||
      launch.getRegionKinds().front() != 0 ||
      (hasNode && launch.getRegionKinds()[1] != 1))
    return reject(launch, "generic dense algebra requires one edge region, "
                          "an optional node region, one reducer, and one output");
  if (!allF32(reducer.getMessageTypes()) ||
      !allF32(reducer.getStateTypes()) ||
      !allF32(reducer.getResultTypes()) ||
      reducer.getResultTypes().size() != 1)
    return reject(launch, "generic dense algebra currently requires scalar "
                          "FP32 message, state, and result types");
  if (!isRankOneTensor(launch.getResult(0), f32))
    return reject(launch, "generic dense algebra output must be tensor<?xf32>");
  ArrayAttr roles = launch.getInputRolesAttr();
  ArrayAttr names = launch.getInputNamesAttr();
  if (!roles || roles.size() != launch.getInputs().size())
    return reject(launch, "generic dense algebra requires input_roles");
  if (names && names.size() != launch.getInputs().size())
    return reject(launch, "generic dense algebra input_names are malformed");
  DenseI64ArrayAttr nodeIndices = launch.getNodeInputIndicesAttr();
  DenseI64ArrayAttr nodeSegments = launch.getNodeInputSegmentSizesAttr();
  if (hasNode && (!nodeIndices || !nodeSegments || nodeSegments.size() != 1 ||
                  nodeSegments.asArrayRef().front() !=
                      static_cast<int64_t>(nodeIndices.size())))
    return reject(launch, "generic dense node requires its explicit indexed "
                          "input ABI");

  SmallVector<std::string> argumentNames;
  llvm::StringMap<unsigned> occurrences;
  for (auto [index, attribute] : llvm::enumerate(names ? names : roles)) {
    StringRef raw = cast<StringAttr>(attribute).getValue();
    std::string name = sanitizeIdentifier(raw, index);
    unsigned &occurrence = occurrences[name];
    if (occurrence++) name += "_" + std::to_string(occurrence);
    argumentNames.push_back(std::move(name));
  }

  for (auto [input, roleAttribute] :
       llvm::zip(launch.getInputs(), roles)) {
    StringRef role = cast<StringAttr>(roleAttribute).getValue();
    if (role == "param") {
      if (input.getType() != f32)
        return reject(launch, "generic dense scalar parameters must be f32");
    } else {
      if (role == "edge")
        return reject(launch, "implicit dense relations have no materialized "
                              "per-edge field in the generic provider path");
      if (role != "src" && role != "dst")
        return reject(launch, "generic dense field role must be src or dst");
      if (!isRankOneTensor(input, f32))
        return reject(launch, "generic dense fields must be tensor<?xf32>");
    }
  }

  Block &edge = launch.getRegions().front().front();
  if (edge.getNumArguments() != launch.getInputs().size())
    return reject(launch, "edge region arguments do not match dense inputs");
  if (cast<kernel::YieldOp>(edge.getTerminator()).getValues().size() !=
      reducer.getMessageTypes().size())
    return reject(launch, "edge message arity does not match reducer ABI");

  output << "// graphforge.launch entry=gf_dense_scalar_reduce block_rows=1 "
            "num_warps=1 abi=";
  for (StringRef name : argumentNames) output << name << ",";
  output << "out\nmodule {\n"
         << "  tt.func public @gf_dense_scalar_reduce(";
  for (auto [index, input] : llvm::enumerate(launch.getInputs())) {
    if (index) output << ", ";
    output << "%" << argumentNames[index] << ": ";
    output << (input.getType() == f32 ? "f32" : "!tt.ptr<f32>");
  }
  if (!launch.getInputs().empty()) output << ", ";
  output << "%out: !tt.ptr<f32>) attributes {noinline = false} {\n"
         << "    %gf_c0_i32 = arith.constant 0 : i32\n"
         << "    %gf_c1_i32 = arith.constant 1 : i32\n"
         << "    %gf_num_src = arith.constant " << launch.getNumSrc()
         << " : i32\n"
         << "    %gf_row = tt.get_program_id x : i32\n";

  ScalarAlgebraEmitter emitter(launch, output);
  FailureOr<SmallVector<std::string>> identity =
      emitter.emit(reducer.getIdentity(), {}, "    ");
  if (failed(identity)) return failure();
  if (identity->size() != reducer.getStateTypes().size())
    return reject(launch, "reducer identity arity does not match state ABI");

  output << "    %gf_state:" << identity->size()
         << " = scf.for %gf_src = %gf_c0_i32 to %gf_num_src step "
            "%gf_c1_i32 iter_args(";
  for (auto [index, value] : llvm::enumerate(*identity)) {
    if (index) output << ", ";
    output << "%gf_acc" << index << " = " << value;
  }
  output << ") -> (";
  llvm::interleaveComma(*identity, output, [&](const std::string &) {
    output << "f32";
  });
  output << ") : i32 {\n";

  SmallVector<std::string> edgeArguments;
  for (auto [index, roleAttribute] : llvm::enumerate(roles)) {
    StringRef role = cast<StringAttr>(roleAttribute).getValue();
    std::string argument = "%" + argumentNames[index];
    if (role == "param") {
      edgeArguments.push_back(std::move(argument));
      continue;
    }
    std::string pointer = "%gf_input_ptr" + std::to_string(index);
    std::string loaded = "%gf_input" + std::to_string(index);
    StringRef coordinate = role == "src" ? "%gf_src" : "%gf_row";
    output << "      " << pointer << " = tt.addptr " << argument << ", "
           << coordinate << " : !tt.ptr<f32>, i32\n"
           << "      " << loaded << " = tt.load " << pointer
           << " : !tt.ptr<f32>\n";
    edgeArguments.push_back(std::move(loaded));
  }
  FailureOr<SmallVector<std::string>> message =
      emitter.emit(launch.getRegions().front(), edgeArguments, "      ");
  if (failed(message)) return failure();
  FailureOr<SmallVector<std::string>> lifted =
      emitter.emit(reducer.getLift(), *message, "      ");
  if (failed(lifted)) return failure();
  if (lifted->size() != identity->size())
    return reject(launch, "reducer lift arity does not match state ABI");
  SmallVector<std::string> combineArguments;
  for (size_t index = 0; index < identity->size(); ++index)
    combineArguments.push_back("%gf_acc" + std::to_string(index));
  llvm::append_range(combineArguments, *lifted);
  FailureOr<SmallVector<std::string>> combined =
      emitter.emit(reducer.getCombine(), combineArguments, "      ");
  if (failed(combined)) return failure();
  output << "      scf.yield ";
  llvm::interleaveComma(*combined, output);
  output << " : ";
  llvm::interleaveComma(*combined, output, [&](const std::string &) {
    output << "f32";
  });
  output << "\n    }\n";

  SmallVector<std::string> finalArguments;
  for (size_t index = 0; index < identity->size(); ++index)
    finalArguments.push_back("%gf_state#" + std::to_string(index));
  FailureOr<SmallVector<std::string>> finalized =
      emitter.emit(reducer.getFinalize(), finalArguments, "    ");
  if (failed(finalized)) return failure();
  if (finalized->size() != 1)
    return reject(launch, "generic dense reducer must finalize one result");
  std::string resultValue = finalized->front();
  if (hasNode) {
    SmallVector<std::string> nodeArguments;
    for (int64_t rawIndex : nodeIndices.asArrayRef()) {
      size_t index = static_cast<size_t>(rawIndex);
      StringRef role = cast<StringAttr>(roles[index]).getValue();
      std::string argument = "%" + argumentNames[index];
      if (role == "param") {
        nodeArguments.push_back(std::move(argument));
        continue;
      }
      if (role != "dst")
        return reject(launch, "node indexed inputs must be destination fields "
                              "or scalar parameters");
      std::string pointer = "%gf_node_ptr" + std::to_string(index);
      std::string loaded = "%gf_node_input" + std::to_string(index);
      output << "    " << pointer << " = tt.addptr " << argument
             << ", %gf_row : !tt.ptr<f32>, i32\n"
             << "    " << loaded << " = tt.load " << pointer
             << " : !tt.ptr<f32>\n";
      nodeArguments.push_back(std::move(loaded));
    }
    nodeArguments.push_back(resultValue);
    FailureOr<SmallVector<std::string>> updated =
        emitter.emit(launch.getRegions()[1], nodeArguments, "    ");
    if (failed(updated)) return failure();
    if (updated->size() != 1)
      return reject(launch, "generic dense node must yield one result");
    resultValue = updated->front();
  }
  output << "    %gf_out_ptr = tt.addptr %out, %gf_row : !tt.ptr<f32>, i32\n"
         << "    tt.store %gf_out_ptr, " << resultValue
         << " : !tt.ptr<f32>\n"
         << "    tt.return\n"
         << "  }\n"
         << "}\n";
  return success();
}

static LogicalResult translateDenseStreaming(
    kernel::DenseLaunchOp launch, llvm::raw_ostream &output) {
  MLIRContext *context = launch.getContext();
  Type f16 = Float16Type::get(context);
  Type f32 = Float32Type::get(context);
  if (launch.getNumSrc() != launch.getNumDst() || launch.getNumSrc() <= 0)
    return reject(launch, "dense streaming lowering requires a non-empty square relation");
  if (launch.getInputs().size() != 4 || launch.getNumResults() != 1 ||
      launch.getNumRegions() != 1)
    return reject(launch, "expected three fields, one scalar parameter, one region and one result");
  auto roles = launch.getInputRolesAttr();
  if (!roles || roles.size() != 4 ||
      cast<StringAttr>(roles[0]).getValue() != "dst" ||
      cast<StringAttr>(roles[1]).getValue() != "src" ||
      cast<StringAttr>(roles[2]).getValue() != "src" ||
      cast<StringAttr>(roles[3]).getValue() != "param")
    return reject(launch, "dense contraction requires dst,src,src,param projections");
  auto field = dyn_cast<RankedTensorType>(launch.getInputs()[0].getType());
  if (!field || field.getRank() != 3 || field.getElementType() != f16 ||
      field.isDynamicDim(1) || field.isDynamicDim(2))
    return reject(launch, "fields must be tensor<?xlanesxwidthxf16>");
  auto sourceField = dyn_cast<RankedTensorType>(launch.getInputs()[1].getType());
  if (!sourceField || sourceField.getRank() != 3 ||
      sourceField.getElementType() != f16 || sourceField.isDynamicDim(1) ||
      sourceField.isDynamicDim(2) ||
      launch.getInputs()[2].getType() != sourceField)
    return reject(launch, "dense source fields must share lanes and width");
  if (sourceField.getDimSize(2) != field.getDimSize(2) ||
      sourceField.getDimSize(1) <= 0 ||
      field.getDimSize(1) % sourceField.getDimSize(1) != 0)
    return reject(
        launch, "source lanes must divide destination lanes at equal width");
  if (launch.getInputs()[3].getType() != f32)
    return reject(launch, "the scalar parameter must be f32");
  auto result = dyn_cast<RankedTensorType>(launch.getResult(0).getType());
  if (result != field)
    return reject(launch, "result type must match the payload field");
  int64_t lanes = field.getDimSize(1);
  int64_t sourceLanes = sourceField.getDimSize(1);
  int64_t width = field.getDimSize(2);
  if (width != 16 && width != 32 && width != 64 && width != 128)
    return reject(launch, "provider supports contraction widths 16, 32, 64, or 128");
  auto laneAttr = launch.getIterationLanesAttr();
  if (!laneAttr || laneAttr.size() != 1 || laneAttr[0] != lanes)
    return reject(launch, "iteration_lanes must match the field lane extent");
  if (launch.getReducers().size() != 1 ||
      reducerKind(launch, launch.getReducers()[0]) != "streaming")
    return reject(launch, "requires one streaming reducer definition");
  auto reducerReference =
      dyn_cast<FlatSymbolRefAttr>(launch.getReducers()[0]);
  ReducerOp reducer = reducerReference
      ? SymbolTable::lookupNearestSymbolFrom<ReducerOp>(launch,
                                                        reducerReference)
      : ReducerOp();
  if (!reducer)
    return reject(launch, "streaming reducer symbol cannot be resolved");
  Block &block = launch.getRegions().front().front();
  auto yield = dyn_cast<kernel::YieldOp>(block.getTerminator());
  auto vectorType = VectorType::get({width}, f16);
  if (block.getNumArguments() != 4 ||
      block.getArgument(0).getType() != vectorType ||
      block.getArgument(1).getType() != vectorType ||
      block.getArgument(2).getType() != vectorType ||
      block.getArgument(3).getType() != f32 || !yield ||
      yield.getValues().size() != 2 || yield.getValues()[0].getType() != f32 ||
      yield.getValues()[1] != block.getArgument(2))
    return reject(launch, "edge region must yield scalar contraction and vector payload");
  auto schedule = launch->getAttrOfType<StringAttr>("schedule_kind");
  if (!schedule || schedule.getValue() != "dense-query-key-tile")
    return reject(launch, "requires a selected dense tile schedule");
  int64_t blockM = launch->getAttrOfType<IntegerAttr>("block_rows").getInt();
  int64_t blockN = launch->getAttrOfType<IntegerAttr>("block_neighbors").getInt();
  int64_t numWarps = launch->getAttrOfType<IntegerAttr>("num_warps").getInt();
  if (blockM <= 0 || blockN <= 0 || blockM > 128 || blockN > 128)
    return reject(launch, "dense tile bounds are invalid");
  double blockPruneThreshold = -1.0;
  if (auto threshold = reducer->getAttrOfType<FloatAttr>(
          "block_prune_threshold"))
    blockPruneThreshold = threshold.getValueAsDouble();
  // Dynamic admission carries the row maximum, denominator and output tile
  // across a control-flow edge.  A 128-row tile spills that live state on the
  // current CUDA provider; 64 rows is the registered occupancy boundary for
  // the approximate schedule.  This remains a target schedule decision: the
  // reducer IR contains only the semantic threshold.
  if (blockPruneThreshold > 0.0)
    blockM = 64;
  StringRef boundary = "full";
  if (auto boundaryAttr = launch->getAttrOfType<StringAttr>("boundary"))
    boundary = boundaryAttr.getValue();
  if (boundary != "full" && boundary != "lower_inclusive")
    return reject(launch, "unsupported dense boundary policy");
  emitDenseStreamingTTIR(output, launch.getNumSrc(), lanes, sourceLanes, width,
                         blockM, blockN, numWarps, blockPruneThreshold,
                         boundary);
  return success();
}

static LogicalResult translateCSRAdditiveTile(
    kernel::LaunchOp launch, ReducerOp reducer, StringRef indexType,
    int64_t blockD, llvm::raw_ostream &output) {
  ArrayAttr roles = launch.getInputRolesAttr();
  ArrayAttr names = launch.getInputNamesAttr();
  bool hasNode = launch.getNumRegions() == 2;
  if (!roles || roles.size() != launch.getInputs().size())
    return reject(launch, "tiled CSR algebra requires input_roles");
  SmallVector<std::string> argumentNames;
  llvm::StringMap<unsigned> occurrences;
  for (auto [position, attribute] : llvm::enumerate(names ? names : roles)) {
    std::string name = sanitizeIdentifier(
        cast<StringAttr>(attribute).getValue(), position);
    unsigned &occurrence = occurrences[name];
    if (occurrence++) name += "_" + std::to_string(occurrence);
    argumentNames.push_back(std::move(name));
  }
  std::string tileF32 = "tensor<" + std::to_string(blockD) + "xf32>";
  std::string tileIndex =
      "tensor<" + std::to_string(blockD) + "x" + indexType.str() + ">";
  output << "// graphforge.launch entry=gf_csr_additive_tile block_rows=1 "
            "num_warps=1 abi=row_ptr,col_idx,";
  for (StringRef name : argumentNames) output << name << ",";
  output << "out\nmodule {\n  tt.func public @gf_csr_additive_tile("
         << "%row_ptr: !tt.ptr<" << indexType << ">, %col_idx: !tt.ptr<"
         << indexType << ">";
  Type f32 = Float32Type::get(launch.getContext());
  for (auto [position, input] : llvm::enumerate(launch.getInputs()))
    output << ", %" << argumentNames[position] << ": "
           << (input.getType() == f32 ? "f32" : "!tt.ptr<f32>");
  output << ", %out: !tt.ptr<f32>) attributes {noinline = false} {\n"
         << "    %gf_zero_index = arith.constant dense<0> : " << tileIndex << "\n"
         << "    %gf_zero = arith.constant dense<0.000000e+00> : "
         << tileF32 << "\n"
         << "    %gf_c1_i32 = arith.constant 1 : i32\n"
         << "    %gf_row = tt.get_program_id x : i32\n"
         << "    %gf_start_ptr = tt.addptr %row_ptr, %gf_row : !tt.ptr<"
         << indexType << ">, i32\n"
         << "    %gf_start = tt.load %gf_start_ptr : !tt.ptr<" << indexType << ">\n"
         << "    %gf_end_ptr = tt.addptr %gf_start_ptr, %gf_c1_i32 : !tt.ptr<"
         << indexType << ">, i32\n"
         << "    %gf_end = tt.load %gf_end_ptr : !tt.ptr<" << indexType << ">\n"
         << "    %gf_lane_i32 = tt.make_range {end = " << blockD
         << " : i32, start = 0 : i32} : tensor<" << blockD << "xi32>\n";
  if (indexType == "i64")
    output << "    %gf_lane = arith.extsi %gf_lane_i32 : tensor<" << blockD
           << "xi32> to " << tileIndex << "\n";
  else
    output << "    %gf_lane = arith.bitcast %gf_lane_i32 : tensor<" << blockD
           << "xi32> to " << tileIndex << "\n";
  output << "    %gf_start_v = tt.splat %gf_start : " << indexType << " -> "
         << tileIndex << "\n"
         << "    %gf_edges = arith.addi %gf_start_v, %gf_lane : " << tileIndex << "\n"
         << "    %gf_end_v = tt.splat %gf_end : " << indexType << " -> "
         << tileIndex << "\n"
         << "    %gf_mask = arith.cmpi slt, %gf_edges, %gf_end_v : "
         << tileIndex << "\n"
         << "    %gf_col_base = tt.splat %col_idx : !tt.ptr<" << indexType
         << "> -> tensor<" << blockD << "x!tt.ptr<" << indexType << ">>\n"
         << "    %gf_col_ptr = tt.addptr %gf_col_base, %gf_edges : tensor<"
         << blockD << "x!tt.ptr<" << indexType << ">>, " << tileIndex << "\n"
         << "    %gf_src = tt.load %gf_col_ptr, %gf_mask, %gf_zero_index : tensor<"
         << blockD << "x!tt.ptr<" << indexType << ">>\n";
  SmallVector<std::string> edgeArguments;
  for (auto [position, roleAttribute] : llvm::enumerate(roles)) {
    StringRef role = cast<StringAttr>(roleAttribute).getValue();
    std::string argument = "%" + argumentNames[position];
    if (role == "param") {
      std::string splat = "%gf_param" + std::to_string(position);
      output << "    " << splat << " = tt.splat " << argument
             << " : f32 -> " << tileF32 << "\n";
      edgeArguments.push_back(std::move(splat));
      continue;
    }
    std::string base = "%gf_input_base" + std::to_string(position);
    std::string pointer = "%gf_input_ptr" + std::to_string(position);
    std::string loaded = "%gf_input" + std::to_string(position);
    output << "    " << base << " = tt.splat " << argument
           << " : !tt.ptr<f32> -> tensor<" << blockD << "x!tt.ptr<f32>>\n";
    if (role == "src")
      output << "    " << pointer << " = tt.addptr " << base
             << ", %gf_src : tensor<" << blockD << "x!tt.ptr<f32>>, "
             << tileIndex << "\n";
    else if (role == "edge")
      output << "    " << pointer << " = tt.addptr " << base
             << ", %gf_edges : tensor<" << blockD << "x!tt.ptr<f32>>, "
             << tileIndex << "\n";
    else if (role == "dst") {
      std::string row = "%gf_row_v" + std::to_string(position);
      output << "    " << row << " = tt.splat %gf_row : i32 -> tensor<"
             << blockD << "xi32>\n"
             << "    " << pointer << " = tt.addptr " << base << ", " << row
             << " : tensor<" << blockD << "x!tt.ptr<f32>>, tensor<" << blockD
             << "xi32>\n";
    } else
      return reject(launch, "tiled CSR input role is unsupported");
    output << "    " << loaded << " = tt.load " << pointer
           << ", %gf_mask, %gf_zero : tensor<" << blockD << "x!tt.ptr<f32>>\n";
    edgeArguments.push_back(std::move(loaded));
  }
  TensorAlgebraEmitter tensorEmitter(launch, output, tileF32);
  auto message = tensorEmitter.emit(launch.getRegions()[0], edgeArguments, "    ");
  if (failed(message)) return failure();
  auto lifted = tensorEmitter.emit(reducer.getLift(), *message, "    ");
  if (failed(lifted) || lifted->size() != reducer.getStateTypes().size())
    return reject(launch, "tiled lift arity does not match reducer state ABI");
  SmallVector<std::string> states;
  for (auto [position, value] : llvm::enumerate(*lifted)) {
    std::string masked = "%gf_masked_state" + std::to_string(position);
    std::string reduced = "%gf_state" + std::to_string(position);
    output << "    " << masked << " = \"arith.select\"(%gf_mask, " << value
           << ", %gf_zero) : (tensor<" << blockD << "xi1>, " << tileF32
           << ", " << tileF32 << ") -> " << tileF32 << "\n"
           << "    " << reduced << " = \"tt.reduce\"(" << masked
           << ") <{axis = 0 : i32}> ({\n"
           << "    ^bb0(%gf_a: f32, %gf_b: f32):\n"
           << "      %gf_sum = arith.addf %gf_a, %gf_b : f32\n"
           << "      tt.reduce.return %gf_sum : f32\n"
           << "    }) : (" << tileF32 << ") -> f32\n";
    states.push_back(std::move(reduced));
  }
  ScalarAlgebraEmitter scalarEmitter(launch, output);
  auto finalized = scalarEmitter.emit(reducer.getFinalize(), states, "    ");
  if (failed(finalized) || finalized->size() != 1)
    return reject(launch, "tiled reducer must finalize one result");
  std::string resultValue = finalized->front();
  if (hasNode) {
    SmallVector<std::string> nodeArguments;
    for (int64_t rawPosition : launch.getNodeInputIndicesAttr().asArrayRef()) {
      size_t position = static_cast<size_t>(rawPosition);
      StringRef role = cast<StringAttr>(roles[position]).getValue();
      std::string argument = "%" + argumentNames[position];
      if (role == "param") nodeArguments.push_back(std::move(argument));
      else if (role == "dst") {
        std::string pointer = "%gf_node_ptr" + std::to_string(position);
        std::string loaded = "%gf_node_input" + std::to_string(position);
        output << "    " << pointer << " = tt.addptr " << argument
               << ", %gf_row : !tt.ptr<f32>, i32\n"
               << "    " << loaded << " = tt.load " << pointer
               << " : !tt.ptr<f32>\n";
        nodeArguments.push_back(std::move(loaded));
      } else return reject(launch, "tiled node input must be dst or param");
    }
    nodeArguments.push_back(resultValue);
    auto updated = scalarEmitter.emit(launch.getRegions()[1], nodeArguments, "    ");
    if (failed(updated) || updated->size() != 1)
      return reject(launch, "tiled node must yield one result");
    resultValue = updated->front();
  }
  output << "    %gf_out_ptr = tt.addptr %out, %gf_row : !tt.ptr<f32>, i32\n"
         << "    tt.store %gf_out_ptr, " << resultValue << " : !tt.ptr<f32>\n"
         << "    tt.return\n  }\n}\n";
  return success();
}

/// Lower a horizontally fused product of zero/add reducers through one CSR
/// neighbor tile. Each edge region retains its own projected input segment and
/// reducer algebra, while relation indices and field loads are shared by the
/// single provider launch. This is operation-driven multi-result codegen, not
/// a workload-specific fused kernel.
static LogicalResult translateCSRProductAdditiveTile(
    kernel::LaunchOp launch, StringRef indexType, int64_t blockD,
    llvm::raw_ostream &output) {
  Type f32 = Float32Type::get(launch.getContext());
  ArrayAttr roles = launch.getInputRolesAttr();
  ArrayAttr names = launch.getInputNamesAttr();
  auto segments = launch.getInputSegmentSizesAttr();
  if (launch.getDeterministic() || launch.getNumResults() < 2 ||
      launch.getReducers().size() != launch.getNumResults() ||
      launch.getNumRegions() != launch.getNumResults() || !roles ||
      roles.size() != launch.getInputs().size() ||
      (names && names.size() != launch.getInputs().size()) ||
      segments.size() != launch.getNumResults() ||
      llvm::any_of(launch.getRegionKinds(), [](int64_t kind) { return kind != 0; }))
    return reject(launch, "product additive tile has an invalid fused ABI");
  int64_t projected = 0;
  for (int64_t size : segments.asArrayRef()) projected += size;
  if (projected != static_cast<int64_t>(launch.getInputs().size()))
    return reject(launch, "product input segments do not cover all inputs");

  SmallVector<ReducerOp> reducers;
  for (auto [position, attribute] : llvm::enumerate(launch.getReducers())) {
    auto reference = dyn_cast<FlatSymbolRefAttr>(attribute);
    auto reducer = reference
        ? SymbolTable::lookupNearestSymbolFrom<ReducerOp>(launch, reference)
        : ReducerOp();
    if (!reducer || !isZeroAdditiveState(reducer) ||
        reducer.getResultTypes().size() != 1 ||
        cast<TypeAttr>(reducer.getResultTypes()[0]).getValue() != f32 ||
        !isRankOneTensor(launch.getResult(position), f32))
      return reject(launch, "every product component must be a scalar FP32 "
                            "zero/add reducer");
    Block &edge = launch.getRegions()[position].front();
    if (edge.getNumArguments() !=
        static_cast<unsigned>(segments.asArrayRef()[position]))
      return reject(launch, "product edge arguments do not match its input segment");
    reducers.push_back(reducer);
  }
  for (auto [input, roleAttribute] : llvm::zip(launch.getInputs(), roles)) {
    StringRef role = cast<StringAttr>(roleAttribute).getValue();
    if (role == "param") {
      if (input.getType() != f32)
        return reject(launch, "product scalar parameters must be f32");
    } else if (role == "src" || role == "dst" || role == "edge") {
      if (!isRankOneTensor(input, f32))
        return reject(launch, "product fields must be tensor<?xf32>");
    } else {
      return reject(launch, "product input role is unsupported");
    }
  }

  SmallVector<std::string> argumentNames;
  llvm::StringMap<unsigned> occurrences;
  for (auto [position, attribute] : llvm::enumerate(names ? names : roles)) {
    std::string name = sanitizeIdentifier(
        cast<StringAttr>(attribute).getValue(), position);
    unsigned &occurrence = occurrences[name];
    if (occurrence++) name += "_" + std::to_string(occurrence);
    argumentNames.push_back(std::move(name));
  }
  std::string tileF32 = "tensor<" + std::to_string(blockD) + "xf32>";
  std::string tileI1 = "tensor<" + std::to_string(blockD) + "xi1>";
  std::string tileIndex = "tensor<" + std::to_string(blockD) + "x" +
                          indexType.str() + ">";
  output << "// graphforge.launch entry=gf_csr_product_additive_tile "
            "block_rows=1 num_warps=1 abi=row_ptr,col_idx,";
  for (StringRef name : argumentNames) output << name << ",";
  for (unsigned result = 0; result < launch.getNumResults(); ++result)
    output << "out" << result << (result + 1 == launch.getNumResults() ? "" : ",");
  output << "\nmodule {\n  tt.func public @gf_csr_product_additive_tile("
         << "%row_ptr: !tt.ptr<" << indexType << ">, %col_idx: !tt.ptr<"
         << indexType << ">";
  for (auto [position, input] : llvm::enumerate(launch.getInputs()))
    output << ", %" << argumentNames[position] << ": "
           << (input.getType() == f32 ? "f32" : "!tt.ptr<f32>");
  for (unsigned result = 0; result < launch.getNumResults(); ++result)
    output << ", %out" << result << ": !tt.ptr<f32>";
  output << ") attributes {noinline = false} {\n"
         << "    %gf_zero_index = arith.constant dense<0> : " << tileIndex << "\n"
         << "    %gf_zero = arith.constant dense<0.000000e+00> : " << tileF32 << "\n"
         << "    %gf_c1_i32 = arith.constant 1 : i32\n"
         << "    %gf_row = tt.get_program_id x : i32\n"
         << "    %gf_start_ptr = tt.addptr %row_ptr, %gf_row : !tt.ptr<"
         << indexType << ">, i32\n"
         << "    %gf_start = tt.load %gf_start_ptr : !tt.ptr<" << indexType << ">\n"
         << "    %gf_end_ptr = tt.addptr %gf_start_ptr, %gf_c1_i32 : !tt.ptr<"
         << indexType << ">, i32\n"
         << "    %gf_end = tt.load %gf_end_ptr : !tt.ptr<" << indexType << ">\n"
         << "    %gf_lane_i32 = tt.make_range {end = " << blockD
         << " : i32, start = 0 : i32} : tensor<" << blockD << "xi32>\n";
  if (indexType == "i64")
    output << "    %gf_lane = arith.extsi %gf_lane_i32 : tensor<" << blockD
           << "xi32> to " << tileIndex << "\n";
  else
    output << "    %gf_lane = arith.bitcast %gf_lane_i32 : tensor<" << blockD
           << "xi32> to " << tileIndex << "\n";
  output << "    %gf_start_v = tt.splat %gf_start : " << indexType << " -> "
         << tileIndex << "\n"
         << "    %gf_edges = arith.addi %gf_start_v, %gf_lane : " << tileIndex << "\n"
         << "    %gf_end_v = tt.splat %gf_end : " << indexType << " -> "
         << tileIndex << "\n"
         << "    %gf_mask = arith.cmpi slt, %gf_edges, %gf_end_v : "
         << tileIndex << "\n"
         << "    %gf_col_base = tt.splat %col_idx : !tt.ptr<" << indexType
         << "> -> tensor<" << blockD << "x!tt.ptr<" << indexType << ">>\n"
         << "    %gf_col_ptr = tt.addptr %gf_col_base, %gf_edges : tensor<"
         << blockD << "x!tt.ptr<" << indexType << ">>, " << tileIndex << "\n"
         << "    %gf_src = tt.load %gf_col_ptr, %gf_mask, %gf_zero_index : tensor<"
         << blockD << "x!tt.ptr<" << indexType << ">>\n";

  SmallVector<std::string> loadedInputs;
  for (auto [position, roleAttribute] : llvm::enumerate(roles)) {
    StringRef role = cast<StringAttr>(roleAttribute).getValue();
    // Fusion preserves each apply's projected ABI, so the same SSA field may
    // occur in more than one segment.  Reuse its already loaded tile when the
    // role also agrees; this is the memory-traffic benefit of horizontal
    // fusion, not merely a reduction in launch count.
    bool reused = false;
    for (unsigned previous = 0; previous < position; ++previous) {
      if (launch.getInputs()[previous] == launch.getInputs()[position] &&
          cast<StringAttr>(roles[previous]).getValue() == role) {
        loadedInputs.push_back(loadedInputs[previous]);
        reused = true;
        break;
      }
    }
    if (reused) continue;
    std::string argument = "%" + argumentNames[position];
    if (role == "param") {
      std::string splat = "%gf_param" + std::to_string(position);
      output << "    " << splat << " = tt.splat " << argument
             << " : f32 -> " << tileF32 << "\n";
      loadedInputs.push_back(std::move(splat));
      continue;
    }
    std::string base = "%gf_input_base" + std::to_string(position);
    std::string pointer = "%gf_input_ptr" + std::to_string(position);
    std::string loaded = "%gf_input" + std::to_string(position);
    output << "    " << base << " = tt.splat " << argument
           << " : !tt.ptr<f32> -> tensor<" << blockD << "x!tt.ptr<f32>>\n";
    if (role == "src")
      output << "    " << pointer << " = tt.addptr " << base
             << ", %gf_src : tensor<" << blockD << "x!tt.ptr<f32>>, "
             << tileIndex << "\n";
    else if (role == "edge")
      output << "    " << pointer << " = tt.addptr " << base
             << ", %gf_edges : tensor<" << blockD << "x!tt.ptr<f32>>, "
             << tileIndex << "\n";
    else {
      std::string row = "%gf_row_v" + std::to_string(position);
      output << "    " << row << " = tt.splat %gf_row : i32 -> tensor<"
             << blockD << "xi32>\n"
             << "    " << pointer << " = tt.addptr " << base << ", " << row
             << " : tensor<" << blockD << "x!tt.ptr<f32>>, tensor<" << blockD
             << "xi32>\n";
    }
    output << "    " << loaded << " = tt.load " << pointer
           << ", %gf_mask, %gf_zero : tensor<" << blockD << "x!tt.ptr<f32>>\n";
    loadedInputs.push_back(std::move(loaded));
  }

  int64_t inputOffset = 0;
  for (auto [component, reducer] : llvm::enumerate(reducers)) {
    int64_t inputCount = segments.asArrayRef()[component];
    ArrayRef<std::string> edgeArguments(loadedInputs.data() + inputOffset,
                                        inputCount);
    TensorAlgebraEmitter tileEmitter(
        launch, output, tileF32,
        "%gf_p" + std::to_string(component) + "_t");
    auto message = tileEmitter.emit(
        launch.getRegions()[component], edgeArguments, "    ");
    if (failed(message)) return failure();
    auto lifted = tileEmitter.emit(reducer.getLift(), *message, "    ");
    if (failed(lifted) || lifted->size() != reducer.getStateTypes().size())
      return reject(launch, "product lift arity does not match reducer state");
    SmallVector<std::string> states;
    for (auto [state, value] : llvm::enumerate(*lifted)) {
      std::string suffix = std::to_string(component) + "_" + std::to_string(state);
      std::string masked = "%gf_product_masked" + suffix;
      std::string reduced = "%gf_product_state" + suffix;
      output << "    " << masked << " = \"arith.select\"(%gf_mask, " << value
             << ", %gf_zero) : (" << tileI1 << ", " << tileF32 << ", "
             << tileF32 << ") -> " << tileF32 << "\n"
             << "    " << reduced << " = \"tt.reduce\"(" << masked
             << ") <{axis = 0 : i32}> ({\n"
             << "    ^bb0(%gf_a: f32, %gf_b: f32):\n"
             << "      %gf_product_sum" << suffix
             << " = arith.addf %gf_a, %gf_b : f32\n"
             << "      tt.reduce.return %gf_product_sum" << suffix << " : f32\n"
             << "    }) : (" << tileF32 << ") -> f32\n";
      states.push_back(std::move(reduced));
    }
    ScalarAlgebraEmitter scalarEmitter(
        launch, output, "%gf_p" + std::to_string(component) + "_s");
    auto finalized = scalarEmitter.emit(reducer.getFinalize(), states, "    ");
    if (failed(finalized) || finalized->size() != 1)
      return reject(launch, "product reducer must finalize one result");
    output << "    %gf_out_ptr" << component << " = tt.addptr %out"
           << component << ", %gf_row : !tt.ptr<f32>, i32\n"
           << "    tt.store %gf_out_ptr" << component << ", "
           << finalized->front() << " : !tt.ptr<f32>\n";
    inputOffset += inputCount;
  }
  output << "    tt.return\n  }\n}\n";
  return success();
}

/// Lower an associative scalar reducer over a two-dimensional row-neighbor
/// tile. Padding lanes carry the reducer's actual identity and all state
/// components enter one variadic tt.reduce whose body is emitted from the
/// combine region. The edge and lift regions run over [blockM, blockD], while
/// finalize and the optional node region run over [blockM]. No reducer symbol
/// or workload name participates in this decision: legality comes from the
/// typed reducer regions.
static LogicalResult translateCSRRowsTile(
    kernel::LaunchOp launch, ReducerOp reducer, StringRef indexType,
    int64_t blockM, int64_t blockD, int64_t numWarps,
    llvm::raw_ostream &output, int64_t bucketOrdinal = -1,
    int64_t bucketCount = 0, bool directFilter = false) {
  Type f32 = Float32Type::get(launch.getContext());
  ArrayAttr roles = launch.getInputRolesAttr();
  ArrayAttr names = launch.getInputNamesAttr();
  bool hasNode = launch.getNumRegions() == 2;
  if (blockM <= 1 || blockD <= 0 || !roles ||
      roles.size() != launch.getInputs().size() ||
      (names && names.size() != launch.getInputs().size()))
    return reject(launch, "multi-row reducer tile has an invalid ABI");
  for (auto [input, roleAttribute] : llvm::zip(launch.getInputs(), roles)) {
    StringRef role = cast<StringAttr>(roleAttribute).getValue();
    if (role == "param") {
      if (input.getType() != f32)
        return reject(launch, "tiled scalar parameters must be f32");
    } else if (role == "src" || role == "dst" || role == "edge") {
      if (!isRankOneTensor(input, f32))
        return reject(launch, "tiled fields must be tensor<?xf32>");
    } else {
      return reject(launch, "multi-row reducer input role is unsupported");
    }
  }
  DenseI64ArrayAttr nodeIndices = launch.getNodeInputIndicesAttr();
  DenseI64ArrayAttr nodeSegments = launch.getNodeInputSegmentSizesAttr();
  if (hasNode && (!nodeIndices || !nodeSegments || nodeSegments.size() != 1 ||
                  nodeSegments.asArrayRef()[0] !=
                      static_cast<int64_t>(nodeIndices.size())))
    return reject(launch, "tiled node requires its explicit indexed ABI");

  SmallVector<std::string> argumentNames;
  llvm::StringMap<unsigned> occurrences;
  for (auto [position, attribute] : llvm::enumerate(names ? names : roles)) {
    std::string name = sanitizeIdentifier(
        cast<StringAttr>(attribute).getValue(), position);
    unsigned &occurrence = occurrences[name];
    if (occurrence++) name += "_" + std::to_string(occurrence);
    argumentNames.push_back(std::move(name));
  }
  std::string rowsI32 = "tensor<" + std::to_string(blockM) + "xi32>";
  std::string rowsI1 = "tensor<" + std::to_string(blockM) + "xi1>";
  std::string rowsIndex = "tensor<" + std::to_string(blockM) + "x" +
                          indexType.str() + ">";
  std::string rowsF32 = "tensor<" + std::to_string(blockM) + "xf32>";
  std::string tileI32 = "tensor<" + std::to_string(blockM) + "x" +
                        std::to_string(blockD) + "xi32>";
  std::string tileIndex = "tensor<" + std::to_string(blockM) + "x" +
                          std::to_string(blockD) + "x" + indexType.str() + ">";
  std::string tileF32 = "tensor<" + std::to_string(blockM) + "x" +
                        std::to_string(blockD) + "xf32>";
  std::string tileI1 = "tensor<" + std::to_string(blockM) + "x" +
                       std::to_string(blockD) + "xi1>";

  bool bucketed = bucketOrdinal >= 0;
  if (bucketed && (bucketCount <= 0 || bucketOrdinal >= bucketCount ||
                   launch.getNumRowsAttr().getInt() >
                       std::numeric_limits<int32_t>::max()))
    return reject(launch, "degree bucket has invalid bounds or row count");
  bool additive = isZeroAdditiveState(reducer);
  bool stableWeighted = isStableWeightedScalarState(reducer);
  int64_t directScoreInput = -1;
  int64_t directValueInput = -1;
  double stableScoreIdentity = -std::numeric_limits<double>::infinity();
  if (stableWeighted) {
    Block &edgeRegion = launch.getRegions()[0].front();
    auto edgeYield = dyn_cast<kernel::YieldOp>(edgeRegion.getTerminator());
    if (edgeYield && edgeYield.getValues().size() == 2) {
      for (auto [position, argument] :
           llvm::enumerate(edgeRegion.getArguments())) {
        if (edgeYield.getValues()[0] == argument)
          directScoreInput = static_cast<int64_t>(position);
        if (edgeYield.getValues()[1] == argument)
          directValueInput = static_cast<int64_t>(position);
      }
    }
    auto identityYield =
        cast<ReducerYieldOp>(reducer.getIdentity().front().getTerminator());
    auto identityConstant = identityYield.getValues()[0]
                                .getDefiningOp<arith::ConstantOp>();
    stableScoreIdentity =
        cast<FloatAttr>(identityConstant.getValue()).getValueAsDouble();
  }
  std::string entry = bucketed
      ? "gf_csr_additive_bucket_" + std::to_string(bucketOrdinal)
      : additive ? "gf_csr_additive_tile"
      : stableWeighted ? "gf_csr_stable_weighted_tile"
                       : "gf_csr_algebra_tile";
  output << "// graphforge.launch entry=" << entry << " block_rows="
         << blockM << " num_warps=" << numWarps
         << " abi=row_ptr,col_idx,";
  for (StringRef name : argumentNames) output << name << ",";
  if (bucketed) output << "row_worklist,";
  output << "out\nmodule {\n  tt.func public @" << entry << "("
         << "%row_ptr: !tt.ptr<" << indexType << ">, %col_idx: !tt.ptr<"
         << indexType << ">";
  for (auto [position, input] : llvm::enumerate(launch.getInputs()))
    output << ", %" << argumentNames[position] << ": "
           << (input.getType() == f32 ? "f32" : "!tt.ptr<f32>");
  if (bucketed) output << ", %row_worklist: !tt.ptr<i64>";
  output << ", %out: !tt.ptr<f32>) attributes {noinline = false} {\n"
         << "    %gf_zero_index = arith.constant dense<0> : " << tileIndex << "\n"
         << "    %gf_zero_row_index = arith.constant dense<0> : " << rowsIndex << "\n"
         << "    %gf_zero = arith.constant dense<0.000000e+00> : " << tileF32 << "\n"
         << "    %gf_zero_rows = arith.constant dense<0.000000e+00> : " << rowsF32 << "\n"
         << "    %gf_block_m = arith.constant " << blockM << " : i32\n"
         << "    %gf_one = arith.constant dense<1> : " << rowsI32 << "\n"
         << "    %gf_pid = tt.get_program_id x : i32\n"
         << "    %gf_base = arith.muli %gf_pid, %gf_block_m : i32\n"
         << "    %gf_row_lane = tt.make_range {end = " << blockM
         << " : i32, start = 0 : i32} : " << rowsI32 << "\n"
         << "    %gf_base_v = tt.splat %gf_base : i32 -> " << rowsI32 << "\n"
         << "    %gf_positions = arith.addi %gf_base_v, %gf_row_lane : "
         << rowsI32 << "\n";
  if (directScoreInput >= 0)
    output << "    %gf_stable_score_pad = arith.constant dense<"
           << llvm::format("%.9e", stableScoreIdentity) << "> : " << tileF32
           << "\n";
  if (bucketed && !directFilter) {
    int64_t rowBase = 2 * bucketCount + 1;
    output
         << "    %gf_bucket = arith.constant " << bucketOrdinal << " : i32\n"
         << "    %gf_bucket_next = arith.constant " << bucketOrdinal + 1
         << " : i32\n"
         << "    %gf_bucket_start_ptr = tt.addptr %row_worklist, %gf_bucket : "
            "!tt.ptr<i64>, i32\n"
         << "    %gf_bucket_end_ptr = tt.addptr %row_worklist, %gf_bucket_next : "
            "!tt.ptr<i64>, i32\n"
         << "    %gf_bucket_start = tt.load %gf_bucket_start_ptr : !tt.ptr<i64>\n"
         << "    %gf_bucket_end = tt.load %gf_bucket_end_ptr : !tt.ptr<i64>\n"
         << "    %gf_bucket_size64 = arith.subi %gf_bucket_end, %gf_bucket_start : i64\n"
         << "    %gf_bucket_size = arith.trunci %gf_bucket_size64 : i64 to i32\n"
         << "    %gf_bucket_size_v = tt.splat %gf_bucket_size : i32 -> "
         << rowsI32 << "\n"
         << "    %gf_row_mask = arith.cmpi slt, %gf_positions, %gf_bucket_size_v : "
         << rowsI32 << "\n"
         << "    %gf_positions64 = arith.extui %gf_positions : " << rowsI32
         << " to tensor<" << blockM << "xi64>\n"
         << "    %gf_bucket_start_v = tt.splat %gf_bucket_start : i64 -> tensor<"
         << blockM << "xi64>\n"
         << "    %gf_row_base = arith.constant dense<" << rowBase << "> : tensor<"
         << blockM << "xi64>\n"
         << "    %gf_row_slots0 = arith.addi %gf_row_base, %gf_bucket_start_v : "
            "tensor<" << blockM << "xi64>\n"
         << "    %gf_row_slots = arith.addi %gf_row_slots0, %gf_positions64 : "
            "tensor<" << blockM << "xi64>\n"
         << "    %gf_worklist_base = tt.splat %row_worklist : !tt.ptr<i64> -> "
            "tensor<" << blockM << "x!tt.ptr<i64>>\n"
         << "    %gf_row_id_ptr = tt.addptr %gf_worklist_base, %gf_row_slots : "
            "tensor<" << blockM << "x!tt.ptr<i64>>, tensor<" << blockM
         << "xi64>\n"
         << "    %gf_zero_rows64 = arith.constant dense<0> : tensor<" << blockM
         << "xi64>\n"
         << "    %gf_rows64 = tt.load %gf_row_id_ptr, %gf_row_mask, %gf_zero_rows64 : "
            "tensor<" << blockM << "x!tt.ptr<i64>>\n"
         << "    %gf_rows = arith.trunci %gf_rows64 : tensor<" << blockM
         << "xi64> to " << rowsI32 << "\n";
  } else {
    output
         << "    %gf_nrows = arith.constant dense<"
         << launch.getNumRowsAttr().getInt() << "> : " << rowsI32 << "\n"
         << "    %gf_zero_rows_i32 = arith.constant dense<0> : " << rowsI32
         << "\n"
         << "    %gf_rows = arith.addi %gf_positions, %gf_zero_rows_i32 : "
         << rowsI32 << "\n"
         << "    %gf_row_mask = arith.cmpi slt, %gf_rows, %gf_nrows : "
         << rowsI32 << "\n";
  }
  output
         << "    %gf_row_ptr_base = tt.splat %row_ptr : !tt.ptr<" << indexType
         << "> -> tensor<" << blockM << "x!tt.ptr<" << indexType << ">>\n"
         << "    %gf_start_ptr = tt.addptr %gf_row_ptr_base, %gf_rows : tensor<"
         << blockM << "x!tt.ptr<" << indexType << ">>, " << rowsI32 << "\n"
         << "    %gf_starts = tt.load %gf_start_ptr, %gf_row_mask, %gf_zero_row_index : tensor<"
         << blockM << "x!tt.ptr<" << indexType << ">>\n"
         << "    %gf_end_ptr = tt.addptr %gf_start_ptr, %gf_one : tensor<"
         << blockM << "x!tt.ptr<" << indexType << ">>, " << rowsI32 << "\n"
         << "    %gf_ends = tt.load %gf_end_ptr, %gf_row_mask, %gf_zero_row_index : tensor<"
         << blockM << "x!tt.ptr<" << indexType << ">>\n"
         ;
  if (directFilter) {
    output
         << "    %gf_degrees = arith.subi %gf_ends, %gf_starts : " << rowsIndex
         << "\n"
         << "    %gf_degree_limit = arith.constant dense<" << blockD << "> : "
         << rowsIndex << "\n"
         << "    %gf_degree_ok = arith.cmpi sle, %gf_degrees, %gf_degree_limit : "
         << rowsIndex << "\n"
         << "    %gf_filtered_row_mask = arith.andi %gf_row_mask, %gf_degree_ok : "
         << rowsI1 << "\n";
  }
  StringRef activeRowMask = directFilter ? "%gf_filtered_row_mask"
                                         : "%gf_row_mask";
  output
         << "    %gf_neighbor_lane = tt.make_range {end = " << blockD
         << " : i32, start = 0 : i32} : tensor<" << blockD << "xi32>\n"
         << "    %gf_neighbors_2d = tt.expand_dims %gf_neighbor_lane {axis = 0 : i32} : tensor<"
         << blockD << "xi32> -> tensor<1x" << blockD << "xi32>\n";
  if (indexType == "i64")
    output << "    %gf_neighbors_index = arith.extsi %gf_neighbors_2d : tensor<1x"
           << blockD << "xi32> to tensor<1x" << blockD << "xi64>\n";
  output << "    %gf_starts_2d = tt.expand_dims %gf_starts {axis = 1 : i32} : "
         << rowsIndex << " -> tensor<" << blockM << "x1x" << indexType << ">\n"
         << "    %gf_starts_b = tt.broadcast %gf_starts_2d : tensor<" << blockM
         << "x1x" << indexType << "> -> " << tileIndex << "\n"
         << "    %gf_neighbors_b = tt.broadcast %gf_neighbors_"
         << (indexType == "i64" ? "index" : "2d") << " : tensor<1x" << blockD
         << "x" << indexType << "> -> " << tileIndex << "\n"
         << "    %gf_edges = arith.addi %gf_starts_b, %gf_neighbors_b : " << tileIndex << "\n"
         << "    %gf_ends_2d = tt.expand_dims %gf_ends {axis = 1 : i32} : "
         << rowsIndex << " -> tensor<" << blockM << "x1x" << indexType << ">\n"
         << "    %gf_ends_b = tt.broadcast %gf_ends_2d : tensor<" << blockM
         << "x1x" << indexType << "> -> " << tileIndex << "\n"
         << "    %gf_edge_mask = arith.cmpi slt, %gf_edges, %gf_ends_b : " << tileIndex << "\n"
         << "    %gf_row_mask_2d = tt.expand_dims " << activeRowMask
         << " {axis = 1 : i32} : tensor<"
         << blockM << "xi1> -> tensor<" << blockM << "x1xi1>\n"
         << "    %gf_row_mask_b = tt.broadcast %gf_row_mask_2d : tensor<" << blockM
         << "x1xi1> -> " << tileI1 << "\n"
         << "    %gf_mask = arith.andi %gf_row_mask_b, %gf_edge_mask : " << tileI1 << "\n"
         << "    %gf_col_base = tt.splat %col_idx : !tt.ptr<" << indexType
         << "> -> tensor<" << blockM << "x" << blockD << "x!tt.ptr<" << indexType << ">>\n"
         << "    %gf_col_ptr = tt.addptr %gf_col_base, %gf_edges : tensor<"
         << blockM << "x" << blockD << "x!tt.ptr<" << indexType << ">>, " << tileIndex << "\n"
         << "    %gf_src = tt.load %gf_col_ptr, %gf_mask, %gf_zero_index : tensor<"
         << blockM << "x" << blockD << "x!tt.ptr<" << indexType << ">>\n"
         << "    %gf_rows_2d = tt.expand_dims %gf_rows {axis = 1 : i32} : "
         << rowsI32 << " -> tensor<" << blockM << "x1xi32>\n"
         << "    %gf_rows_b = tt.broadcast %gf_rows_2d : tensor<" << blockM
         << "x1xi32> -> " << tileI32 << "\n";

  SmallVector<std::string> edgeArguments;
  for (auto [position, roleAttribute] : llvm::enumerate(roles)) {
    StringRef role = cast<StringAttr>(roleAttribute).getValue();
    std::string argument = "%" + argumentNames[position];
    if (role == "param") {
      std::string splat = "%gf_param" + std::to_string(position);
      output << "    " << splat << " = tt.splat " << argument
             << " : f32 -> " << tileF32 << "\n";
      edgeArguments.push_back(std::move(splat));
      continue;
    }
    if (role == "dst") {
      // Destination fields are invariant across all neighbors in a row. Load
      // once per row and broadcast instead of repeating the same global load
      // and address calculation blockD times.
      std::string rowBase = "%gf_dst_row_base" + std::to_string(position);
      std::string rowPointer = "%gf_dst_row_ptr" + std::to_string(position);
      std::string rowLoaded = "%gf_dst_row_value" + std::to_string(position);
      std::string expanded = "%gf_dst_row_2d" + std::to_string(position);
      std::string broadcast = "%gf_input" + std::to_string(position);
      output << "    " << rowBase << " = tt.splat " << argument
             << " : !tt.ptr<f32> -> tensor<" << blockM
             << "x!tt.ptr<f32>>\n"
             << "    " << rowPointer << " = tt.addptr " << rowBase
             << ", %gf_rows : tensor<" << blockM
             << "x!tt.ptr<f32>>, " << rowsI32 << "\n"
             << "    " << rowLoaded << " = tt.load " << rowPointer
             << ", " << activeRowMask << ", %gf_zero_rows : tensor<"
             << blockM << "x!tt.ptr<f32>>\n"
             << "    " << expanded << " = tt.expand_dims " << rowLoaded
             << " {axis = 1 : i32} : " << rowsF32 << " -> tensor<"
             << blockM << "x1xf32>\n"
             << "    " << broadcast << " = tt.broadcast " << expanded
             << " : tensor<" << blockM << "x1xf32> -> " << tileF32 << "\n";
      edgeArguments.push_back(std::move(broadcast));
      continue;
    }
    std::string base = "%gf_input_base" + std::to_string(position);
    std::string pointer = "%gf_input_ptr" + std::to_string(position);
    std::string loaded = "%gf_input" + std::to_string(position);
    StringRef padding =
        static_cast<int64_t>(position) == directScoreInput
            ? StringRef("%gf_stable_score_pad")
            : StringRef("%gf_zero");
    output << "    " << base << " = tt.splat " << argument << " : !tt.ptr<f32> -> tensor<"
           << blockM << "x" << blockD << "x!tt.ptr<f32>>\n"
           << "    " << pointer << " = tt.addptr " << base << ", "
           << (role == "src" ? "%gf_src" : "%gf_edges")
           << " : tensor<" << blockM << "x" << blockD << "x!tt.ptr<f32>>, "
           << tileIndex << "\n"
           << "    " << loaded << " = tt.load " << pointer
           << ", %gf_mask, " << padding << " : tensor<" << blockM << "x" << blockD
           << "x!tt.ptr<f32>>\n";
    edgeArguments.push_back(std::move(loaded));
  }

  TensorAlgebraEmitter tileEmitter(launch, output, tileF32, "%gf_tile_v");
  auto message = tileEmitter.emit(launch.getRegions()[0], edgeArguments, "    ");
  if (failed(message)) return failure();
  SmallVector<std::string> states;
  if (stableWeighted) {
    if (message->size() != 2)
      return reject(launch, "stable weighted tile requires score and value messages");
    auto identity = tileEmitter.emit(reducer.getIdentity(), {}, "    ");
    if (failed(identity) || identity->size() != 3)
      return reject(launch, "stable weighted identity must have three states");
    std::string stableScore = (*message)[0];
    if (directScoreInput < 0) {
      stableScore = "%gf_stable_score";
      output << "    %gf_stable_score = \"arith.select\"(%gf_mask, "
             << (*message)[0] << ", " << (*identity)[0] << ") : (" << tileI1
             << ", " << tileF32 << ", " << tileF32 << ") -> " << tileF32
             << "\n";
    }
    output
        << "    %gf_stable_max = \"tt.reduce\"(" << stableScore << ") "
           "<{axis = 1 : i32}> ({\n"
        << "    ^bb0(%gf_max_left: f32, %gf_max_right: f32):\n"
        << "      %gf_max_value = arith.maximumf %gf_max_left, "
           "%gf_max_right : f32\n"
        << "      tt.reduce.return %gf_max_value : f32\n"
        << "    }) : (" << tileF32 << ") -> " << rowsF32 << "\n"
        << "    %gf_stable_max_2d = tt.expand_dims %gf_stable_max "
           "{axis = 1 : i32} : " << rowsF32 << " -> tensor<" << blockM
        << "x1xf32>\n"
        << "    %gf_stable_max_tile = tt.broadcast %gf_stable_max_2d : "
           "tensor<" << blockM << "x1xf32> -> " << tileF32 << "\n"
        << "    %gf_stable_shift = arith.subf " << stableScore << ", "
           "%gf_stable_max_tile : " << tileF32 << "\n"
        << "    %gf_stable_weight_raw = math.exp %gf_stable_shift : "
        << tileF32 << "\n"
        << "    %gf_stable_weight = \"arith.select\"(%gf_mask, "
           "%gf_stable_weight_raw, %gf_zero) : (" << tileI1 << ", "
        << tileF32 << ", " << tileF32 << ") -> " << tileF32 << "\n"
        << "    %gf_stable_weighted_raw = arith.mulf %gf_stable_weight, "
        << (*message)[1] << " : " << tileF32 << "\n";
    std::string stableWeightedValue = "%gf_stable_weighted_raw";
    if (directValueInput < 0) {
      stableWeightedValue = "%gf_stable_weighted";
      output << "    %gf_stable_weighted = \"arith.select\"(%gf_mask, "
                "%gf_stable_weighted_raw, %gf_zero) : (" << tileI1 << ", "
             << tileF32 << ", " << tileF32 << ") -> " << tileF32 << "\n";
    }
    output
        << "    %gf_stable_den = \"tt.reduce\"(%gf_stable_weight) "
           "<{axis = 1 : i32}> ({\n"
        << "    ^bb0(%gf_den_left: f32, %gf_den_right: f32):\n"
        << "      %gf_den_sum = arith.addf %gf_den_left, %gf_den_right : f32\n"
        << "      tt.reduce.return %gf_den_sum : f32\n"
        << "    }) : (" << tileF32 << ") -> " << rowsF32 << "\n"
        << "    %gf_stable_num = \"tt.reduce\"(" << stableWeightedValue << ") "
           "<{axis = 1 : i32}> ({\n"
        << "    ^bb0(%gf_num_left: f32, %gf_num_right: f32):\n"
        << "      %gf_num_sum = arith.addf %gf_num_left, %gf_num_right : f32\n"
        << "      tt.reduce.return %gf_num_sum : f32\n"
        << "    }) : (" << tileF32 << ") -> " << rowsF32 << "\n";
    states = {"%gf_stable_max", "%gf_stable_den", "%gf_stable_num"};
  } else {
    auto lifted = tileEmitter.emit(reducer.getLift(), *message, "    ");
    if (failed(lifted) || lifted->size() != reducer.getStateTypes().size())
      return reject(launch, "multi-row lift arity does not match reducer state ABI");
    auto identity = tileEmitter.emit(reducer.getIdentity(), {}, "    ");
    if (failed(identity) || identity->size() != lifted->size())
      return reject(launch, "multi-row identity arity does not match reducer state ABI");
    SmallVector<std::string> maskedStates;
    for (auto [position, value] : llvm::enumerate(*lifted)) {
      std::string masked = "%gf_masked_state" + std::to_string(position);
      output << "    " << masked << " = \"arith.select\"(%gf_mask, " << value
             << ", " << (*identity)[position] << ") : (" << tileI1 << ", "
             << tileF32 << ", " << tileF32 << ") -> " << tileF32 << "\n";
      maskedStates.push_back(std::move(masked));
    }
    output << "    %gf_state:" << maskedStates.size() << " = \"tt.reduce\"(";
    llvm::interleaveComma(maskedStates, output);
    output << ") <{axis = 1 : i32}> ({\n    ^bb0(";
    for (size_t position = 0; position < maskedStates.size(); ++position) {
      if (position) output << ", ";
      output << "%gf_left" << position << ": f32";
    }
    for (size_t position = 0; position < maskedStates.size(); ++position)
      output << ", %gf_right" << position << ": f32";
    output << "):\n";
    SmallVector<std::string> combineArguments;
    for (size_t position = 0; position < maskedStates.size(); ++position)
      combineArguments.push_back("%gf_left" + std::to_string(position));
    for (size_t position = 0; position < maskedStates.size(); ++position)
      combineArguments.push_back("%gf_right" + std::to_string(position));
    ScalarAlgebraEmitter combineEmitter(launch, output, "%gf_combine_v");
    auto combined =
        combineEmitter.emit(reducer.getCombine(), combineArguments, "      ");
    if (failed(combined) || combined->size() != maskedStates.size())
      return reject(launch, "multi-row combine arity does not match reducer state ABI");
    output << "      tt.reduce.return ";
    llvm::interleaveComma(*combined, output);
    output << " : ";
    llvm::interleaveComma(*combined, output,
                          [&](const std::string &) { output << "f32"; });
    output << "\n    }) : (";
    llvm::interleaveComma(maskedStates, output,
                          [&](const std::string &) { output << tileF32; });
    output << ") -> (";
    llvm::interleaveComma(maskedStates, output,
                          [&](const std::string &) { output << rowsF32; });
    output << ")\n";
    for (size_t position = 0; position < maskedStates.size(); ++position)
      states.push_back("%gf_state#" + std::to_string(position));
  }
  TensorAlgebraEmitter rowEmitter(launch, output, rowsF32, "%gf_row_v");
  auto finalized = rowEmitter.emit(reducer.getFinalize(), states, "    ");
  if (failed(finalized) || finalized->size() != 1)
    return reject(launch, "multi-row reducer must finalize one result");
  std::string resultValue = finalized->front();
  if (hasNode) {
    SmallVector<std::string> nodeArguments;
    for (int64_t rawPosition : nodeIndices.asArrayRef()) {
      size_t position = static_cast<size_t>(rawPosition);
      StringRef role = cast<StringAttr>(roles[position]).getValue();
      std::string argument = "%" + argumentNames[position];
      if (role == "param") {
        std::string splat = "%gf_node_param" + std::to_string(position);
        output << "    " << splat << " = tt.splat " << argument
               << " : f32 -> " << rowsF32 << "\n";
        nodeArguments.push_back(std::move(splat));
      } else if (role == "dst") {
        std::string base = "%gf_node_base" + std::to_string(position);
        std::string pointer = "%gf_node_ptr" + std::to_string(position);
        std::string loaded = "%gf_node_input" + std::to_string(position);
        output << "    " << base << " = tt.splat " << argument
               << " : !tt.ptr<f32> -> tensor<" << blockM << "x!tt.ptr<f32>>\n"
               << "    " << pointer << " = tt.addptr " << base
               << ", %gf_rows : tensor<" << blockM << "x!tt.ptr<f32>>, "
               << rowsI32 << "\n"
               << "    " << loaded << " = tt.load " << pointer
               << ", " << activeRowMask << ", %gf_zero_rows : tensor<" << blockM
               << "x!tt.ptr<f32>>\n";
        nodeArguments.push_back(std::move(loaded));
      } else {
        return reject(launch, "multi-row node input must be dst or param");
      }
    }
    nodeArguments.push_back(resultValue);
    auto updated = rowEmitter.emit(launch.getRegions()[1], nodeArguments, "    ");
    if (failed(updated) || updated->size() != 1)
      return reject(launch, "multi-row node must yield one result");
    resultValue = updated->front();
  }
  output << "    %gf_out_base = tt.splat %out : !tt.ptr<f32> -> tensor<" << blockM
         << "x!tt.ptr<f32>>\n"
         << "    %gf_out_ptr = tt.addptr %gf_out_base, %gf_rows : tensor<" << blockM
         << "x!tt.ptr<f32>>, " << rowsI32 << "\n"
         << "    tt.store %gf_out_ptr, " << resultValue
         << ", " << activeRowMask << " : tensor<" << blockM << "x!tt.ptr<f32>>\n"
         << "    tt.return\n  }\n}\n";
  return success();
}

static LogicalResult translateCSRScalarAlgebra(
    kernel::LaunchOp launch, ReducerOp reducer, StringRef indexType,
    llvm::raw_ostream &output) {
  Type f32 = Float32Type::get(launch.getContext());
  auto allF32 = [&](ArrayAttr attributes) {
    return llvm::all_of(attributes, [&](Attribute attribute) {
      auto type = dyn_cast<TypeAttr>(attribute);
      return type && type.getValue() == f32;
    });
  };
  bool hasNode = launch.getNumRegions() == 2;
  if (launch.getNumResults() != 1 ||
      (launch.getNumRegions() != 1 && !hasNode) ||
      launch.getRegionKinds().size() != launch.getNumRegions() ||
      launch.getRegionKinds()[0] != 0 ||
      (hasNode && launch.getRegionKinds()[1] != 1))
    return reject(launch, "generic CSR algebra requires one edge region, an "
                          "optional node region, and one output");
  if (!allF32(reducer.getMessageTypes()) ||
      !allF32(reducer.getStateTypes()) ||
      !allF32(reducer.getResultTypes()) ||
      reducer.getResultTypes().size() != 1 ||
      !isRankOneTensor(launch.getResult(0), f32))
    return reject(launch, "generic CSR algebra currently requires scalar "
                          "FP32 message, state, and result types");
  ArrayAttr roles = launch.getInputRolesAttr();
  ArrayAttr names = launch.getInputNamesAttr();
  if (!roles || roles.size() != launch.getInputs().size())
    return reject(launch, "generic CSR algebra requires input_roles");
  if (names && names.size() != launch.getInputs().size())
    return reject(launch, "generic CSR input_names are malformed");

  SmallVector<std::string> argumentNames;
  llvm::StringMap<unsigned> occurrences;
  for (auto [position, attribute] : llvm::enumerate(names ? names : roles)) {
    std::string name = sanitizeIdentifier(
        cast<StringAttr>(attribute).getValue(), position);
    unsigned &occurrence = occurrences[name];
    if (occurrence++) name += "_" + std::to_string(occurrence);
    argumentNames.push_back(std::move(name));
  }
  for (auto [input, roleAttribute] : llvm::zip(launch.getInputs(), roles)) {
    StringRef role = cast<StringAttr>(roleAttribute).getValue();
    if (role == "param") {
      if (input.getType() != f32)
        return reject(launch, "generic CSR scalar parameters must be f32");
    } else if (role == "src" || role == "dst" || role == "edge") {
      if (!isRankOneTensor(input, f32))
        return reject(launch, "generic CSR fields must be tensor<?xf32>");
    } else {
      return reject(launch, "generic CSR input role is unsupported");
    }
  }
  Block &edge = launch.getRegions()[0].front();
  if (edge.getNumArguments() != launch.getInputs().size())
    return reject(launch, "edge region arguments do not match CSR inputs");

  DenseI64ArrayAttr nodeIndices = launch.getNodeInputIndicesAttr();
  DenseI64ArrayAttr nodeSegments = launch.getNodeInputSegmentSizesAttr();
  if (hasNode && (!nodeIndices || !nodeSegments || nodeSegments.size() != 1 ||
                  nodeSegments.asArrayRef()[0] !=
                      static_cast<int64_t>(nodeIndices.size())))
    return reject(launch, "generic CSR node requires its explicit indexed ABI");

  output << "// graphforge.launch entry=gf_csr_scalar_reduce block_rows=1 "
            "num_warps=1 abi=row_ptr,col_idx,";
  for (StringRef name : argumentNames) output << name << ",";
  output << "out\nmodule {\n"
         << "  tt.func public @gf_csr_scalar_reduce("
         << "%row_ptr: !tt.ptr<" << indexType << ">, "
         << "%col_idx: !tt.ptr<" << indexType << ">";
  for (auto [position, input] : llvm::enumerate(launch.getInputs())) {
    output << ", %" << argumentNames[position] << ": "
           << (input.getType() == f32 ? "f32" : "!tt.ptr<f32>");
  }
  output << ", %out: !tt.ptr<f32>) attributes {noinline = false} {\n"
         << "    %gf_c1_i32 = arith.constant 1 : i32\n"
         << "    %gf_c1_index = arith.constant 1 : " << indexType << "\n"
         << "    %gf_row = tt.get_program_id x : i32\n"
         << "    %gf_start_ptr = tt.addptr %row_ptr, %gf_row : !tt.ptr<"
         << indexType << ">, i32\n"
         << "    %gf_start = tt.load %gf_start_ptr : !tt.ptr<" << indexType
         << ">\n"
         << "    %gf_end_ptr = tt.addptr %gf_start_ptr, %gf_c1_i32 : "
            "!tt.ptr<" << indexType << ">, i32\n"
         << "    %gf_end = tt.load %gf_end_ptr : !tt.ptr<" << indexType
         << ">\n";
  ScalarAlgebraEmitter emitter(launch, output);
  FailureOr<SmallVector<std::string>> identity =
      emitter.emit(reducer.getIdentity(), {}, "    ");
  if (failed(identity)) return failure();
  output << "    %gf_state:" << identity->size()
         << " = scf.for %gf_edge = %gf_start to %gf_end step %gf_c1_index "
            "iter_args(";
  for (auto [position, value] : llvm::enumerate(*identity)) {
    if (position) output << ", ";
    output << "%gf_acc" << position << " = " << value;
  }
  output << ") -> (";
  llvm::interleaveComma(*identity, output,
                        [&](const std::string &) { output << "f32"; });
  output << ") : " << indexType << " {\n"
         << "      %gf_col_ptr = tt.addptr %col_idx, %gf_edge : !tt.ptr<"
         << indexType << ">, " << indexType << "\n"
         << "      %gf_src = tt.load %gf_col_ptr : !tt.ptr<" << indexType
         << ">\n";
  SmallVector<std::string> edgeArguments;
  for (auto [position, roleAttribute] : llvm::enumerate(roles)) {
    StringRef role = cast<StringAttr>(roleAttribute).getValue();
    std::string argument = "%" + argumentNames[position];
    if (role == "param") {
      edgeArguments.push_back(std::move(argument));
      continue;
    }
    StringRef coordinate = role == "src" ? "%gf_src" :
                           role == "dst" ? "%gf_row" : "%gf_edge";
    StringRef coordinateType = role == "dst" ? "i32" : indexType;
    std::string pointer = "%gf_input_ptr" + std::to_string(position);
    std::string loaded = "%gf_input" + std::to_string(position);
    output << "      " << pointer << " = tt.addptr " << argument << ", "
           << coordinate << " : !tt.ptr<f32>, " << coordinateType << "\n"
           << "      " << loaded << " = tt.load " << pointer
           << " : !tt.ptr<f32>\n";
    edgeArguments.push_back(std::move(loaded));
  }
  auto message = emitter.emit(launch.getRegions()[0], edgeArguments, "      ");
  if (failed(message)) return failure();
  auto lifted = emitter.emit(reducer.getLift(), *message, "      ");
  if (failed(lifted) || lifted->size() != identity->size())
    return reject(launch, "reducer lift arity does not match state ABI");
  SmallVector<std::string> combineArguments;
  for (size_t position = 0; position < identity->size(); ++position)
    combineArguments.push_back("%gf_acc" + std::to_string(position));
  llvm::append_range(combineArguments, *lifted);
  auto combined = emitter.emit(reducer.getCombine(), combineArguments, "      ");
  if (failed(combined)) return failure();
  output << "      scf.yield ";
  llvm::interleaveComma(*combined, output);
  output << " : ";
  llvm::interleaveComma(*combined, output,
                        [&](const std::string &) { output << "f32"; });
  output << "\n    }\n";
  SmallVector<std::string> finalArguments;
  for (size_t position = 0; position < identity->size(); ++position)
    finalArguments.push_back("%gf_state#" + std::to_string(position));
  auto finalized = emitter.emit(reducer.getFinalize(), finalArguments, "    ");
  if (failed(finalized) || finalized->size() != 1)
    return reject(launch, "generic CSR reducer must finalize one result");
  std::string resultValue = finalized->front();
  if (hasNode) {
    SmallVector<std::string> nodeArguments;
    for (int64_t rawPosition : nodeIndices.asArrayRef()) {
      size_t position = static_cast<size_t>(rawPosition);
      StringRef role = cast<StringAttr>(roles[position]).getValue();
      std::string argument = "%" + argumentNames[position];
      if (role == "param") {
        nodeArguments.push_back(std::move(argument));
      } else if (role == "dst") {
        std::string pointer = "%gf_node_ptr" + std::to_string(position);
        std::string loaded = "%gf_node_input" + std::to_string(position);
        output << "    " << pointer << " = tt.addptr " << argument
               << ", %gf_row : !tt.ptr<f32>, i32\n"
               << "    " << loaded << " = tt.load " << pointer
               << " : !tt.ptr<f32>\n";
        nodeArguments.push_back(std::move(loaded));
      } else {
        return reject(launch, "node indexed inputs must be destination fields "
                              "or scalar parameters");
      }
    }
    nodeArguments.push_back(resultValue);
    auto updated = emitter.emit(launch.getRegions()[1], nodeArguments, "    ");
    if (failed(updated) || updated->size() != 1)
      return reject(launch, "generic CSR node must yield one result");
    resultValue = updated->front();
  }
  output << "    %gf_out_ptr = tt.addptr %out, %gf_row : !tt.ptr<f32>, i32\n"
         << "    tt.store %gf_out_ptr, " << resultValue
         << " : !tt.ptr<f32>\n"
         << "    tt.return\n  }\n}\n";
  return success();
}

static LogicalResult translateKernelToTriton(Operation *root,
                                              llvm::raw_ostream &output) {
  SmallVector<kernel::LaunchOp> launches;
  SmallVector<kernel::GeneratedLaunchOp> generatedLaunches;
  SmallVector<kernel::DenseLaunchOp> denseLaunches;
  root->walk([&](kernel::LaunchOp launch) { launches.push_back(launch); });
  root->walk([&](kernel::GeneratedLaunchOp launch) {
    generatedLaunches.push_back(launch);
  });
  root->walk([&](kernel::DenseLaunchOp launch) {
    denseLaunches.push_back(launch);
  });
  if (launches.size() + generatedLaunches.size() + denseLaunches.size() != 1) {
    root->emitError() << "gf-kernel-to-ttir requires exactly one "
                         "GraphForge Kernel launch, found "
                      << launches.size() + generatedLaunches.size() +
                             denseLaunches.size();
    return failure();
  }
  if (!generatedLaunches.empty())
    return translateGeneratedRadius(generatedLaunches.front(), output);
  if (!denseLaunches.empty()) {
    kernel::DenseLaunchOp launch = denseLaunches.front();
    if (launch.getReducers().size() != 1)
      return reject(launch, "dense TTIR lowering requires one reducer");
    StringRef kind = reducerKind(launch, launch.getReducers()[0]);
    if (kind == "streaming")
      return translateDenseStreaming(launch, output);
    auto reference = dyn_cast<FlatSymbolRefAttr>(launch.getReducers()[0]);
    auto reducer = reference
        ? SymbolTable::lookupNearestSymbolFrom<ReducerOp>(launch, reference)
        : ReducerOp();
    if (!reducer)
      return reject(launch, "generic dense TTIR lowering requires a typed "
                            "gf.reducer symbol");
    return translateDenseScalarAlgebra(launch, reducer, output);
  }

  kernel::LaunchOp launch = launches.front();
  MLIRContext *context = root->getContext();
  Type f32 = Float32Type::get(context);
  Type i32 = IntegerType::get(context, 32);
  Type i64 = IntegerType::get(context, 64);

  if (launch.getTraversal() != "csr-row")
    return reject(launch, "the bootstrap translator only supports csr-row");

  auto rowType = dyn_cast<RankedTensorType>(launch.getRowPtr().getType());
  auto colType = dyn_cast<RankedTensorType>(launch.getColIdx().getType());
  if (!rowType || !colType || rowType.getRank() != 1 ||
      colType.getRank() != 1 ||
      rowType.getElementType() != colType.getElementType() ||
      (rowType.getElementType() != i32 && rowType.getElementType() != i64))
    return reject(launch, "CSR indices must be rank-one i32 or i64 tensors");

  StringRef index = rowType.getElementType() == i64 ? "i64" : "i32";
  auto sourceVector = launch.getInputs().size() == 2
      ? dyn_cast<RankedTensorType>(launch.getInputs()[0].getType())
      : RankedTensorType();
  auto weightVector = launch.getInputs().size() == 2
      ? dyn_cast<RankedTensorType>(launch.getInputs()[1].getType())
      : RankedTensorType();
  auto outputVector = launch.getNumResults() == 1
      ? dyn_cast<RankedTensorType>(launch.getResult(0).getType())
      : RankedTensorType();
  int64_t vectorWidth =
      sourceVector && sourceVector.getRank() == 2 ? sourceVector.getDimSize(1) : 0;
  bool optimizedVectorWeightedSum =
      vectorWidth > 1 && weightVector && outputVector &&
      (weightVector.getRank() == 1 ||
       (weightVector.getRank() == 2 && weightVector.getDimSize(1) == 1)) &&
      outputVector.getRank() == 2 &&
      sourceVector.getElementType() == f32 &&
      weightVector.getElementType() == f32 &&
      outputVector.getElementType() == f32 &&
      outputVector.getDimSize(1) == vectorWidth &&
      launch.getNumRegions() == 1 && launch.getReducers().size() == 1 &&
      reducerKind(launch, launch.getReducers()[0]) == "sum" &&
      launch.getRegionKinds().size() == 1 && launch.getRegionKinds()[0] == 0 &&
      launch.getInputSegmentSizes().size() == 1 &&
      launch.getInputSegmentSizes()[0] == 2 &&
      isVectorScaleMultiplyRegion(launch.getRegions().front(), vectorWidth);
  if (optimizedVectorWeightedSum) {
    IntegerAttr degreeMin = launch.getDegreeMinAttr();
    IntegerAttr degreeMax = launch.getDegreeMaxAttr();
    auto schedule = launch->getAttrOfType<StringAttr>("schedule_kind");
    auto rows = launch->getAttrOfType<IntegerAttr>("block_rows");
    auto neighbors = launch->getAttrOfType<IntegerAttr>("block_neighbors");
    auto warps = launch->getAttrOfType<IntegerAttr>("num_warps");
    bool fixed = degreeMin && degreeMax &&
                 degreeMin.getInt() == degreeMax.getInt();
    bool boundedRagged = degreeMax && degreeMax.getInt() > 0 &&
                         degreeMax.getInt() <= 64 && !fixed;
    bool legalSchedule =
        schedule &&
        ((fixed && schedule.getValue() == "fixed-row-neighbor-feature") ||
         (boundedRagged &&
          schedule.getValue() == "bounded-ragged-row-neighbor-feature"));
    if (launch.getDeterministic() || !degreeMax || degreeMax.getInt() <= 0 ||
        degreeMax.getInt() > 64 || !legalSchedule || !rows || !neighbors ||
        neighbors.getInt() < degreeMax.getInt() || !warps ||
        launch.getNumRowsAttr().getInt() >
            std::numeric_limits<int32_t>::max() / vectorWidth)
      return reject(launch, "vector weighted sum requires a guarded bounded feature tile");
    emitFixedDegreeVectorTTIR(
        output, index, launch.getNumRowsAttr().getInt(), degreeMax.getInt(),
        vectorWidth, rows.getInt(), neighbors.getInt(), warps.getInt(),
        boundedRagged);
    return success();
  }
  bool optimizedWeightedSum =
      launch.getNumResults() == 1 && launch.getNumRegions() == 1 &&
      launch.getInputs().size() == 2 && launch.getReducers().size() == 1 &&
      reducerKind(launch, launch.getReducers()[0]) == "sum" &&
      launch.getRegionKinds().size() == 1 &&
      launch.getRegionKinds()[0] == 0 &&
      launch.getInputSegmentSizes().size() == 1 &&
      launch.getInputSegmentSizes()[0] == 2 &&
      isRankOneTensor(launch.getInputs()[0], f32) &&
      isRankOneTensor(launch.getInputs()[1], f32) &&
      isRankOneTensor(launch.getResult(0), f32) &&
      isScalarMultiplyRegion(launch.getRegions().front(), 2);
  if (!optimizedWeightedSum) {
    if (launch.getReducers().size() > 1) {
      IntegerAttr minimum = launch.getDegreeMinAttr();
      IntegerAttr maximum = launch.getDegreeMaxAttr();
      auto schedule = launch->getAttrOfType<StringAttr>("schedule_kind");
      auto rows = launch->getAttrOfType<IntegerAttr>("block_rows");
      auto neighbors = launch->getAttrOfType<IntegerAttr>("block_neighbors");
      auto warps = launch->getAttrOfType<IntegerAttr>("num_warps");
      if (!maximum || maximum.getInt() <= 0 || maximum.getInt() > 64 ||
          !schedule ||
          (schedule.getValue() != "fixed-row-neighbor" &&
           schedule.getValue() != "bounded-ragged-row-neighbor") ||
          !neighbors || neighbors.getInt() < maximum.getInt())
        return reject(launch, "multi-result CSR product requires a bounded "
                              "row-neighbor schedule");
      if (minimum && minimum.getInt() == maximum.getInt() && rows && warps &&
          schedule.getValue() == "fixed-row-neighbor" &&
          succeeded(emitFixedDegreeProductMultiplyTTIR(
              launch, index, launch.getNumRowsAttr().getInt(),
              maximum.getInt(), rows.getInt(), neighbors.getInt(),
              warps.getInt(), output)))
        return success();
      return translateCSRProductAdditiveTile(
          launch, index, neighbors.getInt(), output);
    }
    if (launch.getReducers().size() != 1)
      return reject(launch, "generic CSR TTIR lowering requires a reducer");
    auto reference = dyn_cast<FlatSymbolRefAttr>(launch.getReducers()[0]);
    auto reducer = reference
        ? SymbolTable::lookupNearestSymbolFrom<ReducerOp>(launch, reference)
        : ReducerOp();
    if (!reducer)
      return reject(launch, "generic CSR TTIR lowering requires a typed "
                            "gf.reducer symbol");
    IntegerAttr maximum = launch.getDegreeMaxAttr();
    auto schedule = launch->getAttrOfType<StringAttr>("schedule_kind");
    auto neighbors = launch->getAttrOfType<IntegerAttr>("block_neighbors");
    auto rows = launch->getAttrOfType<IntegerAttr>("block_rows");
    auto warps = launch->getAttrOfType<IntegerAttr>("num_warps");
    if (maximum && maximum.getInt() > 0 && maximum.getInt() <= 64 &&
        schedule &&
        (schedule.getValue() == "fixed-row-neighbor" ||
         schedule.getValue() == "bounded-ragged-row-neighbor") &&
        neighbors && neighbors.getInt() >= maximum.getInt() &&
        !launch.getDeterministic()) {
      // Batch rows while bounding the live [row, neighbor] tile. Reducer
      // identity/combine semantics are preserved by the generic variadic
      // reduction, so tuple-state monoids use the same schedule as sum.
      bool stableWeighted = isStableWeightedScalarState(reducer);
      int64_t tileRows = stableWeighted
          ? 32
          : rows
              ? std::min<int64_t>(
                    rows.getInt(),
                    std::max<int64_t>(1, 512 / neighbors.getInt()))
              : 1;
      if (tileRows > 1)
        return translateCSRRowsTile(
            launch, reducer, index, tileRows, neighbors.getInt(),
            warps ? warps.getInt() : 1, output);
      return translateCSRAdditiveTile(launch, reducer, index,
                                      neighbors.getInt(), output);
    }
    return translateCSRScalarAlgebra(launch, reducer, index, output);
  }
  if (!isRankOneTensor(launch.getInputs()[0], f32) ||
      !isRankOneTensor(launch.getInputs()[1], f32) ||
      !isRankOneTensor(launch.getResult(0), f32))
    return reject(launch, "the bootstrap ABI requires scalar FP32 fields");

  if (!isScalarMultiplyRegion(launch.getRegions().front(), 2))
    return reject(launch, "edge region is not weight * source");

  IntegerAttr degreeMin = launch.getDegreeMinAttr();
  IntegerAttr degreeMax = launch.getDegreeMaxAttr();
  auto scheduleKind = launch->getAttrOfType<StringAttr>("schedule_kind");
  auto scheduledRows = launch->getAttrOfType<IntegerAttr>("block_rows");
  auto scheduledNeighbors =
      launch->getAttrOfType<IntegerAttr>("block_neighbors");
  auto scheduledWarps = launch->getAttrOfType<IntegerAttr>("num_warps");
  auto scheduleValue = [&](IntegerAttr attribute, int64_t fallback) {
    return attribute ? attribute.getInt() : fallback;
  };
  if (!launch.getDeterministic() && degreeMin && degreeMax &&
      degreeMin.getInt() == degreeMax.getInt() &&
      degreeMin.getInt() > 0 && degreeMin.getInt() <= 64 &&
      launch.getNumRowsAttr().getInt() <=
          std::numeric_limits<int32_t>::max() / degreeMin.getInt()) {
    if (scheduleKind && scheduleKind.getValue() != "fixed-row-neighbor")
      return reject(launch, "fixed-degree launch has an incompatible schedule");
    int64_t blockRows = scheduleValue(scheduledRows, 16);
    int64_t blockNeighbors = scheduleValue(
        scheduledNeighbors, nextPowerOfTwo(degreeMin.getInt()));
    int64_t numWarps = scheduleValue(scheduledWarps, 1);
    if (blockRows <= 0 || blockNeighbors < degreeMin.getInt() || numWarps <= 0)
      return reject(launch, "fixed-degree schedule has invalid tile bounds");
    emitFixedDegreeTTIR(output, index, launch.getNumRowsAttr().getInt(),
                        degreeMin.getInt(), blockRows, blockNeighbors,
                        numWarps);
    return success();
  }
  if (!launch.getDeterministic() && degreeMax && degreeMax.getInt() > 0 &&
      degreeMax.getInt() <= 64 &&
      launch.getNumRowsAttr().getInt() <= std::numeric_limits<int32_t>::max()) {
    if (scheduleKind &&
        scheduleKind.getValue() != "bounded-ragged-row-neighbor")
      return reject(launch, "ragged launch has an incompatible schedule");
    bool wide = degreeMax.getInt() > 32;
    int64_t blockRows = scheduleValue(scheduledRows, wide ? 32 : 16);
    int64_t blockNeighbors = scheduleValue(
        scheduledNeighbors, nextPowerOfTwo(degreeMax.getInt()));
    int64_t numWarps = scheduleValue(scheduledWarps, wide ? 1 : 4);
    if (blockRows <= 0 || blockNeighbors < degreeMax.getInt() || numWarps <= 0)
      return reject(launch, "ragged schedule has invalid tile bounds");
    emitBoundedRaggedTTIR(output, index, launch.getNumRowsAttr().getInt(),
                          blockRows, blockNeighbors, numWarps);
    return success();
  }
  output << "// graphforge.launch entry=gf_csr_weighted_sum block_rows=1 "
            "num_warps=4 abi=row_ptr,col_idx,x,weight,out\n"
         << "module {\n"
         << "  tt.func public @gf_csr_weighted_sum("
         << "%row_ptr: !tt.ptr<" << index << ">, "
         << "%col_idx: !tt.ptr<" << index << ">, "
         << "%x: !tt.ptr<f32>, %weight: !tt.ptr<f32>, "
         << "%out: !tt.ptr<f32>) attributes {noinline = false} {\n"
         << "    %c1_index = arith.constant 1 : " << index << "\n"
         << "    %zero = arith.constant 0.000000e+00 : f32\n"
         << "    %c1_i32 = arith.constant 1 : i32\n"
         << "    %row = tt.get_program_id x : i32\n"
         << "    %start_ptr = tt.addptr %row_ptr, %row : !tt.ptr<"
         << index << ">, i32\n"
         << "    %start = tt.load %start_ptr : !tt.ptr<" << index << ">\n"
         << "    %end_ptr = tt.addptr %start_ptr, %c1_i32 : !tt.ptr<"
         << index << ">, i32\n"
         << "    %end = tt.load %end_ptr : !tt.ptr<" << index << ">\n"
         << "    %sum = scf.for %edge = %start to %end step %c1_index "
            "iter_args(%acc = %zero) -> (f32) : "
         << index << " {\n"
         << "      %col_ptr = tt.addptr %col_idx, %edge : !tt.ptr<"
         << index << ">, " << index << "\n"
         << "      %src = tt.load %col_ptr : !tt.ptr<" << index << ">\n"
         << "      %x_ptr = tt.addptr %x, %src : !tt.ptr<f32>, " << index
         << "\n"
         << "      %x_value = tt.load %x_ptr : !tt.ptr<f32>\n"
         << "      %w_ptr = tt.addptr %weight, %edge : !tt.ptr<f32>, "
         << index << "\n"
         << "      %w_value = tt.load %w_ptr : !tt.ptr<f32>\n"
         << "      %message = arith.mulf %x_value, %w_value : f32\n"
         << "      %next = arith.addf %acc, %message : f32\n"
         << "      scf.yield %next : f32\n"
         << "    }\n"
         << "    %out_ptr = tt.addptr %out, %row : !tt.ptr<f32>, i32\n"
         << "    tt.store %out_ptr, %sum : !tt.ptr<f32>\n"
         << "    tt.return\n"
         << "  }\n"
         << "}\n";
  return success();
}

} // namespace

void registerKernelToTritonTranslation() {
  TranslateFromMLIRRegistration(
      "gf-kernel-to-ttir",
      "lower a supported GraphForge Kernel module to serialized Triton IR",
      translateKernelToTriton, [](DialectRegistry &registry) {
        registry.insert<GraphForgeDomainDialect,
                        kernel::GraphForgeKernelDialect,
                        arith::ArithDialect, func::FuncDialect,
                        math::MathDialect, vector::VectorDialect>();
      });
}

namespace {

static LogicalResult translateTaskToBundle(Operation *root,
                                           llvm::raw_ostream &output) {
  SmallVector<task::DegreeWorklistOp> worklists;
  SmallVector<task::DegreeHistogramOp> histograms;
  SmallVector<task::DegreeResetOp> resets;
  SmallVector<task::DegreePrefixOp> prefixes;
  SmallVector<task::DegreeScatterOp> scatters;
  SmallVector<task::DegreeBucketLaunchOp> buckets;
  SmallVector<task::RowSplitPartialOp> splitPartials;
  SmallVector<task::RowSplitFinalizeOp> splitFinalizes;
  SmallVector<task::LaunchOp> launches;
  SmallVector<task::HaloPackOp> haloPacks;
  SmallVector<task::HaloExchangeOp> haloExchanges;
  SmallVector<task::HaloUnpackOp> haloUnpacks;
  SmallVector<storage::TransferOp> transfers;
  SmallVector<storage::ReleaseOp> releases;
  SmallVector<storage::JoinOp> joins;
  root->walk([&](task::DegreeWorklistOp op) { worklists.push_back(op); });
  root->walk([&](task::DegreeHistogramOp op) { histograms.push_back(op); });
  root->walk([&](task::DegreeResetOp op) { resets.push_back(op); });
  root->walk([&](task::DegreePrefixOp op) { prefixes.push_back(op); });
  root->walk([&](task::DegreeScatterOp op) { scatters.push_back(op); });
  root->walk([&](task::DegreeBucketLaunchOp op) { buckets.push_back(op); });
  root->walk([&](task::RowSplitPartialOp op) { splitPartials.push_back(op); });
  root->walk([&](task::RowSplitFinalizeOp op) { splitFinalizes.push_back(op); });
  root->walk([&](task::LaunchOp op) { launches.push_back(op); });
  root->walk([&](task::HaloPackOp op) { haloPacks.push_back(op); });
  root->walk([&](task::HaloExchangeOp op) { haloExchanges.push_back(op); });
  root->walk([&](task::HaloUnpackOp op) { haloUnpacks.push_back(op); });
  root->walk([&](storage::TransferOp op) { transfers.push_back(op); });
  root->walk([&](storage::ReleaseOp op) { releases.push_back(op); });
  root->walk([&](storage::JoinOp op) { joins.push_back(op); });
  if (worklists.empty() && resets.empty() && histograms.empty() && prefixes.empty() &&
      scatters.empty() && buckets.empty() && splitPartials.empty() &&
      splitFinalizes.empty() && launches.empty() && haloPacks.empty() &&
      haloExchanges.empty() && haloUnpacks.empty() && transfers.empty() && releases.empty() &&
      joins.empty())
    return root->emitError("gf-task-to-bundle found no executable task plan");

  llvm::DenseMap<Operation *, std::string> names;
  for (auto [index, op] : llvm::enumerate(worklists))
    names[op.getOperation()] = "degree-worklist:" + std::to_string(index);
  for (auto [index, op] : llvm::enumerate(histograms))
    names[op.getOperation()] = "degree-histogram:" + std::to_string(index);
  for (auto [index, op] : llvm::enumerate(resets))
    names[op.getOperation()] = "degree-reset:" + std::to_string(index);
  for (auto [index, op] : llvm::enumerate(prefixes))
    names[op.getOperation()] = "degree-prefix:" + std::to_string(index);
  for (auto [index, op] : llvm::enumerate(scatters))
    names[op.getOperation()] = "degree-scatter:" + std::to_string(index);
  for (auto [index, op] : llvm::enumerate(buckets))
    names[op.getOperation()] = "degree-bucket:" + std::to_string(index);
  for (auto [index, op] : llvm::enumerate(splitPartials))
    names[op.getOperation()] = "row-split-partial:" + std::to_string(index);
  for (auto [index, op] : llvm::enumerate(splitFinalizes))
    names[op.getOperation()] = "row-split-finalize:" + std::to_string(index);
  for (auto [index, op] : llvm::enumerate(launches))
    names[op.getOperation()] = "task:" + std::to_string(index);
  for (auto [index, op] : llvm::enumerate(haloPacks))
    names[op.getOperation()] = "halo-pack:" + std::to_string(index);
  for (auto [index, op] : llvm::enumerate(haloExchanges))
    names[op.getOperation()] = "halo-exchange:" + std::to_string(index);
  for (auto [index, op] : llvm::enumerate(haloUnpacks))
    names[op.getOperation()] = "halo-unpack:" + std::to_string(index);
  for (auto [index, op] : llvm::enumerate(transfers))
    names[op.getOperation()] = "transfer:" + std::to_string(index);
  for (auto [index, op] : llvm::enumerate(releases))
    names[op.getOperation()] = "release:" + std::to_string(index);
  for (auto [index, op] : llvm::enumerate(joins))
    names[op.getOperation()] = "join:" + std::to_string(index);

  auto dependencies = [&](ValueRange events) {
    llvm::json::Array result;
    for (Value event : events) {
      auto found = names.find(event.getDefiningOp());
      if (found != names.end())
        result.push_back(found->second);
    }
    return result;
  };
  auto access = [](StringRef binding, StringRef mode, int64_t version,
                   StringRef partitioning = {}, StringRef partition = {}) {
    llvm::json::Object result{{"binding", binding.str()},
                              {"mode", mode},
                              {"snapshot_version", version}};
    if (!partitioning.empty()) {
      result["partitioning"] = partitioning;
      result["partition"] = partition;
    }
    return result;
  };
  auto indexType = [](Value relationValue) -> StringRef {
    auto relation = relationValue.getDefiningOp<RelationOp>();
    if (!relation) return {};
    auto tensor = dyn_cast<RankedTensorType>(relation.getRowPtr().getType());
    if (!tensor) return {};
    auto integer = dyn_cast<IntegerType>(tensor.getElementType());
    if (!integer) return {};
    return integer.getWidth() == 64 ? StringRef("i64")
                                    : integer.getWidth() == 32
                                          ? StringRef("i32") : StringRef();
  };
  auto resetTTIR = [](int64_t buckets) {
    std::string text;
    llvm::raw_string_ostream o(text);
    int64_t elements = 2 * buckets + 1;
    int64_t block = nextPowerOfTwo(elements);
    o << "// graphforge.launch entry=gf_degree_reset block_rows=" << block
      << " num_warps=1 abi=row_worklist\nmodule {\n"
      << "  tt.func public @gf_degree_reset(%row_worklist: !tt.ptr<i64>) "
         "attributes {noinline = false} {\n"
      << "    %zero = arith.constant dense<0> : tensor<" << block << "xi64>\n"
      << "    %limit = arith.constant dense<" << elements << "> : tensor<"
      << block << "xi32>\n"
      << "    %range = tt.make_range {end = " << block
      << " : i32, start = 0 : i32} : tensor<" << block << "xi32>\n"
      << "    %mask = arith.cmpi slt, %range, %limit : tensor<" << block
      << "xi32>\n"
      << "    %base = tt.splat %row_worklist : !tt.ptr<i64> -> tensor<"
      << block << "x!tt.ptr<i64>>\n"
      << "    %ptr = tt.addptr %base, %range : tensor<" << block
      << "x!tt.ptr<i64>>, tensor<" << block << "xi32>\n"
      << "    tt.store %ptr, %zero, %mask : tensor<" << block
      << "x!tt.ptr<i64>>\n    tt.return\n  }\n}\n";
    return text;
  };
  auto bucketExpression = [](llvm::raw_ostream &o, StringRef degree,
                             StringRef index,
                             ArrayRef<int64_t> bounds) -> std::string {
    o << "    %gf_bucket_last = arith.constant " << bounds.size() - 1
      << " : i32\n";
    std::string current = "%gf_bucket_last";
    for (int64_t ordinal = static_cast<int64_t>(bounds.size()) - 2;
         ordinal >= 0; --ordinal) {
      o << "    %gf_bound" << ordinal << " = arith.constant "
        << bounds[ordinal] << " : " << index << "\n"
        << "    %gf_in_bucket" << ordinal << " = arith.cmpi sle, " << degree
        << ", %gf_bound" << ordinal << " : " << index << "\n"
        << "    %gf_ordinal" << ordinal << " = arith.constant " << ordinal
        << " : i32\n"
        << "    %gf_bucket" << ordinal << " = arith.select %gf_in_bucket"
        << ordinal << ", %gf_ordinal" << ordinal << ", " << current
        << " : i32\n";
      current = "%gf_bucket" + std::to_string(ordinal);
    }
    return current;
  };
  auto histogramTTIR = [&](StringRef index, int64_t rows,
                           ArrayRef<int64_t> bounds) {
    std::string text;
    llvm::raw_string_ostream o(text);
    o << "// graphforge.launch entry=gf_degree_histogram block_rows=1 "
         "num_warps=1 abi=row_ptr,row_worklist\nmodule {\n"
      << "  tt.func public @gf_degree_histogram(%row_ptr: !tt.ptr<" << index
      << ">, %row_worklist: !tt.ptr<i64>) attributes {noinline = false} {\n"
      << "    %one_i32 = arith.constant 1 : i32\n"
      << "    %one_i64 = arith.constant 1 : i64\n"
      << "    %true = arith.constant true\n"
      << "    %row = tt.get_program_id x : i32\n"
      << "    %start_ptr = tt.addptr %row_ptr, %row : !tt.ptr<" << index
      << ">, i32\n"
      << "    %start = tt.load %start_ptr : !tt.ptr<" << index << ">\n"
      << "    %next = arith.addi %row, %one_i32 : i32\n"
      << "    %end_ptr = tt.addptr %row_ptr, %next : !tt.ptr<" << index
      << ">, i32\n"
      << "    %end = tt.load %end_ptr : !tt.ptr<" << index << ">\n"
      << "    %degree = arith.subi %end, %start : " << index << "\n";
    std::string bucket = bucketExpression(o, "%degree", index, bounds);
    o << "    %count_ptr = tt.addptr %row_worklist, " << bucket
      << " : !tt.ptr<i64>, i32\n"
      << "    %old = tt.atomic_rmw add, acq_rel, gpu, %count_ptr, %one_i64, "
         "%true : (!tt.ptr<i64>, i64, i1) -> i64\n"
      << "    tt.return\n  }\n}\n";
    (void)rows;
    return text;
  };
  auto prefixTTIR = [](int64_t buckets) {
    std::string text;
    llvm::raw_string_ostream o(text);
    o << "// graphforge.launch entry=gf_degree_prefix block_rows=1 num_warps=1 "
         "abi=row_worklist\nmodule {\n"
      << "  tt.func public @gf_degree_prefix(%row_worklist: !tt.ptr<i64>) "
         "attributes {noinline = false} {\n"
      << "    %zero = arith.constant 0 : i64\n"
      << "    %one = arith.constant 1 : index\n"
      << "    %one_i32 = arith.constant 1 : i32\n"
      << "    %begin = arith.constant 0 : index\n"
      << "    %end = arith.constant " << buckets << " : index\n"
      << "    %cursor_base = arith.constant " << buckets + 1 << " : index\n"
      << "    %total = scf.for %i = %begin to %end step %one "
         "iter_args(%prefix = %zero) -> (i64) {\n"
      << "      %i32 = arith.index_cast %i : index to i32\n"
      << "      %count_ptr = tt.addptr %row_worklist, %i32 : !tt.ptr<i64>, i32\n"
      << "      %count = tt.load %count_ptr : !tt.ptr<i64>\n"
      << "      %offset_ptr = tt.addptr %count_ptr, %one_i32 : !tt.ptr<i64>, i32\n"
      << "      %cursor_i = arith.addi %cursor_base, %i : index\n"
      << "      %cursor_i32 = arith.index_cast %cursor_i : index to i32\n"
      << "      %cursor_ptr = tt.addptr %row_worklist, %cursor_i32 : "
         "!tt.ptr<i64>, i32\n"
      << "      tt.store %count_ptr, %prefix : !tt.ptr<i64>\n"
      << "      tt.store %cursor_ptr, %prefix : !tt.ptr<i64>\n"
      << "      %next = arith.addi %prefix, %count : i64\n"
      << "      scf.yield %next : i64\n    }\n"
      << "    %last = arith.constant " << buckets << " : i32\n"
      << "    %last_ptr = tt.addptr %row_worklist, %last : !tt.ptr<i64>, i32\n"
      << "    tt.store %last_ptr, %total : !tt.ptr<i64>\n"
      << "    tt.return\n  }\n}\n";
    return text;
  };
  auto scatterTTIR = [&](StringRef index, int64_t rows,
                         ArrayRef<int64_t> bounds) {
    std::string text;
    llvm::raw_string_ostream o(text);
    int64_t buckets = bounds.size();
    o << "// graphforge.launch entry=gf_degree_scatter block_rows=1 num_warps=1 "
         "abi=row_ptr,row_worklist\nmodule {\n"
      << "  tt.func public @gf_degree_scatter(%row_ptr: !tt.ptr<" << index
      << ">, %row_worklist: !tt.ptr<i64>) attributes {noinline = false} {\n"
      << "    %one_i32 = arith.constant 1 : i32\n"
      << "    %one_i64 = arith.constant 1 : i64\n"
      << "    %true = arith.constant true\n"
      << "    %row = tt.get_program_id x : i32\n"
      << "    %start_ptr = tt.addptr %row_ptr, %row : !tt.ptr<" << index
      << ">, i32\n"
      << "    %start = tt.load %start_ptr : !tt.ptr<" << index << ">\n"
      << "    %next = arith.addi %row, %one_i32 : i32\n"
      << "    %end_ptr = tt.addptr %row_ptr, %next : !tt.ptr<" << index
      << ">, i32\n"
      << "    %end = tt.load %end_ptr : !tt.ptr<" << index << ">\n"
      << "    %degree = arith.subi %end, %start : " << index << "\n";
    std::string bucket = bucketExpression(o, "%degree", index, bounds);
    o << "    %cursor_base = arith.constant " << buckets + 1 << " : i32\n"
      << "    %cursor_index = arith.addi %cursor_base, " << bucket << " : i32\n"
      << "    %cursor_ptr = tt.addptr %row_worklist, %cursor_index : "
         "!tt.ptr<i64>, i32\n"
      << "    %position = tt.atomic_rmw add, acq_rel, gpu, %cursor_ptr, "
         "%one_i64, %true : (!tt.ptr<i64>, i64, i1) -> i64\n"
      << "    %row_base = arith.constant " << 2 * buckets + 1 << " : i64\n"
      << "    %slot = arith.addi %row_base, %position : i64\n"
      << "    %row64 = arith.extui %row : i32 to i64\n"
      << "    %row_ptr_out = tt.addptr %row_worklist, %slot : "
         "!tt.ptr<i64>, i64\n"
      << "    tt.store %row_ptr_out, %row64 : !tt.ptr<i64>\n"
      << "    tt.return\n  }\n}\n";
    (void)rows;
    return text;
  };

  auto splitPartialTTIR = [&](task::RowSplitPartialOp op,
                              kernel::LaunchOp launch, ReducerOp reducer,
                              StringRef index) -> FailureOr<std::string> {
    constexpr int64_t blockD = 64;
    if (op.getChunkEdgesAttr().getInt() != blockD ||
        !isZeroAdditiveState(reducer))
      return failure();
    ArrayAttr roles = launch.getInputRolesAttr();
    ArrayAttr names = launch.getInputNamesAttr();
    if (!roles || roles.size() != launch.getInputs().size())
      return failure();
    SmallVector<std::string> argumentNames;
    llvm::StringMap<unsigned> occurrences;
    for (auto [position, attribute] : llvm::enumerate(names ? names : roles)) {
      std::string name = sanitizeIdentifier(
          cast<StringAttr>(attribute).getValue(), position);
      unsigned &occurrence = occurrences[name];
      if (occurrence++) name += "_" + std::to_string(occurrence);
      argumentNames.push_back(std::move(name));
    }
    std::string tileIndex = "tensor<64x" + index.str() + ">";
    std::string text;
    llvm::raw_string_ostream o(text);
    o << "// graphforge.launch entry=gf_row_split_partial block_rows=1 "
         "num_warps=1 abi=row_ptr,col_idx,";
    for (StringRef name : argumentNames) o << name << ",";
    o << "partials,row_worklist\nmodule {\n  tt.func public @gf_row_split_partial("
      << "%row_ptr: !tt.ptr<" << index << ">, %col_idx: !tt.ptr<" << index
      << ">";
    Type f32 = Float32Type::get(launch.getContext());
    for (auto [position, input] : llvm::enumerate(launch.getInputs()))
      o << ", %" << argumentNames[position] << ": "
        << (input.getType() == f32 ? "f32" : "!tt.ptr<f32>");
    o << ", %partials: !tt.ptr<f32>, %row_worklist: !tt.ptr<i64>) "
         "attributes {noinline = false} {\n"
      << "    %gf_zero_index = arith.constant dense<0> : " << tileIndex << "\n"
      << "    %gf_zero = arith.constant dense<0.000000e+00> : tensor<64xf32>\n"
      << "    %gf_partial_count = arith.constant "
      << op.getPartialsPerRowAttr().getInt() << " : i32\n"
      << "    %gf_chunk = tt.get_program_id x : i32\n"
      << "    %gf_active_row = arith.divsi %gf_chunk, %gf_partial_count : i32\n"
      << "    %gf_partial = arith.remsi %gf_chunk, %gf_partial_count : i32\n"
      << "    %gf_bucket = arith.constant " << op.getBucketOrdinalAttr().getInt()
      << " : i32\n"
      << "    %gf_bucket_start_ptr = tt.addptr %row_worklist, %gf_bucket : !tt.ptr<i64>, i32\n"
      << "    %gf_bucket_start = tt.load %gf_bucket_start_ptr : !tt.ptr<i64>\n"
      << "    %gf_row_base = arith.constant " << 2 * op.getBucketsAttr().getInt() + 1
      << " : i64\n"
      << "    %gf_active_row64 = arith.extui %gf_active_row : i32 to i64\n"
      << "    %gf_row_slot0 = arith.addi %gf_row_base, %gf_bucket_start : i64\n"
      << "    %gf_row_slot = arith.addi %gf_row_slot0, %gf_active_row64 : i64\n"
      << "    %gf_row_id_ptr = tt.addptr %row_worklist, %gf_row_slot : !tt.ptr<i64>, i64\n"
      << "    %gf_row64 = tt.load %gf_row_id_ptr : !tt.ptr<i64>\n"
      << "    %gf_row = arith.trunci %gf_row64 : i64 to i32\n"
      << "    %gf_one = arith.constant 1 : i32\n"
      << "    %gf_start_ptr = tt.addptr %row_ptr, %gf_row : !tt.ptr<" << index
      << ">, i32\n"
      << "    %gf_row_start = tt.load %gf_start_ptr : !tt.ptr<" << index << ">\n"
      << "    %gf_next = arith.addi %gf_row, %gf_one : i32\n"
      << "    %gf_end_ptr = tt.addptr %row_ptr, %gf_next : !tt.ptr<" << index
      << ">, i32\n"
      << "    %gf_row_end = tt.load %gf_end_ptr : !tt.ptr<" << index << ">\n"
      << "    %gf_chunk_size = arith.constant 64 : " << index << "\n";
    if (index == "i64")
      o << "    %gf_partial_index = arith.extui %gf_partial : i32 to i64\n"
        << "    %gf_chunk_offset = arith.muli %gf_partial_index, %gf_chunk_size : i64\n";
    else
      o << "    %gf_chunk_offset = arith.muli %gf_partial, %gf_chunk_size : i32\n";
    o << "    %gf_start = arith.addi %gf_row_start, %gf_chunk_offset : "
      << index << "\n"
      << "    %gf_lane_i32 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>\n";
    if (index == "i64")
      o << "    %gf_lane = arith.extsi %gf_lane_i32 : tensor<64xi32> to tensor<64xi64>\n";
    else
      o << "    %gf_lane = arith.bitcast %gf_lane_i32 : tensor<64xi32> to tensor<64xi32>\n";
    o << "    %gf_start_v = tt.splat %gf_start : " << index << " -> "
      << tileIndex << "\n"
      << "    %gf_edges = arith.addi %gf_start_v, %gf_lane : " << tileIndex << "\n"
      << "    %gf_end_v = tt.splat %gf_row_end : " << index << " -> "
      << tileIndex << "\n"
      << "    %gf_mask = arith.cmpi slt, %gf_edges, %gf_end_v : " << tileIndex << "\n"
      << "    %gf_col_base = tt.splat %col_idx : !tt.ptr<" << index
      << "> -> tensor<64x!tt.ptr<" << index << ">>\n"
      << "    %gf_col_ptr = tt.addptr %gf_col_base, %gf_edges : tensor<64x!tt.ptr<"
      << index << ">>, " << tileIndex << "\n"
      << "    %gf_src = tt.load %gf_col_ptr, %gf_mask, %gf_zero_index : tensor<64x!tt.ptr<"
      << index << ">>\n";
    SmallVector<std::string> edgeArguments;
    for (auto [position, roleAttribute] : llvm::enumerate(roles)) {
      StringRef role = cast<StringAttr>(roleAttribute).getValue();
      std::string argument = "%" + argumentNames[position];
      if (role == "param") {
        std::string splat = "%gf_param" + std::to_string(position);
        o << "    " << splat << " = tt.splat " << argument
          << " : f32 -> tensor<64xf32>\n";
        edgeArguments.push_back(std::move(splat));
        continue;
      }
      std::string base = "%gf_input_base" + std::to_string(position);
      std::string pointer = "%gf_input_ptr" + std::to_string(position);
      std::string loaded = "%gf_input" + std::to_string(position);
      o << "    " << base << " = tt.splat " << argument
        << " : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>\n";
      if (role == "src")
        o << "    " << pointer << " = tt.addptr " << base
          << ", %gf_src : tensor<64x!tt.ptr<f32>>, " << tileIndex << "\n";
      else if (role == "edge")
        o << "    " << pointer << " = tt.addptr " << base
          << ", %gf_edges : tensor<64x!tt.ptr<f32>>, " << tileIndex << "\n";
      else if (role == "dst") {
        std::string row = "%gf_row_v" + std::to_string(position);
        o << "    " << row << " = tt.splat %gf_row : i32 -> tensor<64xi32>\n"
          << "    " << pointer << " = tt.addptr " << base << ", " << row
          << " : tensor<64x!tt.ptr<f32>>, tensor<64xi32>\n";
      } else return failure();
      o << "    " << loaded << " = tt.load " << pointer
        << ", %gf_mask, %gf_zero : tensor<64x!tt.ptr<f32>>\n";
      edgeArguments.push_back(std::move(loaded));
    }
    TensorAlgebraEmitter emitter(launch, o, "tensor<64xf32>", "%gf_split_v");
    auto message = emitter.emit(launch.getRegions()[0], edgeArguments, "    ");
    if (failed(message)) return failure();
    auto lifted = emitter.emit(reducer.getLift(), *message, "    ");
    if (failed(lifted) || lifted->size() != 1) return failure();
    o << "    %gf_masked = \"arith.select\"(%gf_mask, " << lifted->front()
      << ", %gf_zero) : (tensor<64xi1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>\n"
      << "    %gf_state = \"tt.reduce\"(%gf_masked) <{axis = 0 : i32}> ({\n"
      << "    ^bb0(%gf_a: f32, %gf_b: f32):\n"
      << "      %gf_sum = arith.addf %gf_a, %gf_b : f32\n"
      << "      tt.reduce.return %gf_sum : f32\n"
      << "    }) : (tensor<64xf32>) -> f32\n"
      << "    %gf_partial_ptr = tt.addptr %partials, %gf_chunk : !tt.ptr<f32>, i32\n"
      << "    tt.store %gf_partial_ptr, %gf_state : !tt.ptr<f32>\n"
      << "    tt.return\n  }\n}\n";
    o.flush();
    return text;
  };

  auto splitFinalizeTTIR = [&](task::RowSplitFinalizeOp op,
                               kernel::LaunchOp launch,
                               ReducerOp reducer) -> FailureOr<std::string> {
    int64_t partials = op.getPartialsPerRowAttr().getInt();
    int64_t block = nextPowerOfTwo(partials);
    if (block <= 1 || block > 1024 || !isZeroAdditiveState(reducer) ||
        launch.getNumRegions() != 1)
      return failure();
    std::string text;
    llvm::raw_string_ostream o(text);
    o << "// graphforge.launch entry=gf_row_split_finalize block_rows=1 "
         "num_warps=1 abi=partials,row_worklist,out\nmodule {\n"
      << "  tt.func public @gf_row_split_finalize(%partials: !tt.ptr<f32>, "
         "%row_worklist: !tt.ptr<i64>, %out: !tt.ptr<f32>) attributes {noinline = false} {\n"
      << "    %gf_active_row = tt.get_program_id x : i32\n"
      << "    %gf_bucket = arith.constant " << op.getBucketOrdinalAttr().getInt()
      << " : i32\n"
      << "    %gf_bucket_start_ptr = tt.addptr %row_worklist, %gf_bucket : !tt.ptr<i64>, i32\n"
      << "    %gf_bucket_start = tt.load %gf_bucket_start_ptr : !tt.ptr<i64>\n"
      << "    %gf_row_base = arith.constant " << 2 * op.getBucketsAttr().getInt() + 1
      << " : i64\n"
      << "    %gf_active_row64 = arith.extui %gf_active_row : i32 to i64\n"
      << "    %gf_row_slot0 = arith.addi %gf_row_base, %gf_bucket_start : i64\n"
      << "    %gf_row_slot = arith.addi %gf_row_slot0, %gf_active_row64 : i64\n"
      << "    %gf_row_id_ptr = tt.addptr %row_worklist, %gf_row_slot : !tt.ptr<i64>, i64\n"
      << "    %gf_row64 = tt.load %gf_row_id_ptr : !tt.ptr<i64>\n"
      << "    %gf_row = arith.trunci %gf_row64 : i64 to i32\n"
      << "    %gf_partial_count = arith.constant " << partials << " : i32\n"
      << "    %gf_base = arith.muli %gf_active_row, %gf_partial_count : i32\n"
      << "    %gf_lane = tt.make_range {end = " << block
      << " : i32, start = 0 : i32} : tensor<" << block << "xi32>\n"
      << "    %gf_limit = arith.constant dense<" << partials << "> : tensor<"
      << block << "xi32>\n"
      << "    %gf_mask = arith.cmpi slt, %gf_lane, %gf_limit : tensor<"
      << block << "xi32>\n"
      << "    %gf_base_v = tt.splat %gf_base : i32 -> tensor<" << block
      << "xi32>\n"
      << "    %gf_slots = arith.addi %gf_base_v, %gf_lane : tensor<" << block
      << "xi32>\n"
      << "    %gf_ptr_base = tt.splat %partials : !tt.ptr<f32> -> tensor<"
      << block << "x!tt.ptr<f32>>\n"
      << "    %gf_ptr = tt.addptr %gf_ptr_base, %gf_slots : tensor<" << block
      << "x!tt.ptr<f32>>, tensor<" << block << "xi32>\n"
      << "    %gf_zero = arith.constant dense<0.000000e+00> : tensor<" << block
      << "xf32>\n"
      << "    %gf_values = tt.load %gf_ptr, %gf_mask, %gf_zero : tensor<"
      << block << "x!tt.ptr<f32>>\n"
      << "    %gf_state = \"tt.reduce\"(%gf_values) <{axis = 0 : i32}> ({\n"
      << "    ^bb0(%gf_a: f32, %gf_b: f32):\n"
      << "      %gf_sum = arith.addf %gf_a, %gf_b : f32\n"
      << "      tt.reduce.return %gf_sum : f32\n"
      << "    }) : (tensor<" << block << "xf32>) -> f32\n";
    ScalarAlgebraEmitter emitter(launch, o);
    auto finalized = emitter.emit(
        reducer.getFinalize(), ArrayRef<std::string>{"%gf_state"}, "    ");
    if (failed(finalized) || finalized->size() != 1)
      return failure();
    o << "    %gf_out_ptr = tt.addptr %out, %gf_row : !tt.ptr<f32>, i32\n"
      << "    tt.store %gf_out_ptr, " << finalized->front()
      << " : !tt.ptr<f32>\n"
      << "    tt.return\n  }\n}\n";
    o.flush();
    return text;
  };

  auto chunkedBucketTTIR = [&](task::DegreeBucketLaunchOp op,
                               kernel::LaunchOp launch, ReducerOp reducer,
                               StringRef index,
                               int64_t buckets) -> FailureOr<std::string> {
    if (!isZeroAdditiveState(reducer) || launch.getNumRegions() != 1 ||
        buckets <= 0)
      return failure();
    ArrayAttr roles = launch.getInputRolesAttr();
    ArrayAttr inputNames = launch.getInputNamesAttr();
    if (!roles || roles.size() != launch.getInputs().size())
      return failure();
    Type f32 = Float32Type::get(launch.getContext());
    SmallVector<std::string> argumentNames;
    llvm::StringMap<unsigned> occurrences;
    for (auto [position, attribute] :
         llvm::enumerate(inputNames ? inputNames : roles)) {
      std::string name = sanitizeIdentifier(
          cast<StringAttr>(attribute).getValue(), position);
      unsigned &occurrence = occurrences[name];
      if (occurrence++) name += "_" + std::to_string(occurrence);
      argumentNames.push_back(std::move(name));
    }
    std::string tileIndex = "tensor<64x" + index.str() + ">";
    std::string text;
    llvm::raw_string_ostream o(text);
    std::string entry = "gf_csr_additive_bucket_chunked_" +
                        std::to_string(op.getBucketOrdinalAttr().getInt());
    o << "// graphforge.launch entry=" << entry
      << " block_rows=1 num_warps=1 abi=row_ptr,col_idx,";
    for (StringRef name : argumentNames) o << name << ",";
    o << "row_worklist,out\nmodule {\n  tt.func public @" << entry << "("
      << "%row_ptr: !tt.ptr<" << index << ">, %col_idx: !tt.ptr<" << index
      << ">";
    for (auto [position, input] : llvm::enumerate(launch.getInputs()))
      o << ", %" << argumentNames[position] << ": "
        << (input.getType() == f32 ? "f32" : "!tt.ptr<f32>");
    o << ", %row_worklist: !tt.ptr<i64>, %out: !tt.ptr<f32>) "
         "attributes {noinline = false} {\n"
      << "    %gf_zero_index = arith.constant dense<0> : " << tileIndex << "\n"
      << "    %gf_zero_tile = arith.constant dense<0.000000e+00> : tensor<64xf32>\n"
      << "    %gf_zero_scalar = arith.constant 0.000000e+00 : f32\n"
      << "    %gf_step = arith.constant 64 : " << index << "\n"
      << "    %gf_active_row = tt.get_program_id x : i32\n"
      << "    %gf_bucket = arith.constant "
      << op.getBucketOrdinalAttr().getInt() << " : i32\n"
      << "    %gf_bucket_start_ptr = tt.addptr %row_worklist, %gf_bucket : !tt.ptr<i64>, i32\n"
      << "    %gf_bucket_start = tt.load %gf_bucket_start_ptr : !tt.ptr<i64>\n"
      << "    %gf_row_base = arith.constant " << 2 * buckets + 1 << " : i64\n"
      << "    %gf_active_row64 = arith.extui %gf_active_row : i32 to i64\n"
      << "    %gf_row_slot0 = arith.addi %gf_row_base, %gf_bucket_start : i64\n"
      << "    %gf_row_slot = arith.addi %gf_row_slot0, %gf_active_row64 : i64\n"
      << "    %gf_row_id_ptr = tt.addptr %row_worklist, %gf_row_slot : !tt.ptr<i64>, i64\n"
      << "    %gf_row64 = tt.load %gf_row_id_ptr : !tt.ptr<i64>\n"
      << "    %gf_row = arith.trunci %gf_row64 : i64 to i32\n"
      << "    %gf_one = arith.constant 1 : i32\n"
      << "    %gf_start_ptr = tt.addptr %row_ptr, %gf_row : !tt.ptr<" << index
      << ">, i32\n"
      << "    %gf_row_start = tt.load %gf_start_ptr : !tt.ptr<" << index << ">\n"
      << "    %gf_next_row = arith.addi %gf_row, %gf_one : i32\n"
      << "    %gf_end_ptr = tt.addptr %row_ptr, %gf_next_row : !tt.ptr<" << index
      << ">, i32\n"
      << "    %gf_row_end = tt.load %gf_end_ptr : !tt.ptr<" << index << ">\n"
      << "    %gf_lane_i32 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>\n";
    if (index == "i64")
      o << "    %gf_lane = arith.extsi %gf_lane_i32 : tensor<64xi32> to tensor<64xi64>\n";
    else
      o << "    %gf_lane = arith.bitcast %gf_lane_i32 : tensor<64xi32> to tensor<64xi32>\n";
    o << "    %gf_state = scf.for %gf_start = %gf_row_start to %gf_row_end step %gf_step "
         "iter_args(%gf_acc = %gf_zero_scalar) -> (f32) : " << index << " {\n"
      << "      %gf_start_v = tt.splat %gf_start : " << index << " -> "
      << tileIndex << "\n"
      << "      %gf_edges = arith.addi %gf_start_v, %gf_lane : " << tileIndex << "\n"
      << "      %gf_end_v = tt.splat %gf_row_end : " << index << " -> "
      << tileIndex << "\n"
      << "      %gf_mask = arith.cmpi slt, %gf_edges, %gf_end_v : "
      << tileIndex << "\n"
      << "      %gf_col_base = tt.splat %col_idx : !tt.ptr<" << index
      << "> -> tensor<64x!tt.ptr<" << index << ">>\n"
      << "      %gf_col_ptr = tt.addptr %gf_col_base, %gf_edges : tensor<64x!tt.ptr<"
      << index << ">>, " << tileIndex << "\n"
      << "      %gf_src = tt.load %gf_col_ptr, %gf_mask, %gf_zero_index : tensor<64x!tt.ptr<"
      << index << ">>\n";
    SmallVector<std::string> edgeArguments;
    for (auto [position, roleAttribute] : llvm::enumerate(roles)) {
      StringRef role = cast<StringAttr>(roleAttribute).getValue();
      std::string argument = "%" + argumentNames[position];
      if (role == "param") {
        std::string splat = "%gf_param" + std::to_string(position);
        o << "      " << splat << " = tt.splat " << argument
          << " : f32 -> tensor<64xf32>\n";
        edgeArguments.push_back(std::move(splat));
        continue;
      }
      std::string base = "%gf_input_base" + std::to_string(position);
      std::string pointer = "%gf_input_ptr" + std::to_string(position);
      std::string loaded = "%gf_input" + std::to_string(position);
      o << "      " << base << " = tt.splat " << argument
        << " : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>\n";
      if (role == "src")
        o << "      " << pointer << " = tt.addptr " << base
          << ", %gf_src : tensor<64x!tt.ptr<f32>>, " << tileIndex << "\n";
      else if (role == "edge")
        o << "      " << pointer << " = tt.addptr " << base
          << ", %gf_edges : tensor<64x!tt.ptr<f32>>, " << tileIndex << "\n";
      else if (role == "dst") {
        std::string row = "%gf_row_v" + std::to_string(position);
        o << "      " << row << " = tt.splat %gf_row : i32 -> tensor<64xi32>\n"
          << "      " << pointer << " = tt.addptr " << base << ", " << row
          << " : tensor<64x!tt.ptr<f32>>, tensor<64xi32>\n";
      } else return failure();
      o << "      " << loaded << " = tt.load " << pointer
        << ", %gf_mask, %gf_zero_tile : tensor<64x!tt.ptr<f32>>\n";
      edgeArguments.push_back(std::move(loaded));
    }
    TensorAlgebraEmitter tileEmitter(
        launch, o, "tensor<64xf32>", "%gf_chunk_v");
    auto message = tileEmitter.emit(
        launch.getRegions().front(), edgeArguments, "      ");
    if (failed(message)) return failure();
    auto lifted = tileEmitter.emit(reducer.getLift(), *message, "      ");
    if (failed(lifted) || lifted->size() != 1) return failure();
    o << "      %gf_masked = \"arith.select\"(%gf_mask, " << lifted->front()
      << ", %gf_zero_tile) : (tensor<64xi1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>\n"
      << "      %gf_chunk_state = \"tt.reduce\"(%gf_masked) <{axis = 0 : i32}> ({\n"
      << "      ^bb0(%gf_a: f32, %gf_b: f32):\n"
      << "        %gf_add = arith.addf %gf_a, %gf_b : f32\n"
      << "        tt.reduce.return %gf_add : f32\n"
      << "      }) : (tensor<64xf32>) -> f32\n"
      << "      %gf_next = arith.addf %gf_acc, %gf_chunk_state : f32\n"
      << "      scf.yield %gf_next : f32\n"
      << "    }\n";
    ScalarAlgebraEmitter scalarEmitter(launch, o);
    auto finalized = scalarEmitter.emit(
        reducer.getFinalize(), ArrayRef<std::string>{"%gf_state"}, "    ");
    if (failed(finalized) || finalized->size() != 1) return failure();
    o << "    %gf_out_ptr = tt.addptr %out, %gf_row : !tt.ptr<f32>, i32\n"
      << "    tt.store %gf_out_ptr, " << finalized->front()
      << " : !tt.ptr<f32>\n"
      << "    tt.return\n  }\n}\n";
    o.flush();
    return text;
  };

  llvm::json::Array invocations;
  llvm::json::Array resources;
  llvm::StringSet<> resourceNames;
  llvm::DenseMap<Value, std::string> instanceBindings;
  auto ensureInstanceRequirement = [&](Value value) -> std::string {
    auto found = instanceBindings.find(value);
    if (found != instanceBindings.end()) return found->second;
    auto instance = value.getDefiningOp<storage::InstanceOp>();
    if (!instance) return "instance";
    auto region = instance.getRegion().getDefiningOp<storage::RegionOp>();
    std::string logical = region ? region.getLogicalId().str() : "instance";
    std::string binding = logical + "@" + instance.getMemorySpace().str() +
                          ":" + instance.getDevice().str();
    instanceBindings[value] = binding;
    if (resourceNames.insert(binding).second)
      resources.push_back(llvm::json::Object{
          {"name", binding},
          {"memory_space", instance.getMemorySpace()},
          {"layout", instance.getLayout()},
          {"device", instance.getDevice()},
          {"capacity_bytes", instance.getCapacityBytesAttr().getInt()},
          {"snapshot_version", region ? region.getVersionAttr().getInt() : 0},
          {"external", instance.getExternal()}});
    return binding;
  };
  auto appendCommunicationRequirement = [&](StringRef name, int64_t bytes,
                                            int64_t version) {
    if (resourceNames.insert(name).second)
      resources.push_back(llvm::json::Object{
          {"name", name.str()}, {"memory_space", "ram"},
          {"layout", "packed-halo"}, {"device", "host:0"},
          {"capacity_bytes", bytes}, {"snapshot_version", version},
          {"external", false}});
  };
  auto appendStorageRequirement = [&](Value storageValue, int64_t version) {
    if (!resourceNames.insert("row_worklist").second)
      return;
    auto instance = storageValue.getDefiningOp<storage::InstanceOp>();
    if (!instance)
      return;
    resources.push_back(llvm::json::Object{
        {"name", "row_worklist"},
        {"memory_space", instance.getMemorySpace()},
        {"layout", instance.getLayout()},
        {"device", instance.getDevice()},
        {"capacity_bytes", instance.getCapacityBytesAttr().getInt()},
        {"snapshot_version", version},
        {"external", instance.getExternal()}});
  };
  auto appendPartialRequirement = [&](Value storageValue, int64_t version) {
    if (!resourceNames.insert("row_partials").second)
      return;
    auto instance = storageValue.getDefiningOp<storage::InstanceOp>();
    if (!instance)
      return;
    resources.push_back(llvm::json::Object{
        {"name", "row_partials"},
        {"memory_space", instance.getMemorySpace()},
        {"layout", instance.getLayout()},
        {"device", instance.getDevice()},
        {"capacity_bytes", instance.getCapacityBytesAttr().getInt()},
        {"snapshot_version", version},
        {"external", instance.getExternal()}});
  };
  for (task::DegreeWorklistOp op : worklists) {
    auto instance = op.getStorage().getDefiningOp<storage::InstanceOp>();
    appendStorageRequirement(op.getStorage(),
                             op.getSnapshotVersionAttr().getInt());
    llvm::json::Array bounds;
    for (int64_t bound : op.getUpperBounds())
      bounds.push_back(bound);
    llvm::json::Array accesses;
    accesses.push_back(access("row_ptr", "read",
                              op.getSnapshotVersionAttr().getInt()));
    accesses.push_back(access("row_worklist", "write",
                              op.getSnapshotVersionAttr().getInt()));
    llvm::json::Array arguments;
    arguments.push_back(
        llvm::json::Object{{"parameter", "row_ptr"}, {"resource", "row_ptr"}});
    arguments.push_back(llvm::json::Object{{"parameter", "row_worklist"},
                                           {"resource", "row_worklist"}});
    invocations.push_back(llvm::json::Object{
        {"name", names[op.getOperation()]},
        {"task_kind", "degree-worklist"},
        {"phase", "materialize"},
        {"executable_symbol", "__gf_build_degree_worklist"},
        {"depends_on", llvm::json::Array()},
        {"arguments", std::move(arguments)},
        {"accesses", std::move(accesses)},
        {"rows", op.getRowsAttr().getInt()},
        {"upper_bounds", std::move(bounds)},
        {"snapshot_version", op.getSnapshotVersionAttr().getInt()},
        {"storage",
         llvm::json::Object{{"memory_space", instance.getMemorySpace()},
                            {"layout", instance.getLayout()},
                            {"device", instance.getDevice()},
                            {"capacity_bytes",
                             instance.getCapacityBytesAttr().getInt()}}}});
  }
  for (task::DegreeHistogramOp op : histograms) {
    StringRef index = indexType(op.getRelation());
    if (index.empty())
      return op.emitError("degree histogram requires i32/i64 CSR row_ptr");
    appendStorageRequirement(op.getStorage(),
                             op.getSnapshotVersionAttr().getInt());
    llvm::json::Array bounds;
    for (int64_t bound : op.getUpperBounds()) bounds.push_back(bound);
    invocations.push_back(llvm::json::Object{
        {"name", names[op.getOperation()]},
        {"task_kind", "degree-histogram"},
        {"phase", "materialize"},
        {"executable_symbol", "__gf_degree_histogram"},
        {"depends_on", dependencies(ValueRange{op.getDependsOn()})},
        {"arguments", llvm::json::Array{
             llvm::json::Object{{"parameter", "row_ptr"},
                                {"resource", "row_ptr"}},
             llvm::json::Object{{"parameter", "row_worklist"},
                                {"resource", "row_worklist"}}}},
        {"accesses", llvm::json::Array{
             access("row_ptr", "read", op.getSnapshotVersionAttr().getInt()),
             access("row_worklist", "write",
                    op.getSnapshotVersionAttr().getInt())}},
        {"rows", op.getRowsAttr().getInt()},
        {"index_dtype", index},
        {"upper_bounds", std::move(bounds)},
        {"provider_ttir", histogramTTIR(index, op.getRowsAttr().getInt(),
                                         op.getUpperBounds())},
        {"snapshot_version", op.getSnapshotVersionAttr().getInt()}});
  }
  for (task::DegreeResetOp op : resets) {
    appendStorageRequirement(op.getStorage(),
                             op.getSnapshotVersionAttr().getInt());
    invocations.push_back(llvm::json::Object{
        {"name", names[op.getOperation()]},
        {"task_kind", "degree-reset"},
        {"phase", "materialize"},
        {"executable_symbol", "__gf_degree_reset"},
        {"depends_on", llvm::json::Array()},
        {"arguments", llvm::json::Array{
             llvm::json::Object{{"parameter", "row_worklist"},
                                {"resource", "row_worklist"}}}},
        {"accesses", llvm::json::Array{
             access("row_worklist", "write",
                    op.getSnapshotVersionAttr().getInt())}},
        {"buckets", op.getBucketsAttr().getInt()},
        {"provider_ttir", resetTTIR(op.getBucketsAttr().getInt())},
        {"snapshot_version", op.getSnapshotVersionAttr().getInt()}});
  }
  for (task::DegreePrefixOp op : prefixes) {
    invocations.push_back(llvm::json::Object{
        {"name", names[op.getOperation()]},
        {"task_kind", "degree-prefix"},
        {"phase", "materialize"},
        {"executable_symbol", "__gf_degree_prefix"},
        {"depends_on", dependencies(ValueRange{op.getDependsOn()})},
        {"arguments", llvm::json::Array{
             llvm::json::Object{{"parameter", "row_worklist"},
                                {"resource", "row_worklist"}}}},
        {"accesses", llvm::json::Array{
             access("row_worklist", "read_write",
                    op.getSnapshotVersionAttr().getInt())}},
        {"buckets", op.getBucketsAttr().getInt()},
        {"provider_ttir", prefixTTIR(op.getBucketsAttr().getInt())},
        {"snapshot_version", op.getSnapshotVersionAttr().getInt()}});
  }
  for (task::DegreeScatterOp op : scatters) {
    StringRef index = indexType(op.getRelation());
    if (index.empty())
      return op.emitError("degree scatter requires i32/i64 CSR row_ptr");
    llvm::json::Array bounds;
    for (int64_t bound : op.getUpperBounds()) bounds.push_back(bound);
    invocations.push_back(llvm::json::Object{
        {"name", names[op.getOperation()]},
        {"task_kind", "degree-scatter"},
        {"phase", "materialize"},
        {"executable_symbol", "__gf_degree_scatter"},
        {"depends_on", dependencies(ValueRange{op.getDependsOn()})},
        {"arguments", llvm::json::Array{
             llvm::json::Object{{"parameter", "row_ptr"},
                                {"resource", "row_ptr"}},
             llvm::json::Object{{"parameter", "row_worklist"},
                                {"resource", "row_worklist"}}}},
        {"accesses", llvm::json::Array{
             access("row_ptr", "read", op.getSnapshotVersionAttr().getInt()),
             access("row_worklist", "read_write",
                    op.getSnapshotVersionAttr().getInt())}},
        {"rows", op.getRowsAttr().getInt()},
        {"index_dtype", index},
        {"upper_bounds", std::move(bounds)},
        {"provider_ttir", scatterTTIR(index, op.getRowsAttr().getInt(),
                                       op.getUpperBounds())},
        {"snapshot_version", op.getSnapshotVersionAttr().getInt()}});
  }
  for (task::DegreeBucketLaunchOp op : buckets) {
    auto function = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        op, op.getCalleeAttr());
    kernel::LaunchOp kernelLaunch;
    if (function)
      function.walk([&](kernel::LaunchOp candidate) {
        if (!kernelLaunch) kernelLaunch = candidate;
      });
    if (!kernelLaunch || kernelLaunch.getReducers().size() != 1)
      return op.emitError(
          "degree bucket requires one visible canonical kernel launch");
    auto reducerReference =
        dyn_cast<FlatSymbolRefAttr>(kernelLaunch.getReducers()[0]);
    auto reducer = reducerReference
        ? SymbolTable::lookupNearestSymbolFrom<ReducerOp>(
              kernelLaunch, reducerReference)
        : ReducerOp();
    auto rowType = dyn_cast<RankedTensorType>(kernelLaunch.getRowPtr().getType());
    auto rowInteger = rowType
        ? dyn_cast<IntegerType>(rowType.getElementType()) : IntegerType();
    if (!rowInteger ||
        (rowInteger.getWidth() != 32 && rowInteger.getWidth() != 64))
      return op.emitError(
          "degree bucket TTIR requires i32/i64 CSR indices");
    int64_t blockD = nextPowerOfTwo(
        op.getDegreeUpperInclusiveAttr().getInt());
    bool chunked = op.getRowMapping() == "worklist-chunked";
    if (blockD <= 0 || (blockD > 64 && !chunked))
      return op.emitError("degree bucket upper bound exceeds tiled TTIR limit");
    int64_t blockM = chunked ? 1 : std::min<int64_t>(
        16, std::max<int64_t>(1, 256 / blockD));
    int64_t rowsInBucket = op.getRowsInBucketAttr().getInt();
    int64_t worklistBucketCount = 0;
    if (auto logical = op.getWorklist().getDefiningOp<task::DegreeWorklistOp>())
      worklistBucketCount = logical.getUpperBounds().size();
    else if (auto scatter =
                 op.getWorklist().getDefiningOp<task::DegreeScatterOp>())
      worklistBucketCount = scatter.getUpperBounds().size();
    if (worklistBucketCount <= 0)
      return op.emitError("degree bucket has no visible worklist bounds");
    std::optional<std::string> bucketTTIR;
    if (reducer && isZeroAdditiveState(reducer) &&
        !kernelLaunch.getDeterministic()) {
      StringRef index = rowInteger.getWidth() == 64 ? StringRef("i64")
                                                    : StringRef("i32");
      if (chunked) {
        FailureOr<std::string> generated = chunkedBucketTTIR(
            op, kernelLaunch, reducer, index, worklistBucketCount);
        if (failed(generated)) return failure();
        bucketTTIR = std::move(*generated);
      } else {
        bucketTTIR.emplace();
        llvm::raw_string_ostream bucketStream(*bucketTTIR);
        if (failed(translateCSRRowsTile(
                kernelLaunch, reducer, index, blockM, blockD, 1, bucketStream,
                op.getBucketOrdinalAttr().getInt(), worklistBucketCount,
                op.getRowMapping() == "direct-filter")))
          return failure();
        bucketStream.flush();
      }
    }
    llvm::json::Array accesses;
    for (Attribute name : op.getReads())
      accesses.push_back(access(cast<StringAttr>(name).getValue(), "read",
                                op.getSnapshotVersionAttr().getInt()));
    for (Attribute name : op.getWrites())
      accesses.push_back(access(
          cast<StringAttr>(name).getValue(), "write",
          op.getSnapshotVersionAttr().getInt(), op.getPartitioning(),
          op.getOutputPartition()));
    llvm::json::Array arguments;
    for (Attribute attribute : op.getArguments()) {
      auto binding = cast<DictionaryAttr>(attribute);
      arguments.push_back(llvm::json::Object{
          {"parameter", binding.getAs<StringAttr>("parameter").getValue()},
          {"resource", binding.getAs<StringAttr>("resource").getValue()}});
    }
    llvm::json::Object bucketObject{
        {"name", names[op.getOperation()]},
        {"task_kind", "degree-bucket"},
        {"phase", "execute"},
        {"executable_symbol", op.getCalleeAttr().getValue()},
        {"depends_on", dependencies(op.getDependsOn())},
        {"arguments", std::move(arguments)},
        {"accesses", std::move(accesses)},
        {"bucket_ordinal", op.getBucketOrdinalAttr().getInt()},
        {"degree_lower_exclusive",
         op.getDegreeLowerExclusiveAttr().getInt()},
        {"degree_upper_inclusive",
         op.getDegreeUpperInclusiveAttr().getInt()},
        // A compact worklist maps programs over rows_in_bucket.  A direct
        // filter maps over the original row domain and rejects rows by
        // degree inside the kernel, so shortening its grid would silently
        // leave a suffix of logical rows unvisited.
        {"grid_x",
         std::max<int64_t>(
             1, llvm::divideCeil(
                    op.getRowMapping() == "direct-filter"
                        ? kernelLaunch.getNumRowsAttr().getInt()
                        : rowsInBucket,
                    blockM))},
        {"rows_in_bucket", rowsInBucket},
        {"row_mapping", op.getRowMapping()},
        {"rows", kernelLaunch.getNumRowsAttr().getInt()},
        {"index_dtype", rowInteger.getWidth() == 64 ? "i64" : "i32"},
        {"snapshot_version", op.getSnapshotVersionAttr().getInt()}};
    if (bucketTTIR)
      bucketObject["provider_ttir"] = std::move(*bucketTTIR);
    invocations.push_back(std::move(bucketObject));
  }
  auto taskArguments = [](ArrayAttr bindings) {
    llvm::json::Array result;
    for (Attribute attribute : bindings) {
      auto binding = cast<DictionaryAttr>(attribute);
      result.push_back(llvm::json::Object{
          {"parameter", binding.getAs<StringAttr>("parameter").getValue()},
          {"resource", binding.getAs<StringAttr>("resource").getValue()}});
    }
    return result;
  };
  auto taskAccesses = [&](ArrayAttr reads, ArrayAttr writes, int64_t version,
                          StringRef partitioning = StringRef(),
                          StringRef outputPartition = StringRef()) {
    llvm::json::Array result;
    for (Attribute name : reads)
      result.push_back(access(cast<StringAttr>(name).getValue(), "read", version));
    for (Attribute name : writes)
      result.push_back(access(cast<StringAttr>(name).getValue(), "write", version,
                              partitioning, outputPartition));
    return result;
  };
  for (task::RowSplitPartialOp op : splitPartials) {
    appendPartialRequirement(op.getStorage(),
                             op.getSnapshotVersionAttr().getInt());
    auto function = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        op, op.getCalleeAttr());
    kernel::LaunchOp launch;
    if (function)
      function.walk([&](kernel::LaunchOp candidate) {
        if (!launch) launch = candidate;
      });
    ReducerOp reducer;
    if (launch && launch.getReducers().size() == 1)
      if (auto reference = dyn_cast<FlatSymbolRefAttr>(launch.getReducers()[0]))
        reducer = SymbolTable::lookupNearestSymbolFrom<ReducerOp>(
            launch, reference);
    StringRef index = indexType(op.getRelation());
    FailureOr<std::string> provider =
        launch && reducer && !index.empty()
            ? splitPartialTTIR(op, launch, reducer, index)
            : FailureOr<std::string>(failure());
    llvm::json::Object object{
        {"name", names[op.getOperation()]},
        {"task_kind", "row-split-partial"},
        {"phase", "execute"},
        {"executable_symbol", op.getCalleeAttr().getValue()},
        {"depends_on", dependencies(ValueRange{op.getDependsOn()})},
        {"arguments", taskArguments(op.getArguments())},
        {"accesses", taskAccesses(op.getReads(), op.getWrites(),
                                  op.getSnapshotVersionAttr().getInt())},
        {"chunk_edges", op.getChunkEdgesAttr().getInt()},
        {"partials_per_row", op.getPartialsPerRowAttr().getInt()},
        {"rows", op.getRowsAttr().getInt()},
        {"active_rows", op.getActiveRowsAttr().getInt()},
        {"bucket_ordinal", op.getBucketOrdinalAttr().getInt()},
        {"buckets", op.getBucketsAttr().getInt()},
        {"state_bytes", op.getStateBytesAttr().getInt()},
        {"index_dtype", index},
        {"grid_x", op.getActiveRowsAttr().getInt() *
                       op.getPartialsPerRowAttr().getInt()},
        {"snapshot_version", op.getSnapshotVersionAttr().getInt()}};
    if (succeeded(provider))
      object["provider_ttir"] = std::move(*provider);
    invocations.push_back(std::move(object));
  }
  for (task::RowSplitFinalizeOp op : splitFinalizes) {
    appendPartialRequirement(op.getStorage(),
                             op.getSnapshotVersionAttr().getInt());
    auto function = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        op, op.getCalleeAttr());
    kernel::LaunchOp launch;
    if (function)
      function.walk([&](kernel::LaunchOp candidate) {
        if (!launch) launch = candidate;
      });
    ReducerOp reducer;
    if (launch && launch.getReducers().size() == 1)
      if (auto reference = dyn_cast<FlatSymbolRefAttr>(launch.getReducers()[0]))
        reducer = SymbolTable::lookupNearestSymbolFrom<ReducerOp>(
            launch, reference);
    FailureOr<std::string> provider =
        launch && reducer
            ? splitFinalizeTTIR(op, launch, reducer)
            : FailureOr<std::string>(failure());
    llvm::json::Object object{
        {"name", names[op.getOperation()]},
        {"task_kind", "row-split-finalize"},
        {"phase", "execute"},
        {"executable_symbol", op.getCalleeAttr().getValue()},
        {"depends_on", dependencies(ValueRange{op.getDependsOn()})},
        {"arguments", taskArguments(op.getArguments())},
        {"accesses", taskAccesses(op.getReads(), op.getWrites(),
                                  op.getSnapshotVersionAttr().getInt(),
                                  op.getPartitioning(),
                                  op.getOutputPartition())},
        {"partials_per_row", op.getPartialsPerRowAttr().getInt()},
        {"rows", op.getRowsAttr().getInt()},
        {"active_rows", op.getActiveRowsAttr().getInt()},
        {"bucket_ordinal", op.getBucketOrdinalAttr().getInt()},
        {"buckets", op.getBucketsAttr().getInt()},
        {"state_bytes", op.getStateBytesAttr().getInt()},
        {"grid_x", op.getActiveRowsAttr().getInt()},
        {"snapshot_version", op.getSnapshotVersionAttr().getInt()}};
    if (succeeded(provider))
      object["provider_ttir"] = std::move(*provider);
    invocations.push_back(std::move(object));
  }
  for (task::LaunchOp op : launches) {
    llvm::json::Array accesses;
    for (Attribute name : op.getReads())
      accesses.push_back(access(cast<StringAttr>(name).getValue(), "read",
                                op.getSnapshotVersionAttr().getInt()));
    for (Attribute name : op.getWrites())
      accesses.push_back(access(cast<StringAttr>(name).getValue(), "write",
                                op.getSnapshotVersionAttr().getInt()));
    llvm::json::Array arguments;
    llvm::SmallDenseSet<StringRef> bound;
    auto appendArgument = [&](Attribute name) {
      StringRef resource = cast<StringAttr>(name).getValue();
      if (bound.insert(resource).second)
        arguments.push_back(llvm::json::Object{{"parameter", resource},
                                               {"resource", resource}});
    };
    for (Attribute name : op.getReads())
      appendArgument(name);
    for (Attribute name : op.getWrites())
      appendArgument(name);
    invocations.push_back(llvm::json::Object{
        {"name", names[op.getOperation()]},
        {"task_kind", op.getTaskKind()},
        {"phase", "execute"},
        {"executable_symbol",
         op.getCalleeAttr() ? op.getCalleeAttr().getValue() : StringRef()},
        {"depends_on", dependencies(op.getDependsOn())},
        {"arguments", std::move(arguments)},
        {"accesses", std::move(accesses)},
        {"snapshot_version", op.getSnapshotVersionAttr().getInt()}});
  }
  for (storage::TransferOp op : transfers) {
    std::string source = ensureInstanceRequirement(op.getSource());
    std::string destination = ensureInstanceRequirement(op.getDestination());
    int64_t version = op.getSnapshotVersionAttr().getInt();
    invocations.push_back(llvm::json::Object{
        {"name", names[op.getOperation()]},
        {"task_kind", "storage-transfer"},
        {"phase", "materialize"},
        {"executable_symbol", "__gf_transfer"},
        {"depends_on", dependencies(op.getDependsOn())},
        {"arguments", llvm::json::Array{
             llvm::json::Object{{"parameter", "source"}, {"resource", source}},
             llvm::json::Object{{"parameter", "destination"}, {"resource", destination}}}},
        {"accesses", llvm::json::Array{
             access(source, "read", version),
             access(destination, "write", version)}},
        {"bytes", op.getBytesAttr().getInt()},
        {"engine", op.getEngine()},
        {"snapshot_version", version}});
  }
  for (auto [index, op] : llvm::enumerate(haloPacks)) {
    std::string source = ensureInstanceRequirement(op.getSource());
    std::string send = "halo-send:" + std::to_string(index);
    int64_t version = op.getSnapshotVersionAttr().getInt();
    appendCommunicationRequirement(send, op.getBytesAttr().getInt(), version);
    invocations.push_back(llvm::json::Object{
        {"name", names[op.getOperation()]}, {"task_kind", "halo-pack"},
        {"phase", "execute"}, {"executable_symbol", "__gf_halo_pack"},
        {"depends_on", dependencies(op.getDependsOn())},
        {"arguments", llvm::json::Array{
             llvm::json::Object{{"parameter", "source"}, {"resource", source}},
             llvm::json::Object{{"parameter", "send"}, {"resource", send}}}},
        {"accesses", llvm::json::Array{
             access(source, "read", version), access(send, "write", version)}},
        {"field", op.getField()}, {"bytes", op.getBytesAttr().getInt()},
        {"snapshot_version", version}});
  }
  for (auto [index, op] : llvm::enumerate(haloExchanges)) {
    auto pack = op.getSend().getDefiningOp<task::HaloPackOp>();
    auto packIt = llvm::find(haloPacks, pack);
    if (packIt == haloPacks.end()) return op.emitError("missing halo pack producer");
    size_t packIndex = std::distance(haloPacks.begin(), packIt);
    std::string send = "halo-send:" + std::to_string(packIndex);
    std::string receive = "halo-receive:" + std::to_string(index);
    int64_t version = op.getSnapshotVersionAttr().getInt();
    appendCommunicationRequirement(receive, op.getBytesAttr().getInt(), version);
    invocations.push_back(llvm::json::Object{
        {"name", names[op.getOperation()]}, {"task_kind", "halo-exchange"},
        {"phase", "execute"}, {"executable_symbol", "__gf_halo_exchange"},
        {"depends_on", dependencies(op.getDependsOn())},
        {"arguments", llvm::json::Array{
             llvm::json::Object{{"parameter", "send"}, {"resource", send}},
             llvm::json::Object{{"parameter", "receive"}, {"resource", receive}}}},
        {"accesses", llvm::json::Array{
             access(send, "read", version), access(receive, "write", version)}},
        {"group", op.getGroup()}, {"bytes", op.getBytesAttr().getInt()},
        {"snapshot_version", version}});
  }
  for (auto [index, op] : llvm::enumerate(haloUnpacks)) {
    auto exchange = op.getReceive().getDefiningOp<task::HaloExchangeOp>();
    auto exchangeIt = llvm::find(haloExchanges, exchange);
    if (exchangeIt == haloExchanges.end())
      return op.emitError("missing halo exchange producer");
    size_t exchangeIndex = std::distance(haloExchanges.begin(), exchangeIt);
    std::string receive = "halo-receive:" + std::to_string(exchangeIndex);
    std::string destination = ensureInstanceRequirement(op.getDestination());
    int64_t version = op.getSnapshotVersionAttr().getInt();
    invocations.push_back(llvm::json::Object{
        {"name", names[op.getOperation()]}, {"task_kind", "halo-unpack"},
        {"phase", "execute"}, {"executable_symbol", "__gf_halo_unpack"},
        {"depends_on", dependencies(op.getDependsOn())},
        {"arguments", llvm::json::Array{
             llvm::json::Object{{"parameter", "receive"}, {"resource", receive}},
             llvm::json::Object{{"parameter", "destination"}, {"resource", destination}}}},
        {"accesses", llvm::json::Array{
             access(receive, "read", version), access(destination, "write", version)}},
        {"field", op.getField()}, {"snapshot_version", version}});
  }
  for (storage::ReleaseOp op : releases) {
    auto instance = op.getInstance().getDefiningOp<storage::InstanceOp>();
    std::string resource =
        instance && instance.getLayout() == "row-partial-f32"
            ? std::string("row_partials")
            : ensureInstanceRequirement(op.getInstance());
    invocations.push_back(llvm::json::Object{
        {"name", names[op.getOperation()]},
        {"task_kind", "release"},
        {"phase", "execute"},
        {"executable_symbol", "__gf_release"},
        {"depends_on", dependencies(op.getDependsOn())},
        {"arguments", llvm::json::Array{llvm::json::Object{
             {"parameter", "instance"}, {"resource", resource}}}},
        {"accesses", llvm::json::Array{access(
             resource, "read_write", op.getSnapshotVersionAttr().getInt())}},
        {"snapshot_version", op.getSnapshotVersionAttr().getInt()}});
  }
  for (storage::JoinOp op : joins) {
    invocations.push_back(llvm::json::Object{
        {"name", names[op.getOperation()]},
        {"task_kind", "join"},
        {"phase", "execute"},
        {"executable_symbol", "__gf_join"},
        {"depends_on", dependencies(op.getEvents())},
        {"arguments", llvm::json::Array()},
        {"accesses", llvm::json::Array()},
        {"snapshot_version", op.getSnapshotVersionAttr().getInt()}});
  }

  llvm::json::Array terminals;
  auto appendTerminal = [&](Operation *operation, Value event) {
    if (event.use_empty())
      terminals.push_back(names[operation]);
  };
  for (task::DegreeWorklistOp op : worklists)
    appendTerminal(op.getOperation(), op.getReady());
  for (task::DegreeHistogramOp op : histograms)
    appendTerminal(op.getOperation(), op.getDone());
  for (task::DegreeResetOp op : resets)
    appendTerminal(op.getOperation(), op.getDone());
  for (task::DegreePrefixOp op : prefixes)
    appendTerminal(op.getOperation(), op.getDone());
  for (task::DegreeScatterOp op : scatters)
    appendTerminal(op.getOperation(), op.getDone());
  for (task::DegreeBucketLaunchOp op : buckets)
    appendTerminal(op.getOperation(), op.getDone());
  for (task::RowSplitPartialOp op : splitPartials)
    appendTerminal(op.getOperation(), op.getDone());
  for (task::RowSplitFinalizeOp op : splitFinalizes)
    appendTerminal(op.getOperation(), op.getDone());
  for (task::LaunchOp op : launches)
    appendTerminal(op.getOperation(), op.getDone());
  for (task::HaloPackOp op : haloPacks)
    appendTerminal(op.getOperation(), op.getDone());
  for (task::HaloExchangeOp op : haloExchanges)
    appendTerminal(op.getOperation(), op.getDone());
  for (task::HaloUnpackOp op : haloUnpacks)
    appendTerminal(op.getOperation(), op.getDone());
  for (storage::TransferOp op : transfers)
    appendTerminal(op.getOperation(), op.getReady());
  for (storage::ReleaseOp op : releases)
    appendTerminal(op.getOperation(), op.getReleased());
  for (storage::JoinOp op : joins)
    appendTerminal(op.getOperation(), op.getJoined());

  llvm::json::Object document{
      {"schema", "graphforge.executable-bundle-plan.v1"},
      {"resources", std::move(resources)},
      {"invocations", std::move(invocations)},
      {"terminals", std::move(terminals)}};
  output << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(document)));
  return success();
}

} // namespace

void registerTaskToBundleTranslation() {
  TranslateFromMLIRRegistration(
      "gf-task-to-bundle",
      "serialize verified gf_task IR to a provider-neutral bundle plan",
      translateTaskToBundle, [](DialectRegistry &registry) {
        registry.insert<GraphForgeDomainDialect,
                        kernel::GraphForgeKernelDialect,
                        storage::GraphForgeStorageDialect,
                        task::GraphForgeTaskDialect,
                        arith::ArithDialect, func::FuncDialect,
                        math::MathDialect, vector::VectorDialect>();
      });
}

namespace {

class TensorReductionTTIREmitter {
public:
  TensorReductionTTIREmitter(tensor::ReduceSumOp reduction,
                             func::FuncOp function, int64_t blockSize,
                             llvm::raw_ostream &output)
      : reduction(reduction), function(function), blockSize(blockSize),
        output(output) {}

  LogicalResult emit() {
    auto sourceType = dyn_cast<RankedTensorType>(reduction.getInput().getType());
    auto resultType = dyn_cast<RankedTensorType>(reduction.getResult().getType());
    if (!sourceType || !resultType || sourceType.getRank() != 2 ||
        (resultType.getRank() != 1 && resultType.getRank() != 2) ||
        !sourceType.hasStaticShape() ||
        !resultType.hasStaticShape() ||
        !sourceType.getElementType().isF32() ||
        !resultType.getElementType().isF32())
      return reduction.emitError(
          "gf_tensor TTIR reduction requires static FP32 [M,N] -> [M] or [M,1]");
    if (reduction.getAxes().size() != 1 ||
        (reduction.getAxes().front() != 0 &&
         reduction.getAxes().front() != 1))
      return reduction.emitError(
          "gf_tensor TTIR lowering requires one rank-two reduction axis");
    transposed = reduction.getAxes().front() == 0;
    rows = sourceType.getDimSize(transposed ? 1 : 0);
    columns = sourceType.getDimSize(transposed ? 0 : 1);
    bool validResult =
        (!reduction.getKeepDims() && resultType.getRank() == 1 &&
         resultType.getDimSize(0) == rows) ||
        (reduction.getKeepDims() && resultType.getRank() == 2 &&
         resultType.getDimSize(transposed ? 0 : 1) == 1 &&
         resultType.getDimSize(transposed ? 1 : 0) == rows);
    if (!validResult)
      return reduction.emitError(
          "gf_tensor TTIR reduction result shape does not match its axis");
    if (columns <= 0 || columns > blockSize)
      return reduction.emitError("reduction extent exceeds selected TTIR block");

    SmallVector<tensor::InputOp> inputs;
    function.walk([&](tensor::InputOp input) { inputs.push_back(input); });
    if (inputs.empty())
      return reduction.emitError("requires at least one gf_tensor.input");

    output << "// graphforge.tensor entry=gf_tensor_fused_reduce "
              "block_rows=1 block_elements="
           << blockSize << " num_warps=4 abi=";
    for (auto [index, input] : llvm::enumerate(inputs)) {
      if (index)
        output << ",";
      output << "arg" << index;
      argumentIndex[input.getResult()] = index;
    }
    output << ",out\nmodule {\n"
           << "  tt.func public @gf_tensor_fused_reduce(";
    for (auto [index, input] : llvm::enumerate(inputs)) {
      if (index)
        output << ", ";
      output << "%arg" << index << ": !tt.ptr<"
             << spelling(cast<RankedTensorType>(input.getResult().getType())
                             .getElementType()) << ">";
    }
    output << ", %out: !tt.ptr<f32>) attributes {noinline = false} {\n"
           << "    %zero = arith.constant dense<0.000000e+00> : tensor<"
           << blockSize << "xf32>\n"
           << "    %zero_i32 = arith.constant dense<0> : tensor<"
           << blockSize << "xi32>\n"
           << "    %zero_i64 = arith.constant dense<0> : tensor<"
           << blockSize << "xi64>\n"
           << "    %lane_i32 = tt.make_range {end = " << blockSize
           << " : i32, start = 0 : i32} : tensor<" << blockSize
           << "xi32>\n"
           << "    %lane = arith.extsi %lane_i32 : tensor<" << blockSize
           << "xi32> to tensor<" << blockSize << "xi64>\n"
           << "    %column_bound = arith.constant dense<" << columns
           << "> : tensor<" << blockSize << "xi64>\n"
           << "    %mask = arith.cmpi slt, %lane, %column_bound : tensor<"
           << blockSize << "xi64>\n"
           << "    %row_i32 = tt.get_program_id x : i32\n"
           << "    %row = arith.extsi %row_i32 : i32 to i64\n";

    FailureOr<std::string> vector = emitValue(reduction.getInput());
    if (failed(vector))
      return failure();
    output << "    %reduced = \"tt.reduce\"(" << *vector
           << ") <{axis = 0 : i32}> ({\n"
           << "    ^bb0(%a: f32, %b: f32):\n"
           << "      %combined = arith.addf %a, %b : f32\n"
           << "      tt.reduce.return %combined : f32\n"
           << "    }) : (tensor<" << blockSize << "xf32>) -> f32\n"
           << "    %out_ptr = tt.addptr %out, %row : !tt.ptr<f32>, i64\n"
           << "    tt.store %out_ptr, %reduced : !tt.ptr<f32>\n"
           << "    tt.return\n  }\n}\n";
    return success();
  }

private:
  static StringRef spelling(Type type) {
    if (type.isF32()) return "f32";
    if (type.isInteger(32)) return "i32";
    return "i64";
  }

  FailureOr<std::string> emitValue(Value value) {
    auto found = names.find(value);
    if (found != names.end())
      return found->second;
    Operation *operation = value.getDefiningOp();
    if (!operation)
      return failure();
    if (auto input = dyn_cast<tensor::InputOp>(operation))
      return emitInput(input);
    if (auto gather = dyn_cast<tensor::GatherOp>(operation))
      return emitGather(gather);
    if (auto reshape = dyn_cast<tensor::ReshapeOp>(operation)) {
      auto inputType = dyn_cast<RankedTensorType>(reshape.getInput().getType());
      auto resultType = dyn_cast<RankedTensorType>(reshape.getResult().getType());
      if (!inputType || !resultType ||
          inputType.getNumElements() != resultType.getNumElements())
        return reshape.emitError(
            "reduction reshape must preserve a static element count");
      FailureOr<std::string> rewritten = emitValue(reshape.getInput());
      if (succeeded(rewritten)) names[value] = *rewritten;
      return rewritten;
    }
    if (auto broadcast = dyn_cast<tensor::BroadcastOp>(operation)) {
      auto inputType = dyn_cast<RankedTensorType>(broadcast.getInput().getType());
      if (!inputType || inputType.getNumElements() != 1)
        return broadcast.emitError(
            "fused reduction currently broadcasts scalar-like values only");
      FailureOr<std::string> rewritten = emitValue(broadcast.getInput());
      if (succeeded(rewritten)) names[value] = *rewritten;
      return rewritten;
    }
    if (auto checkpoint = dyn_cast<tensor::CheckpointOp>(operation)) {
      FailureOr<std::string> rewritten = emitValue(checkpoint.getInput());
      if (succeeded(rewritten)) names[value] = *rewritten;
      return rewritten;
    }
    if (auto candidate = dyn_cast<tensor::CheckpointCandidateOp>(operation)) {
      FailureOr<std::string> rewritten = emitValue(candidate.getInput());
      if (succeeded(rewritten)) names[value] = *rewritten;
      return rewritten;
    }
    if (auto add = dyn_cast<tensor::AddOp>(operation))
      return emitBinary(add.getLhs(), add.getRhs(), "arith.addf", value);
    if (auto mul = dyn_cast<tensor::MulOp>(operation))
      return emitBinary(mul.getLhs(), mul.getRhs(), "arith.mulf", value);
    operation->emitError(
        "gf_tensor TTIR reduction supports input/gather/checkpoint/add/mul producers");
    return failure();
  }

  FailureOr<std::string> emitGather(tensor::GatherOp gather) {
    auto source = gather.getInput().getDefiningOp<tensor::InputOp>();
    auto index = gather.getIndex().getDefiningOp<tensor::InputOp>();
    auto sourceType = dyn_cast<RankedTensorType>(gather.getInput().getType());
    auto indexType = dyn_cast<RankedTensorType>(gather.getIndex().getType());
    if (!source || !index || !sourceType || !indexType ||
        sourceType.getRank() != 2 || indexType.getRank() != 1 ||
        !sourceType.getElementType().isF32() ||
        (!indexType.getElementType().isInteger(32) &&
         !indexType.getElementType().isInteger(64)) ||
        sourceType.getDimSize(1) != columns)
      return gather.emitError(
          "reduction gather requires direct FP32 [N,F] source and integer [M] index");

    unsigned indexArgument = argumentIndex.lookup(index.getResult());
    StringRef indexName = spelling(indexType.getElementType());
    std::string indexStride = next("gather_index_stride");
    std::string indexOffset = next("gather_index_offset");
    output << "    " << indexStride << " = arith.constant "
           << index.getStrides()[0] << " : i64\n"
           << "    " << indexOffset << " = arith.muli %row, "
           << indexStride << " : i64\n";
    if (index.getOffsetAttr().getInt() != 0) {
      std::string offset = next("gather_index_offset_constant");
      std::string shifted = next("gather_index_shifted");
      output << "    " << offset << " = arith.constant "
             << index.getOffsetAttr().getInt() << " : i64\n"
             << "    " << shifted << " = arith.addi " << indexOffset
             << ", " << offset << " : i64\n";
      indexOffset = shifted;
    }
    std::string indexPointer = next("gather_index_ptr");
    std::string sourceRow = next("gather_source_row");
    output << "    " << indexPointer << " = tt.addptr %arg" << indexArgument
           << ", " << indexOffset << " : !tt.ptr<" << indexName << ">, i64\n"
           << "    " << sourceRow << " = tt.load " << indexPointer
           << " : !tt.ptr<" << indexName << ">\n";
    if (indexType.getElementType().isInteger(32)) {
      std::string extended = next("gather_source_row_i64");
      output << "    " << extended << " = arith.extsi " << sourceRow
             << " : i32 to i64\n";
      sourceRow = extended;
    }

    int64_t rowStride = source.getStrides()[0];
    int64_t columnStride = source.getStrides()[1];
    std::string rowStrideValue = next("gather_row_stride");
    std::string rowOffset = next("gather_row_offset");
    output << "    " << rowStrideValue << " = arith.constant " << rowStride
           << " : i64\n"
           << "    " << rowOffset << " = arith.muli " << sourceRow
           << ", " << rowStrideValue << " : i64\n";
    if (source.getOffsetAttr().getInt() != 0) {
      std::string offset = next("gather_source_offset");
      std::string shifted = next("gather_row_shifted");
      output << "    " << offset << " = arith.constant "
             << source.getOffsetAttr().getInt() << " : i64\n"
             << "    " << shifted << " = arith.addi " << rowOffset
             << ", " << offset << " : i64\n";
      rowOffset = shifted;
    }
    std::string rowVector = next("gather_row_vector");
    std::string columnStrideValue = next("gather_column_stride");
    std::string columnVector = next("gather_column_vector");
    std::string offsets = next("gather_offsets");
    output << "    " << rowVector << " = tt.splat " << rowOffset
           << " : i64 -> tensor<" << blockSize << "xi64>\n"
           << "    " << columnStrideValue << " = arith.constant dense<"
           << columnStride << "> : tensor<" << blockSize << "xi64>\n"
           << "    " << columnVector << " = arith.muli %lane, "
           << columnStrideValue << " : tensor<"
           << blockSize << "xi64>\n"
           << "    " << offsets << " = arith.addi " << rowVector << ", "
           << columnVector << " : tensor<" << blockSize << "xi64>\n";
    unsigned sourceArgument = argumentIndex.lookup(source.getResult());
    std::string base = next("gather_base");
    std::string pointers = next("gather_ptrs");
    std::string loaded = next("gather_values");
    output << "    " << base << " = tt.splat %arg" << sourceArgument
           << " : !tt.ptr<f32> -> tensor<" << blockSize
           << "x!tt.ptr<f32>>\n"
           << "    " << pointers << " = tt.addptr " << base << ", "
           << offsets << " : tensor<" << blockSize
           << "x!tt.ptr<f32>>, tensor<" << blockSize << "xi64>\n"
           << "    " << loaded << " = tt.load " << pointers
           << ", %mask, %zero : tensor<" << blockSize << "x!tt.ptr<f32>>\n";
    names[gather.getResult()] = loaded;
    return loaded;
  }

  FailureOr<std::string> emitBinary(Value lhs, Value rhs, StringRef mnemonic,
                                    Value result) {
    FailureOr<std::string> left = emitValue(lhs);
    FailureOr<std::string> right = emitValue(rhs);
    if (failed(left) || failed(right))
      return failure();
    std::string name = next("value");
    output << "    " << name << " = " << mnemonic << " " << *left << ", "
           << *right << " : tensor<" << blockSize << "xf32>\n";
    names[result] = name;
    return name;
  }

  FailureOr<std::string> emitInput(tensor::InputOp input) {
    auto type = dyn_cast<RankedTensorType>(input.getResult().getType());
    if (!type || !type.hasStaticShape() || !type.getElementType().isF32() ||
        type.getRank() > 2) {
      input.emitError("GPU fused reduction inputs must be static rank <= 2 FP32");
      return failure();
    }
    ArrayRef<int64_t> strides = input.getStrides();
    int64_t offset = input.getOffsetAttr().getInt();
    int64_t rowStride = 0;
    int64_t columnStride = 0;
    if (type.getRank() == 2) {
      int64_t sourceRows = type.getDimSize(transposed ? 1 : 0);
      int64_t sourceColumns = type.getDimSize(transposed ? 0 : 1);
      if ((sourceRows != 1 && sourceRows != rows) ||
          (sourceColumns != 1 && sourceColumns != columns)) {
        input.emitError("input shape is not broadcast-compatible with [M,N]");
        return failure();
      }
      rowStride = sourceRows == 1 ? 0 : strides[transposed ? 1 : 0];
      columnStride = sourceColumns == 1 ? 0 : strides[transposed ? 0 : 1];
    } else if (type.getRank() == 1) {
      int64_t extent = type.getDimSize(0);
      if (transposed) {
        if (extent != 1 && extent != rows) {
          input.emitError(
              "rank-one input must be scalar-like or match the trailing axis");
          return failure();
        }
        rowStride = extent == 1 ? 0 : strides[0];
      } else {
        if (extent != 1 && extent != columns) {
          input.emitError("rank-one input must be scalar-like or match N");
          return failure();
        }
        columnStride = extent == 1 ? 0 : strides[0];
      }
    }

    unsigned index = argumentIndex.lookup(input.getResult());
    std::string base = next("base");
    output << "    " << base << " = tt.splat %arg" << index
           << " : !tt.ptr<f32> -> tensor<" << blockSize
           << "x!tt.ptr<f32>>\n";
    std::string rowOffset = next("row_offset");
    output << "    " << rowOffset << " = arith.constant " << offset
           << " : i64\n";
    if (rowStride != 0) {
      std::string stride = next("row_stride");
      std::string product = next("row_product");
      output << "    " << stride << " = arith.constant " << rowStride
             << " : i64\n"
             << "    " << product << " = arith.muli %row, " << stride
             << " : i64\n";
      std::string combined = next("row_combined");
      output << "    " << combined << " = arith.addi " << rowOffset << ", "
             << product << " : i64\n";
      rowOffset = combined;
    }
    std::string rowVector = next("row_vector");
    output << "    " << rowVector << " = tt.splat " << rowOffset
           << " : i64 -> tensor<" << blockSize << "xi64>\n";
    std::string indices = rowVector;
    if (columnStride != 0) {
      std::string strideVector = next("column_stride");
      std::string column = next("column_offset");
      std::string combined = next("indices");
      output << "    " << strideVector << " = arith.constant dense<"
             << columnStride << "> : tensor<" << blockSize << "xi64>\n"
             << "    " << column << " = arith.muli %lane, " << strideVector
             << " : tensor<" << blockSize << "xi64>\n"
             << "    " << combined << " = arith.addi " << rowVector << ", "
             << column << " : tensor<" << blockSize << "xi64>\n";
      indices = combined;
    }
    std::string pointers = next("pointers");
    std::string loaded = next("input");
    output << "    " << pointers << " = tt.addptr " << base << ", "
           << indices << " : tensor<" << blockSize
           << "x!tt.ptr<f32>>, tensor<" << blockSize << "xi64>\n"
           << "    " << loaded << " = tt.load " << pointers
           << ", %mask, %zero : tensor<" << blockSize
           << "x!tt.ptr<f32>>\n";
    names[input.getResult()] = loaded;
    return loaded;
  }

  std::string next(StringRef stem) {
    return (Twine("%") + stem + Twine(nextId++)).str();
  }

  tensor::ReduceSumOp reduction;
  func::FuncOp function;
  int64_t blockSize;
  llvm::raw_ostream &output;
  int64_t rows = 0;
  int64_t columns = 0;
  bool transposed = false;
  unsigned nextId = 0;
  DenseMap<Value, std::string> names;
  DenseMap<Value, unsigned> argumentIndex;
};

class TensorSegmentSumTTIREmitter {
public:
  TensorSegmentSumTTIREmitter(tensor::SegmentSumOp segment,
                              func::FuncOp function, int64_t blockSize,
                              llvm::raw_ostream &output)
      : segment(segment), function(function), blockSize(blockSize),
        output(output) {}

  LogicalResult emit() {
    auto inputType = dyn_cast<RankedTensorType>(segment.getInput().getType());
    auto indexType = dyn_cast<RankedTensorType>(segment.getIndex().getType());
    auto resultType = dyn_cast<RankedTensorType>(segment.getResult().getType());
    auto input = segment.getInput().getDefiningOp<tensor::InputOp>();
    auto index = segment.getIndex().getDefiningOp<tensor::InputOp>();
    if (!inputType || !indexType || !resultType || !input || !index ||
        (inputType.getRank() != 1 && inputType.getRank() != 2) ||
        indexType.getRank() != 1 || resultType.getRank() != inputType.getRank() ||
        !inputType.hasStaticShape() ||
        !resultType.hasStaticShape() || !inputType.getElementType().isF32() ||
        !resultType.getElementType().isF32() ||
        (!indexType.getElementType().isInteger(32) &&
         !indexType.getElementType().isInteger(64)))
      return segment.emitError(
          "GPU segment_sum currently requires direct FP32 [E] or [E,F], "
          "integer [E], and matching FP32 [R] or [R,F]");
    int64_t edges = inputType.getDimSize(0);
    int64_t features = inputType.getRank() == 2 ? inputType.getDimSize(1) : 1;
    if (resultType.getRank() == 2 &&
        resultType.getDimSize(1) != features)
      return segment.emitError(
          "GPU segment_sum input and result feature extents must match");
    if (edges != indexType.getDimSize(0) || edges <= 0 || edges > blockSize)
      return segment.emitError(
          "GPU segment_sum edge extent exceeds the selected reduction block");

    SmallVector<tensor::InputOp> inputs;
    function.walk([&](tensor::InputOp value) { inputs.push_back(value); });
    DenseMap<Value, unsigned> arguments;
    output << "// graphforge.tensor entry=gf_tensor_segment_sum block_rows=1 "
              "block_elements="
           << blockSize << " num_warps=4 abi=";
    for (auto [ordinal, value] : llvm::enumerate(inputs)) {
      if (ordinal) output << ",";
      output << "arg" << ordinal;
      arguments[value.getResult()] = ordinal;
    }
    output << ",out\nmodule {\n  tt.func public @gf_tensor_segment_sum(";
    for (auto [ordinal, value] : llvm::enumerate(inputs)) {
      if (ordinal) output << ", ";
      Type element = cast<RankedTensorType>(value.getResult().getType())
                         .getElementType();
      output << "%arg" << ordinal << ": !tt.ptr<"
             << (element.isF32() ? "f32" :
                 element.isInteger(32) ? "i32" : "i64") << ">";
    }
    StringRef indexName = indexType.getElementType().isInteger(32)
                              ? "i32" : "i64";
    int64_t inputRowStride = input.getStrides()[0];
    int64_t inputFeatureStride =
        inputType.getRank() == 2 ? input.getStrides()[1] : 0;
    int64_t indexStride = index.getStrides()[0];
    output << ", %out: !tt.ptr<f32>) attributes {noinline = false} {\n"
           << "    %zero = arith.constant dense<0.000000e+00> : tensor<"
           << blockSize << "xf32>\n"
           << "    %lane_i32 = tt.make_range {end = " << blockSize
           << " : i32, start = 0 : i32} : tensor<" << blockSize
           << "xi32>\n"
           << "    %lane = arith.extsi %lane_i32 : tensor<" << blockSize
           << "xi32> to tensor<" << blockSize << "xi64>\n"
           << "    %edge_bound = arith.constant dense<" << edges
           << "> : tensor<" << blockSize << "xi64>\n"
           << "    %valid = arith.cmpi slt, %lane, %edge_bound : tensor<"
           << blockSize << "xi64>\n"
           << "    %program_i32 = tt.get_program_id x : i32\n";
    if (features == 1) {
      output << "    %feature_i32 = arith.constant 0 : i32\n"
             << "    %destination_i32 = arith.addi %program_i32, "
                "%feature_i32 : i32\n";
    } else {
      output << "    %feature_count_i32 = arith.constant " << features
             << " : i32\n"
             << "    %destination_i32 = arith.divui %program_i32, "
                "%feature_count_i32 : i32\n"
             << "    %feature_i32 = arith.remui %program_i32, "
                "%feature_count_i32 : i32\n";
    }
    output << "    %destination = arith.extsi %destination_i32 : i32 to i64\n"
           << "    %feature = arith.extsi %feature_i32 : i32 to i64\n"
           << "    %destination_vector = tt.splat %destination : i64 -> tensor<"
           << blockSize << "xi64>\n"
           << "    %input_stride = arith.constant dense<" << inputRowStride
           << "> : tensor<" << blockSize << "xi64>\n"
           << "    %input_index0 = arith.muli %lane, %input_stride : tensor<"
           << blockSize << "xi64>\n";
    std::string inputIndex = "%input_index0";
    if (inputType.getRank() == 2) {
      output << "    %input_feature_stride = arith.constant "
             << inputFeatureStride << " : i64\n"
             << "    %input_feature_offset0 = arith.muli %feature, "
                "%input_feature_stride : i64\n"
             << "    %input_feature_offset = tt.splat %input_feature_offset0 "
                ": i64 -> tensor<"
             << blockSize << "xi64>\n"
             << "    %input_index1 = arith.addi %input_index0, "
                "%input_feature_offset : tensor<"
             << blockSize << "xi64>\n";
      inputIndex = "%input_index1";
    }
    if (input.getOffsetAttr().getInt() != 0) {
      output << "    %input_offset = arith.constant dense<"
             << input.getOffsetAttr().getInt() << "> : tensor<" << blockSize
             << "xi64>\n"
             << "    %input_index2 = arith.addi " << inputIndex
             << ", %input_offset "
                ": tensor<"
             << blockSize << "xi64>\n";
      inputIndex = "%input_index2";
    }
    output << "    %input_base = tt.splat %arg"
           << arguments.lookup(input.getResult())
           << " : !tt.ptr<f32> -> tensor<" << blockSize
           << "x!tt.ptr<f32>>\n"
           << "    %input_ptr = tt.addptr %input_base, " << inputIndex
           << " : tensor<" << blockSize << "x!tt.ptr<f32>>, tensor<"
           << blockSize << "xi64>\n"
           << "    %values = tt.load %input_ptr, %valid, %zero : tensor<"
           << blockSize << "x!tt.ptr<f32>>\n"
           << "    %index_stride = arith.constant dense<" << indexStride
           << "> : tensor<" << blockSize << "xi64>\n"
           << "    %index_index0 = arith.muli %lane, %index_stride : tensor<"
           << blockSize << "xi64>\n";
    std::string indexIndex = "%index_index0";
    if (index.getOffsetAttr().getInt() != 0) {
      output << "    %index_offset = arith.constant dense<"
             << index.getOffsetAttr().getInt() << "> : tensor<" << blockSize
             << "xi64>\n"
             << "    %index_index1 = arith.addi %index_index0, %index_offset "
                ": tensor<"
             << blockSize << "xi64>\n";
      indexIndex = "%index_index1";
    }
    output << "    %index_base = tt.splat %arg"
           << arguments.lookup(index.getResult()) << " : !tt.ptr<" << indexName
           << "> -> tensor<" << blockSize << "x!tt.ptr<" << indexName
           << ">>\n"
           << "    %index_ptr = tt.addptr %index_base, " << indexIndex
           << " : tensor<" << blockSize << "x!tt.ptr<" << indexName
           << ">>, tensor<" << blockSize << "xi64>\n"
           << "    %segment_raw = tt.load %index_ptr, %valid : tensor<"
           << blockSize << "x!tt.ptr<" << indexName << ">>\n";
    StringRef segmentName = "%segment_raw";
    if (indexType.getElementType().isInteger(32)) {
      output << "    %segment = arith.extsi %segment_raw : tensor<" << blockSize
             << "xi32> to tensor<" << blockSize << "xi64>\n";
      segmentName = "%segment";
    }
    output << "    %matches0 = arith.cmpi eq, " << segmentName
           << ", %destination_vector : tensor<" << blockSize << "xi64>\n"
           << "    %matches = arith.andi %valid, %matches0 : tensor<"
           << blockSize << "xi1>\n"
           << "    %selected = arith.select %matches, %values, %zero : tensor<"
           << blockSize << "xi1>, tensor<" << blockSize << "xf32>\n"
           << "    %sum = \"tt.reduce\"(%selected) <{axis = 0 : i32}> ({\n"
           << "    ^bb0(%a: f32, %b: f32):\n"
           << "      %next = arith.addf %a, %b : f32\n"
           << "      tt.reduce.return %next : f32\n"
           << "    }) : (tensor<" << blockSize << "xf32>) -> f32\n"
           << "    %out_feature_count = arith.constant " << features
           << " : i64\n"
           << "    %out_row_offset = arith.muli %destination, "
              "%out_feature_count : i64\n"
           << "    %out_index = arith.addi %out_row_offset, %feature : i64\n"
           << "    %out_ptr = tt.addptr %out, %out_index : !tt.ptr<f32>, i64\n"
           << "    tt.store %out_ptr, %sum : !tt.ptr<f32>\n"
           << "    tt.return\n  }\n}\n";
    return success();
  }

private:
  tensor::SegmentSumOp segment;
  func::FuncOp function;
  int64_t blockSize;
  llvm::raw_ostream &output;
};

static StringRef tensorABIElementName(Type element) {
  if (element.isF32()) return "f32";
  if (element.isInteger(32)) return "i32";
  return "i64";
}

class TensorCSRProductTTIREmitter {
public:
  TensorCSRProductTTIREmitter(tensor::CSRSegmentProductOp product,
                              func::FuncOp function, int64_t blockSize,
                              llvm::raw_ostream &output)
      : product(product), function(function), blockSize(blockSize),
        output(output) {}

  LogicalResult emit() {
    auto inputType = dyn_cast<RankedTensorType>(product.getInput().getType());
    auto rowType = dyn_cast<RankedTensorType>(product.getRowPtr().getType());
    auto resultType = dyn_cast<RankedTensorType>(product.getResult().getType());
    auto input = product.getInput().getDefiningOp<tensor::InputOp>();
    auto rowPtr = product.getRowPtr().getDefiningOp<tensor::InputOp>();
    if (!inputType || !rowType || !resultType || !input || !rowPtr ||
        (inputType.getRank() != 1 && inputType.getRank() != 2) ||
        rowType.getRank() != 1 || resultType.getRank() != inputType.getRank() ||
        !inputType.hasStaticShape() || !resultType.hasStaticShape() ||
        !inputType.getElementType().isF32() ||
        !resultType.getElementType().isF32() ||
        (!rowType.getElementType().isInteger(32) &&
         !rowType.getElementType().isInteger(64)))
      return product.emitError(
          "GPU CSR product requires direct FP32 [E] or [E,F], integer "
          "row_ptr and matching FP32 [R] or [R,F]");
    int64_t rows = product.getNumRows();
    int64_t features = inputType.getRank() == 2 ? inputType.getDimSize(1) : 1;
    if (rowType.getDimSize(0) != rows + 1 ||
        resultType.getDimSize(0) != rows ||
        (resultType.getRank() == 2 && resultType.getDimSize(1) != features))
      return product.emitError("CSR product static extents are inconsistent");

    SmallVector<tensor::InputOp> inputs;
    function.walk([&](tensor::InputOp value) { inputs.push_back(value); });
    DenseMap<Value, unsigned> arguments;
    output << "// graphforge.tensor entry=gf_tensor_csr_product block_rows=1 "
              "block_elements=" << blockSize << " num_warps=4 abi=";
    for (auto [ordinal, value] : llvm::enumerate(inputs)) {
      if (ordinal) output << ",";
      output << "arg" << ordinal;
      arguments[value.getResult()] = ordinal;
    }
    output << ",out\nmodule {\n  tt.func public @gf_tensor_csr_product(";
    for (auto [ordinal, value] : llvm::enumerate(inputs)) {
      if (ordinal) output << ", ";
      Type element = cast<RankedTensorType>(value.getResult().getType())
                         .getElementType();
      output << "%arg" << ordinal << ": !tt.ptr<"
             << tensorABIElementName(element) << ">";
    }
    int64_t inputRowStride = input.getStrides()[0];
    int64_t inputFeatureStride =
        inputType.getRank() == 2 ? input.getStrides()[1] : 0;
    int64_t rowStride = rowPtr.getStrides()[0];
    StringRef index = tensorABIElementName(rowType.getElementType());
    output << ", %out: !tt.ptr<f32>) attributes {noinline = false} {\n"
           << "    %lane_i32 = tt.make_range {end = " << blockSize
           << " : i32, start = 0 : i32} : tensor<" << blockSize << "xi32>\n"
           << "    %lane = arith.extsi %lane_i32 : tensor<" << blockSize
           << "xi32> to tensor<" << blockSize << "xi64>\n"
           << "    %one = arith.constant dense<1.000000e+00> : tensor<"
           << blockSize << "xf32>\n"
           << "    %program_i32 = tt.get_program_id x : i32\n";
    if (features == 1) {
      output << "    %feature_i32 = arith.constant 0 : i32\n"
             << "    %row_i32 = arith.addi %program_i32, %feature_i32 : i32\n";
    } else {
      output << "    %feature_count_i32 = arith.constant " << features
             << " : i32\n"
             << "    %row_i32 = arith.divui %program_i32, "
                "%feature_count_i32 : i32\n"
             << "    %feature_i32 = arith.remui %program_i32, "
                "%feature_count_i32 : i32\n";
    }
    output << "    %row = arith.extsi %row_i32 : i32 to i64\n"
           << "    %feature = arith.extsi %feature_i32 : i32 to i64\n";
    StringRef begin, end;
    if (product.getUniformDegree()) {
      output << "    %degree = arith.constant " << product.getMaxDegree()
             << " : i64\n"
             << "    %begin = arith.muli %row, %degree : i64\n"
             << "    %end = arith.addi %begin, %degree : i64\n";
      begin = "%begin";
      end = "%end";
    } else {
      output << "    %row_stride = arith.constant " << rowStride << " : i64\n"
             << "    %row_index0 = arith.muli %row, %row_stride : i64\n"
             << "    %row_index1 = arith.addi %row_index0, %row_stride : i64\n";
      if (rowPtr.getOffsetAttr().getInt() != 0) {
        output << "    %row_offset = arith.constant "
               << rowPtr.getOffsetAttr().getInt() << " : i64\n"
               << "    %row_index2 = arith.addi %row_index0, %row_offset : i64\n"
               << "    %row_index3 = arith.addi %row_index1, %row_offset : i64\n";
      }
      StringRef beginIndex = rowPtr.getOffsetAttr().getInt() == 0
                                 ? "%row_index0" : "%row_index2";
      StringRef endIndex = rowPtr.getOffsetAttr().getInt() == 0
                               ? "%row_index1" : "%row_index3";
      output << "    %begin_ptr = tt.addptr %arg"
             << arguments.lookup(rowPtr.getResult()) << ", " << beginIndex
             << " : !tt.ptr<" << index << ">, i64\n"
             << "    %end_ptr = tt.addptr %arg"
             << arguments.lookup(rowPtr.getResult()) << ", " << endIndex
             << " : !tt.ptr<" << index << ">, i64\n"
             << "    %begin_raw = tt.load %begin_ptr : !tt.ptr<" << index << ">\n"
             << "    %end_raw = tt.load %end_ptr : !tt.ptr<" << index << ">\n";
      begin = "%begin_raw";
      end = "%end_raw";
      if (rowType.getElementType().isInteger(32)) {
        output << "    %begin = arith.extsi %begin_raw : i32 to i64\n"
               << "    %end = arith.extsi %end_raw : i32 to i64\n";
        begin = "%begin";
        end = "%end";
      }
    }
    output << "    %begin_vector = tt.splat " << begin
           << " : i64 -> tensor<" << blockSize << "xi64>\n"
           << "    %end_vector = tt.splat " << end
           << " : i64 -> tensor<" << blockSize << "xi64>\n"
           << "    %edge = arith.addi %begin_vector, %lane : tensor<"
           << blockSize << "xi64>\n"
           << "    %active = arith.cmpi slt, %edge, %end_vector : tensor<"
           << blockSize << "xi64>\n"
           << "    %input_stride = arith.constant dense<" << inputRowStride
           << "> : tensor<" << blockSize << "xi64>\n"
           << "    %input_index0 = arith.muli %edge, %input_stride : tensor<"
           << blockSize << "xi64>\n";
    std::string inputIndex = "%input_index0";
    if (inputType.getRank() == 2) {
      output << "    %input_feature_stride = arith.constant "
             << inputFeatureStride << " : i64\n"
             << "    %feature_offset0 = arith.muli %feature, "
                "%input_feature_stride : i64\n"
             << "    %feature_offset = tt.splat %feature_offset0 : i64 -> "
                "tensor<" << blockSize << "xi64>\n"
             << "    %input_index1 = arith.addi %input_index0, "
                "%feature_offset : tensor<" << blockSize << "xi64>\n";
      inputIndex = "%input_index1";
    }
    if (input.getOffsetAttr().getInt() != 0) {
      output << "    %input_offset = arith.constant dense<"
             << input.getOffsetAttr().getInt() << "> : tensor<" << blockSize
             << "xi64>\n"
             << "    %input_index2 = arith.addi " << inputIndex
             << ", %input_offset : tensor<" << blockSize << "xi64>\n";
      inputIndex = "%input_index2";
    }
    output << "    %input_base = tt.splat %arg"
           << arguments.lookup(input.getResult())
           << " : !tt.ptr<f32> -> tensor<" << blockSize
           << "x!tt.ptr<f32>>\n"
           << "    %input_ptr = tt.addptr %input_base, " << inputIndex
           << " : tensor<" << blockSize << "x!tt.ptr<f32>>, tensor<"
           << blockSize << "xi64>\n"
           << "    %values = tt.load %input_ptr, %active, %one : tensor<"
           << blockSize << "x!tt.ptr<f32>>\n"
           << "    %product = \"tt.reduce\"(%values) <{axis = 0 : i32}> ({\n"
           << "    ^bb0(%a: f32, %b: f32):\n"
           << "      %next = arith.mulf %a, %b : f32\n"
           << "      tt.reduce.return %next : f32\n"
           << "    }) : (tensor<" << blockSize << "xf32>) -> f32\n"
           << "    %out_index = arith.extsi %program_i32 : i32 to i64\n"
           << "    %out_ptr = tt.addptr %out, %out_index : !tt.ptr<f32>, i64\n"
           << "    tt.store %out_ptr, %product : !tt.ptr<f32>\n"
           << "    tt.return\n  }\n}\n";
    return success();
  }

private:
  tensor::CSRSegmentProductOp product;
  func::FuncOp function;
  int64_t blockSize;
  llvm::raw_ostream &output;
};

class TensorCSRProductVJPTTIREmitter {
public:
  TensorCSRProductVJPTTIREmitter(tensor::CSRSegmentProductVJPOp vjp,
                                 func::FuncOp function, int64_t blockSize,
                                 llvm::raw_ostream &output)
      : vjp(vjp), function(function), blockSize(blockSize), output(output) {}

  LogicalResult emit() {
    auto inputType = dyn_cast<RankedTensorType>(vjp.getInput().getType());
    auto rowType = dyn_cast<RankedTensorType>(vjp.getRowPtr().getType());
    auto destinationType =
        dyn_cast<RankedTensorType>(vjp.getDestination().getType());
    auto upstreamType = dyn_cast<RankedTensorType>(vjp.getUpstream().getType());
    auto input = vjp.getInput().getDefiningOp<tensor::InputOp>();
    auto rowPtr = vjp.getRowPtr().getDefiningOp<tensor::InputOp>();
    auto destination =
        vjp.getDestination().getDefiningOp<tensor::InputOp>();
    auto upstream = vjp.getUpstream().getDefiningOp<tensor::InputOp>();
    if (!inputType || !rowType || !destinationType || !upstreamType ||
        !input || !rowPtr || !destination || !upstream ||
        (inputType.getRank() != 1 && inputType.getRank() != 2) ||
        upstreamType.getRank() != inputType.getRank() ||
        !inputType.getElementType().isF32() ||
        !upstreamType.getElementType().isF32() ||
        (!rowType.getElementType().isInteger(32) &&
         !rowType.getElementType().isInteger(64)) ||
        destinationType.getElementType() != rowType.getElementType())
      return vjp.emitError(
          "GPU CSR product VJP requires direct FP32 message/upstream and "
          "matching integer topology inputs");
    int64_t edges = inputType.getDimSize(0);
    int64_t rows = vjp.getNumRows();
    int64_t features = inputType.getRank() == 2 ? inputType.getDimSize(1) : 1;
    if (destinationType.getDimSize(0) != edges ||
        upstreamType.getDimSize(0) != rows ||
        (inputType.getRank() == 2 &&
         upstreamType.getDimSize(1) != features))
      return vjp.emitError("CSR product VJP static extents are inconsistent");
    if (vjp.getUniformDegree() && vjp.getMaxDegree() <= 32)
      return emitUniform(
          inputType, upstreamType, input, upstream, features);

    SmallVector<tensor::InputOp> inputs;
    function.walk([&](tensor::InputOp value) { inputs.push_back(value); });
    DenseMap<Value, unsigned> arguments;
    output << "// graphforge.tensor entry=gf_tensor_csr_product_vjp block_rows=1 "
              "block_elements=" << blockSize << " num_warps=4 abi=";
    for (auto [ordinal, value] : llvm::enumerate(inputs)) {
      if (ordinal) output << ",";
      output << "arg" << ordinal;
      arguments[value.getResult()] = ordinal;
    }
    output << ",out\nmodule {\n  tt.func public @gf_tensor_csr_product_vjp(";
    for (auto [ordinal, value] : llvm::enumerate(inputs)) {
      if (ordinal) output << ", ";
      Type element = cast<RankedTensorType>(value.getResult().getType())
                         .getElementType();
      output << "%arg" << ordinal << ": !tt.ptr<"
             << tensorABIElementName(element) << ">";
    }
    StringRef index = tensorABIElementName(rowType.getElementType());
    int64_t inputRowStride = input.getStrides()[0];
    int64_t inputFeatureStride =
        inputType.getRank() == 2 ? input.getStrides()[1] : 0;
    int64_t rowStride = rowPtr.getStrides()[0];
    int64_t destinationStride = destination.getStrides()[0];
    int64_t upstreamRowStride = upstream.getStrides()[0];
    int64_t upstreamFeatureStride =
        upstreamType.getRank() == 2 ? upstream.getStrides()[1] : 0;
    output << ", %out: !tt.ptr<f32>) attributes {noinline = false} {\n"
           << "    %lane_i32 = tt.make_range {end = " << blockSize
           << " : i32, start = 0 : i32} : tensor<" << blockSize << "xi32>\n"
           << "    %lane = arith.extsi %lane_i32 : tensor<" << blockSize
           << "xi32> to tensor<" << blockSize << "xi64>\n"
           << "    %one = arith.constant dense<1.000000e+00> : tensor<"
           << blockSize << "xf32>\n"
           << "    %program_i32 = tt.get_program_id x : i32\n";
    if (features == 1) {
      output << "    %feature_i32 = arith.constant 0 : i32\n"
             << "    %edge_i32 = arith.addi %program_i32, %feature_i32 : i32\n";
    } else {
      output << "    %feature_count_i32 = arith.constant " << features
             << " : i32\n"
             << "    %edge_i32 = arith.divui %program_i32, "
                "%feature_count_i32 : i32\n"
             << "    %feature_i32 = arith.remui %program_i32, "
                "%feature_count_i32 : i32\n";
    }
    output << "    %edge = arith.extsi %edge_i32 : i32 to i64\n"
           << "    %feature = arith.extsi %feature_i32 : i32 to i64\n"
           << "    %destination_stride = arith.constant "
           << destinationStride << " : i64\n"
           << "    %destination_index0 = arith.muli %edge, "
              "%destination_stride : i64\n";
    std::string destinationIndex = "%destination_index0";
    if (destination.getOffsetAttr().getInt() != 0) {
      output << "    %destination_offset = arith.constant "
             << destination.getOffsetAttr().getInt() << " : i64\n"
             << "    %destination_index1 = arith.addi %destination_index0, "
                "%destination_offset : i64\n";
      destinationIndex = "%destination_index1";
    }
    output << "    %destination_ptr = tt.addptr %arg"
           << arguments.lookup(destination.getResult()) << ", "
           << destinationIndex << " : !tt.ptr<" << index << ">, i64\n"
           << "    %row_raw = tt.load %destination_ptr : !tt.ptr<" << index
           << ">\n";
    StringRef row = "%row_raw";
    if (rowType.getElementType().isInteger(32)) {
      output << "    %row = arith.extsi %row_raw : i32 to i64\n";
      row = "%row";
    }
    StringRef begin, end;
    if (vjp.getUniformDegree()) {
      output << "    %degree = arith.constant " << vjp.getMaxDegree()
             << " : i64\n"
             << "    %begin = arith.muli " << row << ", %degree : i64\n"
             << "    %end = arith.addi %begin, %degree : i64\n";
      begin = "%begin";
      end = "%end";
    } else {
      output << "    %row_stride = arith.constant " << rowStride << " : i64\n"
             << "    %row_index0 = arith.muli " << row
             << ", %row_stride : i64\n"
             << "    %row_index1 = arith.addi %row_index0, %row_stride : i64\n";
      std::string beginIndex = "%row_index0", endIndex = "%row_index1";
      if (rowPtr.getOffsetAttr().getInt() != 0) {
        output << "    %row_offset = arith.constant "
               << rowPtr.getOffsetAttr().getInt() << " : i64\n"
               << "    %row_index2 = arith.addi %row_index0, %row_offset : i64\n"
               << "    %row_index3 = arith.addi %row_index1, %row_offset : i64\n";
        beginIndex = "%row_index2";
        endIndex = "%row_index3";
      }
      output << "    %begin_ptr = tt.addptr %arg"
             << arguments.lookup(rowPtr.getResult()) << ", " << beginIndex
             << " : !tt.ptr<" << index << ">, i64\n"
             << "    %end_ptr = tt.addptr %arg"
             << arguments.lookup(rowPtr.getResult()) << ", " << endIndex
             << " : !tt.ptr<" << index << ">, i64\n"
             << "    %begin_raw = tt.load %begin_ptr : !tt.ptr<" << index << ">\n"
             << "    %end_raw = tt.load %end_ptr : !tt.ptr<" << index << ">\n";
      begin = "%begin_raw";
      end = "%end_raw";
      if (rowType.getElementType().isInteger(32)) {
        output << "    %begin = arith.extsi %begin_raw : i32 to i64\n"
               << "    %end = arith.extsi %end_raw : i32 to i64\n";
        begin = "%begin";
        end = "%end";
      }
    }
    output << "    %begin_vector = tt.splat " << begin
           << " : i64 -> tensor<" << blockSize << "xi64>\n"
           << "    %end_vector = tt.splat " << end
           << " : i64 -> tensor<" << blockSize << "xi64>\n"
           << "    %candidate = arith.addi %begin_vector, %lane : tensor<"
           << blockSize << "xi64>\n"
           << "    %active0 = arith.cmpi slt, %candidate, %end_vector : tensor<"
           << blockSize << "xi64>\n"
           << "    %edge_vector = tt.splat %edge : i64 -> tensor<" << blockSize
           << "xi64>\n"
           << "    %not_self = arith.cmpi ne, %candidate, %edge_vector : tensor<"
           << blockSize << "xi64>\n"
           << "    %active = arith.andi %active0, %not_self : tensor<"
           << blockSize << "xi1>\n"
           << "    %input_stride = arith.constant dense<" << inputRowStride
           << "> : tensor<" << blockSize << "xi64>\n"
           << "    %input_index0 = arith.muli %candidate, %input_stride : tensor<"
           << blockSize << "xi64>\n";
    std::string inputIndex = "%input_index0";
    if (inputType.getRank() == 2) {
      output << "    %input_feature_stride = arith.constant "
             << inputFeatureStride << " : i64\n"
             << "    %feature_offset0 = arith.muli %feature, "
                "%input_feature_stride : i64\n"
             << "    %feature_offset = tt.splat %feature_offset0 : i64 -> "
                "tensor<" << blockSize << "xi64>\n"
             << "    %input_index1 = arith.addi %input_index0, "
                "%feature_offset : tensor<" << blockSize << "xi64>\n";
      inputIndex = "%input_index1";
    }
    if (input.getOffsetAttr().getInt() != 0) {
      output << "    %input_offset = arith.constant dense<"
             << input.getOffsetAttr().getInt() << "> : tensor<" << blockSize
             << "xi64>\n"
             << "    %input_index2 = arith.addi " << inputIndex
             << ", %input_offset : tensor<" << blockSize << "xi64>\n";
      inputIndex = "%input_index2";
    }
    output << "    %input_base = tt.splat %arg"
           << arguments.lookup(input.getResult())
           << " : !tt.ptr<f32> -> tensor<" << blockSize
           << "x!tt.ptr<f32>>\n"
           << "    %input_ptr = tt.addptr %input_base, " << inputIndex
           << " : tensor<" << blockSize << "x!tt.ptr<f32>>, tensor<"
           << blockSize << "xi64>\n"
           << "    %values = tt.load %input_ptr, %active, %one : tensor<"
           << blockSize << "x!tt.ptr<f32>>\n"
           << "    %excluded = \"tt.reduce\"(%values) <{axis = 0 : i32}> ({\n"
           << "    ^bb0(%a: f32, %b: f32):\n"
           << "      %next = arith.mulf %a, %b : f32\n"
           << "      tt.reduce.return %next : f32\n"
           << "    }) : (tensor<" << blockSize << "xf32>) -> f32\n"
           << "    %upstream_row_stride = arith.constant "
           << upstreamRowStride << " : i64\n"
           << "    %upstream_index0 = arith.muli " << row
           << ", %upstream_row_stride : i64\n";
    std::string upstreamIndex = "%upstream_index0";
    if (upstreamType.getRank() == 2) {
      output << "    %upstream_feature_stride = arith.constant "
             << upstreamFeatureStride << " : i64\n"
             << "    %upstream_feature_offset = arith.muli %feature, "
                "%upstream_feature_stride : i64\n"
             << "    %upstream_index1 = arith.addi %upstream_index0, "
                "%upstream_feature_offset : i64\n";
      upstreamIndex = "%upstream_index1";
    }
    if (upstream.getOffsetAttr().getInt() != 0) {
      output << "    %upstream_offset = arith.constant "
             << upstream.getOffsetAttr().getInt() << " : i64\n"
             << "    %upstream_index2 = arith.addi " << upstreamIndex
             << ", %upstream_offset : i64\n";
      upstreamIndex = "%upstream_index2";
    }
    output << "    %upstream_ptr = tt.addptr %arg"
           << arguments.lookup(upstream.getResult()) << ", " << upstreamIndex
           << " : !tt.ptr<f32>, i64\n"
           << "    %cotangent = tt.load %upstream_ptr : !tt.ptr<f32>\n"
           << "    %gradient = arith.mulf %excluded, %cotangent : f32\n"
           << "    %out_index = arith.extsi %program_i32 : i32 to i64\n"
           << "    %out_ptr = tt.addptr %out, %out_index : !tt.ptr<f32>, i64\n"
           << "    tt.store %out_ptr, %gradient : !tt.ptr<f32>\n"
           << "    tt.return\n  }\n}\n";
    return success();
  }

private:
  LogicalResult emitUniform(
      RankedTensorType inputType, RankedTensorType upstreamType,
      tensor::InputOp input, tensor::InputOp upstream, int64_t features) {
    SmallVector<tensor::InputOp> inputs;
    function.walk([&](tensor::InputOp value) { inputs.push_back(value); });
    DenseMap<Value, unsigned> arguments;
    output << "// graphforge.tensor entry=gf_tensor_csr_product_vjp block_rows=1 "
              "block_elements=" << blockSize << " num_warps="
           << (blockSize <= 16 ? 1 : 4) << " abi=";
    for (auto [ordinal, value] : llvm::enumerate(inputs)) {
      if (ordinal) output << ",";
      output << "arg" << ordinal;
      arguments[value.getResult()] = ordinal;
    }
    output << ",out\nmodule {\n  tt.func public @gf_tensor_csr_product_vjp(";
    for (auto [ordinal, value] : llvm::enumerate(inputs)) {
      if (ordinal) output << ", ";
      Type element = cast<RankedTensorType>(value.getResult().getType())
                         .getElementType();
      output << "%arg" << ordinal << ": !tt.ptr<"
             << tensorABIElementName(element) << ">";
    }
    int64_t degree = vjp.getMaxDegree();
    int64_t inputRowStride = input.getStrides()[0];
    int64_t inputFeatureStride =
        inputType.getRank() == 2 ? input.getStrides()[1] : 0;
    int64_t upstreamRowStride = upstream.getStrides()[0];
    int64_t upstreamFeatureStride =
        upstreamType.getRank() == 2 ? upstream.getStrides()[1] : 0;
    output << ", %out: !tt.ptr<f32>) attributes {noinline = false} {\n"
           << "    %lane_i32 = tt.make_range {end = " << blockSize
           << " : i32, start = 0 : i32} : tensor<" << blockSize << "xi32>\n"
           << "    %lane = arith.extsi %lane_i32 : tensor<" << blockSize
           << "xi32> to tensor<" << blockSize << "xi64>\n"
           << "    %one = arith.constant dense<1.000000e+00> : tensor<"
           << blockSize << "xf32>\n"
           << "    %one_matrix = arith.constant dense<1.000000e+00> : tensor<"
           << blockSize << "x" << blockSize << "xf32>\n"
           << "    %program_i32 = tt.get_program_id x : i32\n";
    if (features == 1) {
      output << "    %feature_i32 = arith.constant 0 : i32\n"
             << "    %row_i32 = arith.addi %program_i32, %feature_i32 : i32\n";
    } else {
      output << "    %feature_count_i32 = arith.constant " << features
             << " : i32\n"
             << "    %row_i32 = arith.divui %program_i32, "
                "%feature_count_i32 : i32\n"
             << "    %feature_i32 = arith.remui %program_i32, "
                "%feature_count_i32 : i32\n";
    }
    output << "    %row = arith.extsi %row_i32 : i32 to i64\n"
           << "    %feature = arith.extsi %feature_i32 : i32 to i64\n"
           << "    %degree = arith.constant " << degree << " : i64\n"
           << "    %begin = arith.muli %row, %degree : i64\n"
           << "    %begin_vector = tt.splat %begin : i64 -> tensor<"
           << blockSize << "xi64>\n"
           << "    %edge = arith.addi %begin_vector, %lane : tensor<"
           << blockSize << "xi64>\n"
           << "    %degree_vector = arith.constant dense<" << degree
           << "> : tensor<" << blockSize << "xi64>\n"
           << "    %active = arith.cmpi slt, %lane, %degree_vector : tensor<"
           << blockSize << "xi64>\n"
           << "    %input_stride = arith.constant dense<" << inputRowStride
           << "> : tensor<" << blockSize << "xi64>\n"
           << "    %input_index0 = arith.muli %edge, %input_stride : tensor<"
           << blockSize << "xi64>\n";
    std::string inputIndex = "%input_index0";
    if (inputType.getRank() == 2) {
      output << "    %input_feature_stride = arith.constant "
             << inputFeatureStride << " : i64\n"
             << "    %feature_offset0 = arith.muli %feature, "
                "%input_feature_stride : i64\n"
             << "    %feature_offset = tt.splat %feature_offset0 : i64 -> "
                "tensor<" << blockSize << "xi64>\n"
             << "    %input_index1 = arith.addi %input_index0, "
                "%feature_offset : tensor<" << blockSize << "xi64>\n";
      inputIndex = "%input_index1";
    }
    if (input.getOffsetAttr().getInt() != 0) {
      output << "    %input_offset = arith.constant dense<"
             << input.getOffsetAttr().getInt() << "> : tensor<" << blockSize
             << "xi64>\n"
             << "    %input_index2 = arith.addi " << inputIndex
             << ", %input_offset : tensor<" << blockSize << "xi64>\n";
      inputIndex = "%input_index2";
    }
    output << "    %input_base = tt.splat %arg"
           << arguments.lookup(input.getResult())
           << " : !tt.ptr<f32> -> tensor<" << blockSize
           << "x!tt.ptr<f32>>\n"
           << "    %input_ptr = tt.addptr %input_base, " << inputIndex
           << " : tensor<" << blockSize << "x!tt.ptr<f32>>, tensor<"
           << blockSize << "xi64>\n"
           << "    %values = tt.load %input_ptr, %active, %one : tensor<"
           << blockSize << "x!tt.ptr<f32>>\n"
           << "    %values_row = tt.expand_dims %values {axis = 0 : i32} : "
              "tensor<" << blockSize << "xf32> -> tensor<1x" << blockSize
           << "xf32>\n"
           << "    %values_matrix = tt.broadcast %values_row : tensor<1x"
           << blockSize << "xf32> -> tensor<" << blockSize << "x"
           << blockSize << "xf32>\n"
           << "    %self_lane = tt.expand_dims %lane {axis = 1 : i32} : tensor<"
           << blockSize << "xi64> -> tensor<" << blockSize << "x1xi64>\n"
           << "    %candidate_lane = tt.expand_dims %lane {axis = 0 : i32} : "
              "tensor<" << blockSize << "xi64> -> tensor<1x" << blockSize
           << "xi64>\n"
           << "    %self_matrix = tt.broadcast %self_lane : tensor<"
           << blockSize << "x1xi64> -> tensor<" << blockSize << "x"
           << blockSize << "xi64>\n"
           << "    %candidate_matrix = tt.broadcast %candidate_lane : tensor<1x"
           << blockSize << "xi64> -> tensor<" << blockSize << "x"
           << blockSize << "xi64>\n"
           << "    %not_self = arith.cmpi ne, %self_matrix, %candidate_matrix : "
              "tensor<" << blockSize << "x" << blockSize << "xi64>\n"
           << "    %factors = arith.select %not_self, %values_matrix, "
              "%one_matrix : tensor<" << blockSize << "x" << blockSize
           << "xi1>, tensor<" << blockSize << "x" << blockSize << "xf32>\n"
           << "    %excluded = \"tt.reduce\"(%factors) <{axis = 1 : i32}> ({\n"
           << "    ^bb0(%a: f32, %b: f32):\n"
           << "      %next = arith.mulf %a, %b : f32\n"
           << "      tt.reduce.return %next : f32\n"
           << "    }) : (tensor<" << blockSize << "x" << blockSize
           << "xf32>) -> tensor<" << blockSize << "xf32>\n"
           << "    %upstream_row_stride = arith.constant "
           << upstreamRowStride << " : i64\n"
           << "    %upstream_index0 = arith.muli %row, "
              "%upstream_row_stride : i64\n";
    std::string upstreamIndex = "%upstream_index0";
    if (upstreamType.getRank() == 2) {
      output << "    %upstream_feature_stride = arith.constant "
             << upstreamFeatureStride << " : i64\n"
             << "    %upstream_feature_offset = arith.muli %feature, "
                "%upstream_feature_stride : i64\n"
             << "    %upstream_index1 = arith.addi %upstream_index0, "
                "%upstream_feature_offset : i64\n";
      upstreamIndex = "%upstream_index1";
    }
    if (upstream.getOffsetAttr().getInt() != 0) {
      output << "    %upstream_offset = arith.constant "
             << upstream.getOffsetAttr().getInt() << " : i64\n"
             << "    %upstream_index2 = arith.addi " << upstreamIndex
             << ", %upstream_offset : i64\n";
      upstreamIndex = "%upstream_index2";
    }
    output << "    %upstream_ptr = tt.addptr %arg"
           << arguments.lookup(upstream.getResult()) << ", " << upstreamIndex
           << " : !tt.ptr<f32>, i64\n"
           << "    %cotangent = tt.load %upstream_ptr : !tt.ptr<f32>\n"
           << "    %cotangent_vector = tt.splat %cotangent : f32 -> tensor<"
           << blockSize << "xf32>\n"
           << "    %gradient = arith.mulf %excluded, %cotangent_vector : tensor<"
           << blockSize << "xf32>\n";
    StringRef outputIndex = "%edge";
    if (features != 1) {
      output << "    %feature_count = arith.constant dense<" << features
             << "> : tensor<" << blockSize << "xi64>\n"
             << "    %out_index0 = arith.muli %edge, %feature_count : tensor<"
             << blockSize << "xi64>\n"
             << "    %feature_vector = tt.splat %feature : i64 -> tensor<"
             << blockSize << "xi64>\n"
             << "    %out_index = arith.addi %out_index0, %feature_vector : tensor<"
             << blockSize << "xi64>\n";
      outputIndex = "%out_index";
    }
    output << "    %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<"
           << blockSize << "x!tt.ptr<f32>>\n"
           << "    %out_ptr = tt.addptr %out_base, " << outputIndex
           << " : tensor<"
           << blockSize << "x!tt.ptr<f32>>, tensor<" << blockSize << "xi64>\n"
           << "    tt.store %out_ptr, %gradient, %active : tensor<" << blockSize
           << "x!tt.ptr<f32>>\n"
           << "    tt.return\n  }\n}\n";
    return success();
  }

  tensor::CSRSegmentProductVJPOp vjp;
  func::FuncOp function;
  int64_t blockSize;
  llvm::raw_ostream &output;
};

class TensorMatmulTTIREmitter {
public:
  TensorMatmulTTIREmitter(tensor::MatmulOp matmul, func::FuncOp function,
                          llvm::raw_ostream &output)
      : matmul(matmul), function(function), output(output) {}

  LogicalResult emit() {
    auto lhsType = dyn_cast<RankedTensorType>(matmul.getLhs().getType());
    auto rhsType = dyn_cast<RankedTensorType>(matmul.getRhs().getType());
    auto resultType = dyn_cast<RankedTensorType>(matmul.getResult().getType());
    if (!lhsType || !rhsType || !resultType || !lhsType.hasStaticShape() ||
        !rhsType.hasStaticShape() || !resultType.hasStaticShape() ||
        lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
        !lhsType.getElementType().isF16() ||
        rhsType.getElementType() != lhsType.getElementType() ||
        resultType.getElementType() != lhsType.getElementType())
      return matmul.emitError(
          "GPU tensor-core matmul requires static rank-two FP16 tensors");
    struct MatrixBinding {
      tensor::InputOp input;
      int64_t offset;
      int64_t rowStride;
      int64_t columnStride;
    };
    auto resolve = [&](Value value) -> std::optional<MatrixBinding> {
      if (auto input = value.getDefiningOp<tensor::InputOp>()) {
        if (input.getStrides().size() != 2) return std::nullopt;
        return MatrixBinding{input, input.getOffsetAttr().getInt(),
                             input.getStrides()[0], input.getStrides()[1]};
      }
      if (auto permutation = value.getDefiningOp<tensor::PermuteOp>()) {
        auto input = permutation.getInput().getDefiningOp<tensor::InputOp>();
        if (!input || input.getStrides().size() != 2 ||
            permutation.getAxes().size() != 2 ||
            permutation.getAxes()[0] != 1 || permutation.getAxes()[1] != 0)
          return std::nullopt;
        return MatrixBinding{input, input.getOffsetAttr().getInt(),
                             input.getStrides()[1], input.getStrides()[0]};
      }
      return std::nullopt;
    };
    auto lhs = resolve(matmul.getLhs());
    auto rhs = resolve(matmul.getRhs());
    if (!lhs || !rhs)
      return matmul.emitError(
          "GPU tensor-core matmul requires direct or transposed ABI inputs");
    SmallVector<tensor::InputOp> inputs;
    function.walk([&](tensor::InputOp input) { inputs.push_back(input); });
    if (inputs.size() != 2 || inputs[0] != lhs->input ||
        inputs[1] != rhs->input)
      return matmul.emitError("matmul ABI must contain lhs then rhs");

    constexpr int64_t BM = 128, BN = 128, BK = 32;
    int64_t m = lhsType.getDimSize(0);
    int64_t k = lhsType.getDimSize(1);
    int64_t n = rhsType.getDimSize(1);
    int64_t gridN = (n + BN - 1) / BN;
    output << "// graphforge.tensor entry=gf_tensor_matmul block_rows=" << BM
           << " block_elements=" << BN
           << " num_warps=4 abi=arg0,arg1,out\nmodule {\n"
           << "  tt.func public @gf_tensor_matmul(%arg0: !tt.ptr<f16>, "
              "%arg1: !tt.ptr<f16>, %out: !tt.ptr<f16>) attributes "
              "{noinline = false} {\n"
           << "    %zero_a = arith.constant dense<0.000000e+00> : tensor<"
           << BM << "x" << BK << "xf16>\n"
           << "    %zero_b = arith.constant dense<0.000000e+00> : tensor<"
           << BK << "x" << BN << "xf16>\n"
           << "    %zero_acc = arith.constant dense<0.000000e+00> : tensor<"
           << BM << "x" << BN << "xf32>\n"
           << "    %pid = tt.get_program_id x : i32\n"
           << "    %grid_n = arith.constant " << gridN << " : i32\n"
           << "    %pid_m = arith.divui %pid, %grid_n : i32\n"
           << "    %pid_n = arith.remui %pid, %grid_n : i32\n"
           << "    %bm = arith.constant " << BM << " : i32\n"
           << "    %bn = arith.constant " << BN << " : i32\n"
           << "    %row_start = arith.muli %pid_m, %bm : i32\n"
           << "    %col_start = arith.muli %pid_n, %bn : i32\n"
           << "    %row_lane = tt.make_range {end = " << BM
           << " : i32, start = 0 : i32} : tensor<" << BM << "xi32>\n"
           << "    %col_lane = tt.make_range {end = " << BN
           << " : i32, start = 0 : i32} : tensor<" << BN << "xi32>\n"
           << "    %row_base = tt.splat %row_start : i32 -> tensor<" << BM
           << "xi32>\n"
           << "    %col_base = tt.splat %col_start : i32 -> tensor<" << BN
           << "xi32>\n"
           << "    %rows = arith.addi %row_base, %row_lane : tensor<" << BM
           << "xi32>\n"
           << "    %cols = arith.addi %col_base, %col_lane : tensor<" << BN
           << "xi32>\n"
           << "    %m_bound = arith.constant dense<" << m << "> : tensor<"
           << BM << "xi32>\n"
           << "    %n_bound = arith.constant dense<" << n << "> : tensor<"
           << BN << "xi32>\n"
           << "    %row_mask = arith.cmpi slt, %rows, %m_bound : tensor<" << BM
           << "xi32>\n"
           << "    %col_mask = arith.cmpi slt, %cols, %n_bound : tensor<" << BN
           << "xi32>\n"
           << "    %c0 = arith.constant 0 : i32\n"
           << "    %cend = arith.constant " << k << " : i32\n"
           << "    %cstep = arith.constant " << BK << " : i32\n"
           << "    %state = scf.for %start = %c0 to %cend step %cstep "
              "iter_args(%acc = %zero_acc) -> (tensor<"
           << BM << "x" << BN << "xf32>) : i32 {\n"
           << "      %k_lane = tt.make_range {end = " << BK
           << " : i32, start = 0 : i32} : tensor<" << BK << "xi32>\n"
           << "      %k_base = tt.splat %start : i32 -> tensor<" << BK
           << "xi32>\n"
           << "      %ks = arith.addi %k_base, %k_lane : tensor<" << BK
           << "xi32>\n"
           << "      %k_bound = arith.constant dense<" << k << "> : tensor<"
           << BK << "xi32>\n"
           << "      %k_mask = arith.cmpi slt, %ks, %k_bound : tensor<" << BK
           << "xi32>\n"
           << "      %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<"
           << BM << "xi32> -> tensor<" << BM << "x1xi32>\n"
           << "      %ks_a = tt.expand_dims %ks {axis = 0 : i32} : tensor<"
           << BK << "xi32> -> tensor<1x" << BK << "xi32>\n"
           << "      %k_stride_a = arith.constant dense<" << lhs->rowStride
           << "> : tensor<"
           << BM << "x1xi32>\n"
           << "      %a_row = arith.muli %rows_2d, %k_stride_a : tensor<" << BM
           << "x1xi32>\n"
           << "      %a_row_b = tt.broadcast %a_row : tensor<" << BM
           << "x1xi32> -> tensor<" << BM << "x" << BK << "xi32>\n"
           << "      %a_col_stride = arith.constant dense<"
           << lhs->columnStride << "> : tensor<1x" << BK << "xi32>\n"
           << "      %ks_a_scaled = arith.muli %ks_a, %a_col_stride : "
              "tensor<1x" << BK << "xi32>\n"
           << "      %ks_a_b = tt.broadcast %ks_a_scaled : tensor<1x" << BK
           << "xi32> -> tensor<" << BM << "x" << BK << "xi32>\n"
           << "      %a_offset0 = arith.addi %a_row_b, %ks_a_b : tensor<" << BM
           << "x" << BK << "xi32>\n"
           << "      %a_storage_offset = arith.constant dense<" << lhs->offset
           << "> : tensor<" << BM << "x" << BK << "xi32>\n"
           << "      %a_offset = arith.addi %a_offset0, %a_storage_offset : "
              "tensor<" << BM << "x" << BK << "xi32>\n"
           << "      %a_base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<" << BM
           << "x" << BK << "x!tt.ptr<f16>>\n"
           << "      %a_ptr = tt.addptr %a_base, %a_offset : tensor<" << BM
           << "x" << BK << "x!tt.ptr<f16>>, tensor<" << BM << "x" << BK
           << "xi32>\n"
           << "      %row_mask_2d = tt.expand_dims %row_mask {axis = 1 : i32} "
              ": tensor<" << BM << "xi1> -> tensor<" << BM << "x1xi1>\n"
           << "      %k_mask_a = tt.expand_dims %k_mask {axis = 0 : i32} : "
              "tensor<" << BK << "xi1> -> tensor<1x" << BK << "xi1>\n"
           << "      %row_mask_b = tt.broadcast %row_mask_2d : tensor<" << BM
           << "x1xi1> -> tensor<" << BM << "x" << BK << "xi1>\n"
           << "      %k_mask_a_b = tt.broadcast %k_mask_a : tensor<1x" << BK
           << "xi1> -> tensor<" << BM << "x" << BK << "xi1>\n"
           << "      %a_mask = arith.andi %row_mask_b, %k_mask_a_b : tensor<"
           << BM << "x" << BK << "xi1>\n"
           << "      %a = tt.load %a_ptr, %a_mask, %zero_a : tensor<" << BM
           << "x" << BK << "x!tt.ptr<f16>>\n"
           << "      %ks_b = tt.expand_dims %ks {axis = 1 : i32} : tensor<"
           << BK << "xi32> -> tensor<" << BK << "x1xi32>\n"
           << "      %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<"
           << BN << "xi32> -> tensor<1x" << BN << "xi32>\n"
           << "      %n_stride = arith.constant dense<" << rhs->rowStride
           << "> : tensor<"
           << BK << "x1xi32>\n"
           << "      %b_row = arith.muli %ks_b, %n_stride : tensor<" << BK
           << "x1xi32>\n"
           << "      %b_row_b = tt.broadcast %b_row : tensor<" << BK
           << "x1xi32> -> tensor<" << BK << "x" << BN << "xi32>\n"
           << "      %b_col_stride = arith.constant dense<"
           << rhs->columnStride << "> : tensor<1x" << BN << "xi32>\n"
           << "      %cols_scaled = arith.muli %cols_2d, %b_col_stride : "
              "tensor<1x" << BN << "xi32>\n"
           << "      %cols_b = tt.broadcast %cols_scaled : tensor<1x" << BN
           << "xi32> -> tensor<" << BK << "x" << BN << "xi32>\n"
           << "      %b_offset0 = arith.addi %b_row_b, %cols_b : tensor<" << BK
           << "x" << BN << "xi32>\n"
           << "      %b_storage_offset = arith.constant dense<" << rhs->offset
           << "> : tensor<" << BK << "x" << BN << "xi32>\n"
           << "      %b_offset = arith.addi %b_offset0, %b_storage_offset : "
              "tensor<" << BK << "x" << BN << "xi32>\n"
           << "      %b_base = tt.splat %arg1 : !tt.ptr<f16> -> tensor<" << BK
           << "x" << BN << "x!tt.ptr<f16>>\n"
           << "      %b_ptr = tt.addptr %b_base, %b_offset : tensor<" << BK
           << "x" << BN << "x!tt.ptr<f16>>, tensor<" << BK << "x" << BN
           << "xi32>\n"
           << "      %k_mask_b = tt.expand_dims %k_mask {axis = 1 : i32} : "
              "tensor<" << BK << "xi1> -> tensor<" << BK << "x1xi1>\n"
           << "      %col_mask_2d = tt.expand_dims %col_mask {axis = 0 : i32} "
              ": tensor<" << BN << "xi1> -> tensor<1x" << BN << "xi1>\n"
           << "      %k_mask_b_b = tt.broadcast %k_mask_b : tensor<" << BK
           << "x1xi1> -> tensor<" << BK << "x" << BN << "xi1>\n"
           << "      %col_mask_b = tt.broadcast %col_mask_2d : tensor<1x" << BN
           << "xi1> -> tensor<" << BK << "x" << BN << "xi1>\n"
           << "      %b_mask = arith.andi %k_mask_b_b, %col_mask_b : tensor<"
           << BK << "x" << BN << "xi1>\n"
           << "      %b = tt.load %b_ptr, %b_mask, %zero_b : tensor<" << BK
           << "x" << BN << "x!tt.ptr<f16>>\n"
           << "      %next = tt.dot %a, %b, %acc, inputPrecision = ieee : "
              "tensor<" << BM << "x" << BK << "xf16> * tensor<" << BK
           << "x" << BN << "xf16> -> tensor<" << BM << "x" << BN
           << "xf32>\n"
           << "      scf.yield %next : tensor<" << BM << "x" << BN
           << "xf32>\n    }\n"
           << "    %result = arith.truncf %state : tensor<" << BM << "x" << BN
           << "xf32> to tensor<" << BM << "x" << BN << "xf16>\n"
           << "    %rows_out = tt.expand_dims %rows {axis = 1 : i32} : tensor<"
           << BM << "xi32> -> tensor<" << BM << "x1xi32>\n"
           << "    %n_stride_out = arith.constant dense<" << n << "> : tensor<"
           << BM << "x1xi32>\n"
           << "    %out_row = arith.muli %rows_out, %n_stride_out : tensor<"
           << BM << "x1xi32>\n"
           << "    %out_row_b = tt.broadcast %out_row : tensor<" << BM
           << "x1xi32> -> tensor<" << BM << "x" << BN << "xi32>\n"
           << "    %cols_out = tt.expand_dims %cols {axis = 0 : i32} : tensor<"
           << BN << "xi32> -> tensor<1x" << BN << "xi32>\n"
           << "    %cols_out_b = tt.broadcast %cols_out : tensor<1x" << BN
           << "xi32> -> tensor<" << BM << "x" << BN << "xi32>\n"
           << "    %out_offset = arith.addi %out_row_b, %cols_out_b : tensor<"
           << BM << "x" << BN << "xi32>\n"
           << "    %out_base = tt.splat %out : !tt.ptr<f16> -> tensor<" << BM
           << "x" << BN << "x!tt.ptr<f16>>\n"
           << "    %out_ptr = tt.addptr %out_base, %out_offset : tensor<" << BM
           << "x" << BN << "x!tt.ptr<f16>>, tensor<" << BM << "x" << BN
           << "xi32>\n"
           << "    %row_mask_out_2d = tt.expand_dims %row_mask {axis = 1 : i32} "
              ": tensor<" << BM << "xi1> -> tensor<" << BM << "x1xi1>\n"
           << "    %col_mask_out_2d = tt.expand_dims %col_mask {axis = 0 : i32} "
              ": tensor<" << BN << "xi1> -> tensor<1x" << BN << "xi1>\n"
           << "    %row_out_b = tt.broadcast %row_mask_out_2d : tensor<" << BM
           << "x1xi1> -> tensor<" << BM << "x" << BN << "xi1>\n"
           << "    %col_out_b = tt.broadcast %col_mask_out_2d : tensor<1x" << BN
           << "xi1> -> tensor<" << BM << "x" << BN << "xi1>\n"
           << "    %out_mask = arith.andi %row_out_b, %col_out_b : tensor<"
           << BM << "x" << BN << "xi1>\n"
           << "    tt.store %out_ptr, %result, %out_mask : tensor<" << BM
           << "x" << BN << "x!tt.ptr<f16>>\n"
           << "    tt.return\n  }\n}\n";
    return success();
  }

private:
  tensor::MatmulOp matmul;
  func::FuncOp function;
  llvm::raw_ostream &output;
};

class ComplexTensorPointwiseTTIREmitter {
public:
  using Pair = std::pair<std::string, std::string>;

  ComplexTensorPointwiseTTIREmitter(func::FuncOp function, Value result,
                                    int64_t blockSize,
                                    llvm::raw_ostream &output)
      : function(function), result(result), blockSize(blockSize), output(output) {}

  LogicalResult emit() {
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    auto complexType = resultType
                           ? dyn_cast<ComplexType>(resultType.getElementType())
                           : ComplexType();
    auto component = complexType
                         ? dyn_cast<FloatType>(complexType.getElementType())
                         : FloatType();
    if (!resultType || !resultType.hasStaticShape() ||
        (resultType.getRank() != 1 && resultType.getRank() != 2) ||
        !component || (component.getWidth() != 32 && component.getWidth() != 64))
      return function.emitError(
          "complex pointwise TTIR requires a static rank-one/rank-two "
          "complex64/complex128 result");
    elementType = complexType;
    componentName = component.getWidth() == 32 ? "f32" : "f64";
    elements = resultType.getNumElements();
    function.walk([&](tensor::InputOp input) {
      argumentIndex[input.getResult()] = inputs.size();
      inputs.push_back(input);
    });
    if (inputs.empty()) return function.emitError("requires gf_tensor.input");
    for (tensor::InputOp input : inputs) {
      auto type = dyn_cast<RankedTensorType>(input.getResult().getType());
      if (!type || !type.hasStaticShape() || type.getRank() > 2 ||
          type.getElementType() != elementType ||
          (type.getNumElements() != 1 && type.getNumElements() != elements))
        return input.emitError(
            "complex pointwise input must be scalar-like or match result storage");
      if (type.getNumElements() != 1) {
        auto strides = input.getStrides();
        if ((type.getRank() == 1 && strides[0] != 1) ||
            (type.getRank() == 2 &&
             (strides[1] != 1 || strides[0] != type.getDimSize(1))))
          return input.emitError(
              "complex CUDA pointwise currently requires contiguous inputs");
      }
    }
    output << "// graphforge.tensor entry=gf_tensor_pointwise "
              "block_rows=" << blockSize << " block_elements=" << blockSize
           << " num_warps=8 abi=";
    for (auto [index, input] : llvm::enumerate(inputs)) {
      if (index) output << ",";
      output << "arg" << index;
    }
    output << ",out\nmodule {\n  tt.func public @gf_tensor_pointwise(";
    for (auto [index, input] : llvm::enumerate(inputs)) {
      if (index) output << ", ";
      output << "%arg" << index << ": !tt.ptr<" << componentName << ">";
    }
    output << ", %out: !tt.ptr<" << componentName
           << ">) attributes {noinline = false} {\n"
           << "    %lane_i32 = tt.make_range {end = " << blockSize
           << " : i32, start = 0 : i32} : tensor<" << blockSize << "xi32>\n"
           << "    %pid = tt.get_program_id x : i32\n"
           << "    %block = arith.constant " << blockSize << " : i32\n"
           << "    %base_i32 = arith.muli %pid, %block : i32\n"
           << "    %base = arith.extsi %base_i32 : i32 to i64\n"
           << "    %base_v = tt.splat %base : i64 -> tensor<" << blockSize << "xi64>\n"
           << "    %lane = arith.extsi %lane_i32 : tensor<" << blockSize
           << "xi32> to tensor<" << blockSize << "xi64>\n"
           << "    %index = arith.addi %base_v, %lane : tensor<" << blockSize << "xi64>\n"
           << "    %bound = arith.constant dense<" << elements << "> : tensor<"
           << blockSize << "xi64>\n"
           << "    %mask = arith.cmpi slt, %index, %bound : tensor<"
           << blockSize << "xi64>\n"
           << "    %zero = arith.constant dense<0.000000e+00> : tensor<"
           << blockSize << "x" << componentName << ">\n"
           << "    %two = arith.constant dense<2> : tensor<" << blockSize << "xi64>\n"
           << "    %one = arith.constant dense<1> : tensor<" << blockSize << "xi64>\n";
    FailureOr<Pair> value = emitValue(result);
    if (failed(value)) return failure();
    output << "    %out_physical = arith.muli %index, %two : tensor<"
           << blockSize << "xi64>\n";
    emitStore("%out_physical", value->first);
    output << "    %out_imag_index = arith.addi %out_physical, %one : tensor<"
           << blockSize << "xi64>\n";
    emitStore("%out_imag_index", value->second);
    output << "    tt.return\n  }\n}\n";
    return success();
  }

private:
  FailureOr<Pair> emitValue(Value value) {
    auto found = names.find(value);
    if (found != names.end()) return found->second;
    Operation *operation = value.getDefiningOp();
    if (auto input = dyn_cast_or_null<tensor::InputOp>(operation))
      return emitInput(input, value);
    if (auto reshape = dyn_cast_or_null<tensor::ReshapeOp>(operation))
      return alias(value, reshape.getInput());
    if (auto broadcast = dyn_cast_or_null<tensor::BroadcastOp>(operation))
      return alias(value, broadcast.getInput());
    if (auto checkpoint = dyn_cast_or_null<tensor::CheckpointOp>(operation))
      return alias(value, checkpoint.getInput());
    if (auto candidate = dyn_cast_or_null<tensor::CheckpointCandidateOp>(operation))
      return alias(value, candidate.getInput());
    if (auto add = dyn_cast_or_null<tensor::AddOp>(operation))
      return emitBinary(value, add.getLhs(), add.getRhs(), "add");
    if (auto mul = dyn_cast_or_null<tensor::MulOp>(operation))
      return emitBinary(value, mul.getLhs(), mul.getRhs(), "mul");
    if (auto divide = dyn_cast_or_null<tensor::DivOp>(operation))
      return emitBinary(value, divide.getLhs(), divide.getRhs(), "div");
    if (auto neg = dyn_cast_or_null<tensor::NegOp>(operation)) {
      auto operand = emitValue(neg.getInput());
      if (failed(operand)) return failure();
      Pair result{binary("%zero", operand->first, "arith.subf", "neg_re"),
                  binary("%zero", operand->second, "arith.subf", "neg_im")};
      names[value] = result;
      return result;
    }
    if (auto conjugate = dyn_cast_or_null<tensor::ConjOp>(operation)) {
      auto operand = emitValue(conjugate.getInput());
      if (failed(operand)) return failure();
      Pair result{operand->first,
                  binary("%zero", operand->second, "arith.subf", "conj_im")};
      names[value] = result;
      return result;
    }
    if (operation)
      operation->emitError(
          "complex pointwise TTIR supports input/view/add/mul/div/neg/conj producers");
    return failure();
  }

  FailureOr<Pair> alias(Value resultValue, Value input) {
    auto result = emitValue(input);
    if (succeeded(result)) names[resultValue] = *result;
    return result;
  }

  FailureOr<Pair> emitInput(tensor::InputOp input, Value resultValue) {
    auto type = cast<RankedTensorType>(input.getResult().getType());
    std::string logical = "%index";
    if (type.getNumElements() == 1) {
      logical = next("scalar_index");
      output << "    " << logical << " = arith.constant dense<"
             << input.getOffsetAttr().getInt() << "> : tensor<" << blockSize
             << "xi64>\n";
    } else if (input.getOffsetAttr().getInt() != 0) {
      std::string offset = next("offset");
      logical = next("logical_index");
      output << "    " << offset << " = arith.constant dense<"
             << input.getOffsetAttr().getInt() << "> : tensor<" << blockSize
             << "xi64>\n    " << logical << " = arith.addi %index, "
             << offset << " : tensor<" << blockSize << "xi64>\n";
    }
    std::string physical = next("physical_index");
    std::string imaginary = next("imaginary_index");
    output << "    " << physical << " = arith.muli " << logical
           << ", %two : tensor<" << blockSize << "xi64>\n"
           << "    " << imaginary << " = arith.addi " << physical
           << ", %one : tensor<" << blockSize << "xi64>\n";
    unsigned index = argumentIndex.lookup(input.getResult());
    Pair result{emitLoad(index, physical), emitLoad(index, imaginary)};
    names[resultValue] = result;
    return result;
  }

  FailureOr<Pair> emitBinary(Value value, Value lhs, Value rhs, StringRef kind) {
    auto left = emitValue(lhs);
    auto right = emitValue(rhs);
    if (failed(left) || failed(right)) return failure();
    Pair result;
    if (kind == "add") {
      result = {binary(left->first, right->first, "arith.addf", "add_re"),
                binary(left->second, right->second, "arith.addf", "add_im")};
    } else if (kind == "mul") {
      auto ac = binary(left->first, right->first, "arith.mulf", "ac");
      auto bd = binary(left->second, right->second, "arith.mulf", "bd");
      auto ad = binary(left->first, right->second, "arith.mulf", "ad");
      auto bc = binary(left->second, right->first, "arith.mulf", "bc");
      result = {binary(ac, bd, "arith.subf", "mul_re"),
                binary(ad, bc, "arith.addf", "mul_im")};
    } else {
      auto cc = binary(right->first, right->first, "arith.mulf", "cc");
      auto dd = binary(right->second, right->second, "arith.mulf", "dd");
      auto denominator = binary(cc, dd, "arith.addf", "denominator");
      auto ac = binary(left->first, right->first, "arith.mulf", "ac");
      auto bd = binary(left->second, right->second, "arith.mulf", "bd");
      auto bc = binary(left->second, right->first, "arith.mulf", "bc");
      auto ad = binary(left->first, right->second, "arith.mulf", "ad");
      result = {
          binary(binary(ac, bd, "arith.addf", "numerator_re"), denominator,
                 "arith.divf", "div_re"),
          binary(binary(bc, ad, "arith.subf", "numerator_im"), denominator,
                 "arith.divf", "div_im")};
    }
    names[value] = result;
    return result;
  }

  std::string unary(StringRef operand, StringRef mnemonic, StringRef stem) {
    std::string name = next(stem);
    output << "    " << name << " = " << mnemonic << " " << operand
           << " : tensor<" << blockSize << "x" << componentName << ">\n";
    return name;
  }
  std::string binary(StringRef lhs, StringRef rhs, StringRef mnemonic,
                     StringRef stem) {
    std::string name = next(stem);
    output << "    " << name << " = " << mnemonic << " " << lhs << ", "
           << rhs << " : tensor<" << blockSize << "x" << componentName << ">\n";
    return name;
  }
  std::string emitLoad(unsigned argument, StringRef index) {
    std::string base = next("input_base");
    std::string pointer = next("input_ptr");
    std::string loaded = next("input");
    output << "    " << base << " = tt.splat %arg" << argument
           << " : !tt.ptr<" << componentName << "> -> tensor<" << blockSize
           << "x!tt.ptr<" << componentName << ">>\n    " << pointer
           << " = tt.addptr " << base << ", " << index << " : tensor<"
           << blockSize << "x!tt.ptr<" << componentName << ">>, tensor<"
           << blockSize << "xi64>\n    " << loaded << " = tt.load " << pointer
           << ", %mask, %zero : tensor<" << blockSize << "x!tt.ptr<"
           << componentName << ">>\n";
    return loaded;
  }
  void emitStore(StringRef index, StringRef value) {
    std::string base = next("out_base");
    std::string pointer = next("out_ptr");
    output << "    " << base << " = tt.splat %out : !tt.ptr<" << componentName
           << "> -> tensor<" << blockSize << "x!tt.ptr<" << componentName
           << ">>\n    " << pointer << " = tt.addptr " << base << ", " << index
           << " : tensor<" << blockSize << "x!tt.ptr<" << componentName
           << ">>, tensor<" << blockSize << "xi64>\n    tt.store " << pointer
           << ", " << value << ", %mask : tensor<" << blockSize
           << "x!tt.ptr<" << componentName << ">>\n";
  }
  std::string next(StringRef stem) {
    return (Twine("%") + stem + Twine(nextId++)).str();
  }

  func::FuncOp function;
  Value result;
  int64_t blockSize;
  llvm::raw_ostream &output;
  int64_t elements = 0;
  ComplexType elementType;
  std::string componentName;
  unsigned nextId = 0;
  SmallVector<tensor::InputOp> inputs;
  DenseMap<Value, unsigned> argumentIndex;
  DenseMap<Value, Pair> names;
};

/// Fuse a CSR row reduction with its elementwise node epilogue.  One program
/// owns one destination row and walks an arbitrary-length row in fixed-size
/// tiles, so correctness does not depend on a host-known maximum degree.  The
/// producer and epilogue are matched structurally; there are deliberately no
/// PageRank- or workload-named cases here.
class TensorCSRSumEpilogueTTIREmitter {
public:
  TensorCSRSumEpilogueTTIREmitter(tensor::CSRSegmentSumOp reduction,
                                  func::FuncOp function, Value result,
                                  int64_t blockSize,
                                  llvm::raw_ostream &output)
      : reduction(reduction), function(function), result(result),
        blockSize(blockSize), output(output) {}

  LogicalResult emit() {
    auto messageType = dyn_cast<RankedTensorType>(
        reduction.getInput().getType());
    auto rowType = dyn_cast<RankedTensorType>(reduction.getRowPtr().getType());
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    auto rowPtr = reduction.getRowPtr().getDefiningOp<tensor::InputOp>();
    if (!messageType || !rowType || !resultType || !rowPtr ||
        messageType.getRank() != 1 || rowType.getRank() != 1 ||
        resultType.getRank() != 1 || !messageType.hasStaticShape() ||
        !rowType.hasStaticShape() || !resultType.hasStaticShape() ||
        !messageType.getElementType().isF32() ||
        !resultType.getElementType().isF32() ||
        (!rowType.getElementType().isInteger(32) &&
         !rowType.getElementType().isInteger(64)))
      return reduction.emitError(
          "GPU fused CSR sum currently requires rank-one FP32 messages/results "
          "and a direct i32/i64 row_ptr input");
    rows = reduction.getNumRows();
    edges = messageType.getDimSize(0);
    if (rows <= 0 || edges <= 0 || rowType.getDimSize(0) != rows + 1 ||
        resultType.getDimSize(0) != rows)
      return reduction.emitError("fused CSR sum static extents are inconsistent");

    function.walk([&](tensor::InputOp input) {
      argumentIndex[input.getResult()] = inputs.size();
      inputs.push_back(input);
    });
    if (inputs.empty()) return function.emitError("requires gf_tensor.input");
    for (tensor::InputOp input : inputs) {
      auto type = dyn_cast<RankedTensorType>(input.getResult().getType());
      if (!type || !type.hasStaticShape() || type.getRank() > 1 ||
          (!type.getElementType().isF32() &&
           !type.getElementType().isInteger(32) &&
           !type.getElementType().isInteger(64)))
        return input.emitError(
            "fused CSR sum ABI supports static rank-zero/rank-one FP32/i32/i64");
    }

    // A fixed-degree weighted gather is the dominant low-degree graph case.
    // Preserve the generic arbitrary-row loop below as the correctness path,
    // but batch rows here when typed degree bounds prove that one rectangular
    // relation tile is safe.
    if (succeeded(emitUniformWeighted())) return success();

    StringRef indexName = rowType.getElementType().isInteger(32) ? "i32" : "i64";
    int64_t numWarps = std::max<int64_t>(1, blockSize / 32);
    output << "// graphforge.tensor entry=gf_tensor_csr_sum_epilogue "
              "block_rows=1 block_elements="
           << blockSize << " num_warps=" << numWarps << " abi=";
    for (auto [ordinal, input] : llvm::enumerate(inputs)) {
      if (ordinal) output << ",";
      output << "arg" << ordinal;
    }
    output << ",out\nmodule {\n  tt.func public @gf_tensor_csr_sum_epilogue(";
    for (auto [ordinal, input] : llvm::enumerate(inputs)) {
      if (ordinal) output << ", ";
      output << "%arg" << ordinal << ": !tt.ptr<"
             << spelling(cast<RankedTensorType>(input.getResult().getType())
                             .getElementType())
             << ">";
    }
    output << ", %out: !tt.ptr<f32>) attributes {noinline = false} {\n"
           << "    %c0 = arith.constant 0 : i64\n"
           << "    %c1 = arith.constant 1 : i64\n"
           << "    %tile_step = arith.constant " << blockSize << " : i64\n"
           << "    %zero_scalar = arith.constant 0.000000e+00 : f32\n"
           << "    %zero_vector = arith.constant dense<0.000000e+00> : tensor<"
           << blockSize << "xf32>\n"
           << "    %zero_i32 = arith.constant dense<0> : tensor<" << blockSize
           << "xi32>\n"
           << "    %zero_i64 = arith.constant dense<0> : tensor<" << blockSize
           << "xi64>\n"
           << "    %lane_i32 = tt.make_range {end = " << blockSize
           << " : i32, start = 0 : i32} : tensor<" << blockSize << "xi32>\n"
           << "    %lane = arith.extsi %lane_i32 : tensor<" << blockSize
           << "xi32> to tensor<" << blockSize << "xi64>\n"
           << "    %row_i32 = tt.get_program_id x : i32\n"
           << "    %row = arith.extsi %row_i32 : i32 to i64\n";

    unsigned rowArgument = argumentIndex.lookup(rowPtr.getResult());
    int64_t rowStride = rowPtr.getStrides()[0];
    int64_t rowOffset = rowPtr.getOffsetAttr().getInt();
    output << "    %row_stride = arith.constant " << rowStride << " : i64\n"
           << "    %row_offset = arith.constant " << rowOffset << " : i64\n"
           << "    %row_scaled = arith.muli %row, %row_stride : i64\n"
           << "    %begin_index = arith.addi %row_scaled, %row_offset : i64\n"
           << "    %end_index = arith.addi %begin_index, %row_stride : i64\n"
           << "    %begin_ptr = tt.addptr %arg" << rowArgument
           << ", %begin_index : !tt.ptr<" << indexName << ">, i64\n"
           << "    %end_ptr = tt.addptr %arg" << rowArgument
           << ", %end_index : !tt.ptr<" << indexName << ">, i64\n"
           << "    %begin_raw = tt.load %begin_ptr : !tt.ptr<" << indexName
           << ">\n"
           << "    %end_raw = tt.load %end_ptr : !tt.ptr<" << indexName
           << ">\n";
    StringRef begin = "%begin_raw", end = "%end_raw";
    if (rowType.getElementType().isInteger(32)) {
      output << "    %begin = arith.extsi %begin_raw : i32 to i64\n"
             << "    %end = arith.extsi %end_raw : i32 to i64\n";
      begin = "%begin";
      end = "%end";
    }
    output << "    %sum = scf.for %tile = " << begin << " to " << end
           << " step %tile_step iter_args(%acc = %zero_scalar) -> (f32) : i64 {\n"
           << "      %tile_v = tt.splat %tile : i64 -> tensor<" << blockSize
           << "xi64>\n"
           << "      %candidate = arith.addi %tile_v, %lane : tensor<"
           << blockSize << "xi64>\n"
           << "      %end_v = tt.splat " << end << " : i64 -> tensor<"
           << blockSize << "xi64>\n"
           << "      %active = arith.cmpi slt, %candidate, %end_v : tensor<"
           << blockSize << "xi64>\n";
    auto message = emitVector(reduction.getInput(), "%candidate", "%active");
    if (failed(message)) return failure();
    output << "      %tile_sum = \"tt.reduce\"(" << *message
           << ") <{axis = 0 : i32}> ({\n"
           << "      ^bb0(%a: f32, %b: f32):\n"
           << "        %combined = arith.addf %a, %b : f32\n"
           << "        tt.reduce.return %combined : f32\n"
           << "      }) : (tensor<" << blockSize << "xf32>) -> f32\n"
           << "      %next = arith.addf %acc, %tile_sum : f32\n"
           << "      scf.yield %next : f32\n"
           << "    }\n";
    scalarNames[reduction.getResult()] = "%sum";
    auto final = emitScalar(result);
    if (failed(final)) return failure();
    output << "    %out_ptr = tt.addptr %out, %row : !tt.ptr<f32>, i64\n"
           << "    tt.store %out_ptr, " << *final << " : !tt.ptr<f32>\n"
           << "    tt.return\n  }\n}\n";
    return success();
  }

private:
  bool supportsUniformEpilogue(Value value) {
    if (value == reduction.getResult()) return true;
    Operation *operation = value.getDefiningOp();
    if (auto input = dyn_cast_or_null<tensor::InputOp>(operation)) {
      auto type = cast<RankedTensorType>(input.getResult().getType());
      return type.getElementType().isF32() &&
             (type.getNumElements() == 1 ||
              (type.getRank() == 1 && type.getDimSize(0) == rows));
    }
    if (auto reshape = dyn_cast_or_null<tensor::ReshapeOp>(operation))
      return supportsUniformEpilogue(reshape.getInput());
    if (auto broadcast = dyn_cast_or_null<tensor::BroadcastOp>(operation))
      return supportsUniformEpilogue(broadcast.getInput());
    if (auto add = dyn_cast_or_null<tensor::AddOp>(operation))
      return supportsUniformEpilogue(add.getLhs()) &&
             supportsUniformEpilogue(add.getRhs());
    if (auto mul = dyn_cast_or_null<tensor::MulOp>(operation))
      return supportsUniformEpilogue(mul.getLhs()) &&
             supportsUniformEpilogue(mul.getRhs());
    if (auto div = dyn_cast_or_null<tensor::DivOp>(operation))
      return supportsUniformEpilogue(div.getLhs()) &&
             supportsUniformEpilogue(div.getRhs());
    return false;
  }

  FailureOr<std::string> emitUniformEpilogue(Value value, int64_t blockM) {
    auto found = uniformNames.find(value);
    if (found != uniformNames.end()) return found->second;
    if (value == reduction.getResult()) return std::string("%gf_sum");
    Operation *operation = value.getDefiningOp();
    if (auto input = dyn_cast_or_null<tensor::InputOp>(operation)) {
      auto type = cast<RankedTensorType>(input.getResult().getType());
      unsigned ordinal = argumentIndex.lookup(input.getResult());
      std::string loaded = next("uniform_input");
      if (type.getNumElements() == 1) {
        std::string offset = next("uniform_scalar_offset");
        std::string pointer = next("uniform_scalar_ptr");
        std::string scalar = next("uniform_scalar");
        output << "    " << offset << " = arith.constant "
               << input.getOffsetAttr().getInt() << " : i64\n"
               << "    " << pointer << " = tt.addptr %arg" << ordinal
               << ", " << offset << " : !tt.ptr<f32>, i64\n"
               << "    " << scalar << " = tt.load " << pointer
               << " : !tt.ptr<f32>\n"
               << "    " << loaded << " = tt.splat " << scalar
               << " : f32 -> tensor<" << blockM << "xf32>\n";
      } else {
        std::string base = next("uniform_node_base");
        std::string pointer = next("uniform_node_ptr");
        std::string logical = "%gf_rows";
        if (input.getStrides()[0] != 1 ||
            input.getOffsetAttr().getInt() != 0) {
          std::string stride = next("uniform_node_stride");
          std::string scaled = next("uniform_node_scaled");
          output << "    " << stride << " = arith.constant dense<"
                 << input.getStrides()[0] << "> : tensor<" << blockM
                 << "xi32>\n"
                 << "    " << scaled << " = arith.muli %gf_rows, " << stride
                 << " : tensor<" << blockM << "xi32>\n";
          logical = scaled;
          if (input.getOffsetAttr().getInt() != 0) {
            std::string offset = next("uniform_node_offset");
            std::string shifted = next("uniform_node_shifted");
            output << "    " << offset << " = arith.constant dense<"
                   << input.getOffsetAttr().getInt() << "> : tensor<" << blockM
                   << "xi32>\n"
                   << "    " << shifted << " = arith.addi " << logical << ", "
                   << offset << " : tensor<" << blockM << "xi32>\n";
            logical = shifted;
          }
        }
        output << "    " << base << " = tt.splat %arg" << ordinal
               << " : !tt.ptr<f32> -> tensor<" << blockM
               << "x!tt.ptr<f32>>\n"
               << "    " << pointer << " = tt.addptr " << base << ", "
               << logical << " : tensor<" << blockM
               << "x!tt.ptr<f32>>, tensor<" << blockM << "xi32>\n"
               << "    " << loaded << " = tt.load " << pointer
               << ", %gf_row_mask, %gf_row_zero : tensor<" << blockM
               << "x!tt.ptr<f32>>\n";
      }
      uniformNames[value] = loaded;
      return loaded;
    }
    if (auto reshape = dyn_cast_or_null<tensor::ReshapeOp>(operation)) {
      auto emitted = emitUniformEpilogue(reshape.getInput(), blockM);
      if (succeeded(emitted)) uniformNames[value] = *emitted;
      return emitted;
    }
    if (auto broadcast = dyn_cast_or_null<tensor::BroadcastOp>(operation)) {
      auto emitted = emitUniformEpilogue(broadcast.getInput(), blockM);
      if (succeeded(emitted)) uniformNames[value] = *emitted;
      return emitted;
    }
    Value lhs, rhs;
    StringRef mnemonic;
    if (auto add = dyn_cast_or_null<tensor::AddOp>(operation)) {
      lhs = add.getLhs(); rhs = add.getRhs(); mnemonic = "arith.addf";
    } else if (auto mul = dyn_cast_or_null<tensor::MulOp>(operation)) {
      lhs = mul.getLhs(); rhs = mul.getRhs(); mnemonic = "arith.mulf";
    } else if (auto div = dyn_cast_or_null<tensor::DivOp>(operation)) {
      lhs = div.getLhs(); rhs = div.getRhs(); mnemonic = "arith.divf";
    } else {
      return failure();
    }
    auto left = emitUniformEpilogue(lhs, blockM);
    auto right = emitUniformEpilogue(rhs, blockM);
    if (failed(left) || failed(right)) return failure();
    std::string emitted = next("uniform_epilogue");
    output << "    " << emitted << " = " << mnemonic << " " << *left << ", "
           << *right << " : tensor<" << blockM << "xf32>\n";
    uniformNames[value] = emitted;
    return emitted;
  }

  LogicalResult emitUniformWeighted() {
    int64_t degreeMin = reduction.getDegreeMin();
    int64_t degreeMax = reduction.getDegreeMax();
    if (degreeMin <= 0 || degreeMin != degreeMax || degreeMax > 64 ||
        rows > std::numeric_limits<int32_t>::max() ||
        !supportsUniformEpilogue(result))
      return failure();
    auto multiply = reduction.getInput().getDefiningOp<tensor::MulOp>();
    if (!multiply) return failure();
    auto gather = multiply.getLhs().getDefiningOp<tensor::GatherOp>();
    auto weight = multiply.getRhs().getDefiningOp<tensor::InputOp>();
    if (!gather) {
      gather = multiply.getRhs().getDefiningOp<tensor::GatherOp>();
      weight = multiply.getLhs().getDefiningOp<tensor::InputOp>();
    }
    auto source = gather
                      ? gather.getInput().getDefiningOp<tensor::InputOp>()
                      : tensor::InputOp();
    auto column = gather
                      ? gather.getIndex().getDefiningOp<tensor::InputOp>()
                      : tensor::InputOp();
    if (!gather || !weight || !source || !column) return failure();
    auto sourceType = cast<RankedTensorType>(source.getResult().getType());
    auto columnType = cast<RankedTensorType>(column.getResult().getType());
    auto weightType = cast<RankedTensorType>(weight.getResult().getType());
    if (sourceType.getRank() != 1 || columnType.getRank() != 1 ||
        weightType.getRank() != 1 || !sourceType.getElementType().isF32() ||
        !weightType.getElementType().isF32() ||
        (!columnType.getElementType().isInteger(32) &&
         !columnType.getElementType().isInteger(64)) ||
        columnType.getDimSize(0) != edges || weightType.getDimSize(0) != edges ||
        source.getStrides()[0] != 1 || column.getStrides()[0] != 1 ||
        weight.getStrides()[0] != 1 || source.getOffsetAttr().getInt() != 0 ||
        column.getOffsetAttr().getInt() != 0 ||
        weight.getOffsetAttr().getInt() != 0)
      return failure();

    // Bound the rectangular relation tile rather than selecting by workload.
    // The local schedule sweep shows that 16xD is best for short rows, while
    // keeping the product near 256 lanes avoids register pressure once D is
    // 32 or 64. Two warps also hide the extra indirect-load latency there.
    // These are degree-class decisions and remain valid for any structurally
    // matched weighted gather + associative sum + pointwise epilogue.
    int64_t blockM = degreeMax <= 16 ? 16 : 8;
    int64_t blockD = nextPowerOfTwo(degreeMax);
    int64_t numWarps = degreeMax <= 16 ? 1 : 2;
    StringRef index = spelling(columnType.getElementType());
    unsigned columnArg = argumentIndex.lookup(column.getResult());
    unsigned sourceArg = argumentIndex.lookup(source.getResult());
    unsigned weightArg = argumentIndex.lookup(weight.getResult());
    std::string tileI32 = "tensor<" + std::to_string(blockM) + "x" +
                          std::to_string(blockD) + "xi32>";
    std::string tileIndex = "tensor<" + std::to_string(blockM) + "x" +
                            std::to_string(blockD) + "x" + index.str() + ">";
    std::string tileF32 = "tensor<" + std::to_string(blockM) + "x" +
                          std::to_string(blockD) + "xf32>";
    std::string tileI1 = "tensor<" + std::to_string(blockM) + "x" +
                         std::to_string(blockD) + "xi1>";
    std::string rowsI32 = "tensor<" + std::to_string(blockM) + "xi32>";
    std::string rowsI1 = "tensor<" + std::to_string(blockM) + "xi1>";
    std::string rowsF32 = "tensor<" + std::to_string(blockM) + "xf32>";

    output << "// graphforge.tensor entry=gf_tensor_csr_sum_epilogue "
              "block_rows=" << blockM << " block_elements=" << blockD
           << " num_warps=" << numWarps << " abi=";
    for (auto [ordinal, input] : llvm::enumerate(inputs)) {
      if (ordinal) output << ",";
      output << "arg" << ordinal;
    }
    output << ",out\nmodule {\n  tt.func public @gf_tensor_csr_sum_epilogue(";
    for (auto [ordinal, input] : llvm::enumerate(inputs)) {
      if (ordinal) output << ", ";
      output << "%arg" << ordinal << ": !tt.ptr<"
             << spelling(cast<RankedTensorType>(input.getResult().getType())
                             .getElementType()) << ">";
    }
    output << ", %out: !tt.ptr<f32>) attributes {noinline = false} {\n"
           << "    %gf_source_zero = arith.constant dense<0> : " << tileIndex
           << "\n    %gf_value_zero = arith.constant dense<0.000000e+00> : "
           << tileF32
           << "\n    %gf_row_zero = arith.constant dense<0.000000e+00> : "
           << rowsF32
           << "\n    %gf_nrows = arith.constant dense<" << rows << "> : "
           << rowsI32
           << "\n    %gf_degree = arith.constant dense<" << degreeMax << "> : "
           << rowsI32
           << "\n    %gf_degree2 = arith.constant dense<" << degreeMax
           << "> : tensor<1x" << blockD << "xi32>\n"
           << "    %gf_cbm = arith.constant " << blockM << " : i32\n"
           << "    %gf_pid = tt.get_program_id x : i32\n"
           << "    %gf_base = arith.muli %gf_pid, %gf_cbm : i32\n"
           << "    %gf_range_m = tt.make_range {end = " << blockM
           << " : i32, start = 0 : i32} : " << rowsI32 << "\n"
           << "    %gf_base_v = tt.splat %gf_base : i32 -> " << rowsI32
           << "\n    %gf_rows = arith.addi %gf_base_v, %gf_range_m : "
           << rowsI32
           << "\n    %gf_range_d = tt.make_range {end = " << blockD
           << " : i32, start = 0 : i32} : tensor<" << blockD << "xi32>\n"
           << "    %gf_edge_base = arith.muli %gf_rows, %gf_degree : "
           << rowsI32
           << "\n    %gf_edge_base2 = tt.expand_dims %gf_edge_base {axis = 1 : i32} : "
           << rowsI32 << " -> tensor<" << blockM << "x1xi32>\n"
           << "    %gf_neighbors = tt.expand_dims %gf_range_d {axis = 0 : i32} : tensor<"
           << blockD << "xi32> -> tensor<1x" << blockD << "xi32>\n"
           << "    %gf_edge_base_b = tt.broadcast %gf_edge_base2 : tensor<"
           << blockM << "x1xi32> -> " << tileI32
           << "\n    %gf_neighbors_b = tt.broadcast %gf_neighbors : tensor<1x"
           << blockD << "xi32> -> " << tileI32
           << "\n    %gf_edges = arith.addi %gf_edge_base_b, %gf_neighbors_b : "
           << tileI32
           << "\n    %gf_row_mask = arith.cmpi slt, %gf_rows, %gf_nrows : "
           << rowsI32
           << "\n    %gf_degree_mask1 = arith.cmpi slt, %gf_neighbors, %gf_degree2 : tensor<1x"
           << blockD << "xi32>\n"
           << "    %gf_row_mask2 = tt.expand_dims %gf_row_mask {axis = 1 : i32} : "
           << rowsI1 << " -> tensor<" << blockM << "x1xi1>\n"
           << "    %gf_row_mask_b = tt.broadcast %gf_row_mask2 : tensor<"
           << blockM << "x1xi1> -> " << tileI1
           << "\n    %gf_degree_mask = tt.broadcast %gf_degree_mask1 : tensor<1x"
           << blockD << "xi1> -> " << tileI1
           << "\n    %gf_mask = arith.andi %gf_row_mask_b, %gf_degree_mask : "
           << tileI1
           << "\n    %gf_col_base = tt.splat %arg" << columnArg << " : !tt.ptr<"
           << index << "> -> tensor<" << blockM << "x" << blockD
           << "x!tt.ptr<" << index << ">>\n"
           << "    %gf_col_ptr = tt.addptr %gf_col_base, %gf_edges : tensor<"
           << blockM << "x" << blockD << "x!tt.ptr<" << index << ">>, "
           << tileI32
           << "\n    %gf_src = tt.load %gf_col_ptr, %gf_mask, %gf_source_zero : tensor<"
           << blockM << "x" << blockD << "x!tt.ptr<" << index << ">>\n"
           << "    %gf_x_base = tt.splat %arg" << sourceArg
           << " : !tt.ptr<f32> -> tensor<" << blockM << "x" << blockD
           << "x!tt.ptr<f32>>\n"
           << "    %gf_x_ptr = tt.addptr %gf_x_base, %gf_src : tensor<"
           << blockM << "x" << blockD << "x!tt.ptr<f32>>, " << tileIndex
           << "\n    %gf_x = tt.load %gf_x_ptr, %gf_mask, %gf_value_zero : tensor<"
           << blockM << "x" << blockD << "x!tt.ptr<f32>>\n"
           << "    %gf_w_base = tt.splat %arg" << weightArg
           << " : !tt.ptr<f32> -> tensor<" << blockM << "x" << blockD
           << "x!tt.ptr<f32>>\n"
           << "    %gf_w_ptr = tt.addptr %gf_w_base, %gf_edges : tensor<"
           << blockM << "x" << blockD << "x!tt.ptr<f32>>, " << tileI32
           << "\n    %gf_w = tt.load %gf_w_ptr, %gf_mask, %gf_value_zero : tensor<"
           << blockM << "x" << blockD << "x!tt.ptr<f32>>\n"
           << "    %gf_message = arith.mulf %gf_x, %gf_w : " << tileF32
           << "\n    %gf_sum = \"tt.reduce\"(%gf_message) <{axis = 1 : i32}> ({\n"
           << "    ^bb0(%gf_a: f32, %gf_b: f32):\n"
           << "      %gf_combined = arith.addf %gf_a, %gf_b : f32\n"
           << "      tt.reduce.return %gf_combined : f32\n"
           << "    }) : (" << tileF32 << ") -> " << rowsF32 << "\n";
    uniformNames[reduction.getResult()] = "%gf_sum";
    auto final = emitUniformEpilogue(result, blockM);
    if (failed(final)) return failure();
    output << "    %gf_out_base = tt.splat %out : !tt.ptr<f32> -> tensor<"
           << blockM << "x!tt.ptr<f32>>\n"
           << "    %gf_out_ptr = tt.addptr %gf_out_base, %gf_rows : tensor<"
           << blockM << "x!tt.ptr<f32>>, " << rowsI32 << "\n"
           << "    tt.store %gf_out_ptr, " << *final
           << ", %gf_row_mask : tensor<" << blockM << "x!tt.ptr<f32>>\n"
           << "    tt.return\n  }\n}\n";
    return success();
  }

  static StringRef spelling(Type type) {
    if (type.isF32()) return "f32";
    if (type.isInteger(32)) return "i32";
    return "i64";
  }

  FailureOr<std::string> emitVector(Value value, StringRef indices,
                                    StringRef mask) {
    auto found = vectorNames.find(value);
    if (found != vectorNames.end()) return found->second;
    Operation *operation = value.getDefiningOp();
    if (auto input = dyn_cast_or_null<tensor::InputOp>(operation))
      return emitVectorInput(input, indices, mask);
    if (auto gather = dyn_cast_or_null<tensor::GatherOp>(operation)) {
      auto indexInput = gather.getIndex().getDefiningOp<tensor::InputOp>();
      auto sourceInput = gather.getInput().getDefiningOp<tensor::InputOp>();
      if (!indexInput || !sourceInput)
        return gather.emitError("fused CSR gather requires direct ABI inputs");
      auto indexType = cast<RankedTensorType>(indexInput.getResult().getType());
      auto loadedIndex = emitVectorInput(indexInput, indices, mask);
      if (failed(loadedIndex)) return failure();
      std::string sourceIndex = *loadedIndex;
      if (indexType.getElementType().isInteger(32)) {
        std::string extended = next("source_index");
        output << "      " << extended << " = arith.extsi " << sourceIndex
               << " : tensor<" << blockSize << "xi32> to tensor<" << blockSize
               << "xi64>\n";
        sourceIndex = extended;
      }
      auto loaded = emitVectorInput(sourceInput, sourceIndex, mask);
      if (failed(loaded)) return failure();
      vectorNames[value] = *loaded;
      return *loaded;
    }
    if (auto reshape = dyn_cast_or_null<tensor::ReshapeOp>(operation))
      return aliasVector(value, reshape.getInput(), indices, mask);
    if (auto broadcast = dyn_cast_or_null<tensor::BroadcastOp>(operation))
      return aliasVector(value, broadcast.getInput(), indices, mask);
    if (auto add = dyn_cast_or_null<tensor::AddOp>(operation))
      return emitVectorBinary(value, add.getLhs(), add.getRhs(), "arith.addf",
                              indices, mask);
    if (auto mul = dyn_cast_or_null<tensor::MulOp>(operation))
      return emitVectorBinary(value, mul.getLhs(), mul.getRhs(), "arith.mulf",
                              indices, mask);
    if (auto div = dyn_cast_or_null<tensor::DivOp>(operation))
      return emitVectorBinary(value, div.getLhs(), div.getRhs(), "arith.divf",
                              indices, mask);
    if (operation)
      operation->emitError(
          "fused CSR message supports input/gather/broadcast/reshape/add/mul/div");
    return failure();
  }

  FailureOr<std::string> emitVectorInput(tensor::InputOp input,
                                         StringRef indices, StringRef mask) {
    auto type = cast<RankedTensorType>(input.getResult().getType());
    Type element = type.getElementType();
    StringRef elementName = spelling(element);
    std::string effective = indices.str();
    if (type.getNumElements() == 1) {
      effective = next("scalar_indices");
      output << "      " << effective << " = arith.constant dense<"
             << input.getOffsetAttr().getInt() << "> : tensor<" << blockSize
             << "xi64>\n";
    } else {
      int64_t stride = input.getStrides()[0];
      if (stride != 1) {
        std::string strideValue = next("stride");
        std::string scaled = next("scaled_indices");
        output << "      " << strideValue << " = arith.constant dense<"
               << stride << "> : tensor<" << blockSize << "xi64>\n"
               << "      " << scaled << " = arith.muli " << effective << ", "
               << strideValue << " : tensor<" << blockSize << "xi64>\n";
        effective = scaled;
      }
      if (input.getOffsetAttr().getInt() != 0) {
        std::string offset = next("offset");
        std::string shifted = next("shifted_indices");
        output << "      " << offset << " = arith.constant dense<"
               << input.getOffsetAttr().getInt() << "> : tensor<" << blockSize
               << "xi64>\n"
               << "      " << shifted << " = arith.addi " << effective << ", "
               << offset << " : tensor<" << blockSize << "xi64>\n";
        effective = shifted;
      }
    }
    unsigned ordinal = argumentIndex.lookup(input.getResult());
    std::string base = next("input_base");
    std::string pointer = next("input_ptr");
    std::string loaded = next("input");
    output << "      " << base << " = tt.splat %arg" << ordinal
           << " : !tt.ptr<" << elementName << "> -> tensor<" << blockSize
           << "x!tt.ptr<" << elementName << ">>\n"
           << "      " << pointer << " = tt.addptr " << base << ", "
           << effective << " : tensor<" << blockSize << "x!tt.ptr<"
           << elementName << ">>, tensor<" << blockSize << "xi64>\n"
           << "      " << loaded << " = tt.load " << pointer << ", " << mask
           << ", " << (element.isF32() ? "%zero_vector" :
                        element.isInteger(32) ? "%zero_i32" : "%zero_i64")
           << " : tensor<" << blockSize << "x!tt.ptr<" << elementName
           << ">>\n";
    return loaded;
  }

  FailureOr<std::string> aliasVector(Value resultValue, Value input,
                                     StringRef indices, StringRef mask) {
    auto value = emitVector(input, indices, mask);
    if (succeeded(value)) vectorNames[resultValue] = *value;
    return value;
  }

  FailureOr<std::string> emitVectorBinary(Value resultValue, Value lhs,
                                          Value rhs, StringRef mnemonic,
                                          StringRef indices, StringRef mask) {
    auto left = emitVector(lhs, indices, mask);
    auto right = emitVector(rhs, indices, mask);
    if (failed(left) || failed(right)) return failure();
    std::string name = next("message");
    output << "      " << name << " = " << mnemonic << " " << *left << ", "
           << *right << " : tensor<" << blockSize << "xf32>\n";
    vectorNames[resultValue] = name;
    return name;
  }

  FailureOr<std::string> emitScalar(Value value) {
    auto found = scalarNames.find(value);
    if (found != scalarNames.end()) return found->second;
    Operation *operation = value.getDefiningOp();
    if (auto input = dyn_cast_or_null<tensor::InputOp>(operation)) {
      auto type = cast<RankedTensorType>(input.getResult().getType());
      if (!type.getElementType().isF32())
        return input.emitError("CSR epilogue inputs must be FP32");
      std::string index = "%row";
      if (type.getNumElements() == 1) {
        index = next("scalar_index");
        output << "    " << index << " = arith.constant "
               << input.getOffsetAttr().getInt() << " : i64\n";
      } else {
        int64_t stride = input.getStrides()[0];
        std::string scale = next("node_stride");
        std::string scaled = next("node_index");
        output << "    " << scale << " = arith.constant " << stride
               << " : i64\n"
               << "    " << scaled << " = arith.muli %row, " << scale
               << " : i64\n";
        index = scaled;
        if (input.getOffsetAttr().getInt() != 0) {
          std::string offset = next("node_offset");
          std::string shifted = next("node_shifted");
          output << "    " << offset << " = arith.constant "
                 << input.getOffsetAttr().getInt() << " : i64\n"
                 << "    " << shifted << " = arith.addi " << index << ", "
                 << offset << " : i64\n";
          index = shifted;
        }
      }
      unsigned ordinal = argumentIndex.lookup(input.getResult());
      std::string pointer = next("node_ptr");
      std::string loaded = next("node_value");
      output << "    " << pointer << " = tt.addptr %arg" << ordinal << ", "
             << index << " : !tt.ptr<f32>, i64\n"
             << "    " << loaded << " = tt.load " << pointer
             << " : !tt.ptr<f32>\n";
      scalarNames[value] = loaded;
      return loaded;
    }
    if (auto reshape = dyn_cast_or_null<tensor::ReshapeOp>(operation))
      return aliasScalar(value, reshape.getInput());
    if (auto broadcast = dyn_cast_or_null<tensor::BroadcastOp>(operation))
      return aliasScalar(value, broadcast.getInput());
    if (auto add = dyn_cast_or_null<tensor::AddOp>(operation))
      return emitScalarBinary(value, add.getLhs(), add.getRhs(), "arith.addf");
    if (auto mul = dyn_cast_or_null<tensor::MulOp>(operation))
      return emitScalarBinary(value, mul.getLhs(), mul.getRhs(), "arith.mulf");
    if (auto div = dyn_cast_or_null<tensor::DivOp>(operation))
      return emitScalarBinary(value, div.getLhs(), div.getRhs(), "arith.divf");
    if (operation)
      operation->emitError(
          "fused CSR epilogue supports reduction/input/broadcast/reshape/add/mul/div");
    return failure();
  }

  FailureOr<std::string> aliasScalar(Value resultValue, Value input) {
    auto value = emitScalar(input);
    if (succeeded(value)) scalarNames[resultValue] = *value;
    return value;
  }

  FailureOr<std::string> emitScalarBinary(Value resultValue, Value lhs,
                                          Value rhs, StringRef mnemonic) {
    auto left = emitScalar(lhs);
    auto right = emitScalar(rhs);
    if (failed(left) || failed(right)) return failure();
    std::string name = next("epilogue");
    output << "    " << name << " = " << mnemonic << " " << *left << ", "
           << *right << " : f32\n";
    scalarNames[resultValue] = name;
    return name;
  }

  std::string next(StringRef stem) {
    return (Twine("%") + stem + Twine(nextId++)).str();
  }

  tensor::CSRSegmentSumOp reduction;
  func::FuncOp function;
  Value result;
  int64_t blockSize;
  llvm::raw_ostream &output;
  int64_t rows = 0;
  int64_t edges = 0;
  unsigned nextId = 0;
  SmallVector<tensor::InputOp> inputs;
  DenseMap<Value, unsigned> argumentIndex;
  DenseMap<Value, std::string> vectorNames;
  DenseMap<Value, std::string> scalarNames;
  DenseMap<Value, std::string> uniformNames;
};

class TensorPointwiseTTIREmitter {
public:
  TensorPointwiseTTIREmitter(func::FuncOp function, Value result,
                             int64_t blockSize, llvm::raw_ostream &output)
      : function(function), result(result), blockSize(blockSize), output(output) {}

  LogicalResult emit() {
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    if (!resultType || !resultType.hasStaticShape() ||
        (resultType.getRank() != 1 && resultType.getRank() != 2) ||
        !isa<FloatType>(resultType.getElementType()) ||
        (cast<FloatType>(resultType.getElementType()).getWidth() != 16 &&
            cast<FloatType>(resultType.getElementType()).getWidth() != 32 &&
            cast<FloatType>(resultType.getElementType()).getWidth() != 64))
      return function.emitError(
          "gf_tensor pointwise TTIR requires one static rank-one/rank-two "
          "FP16/FP32/FP64 result");
    elementType = resultType.getElementType();
    elementName = spelling(elementType).str();
    elements = resultType.getNumElements();
    resultRows = resultType.getDimSize(0);
    resultColumns = resultType.getRank() == 2 ? resultType.getDimSize(1) : 1;
    if (elements <= 0)
      return function.emitError("pointwise TTIR requires a non-empty result");

    function.walk([&](tensor::InputOp input) {
      argumentIndex[input.getResult()] = inputs.size();
      inputs.push_back(input);
    });
    if (inputs.empty()) return function.emitError("requires gf_tensor.input");
    for (tensor::InputOp input : inputs) {
      auto type = dyn_cast<RankedTensorType>(input.getResult().getType());
      if (!type || !type.hasStaticShape() || type.getRank() > 2 ||
          (type.getElementType() != elementType &&
           !type.getElementType().isInteger(32) &&
           !type.getElementType().isInteger(64)))
        return input.emitError(
            "pointwise TTIR inputs must match the result float type or be "
            "i32/i64 indices");
    }

    output << "// graphforge.tensor entry=gf_tensor_pointwise "
              "block_rows=" << blockSize << " block_elements=" << blockSize
           << " num_warps=8 abi=";
    for (auto [index, input] : llvm::enumerate(inputs)) {
      if (index) output << ",";
      output << "arg" << index;
    }
    output << ",out\nmodule {\n"
           << "  tt.func public @gf_tensor_pointwise(";
    for (auto [index, input] : llvm::enumerate(inputs)) {
      if (index) output << ", ";
      output << "%arg" << index << ": !tt.ptr<"
             << spelling(cast<RankedTensorType>(input.getResult().getType())
                             .getElementType()) << ">";
    }
    output << ", %out: !tt.ptr<" << elementName
           << ">) attributes {noinline = false} {\n"
           << "    %lane_i32 = tt.make_range {end = " << blockSize
           << " : i32, start = 0 : i32} : tensor<" << blockSize << "xi32>\n"
           << "    %pid = tt.get_program_id x : i32\n"
           << "    %block = arith.constant " << blockSize << " : i32\n"
           << "    %base_i32 = arith.muli %pid, %block : i32\n"
           << "    %base = arith.extsi %base_i32 : i32 to i64\n"
           << "    %base_v = tt.splat %base : i64 -> tensor<" << blockSize
           << "xi64>\n"
           << "    %lane = arith.extsi %lane_i32 : tensor<" << blockSize
           << "xi32> to tensor<" << blockSize << "xi64>\n"
           << "    %index = arith.addi %base_v, %lane : tensor<" << blockSize
           << "xi64>\n"
           << "    %bound = arith.constant dense<" << elements << "> : tensor<"
           << blockSize << "xi64>\n"
           << "    %mask = arith.cmpi slt, %index, %bound : tensor<"
           << blockSize << "xi64>\n"
           << "    %zero_float = arith.constant dense<0.000000e+00> : tensor<"
           << blockSize << "x" << elementName << ">\n"
           << "    %zero_i32 = arith.constant dense<0> : tensor<" << blockSize
           << "xi32>\n"
           << "    %zero_i64 = arith.constant dense<0> : tensor<" << blockSize
           << "xi64>\n";
    FailureOr<std::string> value = emitValue(result);
    if (failed(value)) return failure();
    output << "    %out_base = tt.splat %out : !tt.ptr<" << elementName
           << "> -> tensor<" << blockSize << "x!tt.ptr<" << elementName << ">>\n"
           << "    %out_ptr = tt.addptr %out_base, %index : tensor<"
           << blockSize << "x!tt.ptr<" << elementName << ">>, tensor<"
           << blockSize << "xi64>\n"
           << "    tt.store %out_ptr, " << *value << ", %mask : tensor<"
           << blockSize << "x!tt.ptr<" << elementName << ">>\n"
           << "    tt.return\n  }\n}\n";
    return success();
  }

private:
  static StringRef spelling(Type type) {
    if (type.isF16()) return "f16";
    if (type.isF32()) return "f32";
    if (type.isF64()) return "f64";
    if (type.isInteger(32)) return "i32";
    return "i64";
  }

  FailureOr<std::string> emitValue(Value value) {
    auto found = names.find(value);
    if (found != names.end()) return found->second;
    Operation *operation = value.getDefiningOp();
    if (auto input = dyn_cast_or_null<tensor::InputOp>(operation))
      return emitInput(input, "%index", "%mask");
    if (auto gather = dyn_cast_or_null<tensor::GatherOp>(operation)) {
      return emitGather(gather);
    }
    if (auto scatter = dyn_cast_or_null<tensor::ScatterRowsOp>(operation))
      return emitScatterRows(scatter);
    if (auto reshape = dyn_cast_or_null<tensor::ReshapeOp>(operation)) {
      FailureOr<std::string> rewritten = emitValue(reshape.getInput());
      if (succeeded(rewritten)) names[value] = *rewritten;
      return rewritten;
    }
    if (auto broadcast = dyn_cast_or_null<tensor::BroadcastOp>(operation)) {
      auto inputType = cast<RankedTensorType>(broadcast.getInput().getType());
      auto outputType = cast<RankedTensorType>(broadcast.getResult().getType());
      bool scalarLike = inputType.getNumElements() == 1;
      bool sameShape = inputType.getShape() == outputType.getShape();
      bool rowVector =
          (inputType.getRank() == 1 &&
           inputType.getDimSize(0) == resultColumns) ||
          (inputType.getRank() == 2 && inputType.getDimSize(0) == 1 &&
           inputType.getDimSize(1) == resultColumns);
      bool matrixBroadcast =
          inputType.getRank() == 2 &&
          (inputType.getDimSize(0) == 1 ||
           inputType.getDimSize(0) == resultRows) &&
          (inputType.getDimSize(1) == 1 ||
           inputType.getDimSize(1) == resultColumns);
      if (!scalarLike && !sameShape && !rowVector && !matrixBroadcast)
        return broadcast.emitError(
            "pointwise TTIR broadcasts scalar-like values or one row vector");
      FailureOr<std::string> rewritten = emitValue(broadcast.getInput());
      if (succeeded(rewritten)) names[value] = *rewritten;
      return rewritten;
    }
    if (auto checkpoint = dyn_cast_or_null<tensor::CheckpointOp>(operation)) {
      FailureOr<std::string> rewritten = emitValue(checkpoint.getInput());
      if (succeeded(rewritten)) names[value] = *rewritten;
      return rewritten;
    }
    if (auto candidate =
            dyn_cast_or_null<tensor::CheckpointCandidateOp>(operation)) {
      FailureOr<std::string> rewritten = emitValue(candidate.getInput());
      if (succeeded(rewritten)) names[value] = *rewritten;
      return rewritten;
    }
    if (auto add = dyn_cast_or_null<tensor::AddOp>(operation))
      return emitBinary(add.getLhs(), add.getRhs(), "arith.addf", value);
    if (auto mul = dyn_cast_or_null<tensor::MulOp>(operation))
      return emitBinary(mul.getLhs(), mul.getRhs(), "arith.mulf", value);
    if (auto divide = dyn_cast_or_null<tensor::DivOp>(operation))
      return emitBinary(divide.getLhs(), divide.getRhs(), "arith.divf", value);
    if (auto neg = dyn_cast_or_null<tensor::NegOp>(operation)) {
      FailureOr<std::string> operand = emitValue(neg.getInput());
      if (failed(operand)) return failure();
      std::string name = next("neg");
      // Triton's GPU conversion does not legalize vector arith.negf in every
      // supported release. Subtraction from the typed zero is equivalent and
      // belongs to the stable arithmetic subset of all provider pipelines.
      output << "    " << name << " = arith.subf %zero_float, " << *operand
             << " : tensor<" << blockSize << "x" << elementName << ">\n";
      names[value] = name;
      return name;
    }
    if (auto exponential = dyn_cast_or_null<tensor::ExpOp>(operation)) {
      FailureOr<std::string> operand = emitValue(exponential.getInput());
      if (failed(operand)) return failure();
      std::string name = next("exp");
      output << "    " << name << " = math.exp " << *operand << " : tensor<"
             << blockSize << "x" << elementName << ">\n";
      names[value] = name;
      return name;
    }
    if (auto root = dyn_cast_or_null<tensor::SqrtOp>(operation)) {
      FailureOr<std::string> operand = emitValue(root.getInput());
      if (failed(operand)) return failure();
      std::string name = next("sqrt");
      output << "    " << name << " = math.sqrt " << *operand << " : tensor<"
             << blockSize << "x" << elementName << ">\n";
      names[value] = name;
      return name;
    }
    if (operation)
      operation->emitError(
          "pointwise TTIR supports input/gather/scatter_rows/broadcast/reshape/add/mul/div/neg/exp/sqrt producers");
    return failure();
  }

  FailureOr<std::string> emitBinary(Value lhs, Value rhs, StringRef mnemonic,
                                    Value value) {
    FailureOr<std::string> left = emitValue(lhs);
    FailureOr<std::string> right = emitValue(rhs);
    if (failed(left) || failed(right)) return failure();
    std::string name = next("value");
    output << "    " << name << " = " << mnemonic << " " << *left << ", "
           << *right << " : tensor<" << blockSize << "x" << elementName
           << ">\n";
    names[value] = name;
    return name;
  }

  FailureOr<std::string> emitInput(tensor::InputOp input, StringRef indices,
                                   StringRef mask, Value resultValue = {}) {
    auto type = cast<RankedTensorType>(input.getResult().getType());
    std::string effective = indices.str();
    if (type.getRank() == 0 || type.getNumElements() == 1) {
      effective = next("zero_index");
      output << "    " << effective << " = arith.constant dense<"
             << input.getOffsetAttr().getInt() << "> : tensor<" << blockSize
             << "xi64>\n";
    } else if (type.getRank() == 1) {
      int64_t extent = type.getDimSize(0);
      int64_t stride = input.getStrides()[0];
      std::string logical = indices.str();
      if (resultColumns != 1) {
        if (extent != resultColumns)
          return input.emitError(
              "rank-one pointwise input must match the trailing result dimension");
        logical = next("column_index");
        std::string columns = denseI64(resultColumns, "result_columns");
        output << "    " << logical << " = arith.remui " << indices
               << ", " << columns << " : tensor<"
               << blockSize << "xi64>\n";
      } else if (extent != elements) {
        return input.emitError("rank-one pointwise input extent mismatch");
      }
      effective = logical;
      if (stride != 1) {
        std::string strideValue = denseI64(stride, "input_stride");
        std::string scaled = next("scaled_index");
        output << "    " << scaled << " = arith.muli " << logical
               << ", " << strideValue << " : tensor<"
               << blockSize << "xi64>\n";
        effective = scaled;
      }
      if (input.getOffsetAttr().getInt() != 0) {
        std::string offset = denseI64(
            input.getOffsetAttr().getInt(), "input_offset");
        std::string shifted = next("shifted_index");
        output << "    " << shifted << " = arith.addi " << effective
               << ", " << offset << " : tensor<" << blockSize << "xi64>\n";
        effective = shifted;
      }
    } else {
      int64_t sourceRows = type.getDimSize(0);
      int64_t sourceColumns = type.getDimSize(1);
      if ((sourceRows != 1 && sourceRows != resultRows) ||
          (sourceColumns != 1 && sourceColumns != resultColumns))
        return input.emitError(
            "rank-two pointwise input is not broadcast-compatible with result");
      std::string row = next("row_index");
      std::string column = next("column_index");
      std::string columns = denseI64(resultColumns, "result_columns");
      output << "    " << row << " = arith.divui " << indices
             << ", " << columns << " : tensor<"
             << blockSize << "xi64>\n"
             << "    " << column << " = arith.remui " << indices
             << ", " << columns << " : tensor<"
             << blockSize << "xi64>\n";
      if (sourceRows == 1) {
        row = next("zero_row");
        output << "    " << row << " = arith.constant dense<0> : tensor<"
               << blockSize << "xi64>\n";
      }
      if (sourceColumns == 1) {
        column = next("zero_column");
        output << "    " << column << " = arith.constant dense<0> : tensor<"
               << blockSize << "xi64>\n";
      }
      std::string rowOffset = next("row_offset");
      std::string columnOffset = next("column_offset");
      std::string combined = next("physical_index");
      std::string rowStride = denseI64(
          input.getStrides()[0], "row_stride");
      std::string columnStride = denseI64(
          input.getStrides()[1], "column_stride");
      output << "    " << rowOffset << " = arith.muli " << row
             << ", " << rowStride << " : tensor<" << blockSize << "xi64>\n"
             << "    " << columnOffset << " = arith.muli " << column
             << ", " << columnStride << " : tensor<" << blockSize << "xi64>\n"
             << "    " << combined << " = arith.addi " << rowOffset << ", "
             << columnOffset << " : tensor<" << blockSize << "xi64>\n";
      effective = combined;
      if (input.getOffsetAttr().getInt() != 0) {
        std::string offset = denseI64(
            input.getOffsetAttr().getInt(), "input_offset");
        std::string shifted = next("shifted_index");
        output << "    " << shifted << " = arith.addi " << effective
               << ", " << offset << " : tensor<" << blockSize << "xi64>\n";
        effective = shifted;
      }
    }
    return emitRawInput(input, effective, mask, resultValue);
  }

  FailureOr<std::string> emitGather(tensor::GatherOp gather) {
    auto source = gather.getInput().getDefiningOp<tensor::InputOp>();
    auto index = gather.getIndex().getDefiningOp<tensor::InputOp>();
    auto sourceType = dyn_cast<RankedTensorType>(gather.getInput().getType());
    auto indexType = dyn_cast<RankedTensorType>(gather.getIndex().getType());
    auto gatherType = dyn_cast<RankedTensorType>(gather.getResult().getType());
    if (!source || !index || !sourceType || !indexType || !gatherType ||
        sourceType.getRank() != gatherType.getRank() ||
        sourceType.getRank() < 1 || sourceType.getRank() > 2 ||
        indexType.getRank() != 1)
      return gather.emitError(
          "pointwise gather requires direct rank-one/rank-two source and rank-one index");
    int64_t columns = gatherType.getRank() == 2 ? gatherType.getDimSize(1) : 1;
    std::string relationRow = next("relation_row");
    std::string columnsValue = denseI64(columns, "gather_columns");
    if (gatherType.getNumElements() == 1 ||
        (gatherType.getRank() == 2 && gatherType.getDimSize(0) == 1)) {
      // A scalar-like or one-row gather may be broadcast into a larger root.
      // Every output lane must read relation index zero; using %index here
      // walks past the one-element index ABI buffer.
      output << "    " << relationRow
             << " = arith.constant dense<0> : tensor<" << blockSize
             << "xi64>\n";
    } else {
      output << "    " << relationRow << " = arith.divui %index, "
             << columnsValue << " : tensor<"
             << blockSize << "xi64>\n";
    }
    std::string indexPhysical = relationRow;
    if (index.getStrides()[0] != 1) {
      std::string stride = denseI64(
          index.getStrides()[0], "relation_index_stride");
      std::string scaled = next("relation_index_scaled");
      output << "    " << scaled << " = arith.muli " << relationRow
             << ", " << stride << " : tensor<" << blockSize << "xi64>\n";
      indexPhysical = scaled;
    }
    if (index.getOffsetAttr().getInt() != 0) {
      std::string offset = denseI64(
          index.getOffsetAttr().getInt(), "relation_index_offset");
      std::string shifted = next("relation_index_shifted");
      output << "    " << shifted << " = arith.addi " << indexPhysical
             << ", " << offset << " : tensor<" << blockSize << "xi64>\n";
      indexPhysical = shifted;
    }
    FailureOr<std::string> gatheredRow = emitRawInput(
        index, indexPhysical, "%mask", gather.getIndex());
    if (failed(gatheredRow)) return failure();
    std::string gatheredRowI64 = *gatheredRow;
    if (indexType.getElementType().isInteger(32)) {
      gatheredRowI64 = next("gathered_row_i64");
      output << "    " << gatheredRowI64 << " = arith.extsi " << *gatheredRow
             << " : tensor<" << blockSize << "xi32> to tensor<" << blockSize
             << "xi64>\n";
    }
    std::string rowOffset = next("gather_row_offset");
    std::string rowStride = denseI64(
        source.getStrides()[0], "gather_row_stride");
    output << "    " << rowOffset << " = arith.muli " << gatheredRowI64
           << ", " << rowStride << " : tensor<" << blockSize << "xi64>\n";
    std::string physical = rowOffset;
    if (columns != 1) {
      std::string column = next("gather_column");
      std::string columnOffset = next("gather_column_offset");
      std::string combined = next("gather_physical_index");
      std::string columnStride = denseI64(
          source.getStrides()[1], "gather_column_stride");
      output << "    " << column << " = arith.remui %index, "
             << columnsValue << " : tensor<"
             << blockSize << "xi64>\n"
             << "    " << columnOffset << " = arith.muli " << column
             << ", " << columnStride << " : tensor<" << blockSize << "xi64>\n"
             << "    " << combined << " = arith.addi " << rowOffset << ", "
             << columnOffset << " : tensor<" << blockSize << "xi64>\n";
      physical = combined;
    }
    if (source.getOffsetAttr().getInt() != 0) {
      std::string offset = denseI64(
          source.getOffsetAttr().getInt(), "gather_source_offset");
      std::string shifted = next("gather_shifted_index");
      output << "    " << shifted << " = arith.addi " << physical
             << ", " << offset << " : tensor<" << blockSize << "xi64>\n";
      physical = shifted;
    }
    return emitRawInput(source, physical, "%mask", gather.getResult());
  }

  FailureOr<std::string> emitScatterRows(tensor::ScatterRowsOp scatter) {
    auto source = scatter.getInput().getDefiningOp<tensor::InputOp>();
    auto destination =
        scatter.getDestination().getDefiningOp<tensor::InputOp>();
    auto inverse = scatter.getInverse().getDefiningOp<tensor::InputOp>();
    auto sourceType = dyn_cast<RankedTensorType>(scatter.getInput().getType());
    auto destinationType =
        dyn_cast<RankedTensorType>(scatter.getDestination().getType());
    auto inverseType =
        dyn_cast<RankedTensorType>(scatter.getInverse().getType());
    auto resultType = dyn_cast<RankedTensorType>(scatter.getResult().getType());
    if (!source || !destination || !inverse || !sourceType ||
        !destinationType || !inverseType || !resultType ||
        sourceType.getRank() != resultType.getRank() ||
        sourceType.getRank() < 1 || sourceType.getRank() > 2 ||
        destinationType.getRank() != 1 || inverseType.getRank() != 1 ||
        inverseType.getDimSize(0) != resultRows ||
        (sourceType.getRank() == 2 &&
         sourceType.getDimSize(1) != resultColumns))
      return scatter.emitError(
          "pointwise scatter_rows requires direct rank-one/rank-two source "
          "and direct rank-one maps");

    std::string columns = denseI64(resultColumns, "scatter_columns");
    std::string resultRow = next("scatter_result_row");
    output << "    " << resultRow << " = arith.divui %index, " << columns
           << " : tensor<" << blockSize << "xi64>\n";

    std::string inversePhysical = resultRow;
    if (inverse.getStrides()[0] != 1) {
      std::string stride = denseI64(
          inverse.getStrides()[0], "scatter_inverse_stride");
      std::string scaled = next("scatter_inverse_scaled");
      output << "    " << scaled << " = arith.muli " << resultRow << ", "
             << stride << " : tensor<" << blockSize << "xi64>\n";
      inversePhysical = scaled;
    }
    if (inverse.getOffsetAttr().getInt() != 0) {
      std::string offset = denseI64(
          inverse.getOffsetAttr().getInt(), "scatter_inverse_offset");
      std::string shifted = next("scatter_inverse_shifted");
      output << "    " << shifted << " = arith.addi " << inversePhysical
             << ", " << offset << " : tensor<" << blockSize << "xi64>\n";
      inversePhysical = shifted;
    }
    FailureOr<std::string> rawSourceRow = emitRawInput(
        inverse, inversePhysical, "%mask", scatter.getInverse());
    if (failed(rawSourceRow)) return failure();
    std::string sourceRow = *rawSourceRow;
    if (inverseType.getElementType().isInteger(32)) {
      sourceRow = next("scatter_source_row_i64");
      output << "    " << sourceRow << " = arith.extsi " << *rawSourceRow
             << " : tensor<" << blockSize << "xi32> to tensor<"
             << blockSize << "xi64>\n";
    }
    std::string zero = denseI64(0, "scatter_zero");
    std::string active = next("scatter_active");
    std::string valid = next("scatter_valid");
    std::string safeRow = next("scatter_safe_row");
    output << "    " << active << " = arith.cmpi sge, " << sourceRow << ", "
           << zero << " : tensor<" << blockSize << "xi64>\n"
           << "    " << valid << " = arith.andi %mask, " << active
           << " : tensor<" << blockSize << "xi1>\n"
           << "    " << safeRow << " = arith.select " << active << ", "
           << sourceRow << ", " << zero << " : tensor<" << blockSize
           << "xi1>, tensor<" << blockSize << "xi64>\n";

    std::string rowStride = denseI64(
        source.getStrides()[0], "scatter_source_row_stride");
    std::string rowOffset = next("scatter_source_row_offset");
    output << "    " << rowOffset << " = arith.muli " << safeRow << ", "
           << rowStride << " : tensor<" << blockSize << "xi64>\n";
    std::string physical = rowOffset;
    if (sourceType.getRank() == 2) {
      std::string column = next("scatter_column");
      std::string columnStride = denseI64(
          source.getStrides()[1], "scatter_source_column_stride");
      std::string columnOffset = next("scatter_column_offset");
      std::string combined = next("scatter_source_physical");
      output << "    " << column << " = arith.remui %index, " << columns
             << " : tensor<" << blockSize << "xi64>\n"
             << "    " << columnOffset << " = arith.muli " << column << ", "
             << columnStride << " : tensor<" << blockSize << "xi64>\n"
             << "    " << combined << " = arith.addi " << rowOffset << ", "
             << columnOffset << " : tensor<" << blockSize << "xi64>\n";
      physical = combined;
    }
    if (source.getOffsetAttr().getInt() != 0) {
      std::string offset = denseI64(
          source.getOffsetAttr().getInt(), "scatter_source_offset");
      std::string shifted = next("scatter_source_shifted");
      output << "    " << shifted << " = arith.addi " << physical << ", "
             << offset << " : tensor<" << blockSize << "xi64>\n";
      physical = shifted;
    }
    return emitRawInput(source, physical, valid, scatter.getResult());
  }

  FailureOr<std::string> emitRawInput(tensor::InputOp input,
                                      StringRef effective, StringRef mask,
                                      Value resultValue = {}) {
    auto type = cast<RankedTensorType>(input.getResult().getType());
    Type elementType = type.getElementType();
    unsigned index = argumentIndex.lookup(input.getResult());
    std::string base = next("input_base");
    std::string pointer = next("input_ptr");
    std::string loaded = next("input");
    StringRef typeName = spelling(elementType);
    StringRef zero = isa<FloatType>(elementType) ? "%zero_float" :
                     elementType.isInteger(32) ? "%zero_i32" : "%zero_i64";
    output << "    " << base << " = tt.splat %arg" << index << " : !tt.ptr<"
           << typeName << "> -> tensor<" << blockSize << "x!tt.ptr<"
           << typeName << ">>\n"
           << "    " << pointer << " = tt.addptr " << base << ", " << effective
           << " : tensor<" << blockSize << "x!tt.ptr<" << typeName
           << ">>, tensor<" << blockSize << "xi64>\n"
           << "    " << loaded << " = tt.load " << pointer << ", " << mask
           << ", " << zero << " : tensor<" << blockSize << "x!tt.ptr<"
           << typeName << ">>\n";
    names[resultValue ? resultValue : input.getResult()] = loaded;
    return loaded;
  }

  std::string next(StringRef stem) {
    return (Twine("%") + stem + Twine(nextId++)).str();
  }

  std::string denseI64(int64_t value, StringRef stem) {
    std::string name = next(stem);
    output << "    " << name << " = arith.constant dense<" << value
           << "> : tensor<" << blockSize << "xi64>\n";
    return name;
  }

  func::FuncOp function;
  Value result;
  int64_t blockSize;
  llvm::raw_ostream &output;
  int64_t elements = 0;
  int64_t resultRows = 0;
  int64_t resultColumns = 1;
  Type elementType;
  std::string elementName;
  unsigned nextId = 0;
  SmallVector<tensor::InputOp> inputs;
  DenseMap<Value, unsigned> argumentIndex;
  DenseMap<Value, std::string> names;
};

/// Correctness-first provider lowering for an inclusive additive scan. One
/// program owns one logical scan line, so there are no inter-program carries.
/// Later schedule selection may replace long lines by a hierarchical scan
/// without changing the semantic gf_tensor.cumsum operation.
class TensorCumsumTTIREmitter {
public:
  TensorCumsumTTIREmitter(tensor::CumsumOp scan, func::FuncOp function,
                          llvm::raw_ostream &output)
      : scan(scan), function(function), output(output) {}

  LogicalResult emit() {
    auto type = dyn_cast<RankedTensorType>(scan.getResult().getType());
    if (!type || !type.hasStaticShape() || type.getRank() < 1 ||
        !isa<FloatType>(type.getElementType()) ||
        (cast<FloatType>(type.getElementType()).getWidth() != 16 &&
         cast<FloatType>(type.getElementType()).getWidth() != 32 &&
         cast<FloatType>(type.getElementType()).getWidth() != 64))
      return scan.emitError(
          "CUDA cumsum requires a static FP16/FP32/FP64 tensor");
    auto input = scan.getInput().getDefiningOp<tensor::InputOp>();
    if (!input)
      return scan.emitError(
          "initial CUDA cumsum lowering requires a direct ABI input");
    SmallVector<int64_t> expected(type.getRank());
    int64_t stride = 1;
    for (int64_t axis = type.getRank() - 1; axis >= 0; --axis) {
      expected[axis] = stride;
      stride *= type.getDimSize(axis);
    }
    if (input.getOffsetAttr().getInt() != 0 ||
        input.getStrides() != ArrayRef<int64_t>(expected))
      return scan.emitError(
          "initial CUDA cumsum lowering requires contiguous zero-offset input");
    int64_t axis = scan.getAxisAttr().getInt();
    int64_t extent = type.getDimSize(axis);
    int64_t inner = 1;
    for (int64_t index = axis + 1; index < type.getRank(); ++index)
      inner *= type.getDimSize(index);
    int64_t lines = type.getNumElements() / extent;
    StringRef element = type.getElementType().isF16()
                            ? "f16"
                            : type.getElementType().isF32() ? "f32" : "f64";
    output << "// graphforge.tensor entry=gf_tensor_cumsum block_rows=1 "
              "block_elements=1 num_warps=1 abi=arg0,out\n"
           << "module {\n  tt.func public @gf_tensor_cumsum(%arg0: !tt.ptr<"
           << element << ">, %out: !tt.ptr<" << element
           << ">) attributes {noinline = false} {\n"
           << "    %pid_i32 = tt.get_program_id x : i32\n"
           << "    %line = arith.extui %pid_i32 : i32 to i64\n"
           << "    %inner = arith.constant " << inner << " : i64\n"
           << "    %line_outer = arith.divui %line, %inner : i64\n"
           << "    %line_inner = arith.remui %line, %inner : i64\n"
           << "    %outer_stride = arith.constant " << extent * inner
           << " : i64\n"
           << "    %outer_base = arith.muli %line_outer, %outer_stride : i64\n"
           << "    %base = arith.addi %outer_base, %line_inner : i64\n"
           << "    %begin = arith.constant 0 : index\n"
           << "    %end = arith.constant " << extent << " : index\n"
           << "    %one = arith.constant 1 : index\n"
           << "    %zero = arith.constant 0.000000e+00 : " << element << "\n"
           << "    %scan = scf.for %step = %begin to %end step %one "
              "iter_args(%acc = %zero) -> ("
           << element << ") {\n";
    output << "      %step_i64 = arith.index_cast %step : index to i64\n";
    if (scan.getReverse()) {
      output << "      %last = arith.constant " << extent - 1 << " : i64\n"
             << "      %logical = arith.subi %last, %step_i64 : i64\n";
    }
    StringRef logical = scan.getReverse() ? "%logical" : "%step_i64";
    output << "      %axis_offset = arith.muli " << logical
           << ", %inner : i64\n"
           << "      %index = arith.addi %base, %axis_offset : i64\n"
           << "      %in_ptr = tt.addptr %arg0, %index : !tt.ptr<" << element
           << ">, i64\n"
           << "      %item = tt.load %in_ptr : !tt.ptr<" << element << ">\n"
           << "      %next = arith.addf %acc, %item : " << element << "\n"
           << "      %out_ptr = tt.addptr %out, %index : !tt.ptr<" << element
           << ">, i64\n"
           << "      tt.store %out_ptr, %next : !tt.ptr<" << element << ">\n"
           << "      scf.yield %next : " << element << "\n"
           << "    }\n    tt.return\n  }\n}\n";
    (void)lines;
    return success();
  }

private:
  tensor::CumsumOp scan;
  func::FuncOp function;
  llvm::raw_ostream &output;
};

static tensor::InputOp scanContractBaseInput(Value value) {
  while (Operation *operation = value.getDefiningOp()) {
    if (auto input = dyn_cast<tensor::InputOp>(operation)) return input;
    if (auto broadcast = dyn_cast<tensor::BroadcastOp>(operation)) {
      value = broadcast.getInput();
      continue;
    }
    if (auto reshape = dyn_cast<tensor::ReshapeOp>(operation)) {
      value = reshape.getInput();
      continue;
    }
    return {};
  }
  return {};
}

/// Fuse a structurally matched map(outer-product) -> inclusive sum scan ->
/// map(contract) pipeline. The matcher names no workload: it follows Tensor IR
/// shape/algebra only. A program owns one [K,V] recurrent state tile and emits
/// every time step, avoiding materialization of the scanned rank-four tensor.
static LogicalResult translateTensorScanContract(
    tensor::ReduceSumOp reduction, tensor::CumsumOp scan,
    func::FuncOp function, llvm::raw_ostream &output) {
  auto resultType = dyn_cast<RankedTensorType>(reduction.getResult().getType());
  auto stateType = dyn_cast<RankedTensorType>(scan.getResult().getType());
  if (!resultType || !stateType || resultType.getRank() != 3 ||
      stateType.getRank() != 4 || !resultType.hasStaticShape() ||
      !stateType.hasStaticShape() || !resultType.getElementType().isF32() ||
      stateType.getElementType() != resultType.getElementType() ||
      reduction.getAxes().size() != 1 || reduction.getAxes().front() != 2 ||
      reduction.getKeepDims() || scan.getAxisAttr().getInt() != 1 ||
      scan.getReverse())
    return reduction.emitError(
        "scan-contract fusion requires forward FP32 [L,T,K,V] scan and axis-2 contraction");
  auto outputMultiply =
      reduction.getInput().getDefiningOp<tensor::MulOp>();
  if (!outputMultiply)
    return reduction.emitError("scan-contract output must be a multiply reduction");
  Value queryValue;
  if (outputMultiply.getLhs() == scan.getResult())
    queryValue = outputMultiply.getRhs();
  else if (outputMultiply.getRhs() == scan.getResult())
    queryValue = outputMultiply.getLhs();
  else
    return reduction.emitError("contraction multiply must consume the scan result");
  auto liftMultiply = scan.getInput().getDefiningOp<tensor::MulOp>();
  auto queryBroadcast = queryValue.getDefiningOp<tensor::BroadcastOp>();
  if (!liftMultiply || !queryBroadcast)
    return reduction.emitError(
        "scan-contract requires broadcast query and outer-product lift");
  auto leftBroadcast =
      liftMultiply.getLhs().getDefiningOp<tensor::BroadcastOp>();
  auto rightBroadcast =
      liftMultiply.getRhs().getDefiningOp<tensor::BroadcastOp>();
  if (!leftBroadcast || !rightBroadcast)
    return reduction.emitError("scan lift operands must be broadcasts");
  auto leftSource = cast<RankedTensorType>(leftBroadcast.getInput().getType());
  auto rightSource = cast<RankedTensorType>(rightBroadcast.getInput().getType());
  tensor::BroadcastOp keyBroadcast, valueBroadcast;
  if (leftSource.getRank() == 4 && leftSource.getDimSize(3) == 1 &&
      rightSource.getRank() == 4 && rightSource.getDimSize(2) == 1) {
    keyBroadcast = leftBroadcast;
    valueBroadcast = rightBroadcast;
  } else if (rightSource.getRank() == 4 && rightSource.getDimSize(3) == 1 &&
             leftSource.getRank() == 4 && leftSource.getDimSize(2) == 1) {
    keyBroadcast = rightBroadcast;
    valueBroadcast = leftBroadcast;
  } else {
    return reduction.emitError(
        "scan lift must broadcast [L,T,K,1] and [L,T,1,V]");
  }
  tensor::InputOp query = scanContractBaseInput(queryBroadcast.getInput());
  tensor::InputOp key = scanContractBaseInput(keyBroadcast.getInput());
  tensor::InputOp value = scanContractBaseInput(valueBroadcast.getInput());
  if (!query || !key || !value)
    return reduction.emitError("scan-contract fields must originate at ABI inputs");
  auto queryType = cast<RankedTensorType>(query.getResult().getType());
  auto keyType = cast<RankedTensorType>(key.getResult().getType());
  auto valueType = cast<RankedTensorType>(value.getResult().getType());
  int64_t lanes = stateType.getDimSize(0);
  int64_t steps = stateType.getDimSize(1);
  int64_t keyWidth = stateType.getDimSize(2);
  int64_t valueWidth = stateType.getDimSize(3);
  if (keyWidth <= 0 || keyWidth > 16 || valueWidth <= 0 ||
      queryType.getShape() != ArrayRef<int64_t>({lanes, steps, keyWidth}) ||
      keyType.getShape() != queryType.getShape() ||
      valueType.getShape() != ArrayRef<int64_t>({lanes, steps, valueWidth}) ||
      resultType.getShape() != ArrayRef<int64_t>({lanes, steps, valueWidth}))
    return reduction.emitError(
        "initial fused scan-contract supports [L,T,K<=16] x [L,T,V]");
  auto argumentNumber = [&](tensor::InputOp input) -> FailureOr<unsigned> {
    auto argument = dyn_cast<BlockArgument>(input.getBuffer());
    if (!argument || argument.getOwner() != &function.front()) return failure();
    return argument.getArgNumber();
  };
  auto qArg = argumentNumber(query), kArg = argumentNumber(key),
       vArg = argumentNumber(value);
  if (failed(qArg) || failed(kArg) || failed(vArg) ||
      function.getNumArguments() != 3)
    return reduction.emitError("scan-contract requires three function ABI inputs");
  constexpr int64_t BV = 16;
  int64_t valueTiles = (valueWidth + BV - 1) / BV;
  output << "// graphforge.tensor entry=gf_tensor_scan_contract block_rows=1 "
            "block_elements=16 num_warps=1 abi=arg0,arg1,arg2,out\n"
         << "module {\n  tt.func public @gf_tensor_scan_contract("
         << "%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, "
            "%arg2: !tt.ptr<f32>, %out: !tt.ptr<f32>) "
            "attributes {noinline = false} {\n"
         << "    %pid_i32 = tt.get_program_id x : i32\n"
         << "    %pid = arith.extui %pid_i32 : i32 to i64\n"
         << "    %value_tiles = arith.constant " << valueTiles << " : i64\n"
         << "    %lane = arith.divui %pid, %value_tiles : i64\n"
         << "    %value_tile = arith.remui %pid, %value_tiles : i64\n"
         << "    %k_i32 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>\n"
         << "    %v_i32 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>\n"
         << "    %k_lane = arith.extsi %k_i32 : tensor<16xi32> to tensor<16xi64>\n"
         << "    %v_lane = arith.extsi %v_i32 : tensor<16xi32> to tensor<16xi64>\n"
         << "    %k_bound = arith.constant dense<" << keyWidth << "> : tensor<16xi64>\n"
         << "    %k_mask = arith.cmpi slt, %k_lane, %k_bound : tensor<16xi64>\n"
         << "    %sixteen = arith.constant 16 : i64\n"
         << "    %v_base = arith.muli %value_tile, %sixteen : i64\n"
         << "    %v_base_vec = tt.splat %v_base : i64 -> tensor<16xi64>\n"
         << "    %v_column = arith.addi %v_base_vec, %v_lane : tensor<16xi64>\n"
         << "    %v_bound = arith.constant dense<" << valueWidth << "> : tensor<16xi64>\n"
         << "    %v_mask = arith.cmpi slt, %v_column, %v_bound : tensor<16xi64>\n"
         << "    %zero_k = arith.constant dense<0.000000e+00> : tensor<16xf32>\n"
         << "    %zero_v = arith.constant dense<0.000000e+00> : tensor<16xf32>\n"
         << "    %zero_state = arith.constant dense<0.000000e+00> : tensor<16x16xf32>\n"
         << "    %steps_i64 = arith.constant " << steps << " : i64\n"
         << "    %key_width = arith.constant " << keyWidth << " : i64\n"
         << "    %value_width = arith.constant " << valueWidth << " : i64\n"
         << "    %begin = arith.constant 0 : index\n"
         << "    %end = arith.constant " << steps << " : index\n"
         << "    %one = arith.constant 1 : index\n"
         << "    %scan = scf.for %time = %begin to %end step %one "
            "iter_args(%state = %zero_state) -> (tensor<16x16xf32>) {\n"
         << "      %time_i64 = arith.index_cast %time : index to i64\n"
         << "      %lane_time_base = arith.muli %lane, %steps_i64 : i64\n"
         << "      %lane_time = arith.addi %lane_time_base, %time_i64 : i64\n"
         << "      %qk_base = arith.muli %lane_time, %key_width : i64\n"
         << "      %qk_base_v = tt.splat %qk_base : i64 -> tensor<16xi64>\n"
         << "      %qk_index = arith.addi %qk_base_v, %k_lane : tensor<16xi64>\n"
         << "      %q_base = tt.splat %arg" << *qArg
         << " : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>\n"
         << "      %k_base = tt.splat %arg" << *kArg
         << " : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>\n"
         << "      %q_ptr = tt.addptr %q_base, %qk_index : tensor<16x!tt.ptr<f32>>, tensor<16xi64>\n"
         << "      %k_ptr = tt.addptr %k_base, %qk_index : tensor<16x!tt.ptr<f32>>, tensor<16xi64>\n"
         << "      %q = tt.load %q_ptr, %k_mask, %zero_k : tensor<16x!tt.ptr<f32>>\n"
         << "      %k = tt.load %k_ptr, %k_mask, %zero_k : tensor<16x!tt.ptr<f32>>\n"
         << "      %value_time_base = arith.muli %lane_time, %value_width : i64\n"
         << "      %value_time_base_v = tt.splat %value_time_base : i64 -> tensor<16xi64>\n"
         << "      %v_index = arith.addi %value_time_base_v, %v_column : tensor<16xi64>\n"
         << "      %v_input_base = tt.splat %arg" << *vArg
         << " : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>\n"
         << "      %v_ptr = tt.addptr %v_input_base, %v_index : tensor<16x!tt.ptr<f32>>, tensor<16xi64>\n"
         << "      %v = tt.load %v_ptr, %v_mask, %zero_v : tensor<16x!tt.ptr<f32>>\n"
         << "      %k2 = tt.expand_dims %k {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>\n"
         << "      %v2 = tt.expand_dims %v {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>\n"
         << "      %kb = tt.broadcast %k2 : tensor<16x1xf32> -> tensor<16x16xf32>\n"
         << "      %vb = tt.broadcast %v2 : tensor<1x16xf32> -> tensor<16x16xf32>\n"
         << "      %outer = arith.mulf %kb, %vb : tensor<16x16xf32>\n"
         << "      %next_state = arith.addf %state, %outer : tensor<16x16xf32>\n"
         << "      %q2 = tt.expand_dims %q {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>\n"
         << "      %qb = tt.broadcast %q2 : tensor<16x1xf32> -> tensor<16x16xf32>\n"
         << "      %weighted = arith.mulf %next_state, %qb : tensor<16x16xf32>\n"
         << "      %partial = \"tt.reduce\"(%weighted) <{axis = 0 : i32}> ({\n"
         << "      ^bb0(%a: f32, %b: f32):\n"
         << "        %combined = arith.addf %a, %b : f32\n"
         << "        tt.reduce.return %combined : f32\n"
         << "      }) : (tensor<16x16xf32>) -> tensor<16xf32>\n"
         << "      %out_base_scalar = arith.muli %lane_time, %value_width : i64\n"
         << "      %out_base_v = tt.splat %out_base_scalar : i64 -> tensor<16xi64>\n"
         << "      %out_index = arith.addi %out_base_v, %v_column : tensor<16xi64>\n"
         << "      %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>\n"
         << "      %out_ptr = tt.addptr %out_base, %out_index : tensor<16x!tt.ptr<f32>>, tensor<16xi64>\n"
         << "      tt.store %out_ptr, %partial, %v_mask : tensor<16x!tt.ptr<f32>>\n"
         << "      scf.yield %next_state : tensor<16x16xf32>\n"
         << "    }\n    tt.return\n  }\n}\n";
  return success();
}

class TensorCSREuclideanDistanceSumVJPTTIREmitter {
public:
  TensorCSREuclideanDistanceSumVJPTTIREmitter(
      tensor::CSREuclideanDistanceSumVJPOp vjp, func::FuncOp function,
      llvm::raw_ostream &output)
      : vjp(vjp), function(function), output(output) {}

  LogicalResult emit() {
    constexpr int64_t block = 256;
    int64_t dimensions = vjp.getDimensionsAttr().getInt();
    auto sourceType = cast<RankedTensorType>(vjp.getSourceIndex().getType());
    int64_t edges = sourceType.getDimSize(0);
    if (edges <= 0)
      return vjp.emitError("GPU Euclidean CSR VJP requires a non-empty edge set");
    SmallVector<Value> operands = {
        vjp.getPositions(), vjp.getSourceValue(), vjp.getDestination(),
        vjp.getSourceIndex(), vjp.getUpstream(), vjp.getLattice(),
        vjp.getInverseLattice()};
    SmallVector<unsigned> arguments;
    for (Value operand : operands) {
      auto input = operand.getDefiningOp<tensor::InputOp>();
      auto argument = input ? dyn_cast<BlockArgument>(input.getBuffer())
                            : BlockArgument();
      if (!argument || argument.getOwner()->getParentOp() != function)
        return vjp.emitError("GPU Euclidean CSR VJP requires direct ABI inputs");
      arguments.push_back(argument.getArgNumber());
    }
    output << "// graphforge.tensor "
              "entry=gf_tensor_csr_euclidean_distance_sum_vjp "
              "block_rows=" << block << " block_elements=" << block
           << " num_warps=8 abi=";
    for (unsigned index = 0; index < function.getNumArguments(); ++index) {
      if (index) output << ",";
      output << "arg" << index;
    }
    output << ",out\nmodule {\n  tt.func public "
              "@gf_tensor_csr_euclidean_distance_sum_vjp(";
    for (unsigned index = 0; index < function.getNumArguments(); ++index) {
      if (index) output << ", ";
      auto type = cast<RankedTensorType>(function.getArgument(index).getType());
      output << "%arg" << index << ": !tt.ptr<"
             << (type.getElementType().isF32() ? "f32" : "i64") << ">";
    }
    output << ", %out: !tt.ptr<f32>) attributes {noinline = false} {\n"
           << "    %lane_i32 = tt.make_range {end = " << block
           << " : i32, start = 0 : i32} : tensor<" << block << "xi32>\n"
           << "    %pid = tt.get_program_id x : i32\n"
           << "    %block = arith.constant " << block << " : i32\n"
           << "    %base_i32 = arith.muli %pid, %block : i32\n"
           << "    %base = arith.extsi %base_i32 : i32 to i64\n"
           << "    %base_v = tt.splat %base : i64 -> tensor<" << block
           << "xi64>\n"
           << "    %lane = arith.extsi %lane_i32 : tensor<" << block
           << "xi32> to tensor<" << block << "xi64>\n"
           << "    %edge = arith.addi %base_v, %lane : tensor<" << block
           << "xi64>\n"
           << "    %edge_bound = arith.constant dense<" << edges
           << "> : tensor<" << block << "xi64>\n"
           << "    %edge_mask = arith.cmpi slt, %edge, %edge_bound : tensor<"
           << block << "xi64>\n"
           << "    %zero = arith.constant dense<0.000000e+00> : tensor<"
           << block << "xf32>\n"
           << "    %one = arith.constant dense<1.000000e+00> : tensor<"
           << block << "xf32>\n"
           << "    %zero_i64 = arith.constant dense<0> : tensor<" << block
           << "xi64>\n";
    if (vjp.getPeriodic()) {
      output << "    %mask = arith.andi %edge_mask, %edge_mask : tensor<"
             << block << "xi1>\n";
    } else {
      // Non-periodic physical metadata is the identity lattice pair. Touch
      // both ABI operands once per program so provider DCE cannot silently
      // renumber the output pointer; distance arithmetic remains direct.
      output << "    %metadata_zero = arith.constant 0 : i64\n"
             << "    %lattice_probe_ptr = tt.addptr %arg" << arguments[5]
             << ", %metadata_zero : !tt.ptr<f32>, i64\n"
             << "    %inverse_probe_ptr = tt.addptr %arg" << arguments[6]
             << ", %metadata_zero : !tt.ptr<f32>, i64\n"
             << "    %lattice_probe = tt.load %lattice_probe_ptr : !tt.ptr<f32>\n"
             << "    %inverse_probe = tt.load %inverse_probe_ptr : !tt.ptr<f32>\n"
             << "    %metadata_product = arith.mulf %lattice_probe, "
                "%inverse_probe : f32\n"
             << "    %metadata_zero_f32 = arith.constant 0.000000e+00 : f32\n"
             << "    %metadata_valid = arith.cmpf ogt, %metadata_product, "
                "%metadata_zero_f32 : f32\n"
             << "    %metadata_mask = tt.splat %metadata_valid : i1 -> tensor<"
             << block << "xi1>\n"
             << "    %mask = arith.andi %edge_mask, %metadata_mask : tensor<"
             << block << "xi1>\n";
    }
    auto pointer = [&](StringRef name, unsigned operand, StringRef element) {
      output << "    %" << name << "_base = tt.splat %arg" << arguments[operand]
             << " : !tt.ptr<" << element << "> -> tensor<" << block
             << "x!tt.ptr<" << element << ">>\n";
    };
    pointer("destination", 2, "i64");
    pointer("source", 3, "i64");
    output << "    %destination_ptr = tt.addptr %destination_base, %edge : "
              "tensor<" << block << "x!tt.ptr<i64>>, tensor<" << block
           << "xi64>\n"
           << "    %source_ptr = tt.addptr %source_base, %edge : tensor<"
           << block << "x!tt.ptr<i64>>, tensor<" << block << "xi64>\n"
           << "    %destination = tt.load %destination_ptr, %mask, %zero_i64 : "
              "tensor<" << block << "x!tt.ptr<i64>>\n"
           << "    %source = tt.load %source_ptr, %mask, %zero_i64 : tensor<"
           << block << "x!tt.ptr<i64>>\n"
           << "    %dimensions = arith.constant dense<" << dimensions
           << "> : tensor<" << block << "xi64>\n"
           << "    %source_position_base = arith.muli %source, %dimensions : "
              "tensor<" << block << "xi64>\n"
           << "    %destination_position_base = arith.muli %destination, "
              "%dimensions : tensor<" << block << "xi64>\n";
    pointer("positions", 0, "f32");
    for (int64_t axis = 0; axis < dimensions; ++axis) {
      output << "    %axis" << axis << " = arith.constant dense<" << axis
             << "> : tensor<" << block << "xi64>\n"
             << "    %source_position_index" << axis
             << " = arith.addi %source_position_base, %axis" << axis
             << " : tensor<" << block << "xi64>\n"
             << "    %destination_position_index" << axis
             << " = arith.addi %destination_position_base, %axis" << axis
             << " : tensor<" << block << "xi64>\n"
             << "    %source_position_ptr" << axis
             << " = tt.addptr %positions_base, %source_position_index" << axis
             << " : tensor<" << block << "x!tt.ptr<f32>>, tensor<" << block
             << "xi64>\n"
             << "    %destination_position_ptr" << axis
             << " = tt.addptr %positions_base, %destination_position_index"
             << axis << " : tensor<" << block << "x!tt.ptr<f32>>, tensor<"
             << block << "xi64>\n"
             << "    %source_position" << axis << " = tt.load "
             << "%source_position_ptr" << axis << ", %mask, %zero : tensor<"
             << block << "x!tt.ptr<f32>>\n"
             << "    %destination_position" << axis << " = tt.load "
             << "%destination_position_ptr" << axis
             << ", %mask, %zero : tensor<" << block << "x!tt.ptr<f32>>\n"
             << "    %delta" << axis << " = arith.subf %source_position"
             << axis << ", %destination_position" << axis << " : tensor<"
             << block << "xf32>\n";
    }
    if (vjp.getPeriodic()) {
      for (int64_t axis = 0; axis < dimensions; ++axis) {
        output << "    %fractional" << axis
               << "_0 = arith.constant dense<0.000000e+00> : tensor<" << block
               << "xf32>\n";
        for (int64_t component = 0; component < dimensions; ++component) {
          int64_t element = component * dimensions + axis;
          output << "    %inverse_index" << component << "_" << axis
                 << " = arith.constant " << element << " : i64\n"
                 << "    %inverse_ptr" << component << "_" << axis
                 << " = tt.addptr %arg" << arguments[6] << ", %inverse_index"
                 << component << "_" << axis << " : !tt.ptr<f32>, i64\n"
                 << "    %inverse_value" << component << "_" << axis
                 << " = tt.load %inverse_ptr" << component << "_" << axis
                 << " : !tt.ptr<f32>\n"
                 << "    %inverse_v" << component << "_" << axis
                 << " = tt.splat %inverse_value" << component << "_" << axis
                 << " : f32 -> tensor<" << block << "xf32>\n"
                 << "    %fractional_term" << component << "_" << axis
                 << " = arith.mulf %delta" << component << ", %inverse_v"
                 << component << "_" << axis << " : tensor<" << block
                 << "xf32>\n"
                 << "    %fractional" << axis << "_" << component + 1
                 << " = arith.addf %fractional" << axis << "_" << component
                 << ", %fractional_term" << component << "_" << axis
                 << " : tensor<" << block << "xf32>\n";
        }
        output << "    %half" << axis
               << " = arith.constant dense<5.000000e-01> : tensor<" << block
               << "xf32>\n"
               << "    %shifted" << axis << " = arith.addf %fractional"
               << axis << "_" << dimensions << ", %half" << axis
               << " : tensor<" << block << "xf32>\n"
               << "    %rounded" << axis << " = math.floor %shifted" << axis
               << " : tensor<" << block << "xf32>\n"
               << "    %centered" << axis << " = arith.subf %fractional"
               << axis << "_" << dimensions << ", %rounded" << axis
               << " : tensor<" << block << "xf32>\n";
      }
      for (int64_t axis = 0; axis < dimensions; ++axis) {
        output << "    %mapped" << axis
               << "_0 = arith.constant dense<0.000000e+00> : tensor<" << block
               << "xf32>\n";
        for (int64_t component = 0; component < dimensions; ++component) {
          int64_t element = component * dimensions + axis;
          output << "    %lattice_index" << component << "_" << axis
                 << " = arith.constant " << element << " : i64\n"
                 << "    %lattice_ptr" << component << "_" << axis
                 << " = tt.addptr %arg" << arguments[5] << ", %lattice_index"
                 << component << "_" << axis << " : !tt.ptr<f32>, i64\n"
                 << "    %lattice_value" << component << "_" << axis
                 << " = tt.load %lattice_ptr" << component << "_" << axis
                 << " : !tt.ptr<f32>\n"
                 << "    %lattice_v" << component << "_" << axis
                 << " = tt.splat %lattice_value" << component << "_" << axis
                 << " : f32 -> tensor<" << block << "xf32>\n"
                 << "    %mapped_term" << component << "_" << axis
                 << " = arith.mulf %centered" << component << ", %lattice_v"
                 << component << "_" << axis << " : tensor<" << block
                 << "xf32>\n"
                 << "    %mapped" << axis << "_" << component + 1
                 << " = arith.addf %mapped" << axis << "_" << component
                 << ", %mapped_term" << component << "_" << axis
                 << " : tensor<" << block << "xf32>\n";
        }
      }
    }
    output << "    %squared0 = arith.constant dense<0.000000e+00> : tensor<"
           << block << "xf32>\n";
    for (int64_t axis = 0; axis < dimensions; ++axis) {
      std::string value = vjp.getPeriodic()
                              ? (Twine("%mapped") + Twine(axis) + "_" +
                                 Twine(dimensions)).str()
                              : (Twine("%delta") + Twine(axis)).str();
      output << "    %square" << axis << " = arith.mulf " << value << ", "
             << value << " : tensor<" << block << "xf32>\n"
             << "    %squared" << axis + 1 << " = arith.addf %squared" << axis
             << ", %square" << axis << " : tensor<" << block << "xf32>\n";
    }
    pointer("source_value", 1, "f32");
    pointer("upstream", 4, "f32");
    output << "    %source_value_ptr = tt.addptr %source_value_base, %source : "
              "tensor<" << block << "x!tt.ptr<f32>>, tensor<" << block
           << "xi64>\n"
           << "    %upstream_ptr = tt.addptr %upstream_base, %destination : "
              "tensor<" << block << "x!tt.ptr<f32>>, tensor<" << block
           << "xi64>\n"
           << "    %source_value_loaded = tt.load %source_value_ptr, %mask, "
              "%zero : tensor<" << block << "x!tt.ptr<f32>>\n"
           << "    %upstream_loaded = tt.load %upstream_ptr, %mask, %zero : "
              "tensor<" << block << "x!tt.ptr<f32>>\n"
           << "    %distance = math.sqrt %squared" << dimensions
           << " : tensor<" << block << "xf32>\n"
           << "    %positive = arith.cmpf ogt, %distance, %zero : tensor<"
           << block << "xf32>\n"
           << "    %safe_distance = arith.select %positive, %distance, %one : "
              "tensor<" << block << "xi1>, tensor<" << block << "xf32>\n"
           << "    %inverse_distance_raw = arith.divf %one, %safe_distance : tensor<"
           << block << "xf32>\n"
           << "    %inverse_distance = arith.select %positive, "
              "%inverse_distance_raw, %zero : tensor<" << block << "xi1>, "
              "tensor<" << block << "xf32>\n"
           << "    %geometry0 = arith.mulf %upstream_loaded, "
              "%source_value_loaded : tensor<" << block << "xf32>\n"
           << "    %geometry_scale = arith.mulf %geometry0, %inverse_distance : "
              "tensor<" << block << "xf32>\n"
           << "    %packed_stride = arith.constant dense<" << dimensions + 1
           << "> : tensor<" << block << "xi64>\n"
           << "    %source_packed = arith.muli %source, %packed_stride : tensor<"
           << block << "xi64>\n"
           << "    %destination_packed = arith.muli %destination, "
              "%packed_stride : tensor<" << block << "xi64>\n"
           << "    %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<" << block
           << "x!tt.ptr<f32>>\n";
    for (int64_t axis = 0; axis < dimensions; ++axis) {
      std::string delta = vjp.getPeriodic()
                              ? (Twine("%mapped") + Twine(axis) + "_" +
                                 Twine(dimensions)).str()
                              : (Twine("%delta") + Twine(axis)).str();
      output << "    %contribution" << axis
             << " = arith.mulf %geometry_scale, " << delta << " : tensor<"
             << block << "xf32>\n"
             << "    %negative_contribution" << axis
             << " = arith.subf %zero, %contribution" << axis << " : tensor<"
             << block << "xf32>\n"
             << "    %source_output_index" << axis
             << " = arith.addi %source_packed, %axis" << axis << " : tensor<"
             << block << "xi64>\n"
             << "    %destination_output_index" << axis
             << " = arith.addi %destination_packed, %axis" << axis
             << " : tensor<" << block << "xi64>\n"
             << "    %source_output_ptr" << axis
             << " = tt.addptr %out_base, %source_output_index" << axis
             << " : tensor<" << block << "x!tt.ptr<f32>>, tensor<" << block
             << "xi64>\n"
             << "    %destination_output_ptr" << axis
             << " = tt.addptr %out_base, %destination_output_index" << axis
             << " : tensor<" << block << "x!tt.ptr<f32>>, tensor<" << block
             << "xi64>\n"
             << "    %old_source" << axis
             << " = tt.atomic_rmw fadd, acq_rel, gpu, %source_output_ptr" << axis
             << ", %contribution" << axis << ", %mask : (tensor<" << block
             << "x!tt.ptr<f32>>, tensor<" << block << "xf32>, tensor<" << block
             << "xi1>) -> tensor<" << block << "xf32>\n"
             << "    %old_destination" << axis
             << " = tt.atomic_rmw fadd, acq_rel, gpu, %destination_output_ptr"
             << axis << ", %negative_contribution" << axis
             << ", %mask : (tensor<" << block << "x!tt.ptr<f32>>, tensor<"
             << block << "xf32>, tensor<" << block << "xi1>) -> tensor<"
             << block << "xf32>\n";
    }
    output << "    %source_gradient_axis = arith.constant dense<" << dimensions
           << "> : tensor<" << block << "xi64>\n"
           << "    %source_gradient_index = arith.addi %source_packed, "
              "%source_gradient_axis : tensor<" << block << "xi64>\n"
           << "    %source_gradient_ptr = tt.addptr %out_base, "
              "%source_gradient_index : tensor<" << block
           << "x!tt.ptr<f32>>, tensor<" << block << "xi64>\n"
           << "    %source_gradient_value = arith.mulf %upstream_loaded, "
              "%distance : tensor<" << block << "xf32>\n"
           << "    %old_source_gradient = tt.atomic_rmw fadd, acq_rel, gpu, "
              "%source_gradient_ptr, %source_gradient_value, %mask : (tensor<"
           << block << "x!tt.ptr<f32>>, tensor<" << block << "xf32>, tensor<"
           << block << "xi1>) -> tensor<" << block << "xf32>\n"
           << "    tt.return\n  }\n}\n";
    return success();
  }

private:
  tensor::CSREuclideanDistanceSumVJPOp vjp;
  func::FuncOp function;
  llvm::raw_ostream &output;
};

static LogicalResult translateTensorToTriton(Operation *root,
                                             llvm::raw_ostream &output) {
  SmallVector<tensor::ReduceSumOp> reductions;
  SmallVector<tensor::SegmentSumOp> segmentReductions;
  SmallVector<tensor::CSRSegmentSumOp> csrSumReductions;
  SmallVector<tensor::CSRSegmentProductOp> productReductions;
  SmallVector<tensor::CSRSegmentProductVJPOp> productVJPs;
  SmallVector<tensor::MatmulOp> matmuls;
  SmallVector<tensor::CumsumOp> scans;
  SmallVector<tensor::GradOp> gradients;
  SmallVector<tensor::CSREuclideanDistanceSumVJPOp> euclideanVJPs;
  root->walk([&](tensor::ReduceSumOp reduction) {
    reductions.push_back(reduction);
  });
  root->walk([&](tensor::SegmentSumOp reduction) {
    segmentReductions.push_back(reduction);
  });
  root->walk([&](tensor::CSRSegmentSumOp reduction) {
    csrSumReductions.push_back(reduction);
  });
  root->walk([&](tensor::CSRSegmentProductOp reduction) {
    productReductions.push_back(reduction);
  });
  root->walk([&](tensor::CSRSegmentProductVJPOp gradient) {
    productVJPs.push_back(gradient);
  });
  root->walk([&](tensor::GradOp gradient) { gradients.push_back(gradient); });
  root->walk([&](tensor::MatmulOp matmul) { matmuls.push_back(matmul); });
  root->walk([&](tensor::CumsumOp scan) { scans.push_back(scan); });
  root->walk([&](tensor::CSREuclideanDistanceSumVJPOp gradient) {
    euclideanVJPs.push_back(gradient);
  });
  if (!gradients.empty())
    return gradients.front().emitError(
        "run gf-tensor-vjp before provider translation");
  if (!csrSumReductions.empty()) {
    if (csrSumReductions.size() != 1 || !reductions.empty() ||
        !segmentReductions.empty() || !productReductions.empty() ||
        !productVJPs.empty() || !matmuls.empty() || !scans.empty() ||
        !euclideanVJPs.empty())
      return root->emitError(
          "gf-tensor-to-ttir requires one fused CSR sum reduction");
    auto reduction = csrSumReductions.front();
    auto function = reduction->getParentOfType<func::FuncOp>();
    auto returnOp = function && !function.empty()
                        ? dyn_cast<func::ReturnOp>(
                              function.front().getTerminator())
                        : func::ReturnOp();
    if (!returnOp || returnOp.getNumOperands() != 1)
      return reduction.emitError("fused CSR sum requires one returned tensor");
    auto messageType = cast<RankedTensorType>(reduction.getInput().getType());
    int64_t rows = std::max<int64_t>(1, reduction.getNumRows());
    int64_t averageDegree = std::max<int64_t>(
        1, (messageType.getDimSize(0) + rows - 1) / rows);
    int64_t blockSize = std::clamp<int64_t>(
        nextPowerOfTwo(averageDegree), 32, 256);
    return TensorCSRSumEpilogueTTIREmitter(
        reduction, function, returnOp.getOperand(0), blockSize, output).emit();
  }
  if (!scans.empty()) {
    if (scans.size() == 1 && reductions.size() == 1 &&
        segmentReductions.empty() && productReductions.empty() &&
        productVJPs.empty() && matmuls.empty() && euclideanVJPs.empty()) {
      auto reduction = reductions.front();
      auto function = reduction->getParentOfType<func::FuncOp>();
      auto returnOp = function && !function.empty()
                          ? dyn_cast<func::ReturnOp>(
                                function.front().getTerminator())
                          : func::ReturnOp();
      if (returnOp && returnOp.getNumOperands() == 1 &&
          returnOp.getOperand(0) == reduction.getResult())
        return translateTensorScanContract(
            reduction, scans.front(), function, output);
    }
    if (scans.size() != 1 || !reductions.empty() ||
        !segmentReductions.empty() || !productReductions.empty() ||
        !productVJPs.empty() || !matmuls.empty() || !euclideanVJPs.empty())
      return root->emitError(
          "gf-tensor-to-ttir requires one top-level cumsum");
    auto scan = scans.front();
    auto function = scan->getParentOfType<func::FuncOp>();
    auto returnOp = function && !function.empty()
                        ? dyn_cast<func::ReturnOp>(
                              function.front().getTerminator())
                        : func::ReturnOp();
    if (!returnOp || returnOp.getNumOperands() != 1 ||
        returnOp.getOperand(0) != scan.getResult())
      return scan.emitError(
          "GPU cumsum must be the directly returned Tensor value");
    return TensorCumsumTTIREmitter(scan, function, output).emit();
  }
  if (!euclideanVJPs.empty()) {
    if (euclideanVJPs.size() != 1 || !reductions.empty() ||
        !segmentReductions.empty() || !productReductions.empty() ||
        !productVJPs.empty() || !matmuls.empty())
      return root->emitError(
          "gf-tensor-to-ttir requires one top-level Euclidean CSR VJP");
    auto gradient = euclideanVJPs.front();
    auto function = gradient->getParentOfType<func::FuncOp>();
    auto returnOp = function && !function.empty()
                        ? dyn_cast<func::ReturnOp>(
                              function.front().getTerminator())
                        : func::ReturnOp();
    if (!returnOp || returnOp.getNumOperands() != 1 ||
        returnOp.getOperand(0) != gradient.getResult())
      return gradient.emitError(
          "GPU Euclidean CSR VJP must be the directly returned value");
    return TensorCSREuclideanDistanceSumVJPTTIREmitter(
        gradient, function, output).emit();
  }
  if (!productReductions.empty() || !productVJPs.empty()) {
    if (productReductions.size() + productVJPs.size() != 1 ||
        !reductions.empty() || !segmentReductions.empty() || !matmuls.empty())
      return root->emitError(
          "gf-tensor-to-ttir requires one top-level CSR product or VJP");
    Operation *operation = !productReductions.empty()
                               ? productReductions.front().getOperation()
                               : productVJPs.front().getOperation();
    auto function = operation->getParentOfType<func::FuncOp>();
    auto returnOp = function && !function.empty()
                        ? dyn_cast<func::ReturnOp>(
                              function.front().getTerminator())
                        : func::ReturnOp();
    if (!returnOp || returnOp.getNumOperands() != 1 ||
        returnOp.getOperand(0).getDefiningOp() != operation)
      return operation->emitError(
          "GPU CSR product operation must be the directly returned value");
    int64_t maximum = !productReductions.empty()
                          ? productReductions.front().getMaxDegree()
                          : productVJPs.front().getMaxDegree();
    int64_t blockSize = 1;
    while (blockSize < maximum) blockSize *= 2;
    if (blockSize > 65536)
      return operation->emitError("CSR product degree exceeds 65536");
    if (!productReductions.empty())
      return TensorCSRProductTTIREmitter(
          productReductions.front(), function, blockSize, output).emit();
    return TensorCSRProductVJPTTIREmitter(
        productVJPs.front(), function, blockSize, output).emit();
  }
  if (!matmuls.empty()) {
    if (matmuls.size() != 1 || !reductions.empty() ||
        !segmentReductions.empty())
      return root->emitError(
          "gf-tensor-to-ttir currently requires one top-level matmul");
    auto function = matmuls.front()->getParentOfType<func::FuncOp>();
    auto returnOp = function && !function.empty()
                        ? dyn_cast<func::ReturnOp>(
                              function.front().getTerminator())
                        : func::ReturnOp();
    if (!returnOp || returnOp.getNumOperands() != 1 ||
        returnOp.getOperand(0) != matmuls.front().getResult())
      return matmuls.front().emitError(
          "GPU matmul must be the directly returned Tensor value");
    return TensorMatmulTTIREmitter(matmuls.front(), function, output).emit();
  }
  if (!segmentReductions.empty()) {
    if (segmentReductions.size() != 1 || !reductions.empty())
      return root->emitError(
          "gf-tensor-to-ttir requires one top-level segment_sum");
    auto segment = segmentReductions.front();
    auto function = segment->getParentOfType<func::FuncOp>();
    auto returnOp = function && !function.empty()
                        ? dyn_cast<func::ReturnOp>(
                              function.front().getTerminator())
                        : func::ReturnOp();
    if (!returnOp || returnOp.getNumOperands() != 1 ||
        returnOp.getOperand(0) != segment.getResult())
      return segment.emitError(
          "GPU segment_sum must be the directly returned Tensor value");
    int64_t edges = cast<RankedTensorType>(segment.getInput().getType())
                        .getDimSize(0);
    int64_t blockSize = 1;
    while (blockSize < edges) blockSize *= 2;
    if (blockSize > 65536)
      return segment.emitError("segment_sum edge extent is too large");
    return TensorSegmentSumTTIREmitter(
        segment, function, blockSize, output).emit();
  }
  if (reductions.empty()) {
    SmallVector<func::FuncOp> functions;
    root->walk([&](func::FuncOp function) { functions.push_back(function); });
    if (functions.size() != 1)
      return root->emitError("pointwise TTIR requires exactly one function");
    auto returnOp = dyn_cast<func::ReturnOp>(
        functions.front().front().getTerminator());
    if (!returnOp || returnOp.getNumOperands() != 1)
      return functions.front().emitError("requires one returned tensor");
    if (auto type = dyn_cast<RankedTensorType>(returnOp.getOperand(0).getType());
        type && isa<ComplexType>(type.getElementType()))
      return ComplexTensorPointwiseTTIREmitter(
          functions.front(), returnOp.getOperand(0), 512, output).emit();
    return TensorPointwiseTTIREmitter(
        functions.front(), returnOp.getOperand(0), 512, output).emit();
  }
  if (reductions.size() != 1)
    return root->emitError("gf-tensor-to-ttir requires at most one reduction");
  auto function = reductions.front()->getParentOfType<func::FuncOp>();
  if (!function)
    return reductions.front().emitError("must be nested in func.func");
  auto reductionType = cast<RankedTensorType>(
      reductions.front().getInput().getType());
  int64_t reductionAxis = reductions.front().getAxes().size() == 1
                              ? reductions.front().getAxes().front()
                              : 1;
  int64_t columns = reductionType.getRank() == 2 &&
                            (reductionAxis == 0 || reductionAxis == 1)
                        ? reductionType.getDimSize(reductionAxis)
                        : 0;
  int64_t blockSize = 1;
  while (blockSize < columns)
    blockSize *= 2;
  if (blockSize > 65536)
    return reductions.front().emitError("reduction extent is too large");
  return TensorReductionTTIREmitter(reductions.front(), function, blockSize,
                                    output).emit();
}

} // namespace

void registerTensorToTritonTranslation() {
  TranslateFromMLIRRegistration(
      "gf-tensor-to-ttir",
      "lower a supported canonical gf_tensor module to serialized Triton IR",
      translateTensorToTriton, [](DialectRegistry &registry) {
        registry.insert<control::GraphForgeControlDialect,
                        tensor::GraphForgeTensorDialect,
                        func::FuncDialect>();
      });
}

} // namespace mlir::graphforge
