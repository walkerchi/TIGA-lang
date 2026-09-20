#include "tiga/Transforms/Passes.h"

#include "tiga/Dialect/Tensor/TensorDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tiga {

#define GEN_PASS_DEF_GFTENSORVJP
#include "tiga/Transforms/Passes.h.inc"

namespace {

using namespace mlir::tiga::tensor;

static DenseI64ArrayAttr shapeAttr(OpBuilder &builder, RankedTensorType type) {
  return builder.getDenseI64ArrayAttr(type.getShape());
}

template <typename OpTy>
static Value createUnary(IRRewriter &rewriter, Location location, Value input,
                         RankedTensorType resultType,
                         ArrayRef<NamedAttribute> attributes = {}) {
  OperationState state(location, OpTy::getOperationName());
  state.addOperands(input);
  state.addTypes(resultType);
  state.addAttributes(attributes);
  return rewriter.create(state)->getResult(0);
}

template <typename OpTy>
static Value createBinary(IRRewriter &rewriter, Location location, Value lhs,
                          Value rhs, RankedTensorType resultType) {
  OperationState state(location, OpTy::getOperationName());
  state.addOperands({lhs, rhs});
  state.addTypes(resultType);
  return rewriter.create(state)->getResult(0);
}

static Value reshape(IRRewriter &rewriter, Location location, Value input,
                     RankedTensorType resultType) {
  NamedAttribute shape(rewriter.getStringAttr("shape"),
                       shapeAttr(rewriter, resultType));
  return createUnary<ReshapeOp>(rewriter, location, input, resultType, {shape});
}

static Value conjugateIfComplex(IRRewriter &rewriter, Location location,
                                Value value) {
  auto type = cast<RankedTensorType>(value.getType());
  if (!isa<ComplexType>(type.getElementType()))
    return value;
  return createUnary<ConjOp>(rewriter, location, value, type);
}

static Value reduceToShape(IRRewriter &rewriter, Location location, Value value,
                           RankedTensorType targetType) {
  auto sourceType = cast<RankedTensorType>(value.getType());
  if (sourceType == targetType)
    return value;

  int64_t padding = sourceType.getRank() - targetType.getRank();
  if (padding < 0)
    return {};
  SmallVector<int64_t> axes;
  for (int64_t axis = 0; axis < sourceType.getRank(); ++axis) {
    if (axis < padding) {
      axes.push_back(axis);
      continue;
    }
    int64_t target = targetType.getDimSize(axis - padding);
    int64_t source = sourceType.getDimSize(axis);
    if (target == 1 && source != 1)
      axes.push_back(axis);
  }
  Value reduced = value;
  if (!axes.empty()) {
    SmallVector<int64_t> reducedShape(sourceType.getShape());
    for (int64_t axis : axes)
      reducedShape[axis] = 1;
    auto reducedType = RankedTensorType::get(
        reducedShape, sourceType.getElementType(), sourceType.getEncoding());
    NamedAttribute axesAttribute(rewriter.getStringAttr("axes"),
                                 rewriter.getDenseI64ArrayAttr(axes));
    NamedAttribute keepDims(rewriter.getStringAttr("keep_dims"),
                            rewriter.getBoolAttr(true));
    reduced = createUnary<ReduceSumOp>(rewriter, location, value, reducedType,
                                       {axesAttribute, keepDims});
  }
  return reduced.getType() == targetType
             ? reduced
             : reshape(rewriter, location, reduced, targetType);
}

static void accumulate(IRRewriter &rewriter, Location location,
                       llvm::DenseMap<Value, Value> &adjoints, Value primal,
                       Value contribution) {
  auto found = adjoints.find(primal);
  if (found == adjoints.end()) {
    adjoints[primal] = contribution;
    return;
  }
  auto type = cast<RankedTensorType>(primal.getType());
  found->second = createBinary<AddOp>(rewriter, location, found->second,
                                      contribution, type);
}

static bool dependsOn(Value value, Value wrt,
                      llvm::DenseMap<Value, bool> &memo) {
  if (value == wrt)
    return true;
  auto found = memo.find(value);
  if (found != memo.end())
    return found->second;
  Operation *definition = value.getDefiningOp();
  if (!definition || isa<InputOp>(definition)) {
    memo[value] = false;
    return false;
  }
  bool dependent = llvm::any_of(definition->getOperands(), [&](Value operand) {
    return dependsOn(operand, wrt, memo);
  });
  memo[value] = dependent;
  return dependent;
}

static LogicalResult lowerGrad(GradOp grad, IRRewriter &rewriter) {
  llvm::DenseMap<Value, bool> dependencyMemo;
  if (!dependsOn(grad.getOutput(), grad.getWrt(), dependencyMemo))
    return grad.emitError("differentiated input is not connected to output");
  rewriter.setInsertionPoint(grad);
  llvm::DenseMap<Value, Value> adjoints;
  adjoints[grad.getOutput()] = grad.getCotangent();

  SmallVector<Operation *> forward;
  for (Operation *operation = grad->getPrevNode(); operation;
       operation = operation->getPrevNode())
    forward.push_back(operation);

  for (Operation *operation : forward) {
    if (operation->getNumResults() != 1)
      continue;
    Value result = operation->getResult(0);
    auto found = adjoints.find(result);
    if (found == adjoints.end())
      continue;
    Value upstream = found->second;
    Location location = operation->getLoc();

    if (isa<InputOp>(operation))
      continue;
    if (auto add = dyn_cast<AddOp>(operation)) {
      for (Value operand : {add.getLhs(), add.getRhs()}) {
        if (!dependsOn(operand, grad.getWrt(), dependencyMemo))
          continue;
        auto type = cast<RankedTensorType>(operand.getType());
        Value contribution = reduceToShape(rewriter, location, upstream, type);
        if (!contribution)
          return add.emitError("cannot reduce broadcast adjoint to operand shape");
        accumulate(rewriter, location, adjoints, operand, contribution);
      }
      continue;
    }
    if (auto mul = dyn_cast<MulOp>(operation)) {
      for (auto [operand, other] :
           {std::pair<Value, Value>(mul.getLhs(), mul.getRhs()),
            std::pair<Value, Value>(mul.getRhs(), mul.getLhs())}) {
        if (!dependsOn(operand, grad.getWrt(), dependencyMemo))
          continue;
        Value conjugated = conjugateIfComplex(rewriter, location, other);
        auto productType = cast<RankedTensorType>(mul.getResult().getType());
        Value product = createBinary<MulOp>(rewriter, location, upstream,
                                            conjugated, productType);
        auto operandType = cast<RankedTensorType>(operand.getType());
        Value contribution =
            reduceToShape(rewriter, location, product, operandType);
        if (!contribution)
          return mul.emitError("cannot reduce broadcast adjoint to operand shape");
        accumulate(rewriter, location, adjoints, operand, contribution);
      }
      continue;
    }
    if (auto matmul = dyn_cast<MatmulOp>(operation)) {
      Value lhs = matmul.getLhs();
      Value rhs = matmul.getRhs();
      auto transposeConjugate = [&](Value value) {
        auto inputType = cast<RankedTensorType>(value.getType());
        auto transposedType = RankedTensorType::get(
            {inputType.getDimSize(1), inputType.getDimSize(0)},
            inputType.getElementType(), inputType.getEncoding());
        NamedAttribute axes(rewriter.getStringAttr("axes"),
                            rewriter.getDenseI64ArrayAttr({1, 0}));
        Value transposed = createUnary<PermuteOp>(
            rewriter, location, value, transposedType, {axes});
        return conjugateIfComplex(rewriter, location, transposed);
      };
      if (dependsOn(lhs, grad.getWrt(), dependencyMemo)) {
        auto lhsType = cast<RankedTensorType>(lhs.getType());
        Value contribution = createBinary<MatmulOp>(
            rewriter, location, upstream, transposeConjugate(rhs), lhsType);
        accumulate(rewriter, location, adjoints, lhs, contribution);
      }
      if (dependsOn(rhs, grad.getWrt(), dependencyMemo)) {
        auto rhsType = cast<RankedTensorType>(rhs.getType());
        Value contribution = createBinary<MatmulOp>(
            rewriter, location, transposeConjugate(lhs), upstream, rhsType);
        accumulate(rewriter, location, adjoints, rhs, contribution);
      }
      continue;
    }
    if (auto divide = dyn_cast<DivOp>(operation)) {
      Value lhs = divide.getLhs();
      Value rhs = divide.getRhs();
      auto resultType = cast<RankedTensorType>(divide.getResult().getType());
      if (dependsOn(lhs, grad.getWrt(), dependencyMemo)) {
        Value denominator = conjugateIfComplex(rewriter, location, rhs);
        Value quotient = createBinary<DivOp>(
            rewriter, location, upstream, denominator, resultType);
        auto lhsType = cast<RankedTensorType>(lhs.getType());
        Value contribution = reduceToShape(
            rewriter, location, quotient, lhsType);
        if (!contribution)
          return divide.emitError("cannot reduce lhs adjoint to operand shape");
        accumulate(rewriter, location, adjoints, lhs, contribution);
      }
      if (dependsOn(rhs, grad.getWrt(), dependencyMemo)) {
        auto rhsType = cast<RankedTensorType>(rhs.getType());
        Value rhsSquared = createBinary<MulOp>(
            rewriter, location, rhs, rhs, rhsType);
        Value local = createBinary<DivOp>(
            rewriter, location, lhs, rhsSquared, resultType);
        local = conjugateIfComplex(rewriter, location, local);
        local = createUnary<NegOp>(rewriter, location, local, resultType);
        Value product = createBinary<MulOp>(
            rewriter, location, upstream, local, resultType);
        Value contribution = reduceToShape(
            rewriter, location, product, rhsType);
        if (!contribution)
          return divide.emitError("cannot reduce rhs adjoint to operand shape");
        accumulate(rewriter, location, adjoints, rhs, contribution);
      }
      continue;
    }
    if (auto neg = dyn_cast<NegOp>(operation)) {
      if (dependsOn(neg.getInput(), grad.getWrt(), dependencyMemo)) {
        auto type = cast<RankedTensorType>(neg.getInput().getType());
        Value contribution = createUnary<NegOp>(
            rewriter, location, upstream, type);
        accumulate(rewriter, location, adjoints, neg.getInput(), contribution);
      }
      continue;
    }
    if (auto exponential = dyn_cast<ExpOp>(operation)) {
      if (dependsOn(exponential.getInput(), grad.getWrt(), dependencyMemo)) {
        auto type = cast<RankedTensorType>(exponential.getResult().getType());
        Value contribution = createBinary<MulOp>(
            rewriter, location, upstream, exponential.getResult(), type);
        accumulate(rewriter, location, adjoints, exponential.getInput(),
                   contribution);
      }
      continue;
    }
    if (auto root = dyn_cast<SqrtOp>(operation)) {
      if (dependsOn(root.getInput(), grad.getWrt(), dependencyMemo)) {
        auto type = cast<RankedTensorType>(root.getResult().getType());
        Value twiceRoot = createBinary<AddOp>(
            rewriter, location, root.getResult(), root.getResult(), type);
        Value contribution = createBinary<DivOp>(
            rewriter, location, upstream, twiceRoot, type);
        accumulate(rewriter, location, adjoints, root.getInput(), contribution);
      }
      continue;
    }
    if (auto conj = dyn_cast<ConjOp>(operation)) {
      Value contribution = conjugateIfComplex(rewriter, location, upstream);
      accumulate(rewriter, location, adjoints, conj.getInput(), contribution);
      continue;
    }
    if (auto view = dyn_cast<ReshapeOp>(operation)) {
      auto inputType = cast<RankedTensorType>(view.getInput().getType());
      accumulate(rewriter, location, adjoints, view.getInput(),
                 reshape(rewriter, location, upstream, inputType));
      continue;
    }
    if (auto permutation = dyn_cast<PermuteOp>(operation)) {
      SmallVector<int64_t> inverse(permutation.getAxes().size());
      for (auto [outputAxis, inputAxis] : llvm::enumerate(permutation.getAxes()))
        inverse[inputAxis] = outputAxis;
      auto inputType = cast<RankedTensorType>(permutation.getInput().getType());
      NamedAttribute axes(rewriter.getStringAttr("axes"),
                          rewriter.getDenseI64ArrayAttr(inverse));
      Value contribution = createUnary<PermuteOp>(
          rewriter, location, upstream, inputType, {axes});
      accumulate(rewriter, location, adjoints, permutation.getInput(),
                 contribution);
      continue;
    }
    if (auto broadcast = dyn_cast<BroadcastOp>(operation)) {
      auto inputType = cast<RankedTensorType>(broadcast.getInput().getType());
      Value contribution =
          reduceToShape(rewriter, location, upstream, inputType);
      if (!contribution)
        return broadcast.emitError("cannot reduce adjoint to input shape");
      accumulate(rewriter, location, adjoints, broadcast.getInput(),
                 contribution);
      continue;
    }
    if (auto checkpoint = dyn_cast<CheckpointOp>(operation)) {
      if (dependsOn(checkpoint.getInput(), grad.getWrt(), dependencyMemo))
        accumulate(rewriter, location, adjoints, checkpoint.getInput(), upstream);
      continue;
    }
    if (auto candidate = dyn_cast<CheckpointCandidateOp>(operation)) {
      if (dependsOn(candidate.getInput(), grad.getWrt(), dependencyMemo))
        accumulate(rewriter, location, adjoints, candidate.getInput(), upstream);
      continue;
    }
    if (auto reduction = dyn_cast<ReduceSumOp>(operation)) {
      auto inputType = cast<RankedTensorType>(reduction.getInput().getType());
      Value expanded = upstream;
      if (!reduction.getKeepDims()) {
        SmallVector<int64_t> keepShape(inputType.getShape());
        for (int64_t axis : reduction.getAxes())
          keepShape[axis] = 1;
        auto keepType = RankedTensorType::get(
            keepShape, inputType.getElementType(), inputType.getEncoding());
        expanded = reshape(rewriter, location, upstream, keepType);
      }
      NamedAttribute shape(rewriter.getStringAttr("shape"),
                           shapeAttr(rewriter, inputType));
      Value contribution = createUnary<BroadcastOp>(
          rewriter, location, expanded, inputType, {shape});
      accumulate(rewriter, location, adjoints, reduction.getInput(),
                 contribution);
      continue;
    }
    if (auto scan = dyn_cast<CumsumOp>(operation)) {
      if (dependsOn(scan.getInput(), grad.getWrt(), dependencyMemo)) {
        auto type = cast<RankedTensorType>(scan.getInput().getType());
        NamedAttribute axis(rewriter.getStringAttr("axis"),
                            scan.getAxisAttr());
        NamedAttribute reverse(rewriter.getStringAttr("reverse"),
                               rewriter.getBoolAttr(!scan.getReverse()));
        Value contribution = createUnary<CumsumOp>(
            rewriter, location, upstream, type, {axis, reverse});
        accumulate(rewriter, location, adjoints, scan.getInput(), contribution);
      }
      continue;
    }
    if (auto gather = dyn_cast<GatherOp>(operation)) {
      if (dependsOn(gather.getInput(), grad.getWrt(), dependencyMemo)) {
        auto inputType = cast<RankedTensorType>(gather.getInput().getType());
        OperationState state(location, SegmentSumOp::getOperationName());
        state.addOperands({upstream, gather.getIndex()});
        state.addTypes(inputType);
        state.addAttribute("num_segments", rewriter.getI64IntegerAttr(
            inputType.getDimSize(0)));
        Value contribution = rewriter.create(state)->getResult(0);
        accumulate(rewriter, location, adjoints, gather.getInput(), contribution);
      }
      continue;
    }
    if (auto scatter = dyn_cast<ScatterRowsOp>(operation)) {
      if (dependsOn(scatter.getInput(), grad.getWrt(), dependencyMemo)) {
        auto inputType = cast<RankedTensorType>(scatter.getInput().getType());
        OperationState state(location, GatherOp::getOperationName());
        state.addOperands({upstream, scatter.getDestination()});
        state.addTypes(inputType);
        Value contribution = rewriter.create(state)->getResult(0);
        accumulate(rewriter, location, adjoints, scatter.getInput(),
                   contribution);
      }
      continue;
    }
    if (auto segment = dyn_cast<SegmentSumOp>(operation)) {
      if (dependsOn(segment.getInput(), grad.getWrt(), dependencyMemo)) {
        auto inputType = cast<RankedTensorType>(segment.getInput().getType());
        OperationState state(location, GatherOp::getOperationName());
        state.addOperands({upstream, segment.getIndex()});
        state.addTypes(inputType);
        Value contribution = rewriter.create(state)->getResult(0);
        accumulate(rewriter, location, adjoints, segment.getInput(), contribution);
      }
      continue;
    }
    if (auto expand = dyn_cast<CSRExpandRowsOp>(operation)) {
      if (dependsOn(expand.getInput(), grad.getWrt(), dependencyMemo)) {
        auto inputType = cast<RankedTensorType>(expand.getInput().getType());
        OperationState state(location, CSRSegmentSumOp::getOperationName());
        state.addOperands({upstream, expand.getRowPtr()});
        state.addTypes(inputType);
        state.addAttribute("num_rows", rewriter.getI64IntegerAttr(
            inputType.getDimSize(0)));
        // CSRExpandRows does not itself carry a degree proof. Preserve the
        // semantic VJP and let a later relation analysis refine these unknown
        // bounds instead of inventing uniformity.
        state.addAttribute("degree_min", rewriter.getI64IntegerAttr(0));
        state.addAttribute("degree_max", rewriter.getI64IntegerAttr(0));
        Value contribution = rewriter.create(state)->getResult(0);
        accumulate(rewriter, location, adjoints, expand.getInput(), contribution);
      }
      continue;
    }
    if (auto segment = dyn_cast<CSRSegmentSumOp>(operation)) {
      if (dependsOn(segment.getInput(), grad.getWrt(), dependencyMemo)) {
        auto inputType = cast<RankedTensorType>(segment.getInput().getType());
        OperationState state(location, CSRExpandRowsOp::getOperationName());
        state.addOperands({upstream, segment.getRowPtr()});
        state.addTypes(inputType);
        state.addAttribute("num_edges", rewriter.getI64IntegerAttr(
            inputType.getDimSize(0)));
        Value contribution = rewriter.create(state)->getResult(0);
        accumulate(rewriter, location, adjoints, segment.getInput(), contribution);
      }
      continue;
    }
    if (auto product = dyn_cast<CSRSegmentProductOp>(operation)) {
      if (dependsOn(product.getInput(), grad.getWrt(), dependencyMemo)) {
        OperationState state(location,
                             CSRSegmentProductVJPOp::getOperationName());
        state.addOperands({product.getInput(), product.getRowPtr(),
                           product.getDestination(), upstream});
        state.addTypes(product.getInput().getType());
        state.addAttribute("num_rows", product.getNumRowsAttr());
        state.addAttribute("max_degree", product.getMaxDegreeAttr());
        state.addAttribute("uniform_degree", product.getUniformDegreeAttr());
        Value contribution = rewriter.create(state)->getResult(0);
        accumulate(rewriter, location, adjoints, product.getInput(),
                   contribution);
      }
      continue;
    }
    if (isa<CSRSegmentProductVJPOp>(operation))
      return operation->emitError(
          "higher-order product-reducer differentiation is unsupported");
    if (isa<CSRSegmentMaxStopGradientOp>(operation)) {
      // Internal online-softmax numerical shift: the operation contract
      // explicitly detaches the maximum from the reverse graph.
      continue;
    }
    return operation->emitError(
        "has no reverse-mode rule for a value used by gf_tensor.grad");
  }

  auto result = adjoints.find(grad.getWrt());
  if (result == adjoints.end())
    return grad.emitError("differentiated input is not connected to output");
  rewriter.replaceOp(grad, result->second);
  return success();
}

class TensorVJPPass : public impl::GFTensorVJPBase<TensorVJPPass> {
public:
  using impl::GFTensorVJPBase<TensorVJPPass>::GFTensorVJPBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<tensor::TigaTensorDialect>();
  }

  void runOnOperation() final {
    SmallVector<GradOp> requests;
    getOperation().walk([&](GradOp grad) { requests.push_back(grad); });
    IRRewriter rewriter(&getContext());
    for (GradOp grad : requests) {
      if (failed(lowerGrad(grad, rewriter))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
} // namespace mlir::tiga
