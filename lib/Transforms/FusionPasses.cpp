#include "tiga/Transforms/Passes.h"

#include "tiga/Dialect/Domain/DomainDialect.h"
#include "tiga/Transforms/FusionAnalysis.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tiga {

#define GEN_PASS_DEF_GFVERIFYDOMAIN
#define GEN_PASS_DEF_GFFORMAPPLYFUSIONGROUPS
#define GEN_PASS_DEF_GFFUSECOMPATIBLEAPPLIES
#include "tiga/Transforms/Passes.h.inc"

namespace {

class VerifyDomainPass : public impl::GFVerifyDomainBase<VerifyDomainPass> {
public:
  using impl::GFVerifyDomainBase<VerifyDomainPass>::GFVerifyDomainBase;

  void runOnOperation() final {
    if (failed(mlir::verify(getOperation())))
      signalPassFailure();
  }
};

class FormApplyFusionGroupsPass
    : public impl::GFFormApplyFusionGroupsBase<FormApplyFusionGroupsPass> {
public:
  using impl::GFFormApplyFusionGroupsBase<
      FormApplyFusionGroupsPass>::GFFormApplyFusionGroupsBase;

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    getOperation()->walk([](ApplyOp op) {
      op->removeAttr("fusion_group");
    });

    int64_t nextGroup = 0;
    getOperation()->walk([&](ApplyOp first) {
      auto second = dyn_cast_or_null<ApplyOp>(first->getNextNode());
      if (!second)
        return;
      FusionDecision decision = analyzeHorizontalFusion(first, second);
      if (!decision.isLegal()) {
        if (emitRejections)
          first.emitRemark() << "fusion rejected: " << decision.reason();
        return;
      }

      IntegerAttr group = first->getAttrOfType<IntegerAttr>("fusion_group");
      if (!group)
        group = IntegerAttr::get(IntegerType::get(context, 64), nextGroup++);
      first->setAttr("fusion_group", group);
      second->setAttr("fusion_group", group);
    });
  }
};

static void appendArrayAttr(SmallVectorImpl<Attribute> &target,
                            Operation *operation, StringRef name) {
  auto array = operation->getAttrOfType<ArrayAttr>(name);
  llvm::append_range(target, array);
}

static void appendDenseI64(SmallVectorImpl<int64_t> &target,
                           Operation *operation, StringRef name) {
  auto array = operation->getAttrOfType<DenseI64ArrayAttr>(name);
  llvm::append_range(target, array.asArrayRef());
}

class FuseCompatibleApplyPattern : public OpRewritePattern<ApplyOp> {
public:
  using OpRewritePattern<ApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ApplyOp first,
                                PatternRewriter &rewriter) const final {
    auto second = dyn_cast_or_null<ApplyOp>(first->getNextNode());
    if (!second)
      return rewriter.notifyMatchFailure(first, "next operation is not gf.apply");
    FusionDecision decision = analyzeHorizontalFusion(first, second);
    if (!decision.isLegal())
      return rewriter.notifyMatchFailure(first, decision.reason());

    SmallVector<Value> inputs(first.getInputs());
    llvm::append_range(inputs, second.getInputs());
    SmallVector<Type> outputTypes(first.getResultTypes());
    llvm::append_range(outputTypes, second.getResultTypes());

    SmallVector<Attribute> reducers;
    appendArrayAttr(reducers, first, "reducers");
    appendArrayAttr(reducers, second, "reducers");
    SmallVector<Attribute> effects;
    appendArrayAttr(effects, first, "effects");
    appendArrayAttr(effects, second, "effects");
    SmallVector<int64_t> inputSegments;
    appendDenseI64(inputSegments, first, "input_segment_sizes");
    appendDenseI64(inputSegments, second, "input_segment_sizes");
    SmallVector<int64_t> regionKinds;
    appendDenseI64(regionKinds, first, "region_kinds");
    appendDenseI64(regionKinds, second, "region_kinds");
    SmallVector<int64_t> snapshotVersions;
    appendDenseI64(snapshotVersions, first, "snapshot_versions");
    appendDenseI64(snapshotVersions, second, "snapshot_versions");

    OperationState state(
        FusedLoc::get(rewriter.getContext(), {first.getLoc(), second.getLoc()}),
        ApplyOp::getOperationName());
    state.addOperands({first.getRelation()});
    state.addOperands(inputs);
    state.addTypes(outputTypes);
    state.addAttribute("reducers", rewriter.getArrayAttr(reducers));
    state.addAttribute("region_kinds",
                       rewriter.getDenseI64ArrayAttr(regionKinds));
    state.addAttribute("input_segment_sizes",
                       rewriter.getDenseI64ArrayAttr(inputSegments));
    state.addAttribute("snapshot_versions",
                       rewriter.getDenseI64ArrayAttr(snapshotVersions));
    state.addAttribute("effects", rewriter.getArrayAttr(effects));
    state.addAttribute("deterministic", first->getAttr("deterministic"));
    if (first.getInputRolesAttr() && second.getInputRolesAttr()) {
      SmallVector<Attribute> roles;
      llvm::append_range(roles, first.getInputRolesAttr());
      llvm::append_range(roles, second.getInputRolesAttr());
      state.addAttribute("input_roles", rewriter.getArrayAttr(roles));
    }
    if (first.getInputNamesAttr() && second.getInputNamesAttr()) {
      SmallVector<Attribute> names;
      llvm::append_range(names, first.getInputNamesAttr());
      llvm::append_range(names, second.getInputNamesAttr());
      state.addAttribute("input_names", rewriter.getArrayAttr(names));
    }
    if (first.getNodeInputIndicesAttr() && second.getNodeInputIndicesAttr() &&
        first.getNodeInputSegmentSizesAttr() &&
        second.getNodeInputSegmentSizesAttr()) {
      SmallVector<int64_t> nodeIndices(
          first.getNodeInputIndicesAttr().asArrayRef());
      for (int64_t index : second.getNodeInputIndicesAttr().asArrayRef())
        nodeIndices.push_back(index + first.getInputs().size());
      SmallVector<int64_t> nodeSegments(
          first.getNodeInputSegmentSizesAttr().asArrayRef());
      llvm::append_range(nodeSegments,
                         second.getNodeInputSegmentSizesAttr().asArrayRef());
      state.addAttribute("node_input_indices",
                         rewriter.getDenseI64ArrayAttr(nodeIndices));
      state.addAttribute("node_input_segment_sizes",
                         rewriter.getDenseI64ArrayAttr(nodeSegments));
    }
    if (first.getIterationLanesAttr() && second.getIterationLanesAttr() &&
        first.getIterationLanesAttr() == second.getIterationLanesAttr())
      state.addAttribute("iteration_lanes", first.getIterationLanesAttr());
    IntegerAttr firstCount =
        first->getAttrOfType<IntegerAttr>("fusion_count");
    IntegerAttr secondCount =
        second->getAttrOfType<IntegerAttr>("fusion_count");
    int64_t count = firstCount ? firstCount.getInt() : 1;
    count += secondCount ? secondCount.getInt() : 1;
    state.addAttribute("fusion_count", rewriter.getI64IntegerAttr(count));

    for (Operation *operation :
         {first.getOperation(), second.getOperation()}) {
      for (Region &oldRegion : operation->getRegions()) {
        Region *newRegion = state.addRegion();
        IRMapping mapping;
        oldRegion.cloneInto(newRegion, mapping);
      }
    }

    Operation *fusedOperation = rewriter.create(state);
    auto fused = cast<ApplyOp>(fusedOperation);
    unsigned result = 0;
    for (Value oldResult : first.getOutputs())
      oldResult.replaceAllUsesWith(fused.getResult(result++));
    for (Value oldResult : second.getOutputs())
      oldResult.replaceAllUsesWith(fused.getResult(result++));
    first.emitRemark("fused with adjacent apply over the same relation");
    rewriter.eraseOp(second);
    rewriter.eraseOp(first);
    return success();
  }
};

class FuseCompatibleAppliesPass
    : public impl::GFFuseCompatibleAppliesBase<FuseCompatibleAppliesPass> {
public:
  using impl::GFFuseCompatibleAppliesBase<
      FuseCompatibleAppliesPass>::GFFuseCompatibleAppliesBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseCompatibleApplyPattern>(&getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), frozen)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tiga
