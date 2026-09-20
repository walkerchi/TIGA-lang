#include "tiga/Transforms/Passes.h"

#include "tiga/Dialect/Domain/DomainDialect.h"
#include "tiga/Dialect/Iter/IterDialect.h"
#include "tiga/Dialect/Kernel/KernelDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tiga {

#define GEN_PASS_DEF_GFLOWERITERTOKERNEL
#include "tiga/Transforms/Passes.h.inc"

namespace {

static void cloneIterationRegion(Region &source, Region &target,
                                 IRRewriter &rewriter) {
  Block *targetBlock = new Block();
  target.push_back(targetBlock);
  IRMapping mapping;
  for (BlockArgument argument : source.front().getArguments()) {
    BlockArgument clone = targetBlock->addArgument(argument.getType(),
                                                    argument.getLoc());
    mapping.map(argument, clone);
  }
  rewriter.setInsertionPointToEnd(targetBlock);
  for (Operation &operation : source.front().without_terminator())
    rewriter.clone(operation, mapping);
  auto sourceYield = cast<iter::YieldOp>(source.front().getTerminator());
  OperationState yieldState(sourceYield.getLoc(),
                            kernel::YieldOp::getOperationName());
  for (Value value : sourceYield.getValues())
    yieldState.addOperands(mapping.lookup(value));
  rewriter.create(yieldState);
}

class LowerIterToKernelPass
    : public impl::GFLowerIterToKernelBase<LowerIterToKernelPass> {
public:
  using impl::GFLowerIterToKernelBase<
      LowerIterToKernelPass>::GFLowerIterToKernelBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<kernel::TigaKernelDialect>();
  }

  void runOnOperation() final {
    SmallVector<iter::TraverseOp> traversals;
    getOperation()->walk(
        [&](iter::TraverseOp traversal) { traversals.push_back(traversal); });
    IRRewriter rewriter(&getContext());
    for (iter::TraverseOp traversal : traversals) {
      auto relation =
          traversal.getRelation().getDefiningOp<RelationOp>();
      auto generated =
          traversal.getRelation().getDefiningOp<GeneratedRadiusOp>();
      auto cartesian = traversal.getRelation().getDefiningOp<CartesianOp>();
      auto ranked =
          traversal.getRelation().getDefiningOp<RankedRelationOp>();
      if (!relation && !generated && !cartesian && !ranked) {
        traversal.emitError(
            "requires a Tiga relation definition before physical lowering");
        signalPassFailure();
        return;
      }
      StringRef hierarchy = traversal->getAttrOfType<StringAttr>(
          "coordinate_hierarchy").getValue();
      StringRef skeleton;
      if (hierarchy == "compressed-row")
        skeleton = "csr-row";
      else if (hierarchy == "generated-neighborhood")
        skeleton = "generated-tile";
      else if (hierarchy == "cartesian-product")
        skeleton = "dense-tile";
      else if (hierarchy == "ranked-pairs")
        skeleton = "ranked-candidate-tile";
      else {
        traversal.emitError("has no legal kernel skeleton");
        signalPassFailure();
        return;
      }

      rewriter.setInsertionPoint(traversal);
      OperationState state(
          traversal.getLoc(),
          generated ? kernel::GeneratedLaunchOp::getOperationName()
          : ranked ? kernel::RankedLaunchOp::getOperationName()
          : cartesian ? kernel::DenseLaunchOp::getOperationName()
                      : kernel::LaunchOp::getOperationName());
      if (generated) {
        state.addOperands({
            generated.getPositions(), generated.getCellPtr(),
            generated.getParticleOrder(), generated.getCellCoordinates(),
            generated.getExtents(), generated.getStrides(),
            generated.getNeighborOffsets(), generated.getLattice(),
            generated.getInverseLattice()});
      } else if (ranked) {
        state.addOperands(
            {ranked.getQueryPositions(), ranked.getCandidatePositions()});
      } else if (relation) {
        state.addOperands({relation.getRowPtr(), relation.getColIdx()});
      }
      state.addOperands(traversal.getInputs());
      state.addTypes(traversal.getResultTypes());
      for (StringRef name : {"reducers", "region_kinds",
                             "input_segment_sizes", "snapshot_versions",
                             "effects", "deterministic"})
        state.addAttribute(name, traversal->getAttr(name));
      for (StringRef name : {"input_roles", "input_names",
                             "node_input_indices",
                             "node_input_segment_sizes", "iteration_lanes"})
        if (Attribute attribute = traversal->getAttr(name))
          state.addAttribute(name, attribute);
      if (generated) {
        state.addAttribute("num_rows", generated.getNumEntitiesAttr());
        state.addAttribute("cutoff", generated.getCutoffAttr());
        state.addAttribute("dimensions", generated.getDimensionsAttr());
        state.addAttribute("periodic", generated.getPeriodicAttr());
        state.addAttribute("hash_grid", generated.getHashGridAttr());
      } else if (ranked) {
        for (StringRef name : {"num_queries", "num_candidates", "dimensions",
                               "k", "metric", "selection", "tie_break",
                               "exclude_self", "same_entity_domain", "exact"})
          state.addAttribute(name, ranked->getAttr(name));
      } else if (cartesian) {
        state.addAttribute("num_src", cartesian.getNumSrcAttr());
        state.addAttribute("num_dst", cartesian.getNumDstAttr());
        if (Attribute boundary = traversal->getAttr("boundary"))
          state.addAttribute("boundary", boundary);
      } else {
        state.addAttribute("num_rows", relation.getNumDstAttr());
        if (Attribute degreeMin = relation.getDegreeMinAttr())
          state.addAttribute("degree_min", degreeMin);
        if (Attribute degreeMax = relation.getDegreeMaxAttr())
          state.addAttribute("degree_max", degreeMax);
        if (Attribute degreeSum = relation.getDegreeSumAttr())
          state.addAttribute("degree_sum", degreeSum);
        if (Attribute histogram = relation.getDegreeHistogramAttr())
          state.addAttribute("degree_histogram", histogram);
        if (Attribute sourceSpan = relation.getSourceIndexSpanRatioAttr())
          state.addAttribute("source_index_span_ratio", sourceSpan);
      }
      state.addAttribute("traversal", rewriter.getStringAttr(skeleton));
      for (Region &source : traversal.getRegions()) {
        (void)source;
        state.addRegion();
      }
      Operation *launch = rewriter.create(state);
      for (auto [source, target] :
           llvm::zip(traversal.getRegions(), launch->getRegions()))
        cloneIterationRegion(source, target, rewriter);
      rewriter.replaceOp(traversal, launch->getResults());
    }
  }
};

} // namespace
} // namespace mlir::tiga
