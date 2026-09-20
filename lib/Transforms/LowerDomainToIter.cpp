#include "tiga/Transforms/Passes.h"

#include "tiga/Dialect/Domain/DomainDialect.h"
#include "tiga/Dialect/Iter/IterDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tiga {

#define GEN_PASS_DEF_GFLOWERDOMAINTOITER
#include "tiga/Transforms/Passes.h.inc"

namespace {

static void cloneDomainRegion(Region &source, Region &target,
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
  auto sourceYield = cast<YieldOp>(source.front().getTerminator());
  OperationState yieldState(sourceYield.getLoc(),
                            iter::YieldOp::getOperationName());
  for (Value value : sourceYield.getMessages())
    yieldState.addOperands(mapping.lookup(value));
  rewriter.create(yieldState);
}

class LowerDomainToIterPass
    : public impl::GFLowerDomainToIterBase<LowerDomainToIterPass> {
public:
  using impl::GFLowerDomainToIterBase<
      LowerDomainToIterPass>::GFLowerDomainToIterBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<iter::TigaIterDialect>();
  }

  void runOnOperation() final {
    SmallVector<ApplyOp> applies;
    getOperation()->walk([&](ApplyOp apply) { applies.push_back(apply); });
    IRRewriter rewriter(&getContext());
    for (ApplyOp apply : applies) {
      auto relation = apply.getRelation().getDefiningOp<RelationOp>();
      auto generated =
          apply.getRelation().getDefiningOp<GeneratedRadiusOp>();
      auto cartesian = apply.getRelation().getDefiningOp<CartesianOp>();
      auto ranked = apply.getRelation().getDefiningOp<RankedRelationOp>();
      if (!relation && !generated && !cartesian && !ranked) {
        apply.emitError("requires a visible Tiga relation definition");
        signalPassFailure();
        return;
      }
      StringRef hierarchy;
      if (cartesian)
        hierarchy = "cartesian-product";
      else if (ranked)
        hierarchy = "ranked-pairs";
      else if (generated)
        hierarchy = "generated-neighborhood";
      else if (relation.getRealization() == "materialized")
        hierarchy = "compressed-row";
      else if (relation.getRealization() == "generated")
        hierarchy = "generated-neighborhood";
      else {
        apply.emitRemark("paged relation remains in domain IR");
        continue;
      }

      rewriter.setInsertionPoint(apply);
      OperationState state(apply.getLoc(), iter::TraverseOp::getOperationName());
      state.addOperands(apply.getRelation());
      state.addOperands(apply.getInputs());
      state.addTypes(apply.getResultTypes());
      for (StringRef name : {"reducers", "region_kinds",
                             "input_segment_sizes", "snapshot_versions",
                             "effects", "deterministic"})
        state.addAttribute(name, apply->getAttr(name));
      for (StringRef name : {"input_roles", "input_names",
                             "node_input_indices",
                             "node_input_segment_sizes", "iteration_lanes"})
        if (Attribute attribute = apply->getAttr(name))
          state.addAttribute(name, attribute);
      state.addAttribute("coordinate_hierarchy",
                         rewriter.getStringAttr(hierarchy));
      state.addAttribute("ordering",
                         rewriter.getStringAttr("destination-major"));
      if (cartesian)
        if (Attribute boundary = cartesian->getAttr("boundary"))
          state.addAttribute("boundary", boundary);
      for (Region &source : apply.getRegions()) {
        (void)source;
        state.addRegion();
      }
      auto traverse = cast<iter::TraverseOp>(rewriter.create(state));
      for (auto [source, target] :
           llvm::zip(apply.getRegions(), traverse.getRegions()))
        cloneDomainRegion(source, target, rewriter);
      rewriter.replaceOp(apply, traverse.getResults());
    }
  }
};

} // namespace
} // namespace mlir::tiga
