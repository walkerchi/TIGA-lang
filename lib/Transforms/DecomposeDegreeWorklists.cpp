#include "tiga/Transforms/Passes.h"

#include "tiga/Dialect/Task/TaskDialect.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tiga {

#define GEN_PASS_DEF_GFDECOMPOSEDEGREEWORKLISTS
#include "tiga/Transforms/Passes.h.inc"

namespace {
namespace gfs = mlir::tiga::storage;
namespace gft = mlir::tiga::task;

class DecomposeDegreeWorklistsPass
    : public impl::GFDecomposeDegreeWorklistsBase<
          DecomposeDegreeWorklistsPass> {
public:
  using impl::GFDecomposeDegreeWorklistsBase<
      DecomposeDegreeWorklistsPass>::GFDecomposeDegreeWorklistsBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gft::TigaTaskDialect>();
  }

  void runOnOperation() final {
    SmallVector<gft::DegreeWorklistOp> worklists;
    getOperation()->walk(
        [&](gft::DegreeWorklistOp op) { worklists.push_back(op); });
    IRRewriter builder(&getContext());
    for (gft::DegreeWorklistOp worklist : worklists) {
      Location location = worklist.getLoc();
      builder.setInsertionPoint(worklist);

      OperationState resetState(location,
                                gft::DegreeResetOp::getOperationName());
      resetState.addOperands(worklist.getStorage());
      resetState.addTypes(gfs::EventType::get(&getContext()));
      resetState.addAttribute(
          "buckets", builder.getI64IntegerAttr(worklist.getUpperBounds().size()));
      resetState.addAttribute("snapshot_version",
                              worklist.getSnapshotVersionAttr());
      Value reset = builder.create(resetState)->getResult(0);

      OperationState histogramState(
          location, gft::DegreeHistogramOp::getOperationName());
      histogramState.addOperands(
          {worklist.getRelation(), worklist.getStorage(), reset});
      histogramState.addTypes(gfs::EventType::get(&getContext()));
      histogramState.addAttribute("upper_bounds", worklist.getUpperBoundsAttr());
      histogramState.addAttribute("rows", worklist.getRowsAttr());
      histogramState.addAttribute("snapshot_version",
                                  worklist.getSnapshotVersionAttr());
      Value histogram = builder.create(histogramState)->getResult(0);

      OperationState prefixState(location,
                                 gft::DegreePrefixOp::getOperationName());
      prefixState.addOperands({worklist.getStorage(), histogram});
      prefixState.addTypes(gfs::EventType::get(&getContext()));
      prefixState.addAttribute(
          "buckets", builder.getI64IntegerAttr(worklist.getUpperBounds().size()));
      prefixState.addAttribute("snapshot_version",
                               worklist.getSnapshotVersionAttr());
      Value prefix = builder.create(prefixState)->getResult(0);

      OperationState scatterState(location,
                                  gft::DegreeScatterOp::getOperationName());
      scatterState.addOperands(
          {worklist.getRelation(), worklist.getStorage(), prefix});
      scatterState.addTypes({gft::DegreeWorklistType::get(&getContext()),
                             gfs::EventType::get(&getContext())});
      scatterState.addAttribute("upper_bounds", worklist.getUpperBoundsAttr());
      scatterState.addAttribute("rows", worklist.getRowsAttr());
      scatterState.addAttribute("snapshot_version",
                                worklist.getSnapshotVersionAttr());
      Operation *scatter = builder.create(scatterState);
      worklist.getResult().replaceAllUsesWith(scatter->getResult(0));
      worklist.getReady().replaceAllUsesWith(scatter->getResult(1));
      builder.eraseOp(worklist);
    }
  }
};

} // namespace
} // namespace mlir::tiga
