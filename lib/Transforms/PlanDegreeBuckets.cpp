#include "graphforge/Transforms/Passes.h"

#include "graphforge/Dialect/Domain/DomainDialect.h"
#include "graphforge/Dialect/Kernel/KernelDialect.h"
#include "graphforge/Dialect/Storage/StorageDialect.h"
#include "graphforge/Dialect/Task/TaskDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::graphforge {

#define GEN_PASS_DEF_GFPLANDEGREEBUCKETS
#include "graphforge/Transforms/Passes.h.inc"

namespace {
namespace gfk = mlir::graphforge::kernel;
namespace gfs = mlir::graphforge::storage;
namespace gft = mlir::graphforge::task;

static RelationOp findRelation(gfk::LaunchOp launch) {
  RelationOp relation = launch.getRowPtr().getDefiningOp<RelationOp>();
  if (relation)
    return relation;
  launch->getParentOfType<func::FuncOp>().walk([&](RelationOp candidate) {
    if (!relation && candidate.getRowPtr() == launch.getRowPtr() &&
        candidate.getColIdx() == launch.getColIdx())
      relation = candidate;
  });
  return relation;
}

static SmallVector<int64_t> makeUpperBounds(int64_t maximum) {
  SmallVector<int64_t> bounds;
  int64_t bound = 8;
  while (bound < maximum) {
    bounds.push_back(bound);
    if (bound > INT64_MAX / 2)
      break;
    bound *= 2;
  }
  if (bounds.empty() || bounds.back() != maximum)
    bounds.push_back(maximum);
  return bounds;
}

class PlanDegreeBucketsPass
    : public impl::GFPlanDegreeBucketsBase<PlanDegreeBucketsPass> {
public:
  using impl::GFPlanDegreeBucketsBase<
      PlanDegreeBucketsPass>::GFPlanDegreeBucketsBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gfs::GraphForgeStorageDialect,
                    gft::GraphForgeTaskDialect>();
  }

  void runOnOperation() final {
    SmallVector<gfk::LaunchOp> launches;
    getOperation()->walk([&](gfk::LaunchOp launch) {
      auto plan = launch->getAttrOfType<StringAttr>("load_balance_plan");
      if (plan && plan.getValue() == "degree-bucketing-required" &&
          !launch->hasAttr("gf_task.degree_planned"))
        launches.push_back(launch);
    });

    IRRewriter builder(&getContext());
    for (gfk::LaunchOp launch : launches) {
      RelationOp relation = findRelation(launch);
      IntegerAttr maximum = launch.getDegreeMaxAttr();
      if (!relation || !maximum || maximum.getInt() <= 0) {
        launch.emitError(
            "degree bucket planning requires a visible relation and degree_max");
        signalPassFailure();
        return;
      }

      Location location = launch.getLoc();
      int64_t rows = launch.getNumRowsAttr().getInt();
      int64_t version = relation.getVersionAttr().getInt();
      DenseI64ArrayAttr histogram = launch.getDegreeHistogramAttr();
      // Generated TTIR currently requires an exact population for every
      // worklist launch.  Without a histogram, rows above the bounded kernel's
      // 64-edge contract cannot be safely assigned to either a worklist or a
      // compact split buffer.  Preserve the semantic kernel for a provider
      // fallback instead of emitting a task plan that the backend must reject.
      if (!histogram && maximum.getInt() > 64) {
        launch->setAttr("load_balance_plan",
                        builder.getStringAttr("provider-deferred-high-degree"));
        launch->setAttr("gf_task.degree_rejected", builder.getUnitAttr());
        continue;
      }
      SmallVector<int64_t> bounds = makeUpperBounds(maximum.getInt());
      bool tailSplit = false;
      if (histogram && maximum.getInt() > 32) {
        // Keep the fast bounded-row kernel at <=32 for ordinary skew.  For an
        // oversized tail, derive a short-row bound from the exact CDF instead
        // of padding every non-tail row to 64 lanes.  Use the smaller tile
        // only when it covers at least half the rows; otherwise retaining
        // {64, max} avoids specializing for an unrepresentative minority.
        int64_t threshold = maximum.getInt() > 64
                                ? 64
                                : maximum.getInt() / 2;
        int64_t tailRows = 0;
        for (int64_t degree = threshold + 1;
             degree <= maximum.getInt(); ++degree)
          tailRows += histogram[degree];
        if (maximum.getInt() > 64 || tailRows * 10 <= rows) {
          int64_t shortBound = threshold;
          if (maximum.getInt() > 64) {
            int64_t cumulative = 0;
            int64_t previous = -1;
            for (int64_t candidate : {8, 16, 32}) {
              for (int64_t degree = previous + 1; degree <= candidate;
                   ++degree)
                cumulative += histogram[degree];
              previous = candidate;
              if (cumulative * 2 >= rows) {
                shortBound = candidate;
                break;
              }
            }
          }
          // A message-only associative reducer can combine every remaining
          // row in one register-resident chunked worklist.  A node epilogue
          // retains the middle <=64 bucket so split-row finalize executes
          // only for truly oversized rows.
          if (shortBound < threshold && launch.getNumRegions() == 1)
            bounds = {shortBound, maximum.getInt()};
          else if (shortBound < threshold)
            bounds = {shortBound, threshold, maximum.getInt()};
          else
            bounds = {threshold, maximum.getInt()};
          tailSplit = true;
          launch->setAttr("load_balance_plan",
                          builder.getStringAttr("high-degree-tail-split"));
        }
      }
      if (histogram && maximum.getInt() <= 32) {
        launch->setAttr("load_balance_plan",
                        builder.getStringAttr("bounded-row-preferred"));
        launch->setAttr("gf_task.degree_rejected", builder.getUnitAttr());
        continue;
      }
      int64_t bucketCount = static_cast<int64_t>(bounds.size());
      if (bucketCount > (INT64_MAX - rows - 1) / 2 ||
          rows + 2 * bucketCount + 1 > INT64_MAX / 8) {
        launch.emitError("degree worklist storage size overflows i64");
        signalPassFailure();
        return;
      }
      int64_t worklistElements = rows + 2 * bucketCount + 1;
      builder.setInsertionPoint(launch);

      OperationState regionState(location, gfs::RegionOp::getOperationName());
      regionState.addOperands(relation.getResult());
      regionState.addTypes(gfs::RegionType::get(&getContext()));
      regionState.addAttribute(
          "logical_id",
          builder.getStringAttr(relation.getRelationId().str() +
                                ".degree-row-ids"));
      regionState.addAttribute("version", builder.getI64IntegerAttr(version));
      regionState.addAttribute("elements",
                               builder.getI64IntegerAttr(worklistElements));
      Value region = builder.create(regionState)->getResult(0);

      OperationState instanceState(location,
                                   gfs::InstanceOp::getOperationName());
      instanceState.addOperands(region);
      instanceState.addTypes(gfs::InstanceType::get(&getContext()));
      instanceState.addAttribute("memory_space", builder.getStringAttr("device"));
      instanceState.addAttribute("layout",
                                 builder.getStringAttr("degree-row-worklist"));
      instanceState.addAttribute("device", builder.getStringAttr("provider:0"));
      instanceState.addAttribute("capacity_bytes",
                                 builder.getI64IntegerAttr(worklistElements * 8));
      instanceState.addAttribute("external", builder.getBoolAttr(false));
      Value instance = builder.create(instanceState)->getResult(0);

      OperationState worklistState(location,
                                   gft::DegreeWorklistOp::getOperationName());
      worklistState.addOperands({relation.getResult(), instance});
      worklistState.addTypes({gft::DegreeWorklistType::get(&getContext()),
                              gfs::EventType::get(&getContext())});
      worklistState.addAttribute("upper_bounds",
                                 builder.getDenseI64ArrayAttr(bounds));
      worklistState.addAttribute("rows", builder.getI64IntegerAttr(rows));
      worklistState.addAttribute("snapshot_version",
                                 builder.getI64IntegerAttr(version));
      Operation *worklist = builder.create(worklistState);

      SmallVector<Value> completions;
      int64_t lower = -1;
      auto function = launch->getParentOfType<func::FuncOp>();
      auto makeBinding = [&](StringRef parameter, StringRef resource) {
        return builder.getDictionaryAttr({
            builder.getNamedAttr("parameter", builder.getStringAttr(parameter)),
            builder.getNamedAttr("resource", builder.getStringAttr(resource))});
      };
      SmallVector<Attribute> argumentBindings{
          makeBinding("row_ptr", "row_ptr"),
          makeBinding("col_idx", "col_idx")};
      SmallVector<Attribute> reads{builder.getStringAttr("row_ptr"),
                                   builder.getStringAttr("col_idx")};
      ArrayAttr roles = launch.getInputRolesAttr();
      ArrayAttr names = launch.getInputNamesAttr();
      for (auto [index, input] : llvm::enumerate(launch.getInputs())) {
        (void)input;
        std::string parameter = "input" + std::to_string(index);
        std::string resource = parameter;
        if (roles && names && roles.size() == launch.getInputs().size() &&
            names.size() == roles.size())
          resource = cast<StringAttr>(roles[index]).getValue().str() + ":" +
                     cast<StringAttr>(names[index]).getValue().str();
        argumentBindings.push_back(makeBinding(parameter, resource));
        reads.push_back(builder.getStringAttr(resource));
      }
      argumentBindings.push_back(
          makeBinding("row_worklist", "row_worklist"));
      argumentBindings.push_back(makeBinding("output", "output"));
      reads.push_back(builder.getStringAttr("row_worklist"));
      for (auto [ordinal, upper] : llvm::enumerate(bounds)) {
        int64_t rowsInBucket = rows;
        if (histogram) {
          rowsInBucket = 0;
          int64_t begin = lower + 1;
          for (int64_t degree = begin; degree <= upper; ++degree)
            rowsInBucket += histogram[degree];
        }
        OperationState bucketState(
            location, gft::DegreeBucketLaunchOp::getOperationName());
        bucketState.addOperands({worklist->getResult(0), worklist->getResult(1)});
        bucketState.addTypes(gfs::EventType::get(&getContext()));
        bucketState.addAttribute(
            "callee", FlatSymbolRefAttr::get(&getContext(), function.getSymName()));
        bucketState.addAttribute("bucket_ordinal",
                                 builder.getI64IntegerAttr(ordinal));
        bucketState.addAttribute("degree_lower_exclusive",
                                 builder.getI64IntegerAttr(lower));
        bucketState.addAttribute("degree_upper_inclusive",
                                 builder.getI64IntegerAttr(upper));
        bucketState.addAttribute("rows_in_bucket",
                                 builder.getI64IntegerAttr(rowsInBucket));
        bucketState.addAttribute(
            "row_mapping",
            builder.getStringAttr(tailSplit && ordinal == 0
                                      ? "direct-filter" : "worklist"));
        bucketState.addAttribute("arguments",
                                 builder.getArrayAttr(argumentBindings));
        bucketState.addAttribute("reads", builder.getArrayAttr(reads));
        bucketState.addAttribute(
            "writes", builder.getArrayAttr({builder.getStringAttr("output")}));
        bucketState.addAttribute("partitioning",
                                 builder.getStringAttr("degree-worklist"));
        bucketState.addAttribute(
            "output_partition",
            builder.getStringAttr("bucket:" + std::to_string(ordinal)));
        bucketState.addAttribute("snapshot_version",
                                 builder.getI64IntegerAttr(version));
        completions.push_back(builder.create(bucketState)->getResult(0));
        lower = upper;
      }

      OperationState joinState(location, gfs::JoinOp::getOperationName());
      joinState.addOperands(completions);
      joinState.addTypes(gfs::EventType::get(&getContext()));
      joinState.addAttribute("snapshot_version",
                             builder.getI64IntegerAttr(version));
      builder.create(joinState);
      launch->setAttr("gf_task.degree_planned", builder.getUnitAttr());
    }
  }
};

} // namespace
} // namespace mlir::graphforge
