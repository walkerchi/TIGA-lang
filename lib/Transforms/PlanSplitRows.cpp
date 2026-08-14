#include "graphforge/Transforms/Passes.h"

#include "graphforge/Dialect/Domain/DomainDialect.h"
#include "graphforge/Dialect/Kernel/KernelDialect.h"
#include "graphforge/Dialect/Storage/StorageDialect.h"
#include "graphforge/Dialect/Task/TaskDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::graphforge {

#define GEN_PASS_DEF_GFPLANSPLITROWS
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

static ReducerOp findScalarF32Reducer(gfk::LaunchOp launch) {
  if (launch.getDeterministic() ||
      (launch.getNumRegions() != 1 && launch.getNumRegions() != 2))
    return {};
  if (launch.getReducers().size() != 1)
    return {};
  auto reference = dyn_cast<FlatSymbolRefAttr>(launch.getReducers()[0]);
  auto reducer = reference
                     ? SymbolTable::lookupNearestSymbolFrom<ReducerOp>(
                           launch, reference)
                     : ReducerOp();
  if (!reducer || !reducer.getAssociative() ||
      reducer.getStateTypes().size() != 1)
    return {};
  auto state = dyn_cast<TypeAttr>(reducer.getStateTypes()[0]);
  if (!state || !state.getValue().isF32())
    return {};
  Block &identity = reducer.getIdentity().front();
  Block &combine = reducer.getCombine().front();
  auto identityYield = dyn_cast<ReducerYieldOp>(identity.getTerminator());
  auto combineYield = dyn_cast<ReducerYieldOp>(combine.getTerminator());
  auto zero = identityYield && identityYield.getValues().size() == 1
                  ? identityYield.getValues()[0]
                        .getDefiningOp<arith::ConstantOp>()
                  : arith::ConstantOp();
  auto zeroValue = zero ? dyn_cast<FloatAttr>(zero.getValue()) : FloatAttr();
  auto addition = combineYield && combineYield.getValues().size() == 1
                      ? combineYield.getValues()[0]
                            .getDefiningOp<arith::AddFOp>()
                      : arith::AddFOp();
  if (!zeroValue || !zeroValue.getValue().isZero() || !addition ||
      combine.getNumArguments() != 2 ||
      !((addition.getLhs() == combine.getArgument(0) &&
         addition.getRhs() == combine.getArgument(1)) ||
        (addition.getLhs() == combine.getArgument(1) &&
         addition.getRhs() == combine.getArgument(0))))
    return {};
  return reducer;
}

class PlanSplitRowsPass
    : public impl::GFPlanSplitRowsBase<PlanSplitRowsPass> {
public:
  using impl::GFPlanSplitRowsBase<PlanSplitRowsPass>::GFPlanSplitRowsBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gfs::GraphForgeStorageDialect,
                    gft::GraphForgeTaskDialect>();
  }

  void runOnOperation() final {
    SmallVector<gfk::LaunchOp> launches;
    getOperation()->walk([&](gfk::LaunchOp launch) {
      auto plan = launch->getAttrOfType<StringAttr>("load_balance_plan");
      if (plan && plan.getValue() == "high-degree-tail-split" &&
          !launch->hasAttr("gf_task.split_row_planned"))
        launches.push_back(launch);
    });

    IRRewriter builder(&getContext());
    for (gfk::LaunchOp launch : launches) {
      RelationOp relation = findRelation(launch);
      ReducerOp reducer = findScalarF32Reducer(launch);
      IntegerAttr maximum = launch.getDegreeMaxAttr();
      if (!relation || !maximum || maximum.getInt() <= 64 || !reducer) {
        launch->setAttr("gf_task.split_row_rejected", builder.getUnitAttr());
        continue;
      }

      constexpr int64_t chunkEdges = 64;
      int64_t rows = launch.getNumRowsAttr().getInt();
      int64_t partials = llvm::divideCeil(maximum.getInt(), chunkEdges);
      auto function = launch->getParentOfType<func::FuncOp>();
      gft::DegreeBucketLaunchOp tail;
      function.walk([&](gft::DegreeBucketLaunchOp candidate) {
        if (!tail && candidate.getCalleeAttr().getValue() ==
                         function.getSymName() &&
            candidate.getDegreeLowerExclusiveAttr().getInt() == chunkEdges &&
            candidate.getDegreeUpperInclusiveAttr().getInt() ==
                maximum.getInt() &&
            candidate.getRowMapping() == "worklist")
          tail = candidate;
      });
      if (!tail || tail.getDependsOn().size() != 1) {
        launch->setAttr("gf_task.split_row_rejected", builder.getUnitAttr());
        continue;
      }
      int64_t activeRows = tail.getRowsInBucketAttr().getInt();
      auto logicalWorklist =
          tail.getWorklist().getDefiningOp<gft::DegreeWorklistOp>();
      if (!logicalWorklist || activeRows <= 0) {
        launch->setAttr("gf_task.split_row_rejected", builder.getUnitAttr());
        continue;
      }
      int64_t bucketOrdinal = tail.getBucketOrdinalAttr().getInt();
      int64_t bucketCount = logicalWorklist.getUpperBounds().size();
      // A single algebraic reducer with no node epilogue can keep its state in
      // registers while walking 64-edge tiles.  This preserves the compact
      // worklist mapping and removes both the global partial buffer and the
      // finalize launch.  Node epilogues retain the general two-stage plan so
      // finalize still runs exactly once per destination.
      if (launch.getNumRegions() == 1) {
        // A single associative reducer can always keep one row's state in
        // registers across neighbor chunks. Source locality affects which
        // warm executable wins autotuning, but it does not affect legality;
        // forcing local graphs through an otherwise unnecessary partial
        // buffer only removed the executable candidate from auto selection.
        tail.setRowMapping("worklist-chunked");
        launch->setAttr("load_balance_plan",
                        builder.getStringAttr("high-degree-tail-chunked"));
        launch->setAttr("gf_task.split_row_planned", builder.getUnitAttr());
        continue;
      }
      if (rows <= 0 || partials <= 1 ||
          activeRows > INT64_MAX / partials ||
          activeRows * partials > INT64_MAX / 4) {
        launch.emitError("split-row partial-state size overflows i64");
        signalPassFailure();
        return;
      }
      int64_t elements = activeRows * partials;
      int64_t version = relation.getVersionAttr().getInt();
      Location location = launch.getLoc();
      builder.setInsertionPoint(tail);

      OperationState regionState(location, gfs::RegionOp::getOperationName());
      regionState.addOperands(relation.getResult());
      regionState.addTypes(gfs::RegionType::get(&getContext()));
      regionState.addAttribute(
          "logical_id", builder.getStringAttr(
                            relation.getRelationId().str() + ".row-partials"));
      regionState.addAttribute("version", builder.getI64IntegerAttr(version));
      regionState.addAttribute("elements", builder.getI64IntegerAttr(elements));
      Value region = builder.create(regionState)->getResult(0);

      OperationState instanceState(location,
                                   gfs::InstanceOp::getOperationName());
      instanceState.addOperands(region);
      instanceState.addTypes(gfs::InstanceType::get(&getContext()));
      instanceState.addAttribute("memory_space", builder.getStringAttr("device"));
      instanceState.addAttribute("layout",
                                 builder.getStringAttr("row-partial-f32"));
      instanceState.addAttribute("device", builder.getStringAttr("provider:0"));
      instanceState.addAttribute("capacity_bytes",
                                 builder.getI64IntegerAttr(elements * 4));
      instanceState.addAttribute("external", builder.getBoolAttr(false));
      Value instance = builder.create(instanceState)->getResult(0);

      auto binding = [&](StringRef parameter, StringRef resource) {
        return builder.getDictionaryAttr({
            builder.getNamedAttr("parameter", builder.getStringAttr(parameter)),
            builder.getNamedAttr("resource", builder.getStringAttr(resource))});
      };
      SmallVector<Attribute> partialArguments{
          binding("row_ptr", "row_ptr"), binding("col_idx", "col_idx")};
      SmallVector<Attribute> partialReads{builder.getStringAttr("row_ptr"),
                                          builder.getStringAttr("col_idx")};
      SmallVector<Attribute> finalizeArguments;
      SmallVector<Attribute> finalizeReads;
      ArrayAttr roles = launch.getInputRolesAttr();
      ArrayAttr names = launch.getInputNamesAttr();
      DenseI64ArrayAttr segments = launch.getInputSegmentSizesAttr();
      int64_t edgeInputs = segments && !segments.empty()
                               ? segments.asArrayRef().front()
                               : static_cast<int64_t>(launch.getInputs().size());
      for (auto [index, input] : llvm::enumerate(launch.getInputs())) {
        (void)input;
        std::string parameter = "input" + std::to_string(index);
        std::string resource = parameter;
        if (roles && names && roles.size() == launch.getInputs().size() &&
            names.size() == roles.size())
          resource = cast<StringAttr>(roles[index]).getValue().str() + ":" +
                     cast<StringAttr>(names[index]).getValue().str();
        if (static_cast<int64_t>(index) < edgeInputs) {
          partialArguments.push_back(binding(parameter, resource));
          partialReads.push_back(builder.getStringAttr(resource));
        }
      }
      // Only explicit node inputs participate in the second-stage ABI. Edge
      // fields are not re-read merely because they occur in the parent launch.
      if (DenseI64ArrayAttr nodeIndices = launch.getNodeInputIndicesAttr())
        for (int64_t index : nodeIndices.asArrayRef()) {
          std::string parameter = "input" + std::to_string(index);
          std::string resource = parameter;
          if (roles && names && index >= 0 &&
              index < static_cast<int64_t>(roles.size()))
            resource = cast<StringAttr>(roles[index]).getValue().str() + ":" +
                       cast<StringAttr>(names[index]).getValue().str();
          finalizeArguments.push_back(binding(parameter, resource));
          finalizeReads.push_back(builder.getStringAttr(resource));
        }
      partialArguments.push_back(binding("partials", "row_partials"));
      partialArguments.push_back(binding("row_worklist", "row_worklist"));
      partialReads.push_back(builder.getStringAttr("row_worklist"));
      finalizeArguments.push_back(binding("partials", "row_partials"));
      finalizeArguments.push_back(binding("row_worklist", "row_worklist"));
      finalizeArguments.push_back(binding("output", "output"));
      finalizeReads.push_back(builder.getStringAttr("row_worklist"));

      builder.setInsertionPoint(tail);
      OperationState partialState(location,
                                  gft::RowSplitPartialOp::getOperationName());
      partialState.addOperands({relation.getResult(), tail.getWorklist(),
                                instance, tail.getDependsOn().front()});
      partialState.addTypes(gfs::EventType::get(&getContext()));
      partialState.addAttribute(
          "callee", FlatSymbolRefAttr::get(&getContext(), function.getSymName()));
      partialState.addAttribute("chunk_edges",
                                builder.getI64IntegerAttr(chunkEdges));
      partialState.addAttribute("partials_per_row",
                                builder.getI64IntegerAttr(partials));
      partialState.addAttribute("rows", builder.getI64IntegerAttr(rows));
      partialState.addAttribute("active_rows",
                                builder.getI64IntegerAttr(activeRows));
      partialState.addAttribute("bucket_ordinal",
                                builder.getI64IntegerAttr(bucketOrdinal));
      partialState.addAttribute("buckets",
                                builder.getI64IntegerAttr(bucketCount));
      partialState.addAttribute("state_bytes", builder.getI64IntegerAttr(4));
      partialState.addAttribute("arguments",
                                builder.getArrayAttr(partialArguments));
      partialState.addAttribute("reads", builder.getArrayAttr(partialReads));
      partialState.addAttribute(
          "writes", builder.getArrayAttr({builder.getStringAttr("row_partials")}));
      partialState.addAttribute("snapshot_version",
                                builder.getI64IntegerAttr(version));
      Value partialDone = builder.create(partialState)->getResult(0);

      finalizeReads.push_back(builder.getStringAttr("row_partials"));
      OperationState finalizeState(
          location, gft::RowSplitFinalizeOp::getOperationName());
      finalizeState.addOperands(
          {tail.getWorklist(), instance, partialDone});
      finalizeState.addTypes(gfs::EventType::get(&getContext()));
      finalizeState.addAttribute(
          "callee", FlatSymbolRefAttr::get(&getContext(), function.getSymName()));
      finalizeState.addAttribute("partials_per_row",
                                 builder.getI64IntegerAttr(partials));
      finalizeState.addAttribute("rows", builder.getI64IntegerAttr(rows));
      finalizeState.addAttribute("active_rows",
                                 builder.getI64IntegerAttr(activeRows));
      finalizeState.addAttribute("bucket_ordinal",
                                 builder.getI64IntegerAttr(bucketOrdinal));
      finalizeState.addAttribute("buckets",
                                 builder.getI64IntegerAttr(bucketCount));
      finalizeState.addAttribute("state_bytes", builder.getI64IntegerAttr(4));
      finalizeState.addAttribute("arguments",
                                 builder.getArrayAttr(finalizeArguments));
      finalizeState.addAttribute("reads", builder.getArrayAttr(finalizeReads));
      finalizeState.addAttribute(
          "writes", builder.getArrayAttr({builder.getStringAttr("output")}));
      finalizeState.addAttribute("partitioning",
                                 builder.getStringAttr("degree-worklist"));
      finalizeState.addAttribute("output_partition",
                                 tail.getOutputPartitionAttr());
      finalizeState.addAttribute("snapshot_version",
                                 builder.getI64IntegerAttr(version));
      Value finalizeDone = builder.create(finalizeState)->getResult(0);

      tail.getDone().replaceAllUsesWith(finalizeDone);
      builder.eraseOp(tail);

      builder.setInsertionPointAfter(finalizeDone.getDefiningOp());
      OperationState releaseState(location, gfs::ReleaseOp::getOperationName());
      releaseState.addOperands({instance, finalizeDone});
      releaseState.addTypes(gfs::EventType::get(&getContext()));
      releaseState.addAttribute("snapshot_version",
                                builder.getI64IntegerAttr(version));
      builder.create(releaseState);
      launch->setAttr("gf_task.split_row_planned", builder.getUnitAttr());
    }
  }
};

} // namespace
} // namespace mlir::graphforge
