#include "graphforge/Transforms/Passes.h"

#include "graphforge/Dialect/Domain/DomainDialect.h"
#include "graphforge/Dialect/Kernel/KernelDialect.h"
#include "graphforge/Dialect/Storage/StorageDialect.h"
#include "graphforge/Dialect/Task/TaskDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

#include <climits>

namespace mlir::graphforge {

#define GEN_PASS_DEF_GFPLANDISTRIBUTEDTASKS
#include "graphforge/Transforms/Passes.h.inc"

namespace {
namespace gfk = mlir::graphforge::kernel;
namespace gfs = mlir::graphforge::storage;
namespace gft = mlir::graphforge::task;

class PlanDistributedTasksPass
    : public impl::GFPlanDistributedTasksBase<PlanDistributedTasksPass> {
public:
  using impl::GFPlanDistributedTasksBase<
      PlanDistributedTasksPass>::GFPlanDistributedTasksBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gfs::GraphForgeStorageDialect,
                    gft::GraphForgeTaskDialect>();
  }

  void runOnOperation() final {
    SmallVector<gfk::LaunchOp> launches;
    getOperation()->walk([&](gfk::LaunchOp launch) {
      if (!launch->hasAttr("gf_task.planned")) launches.push_back(launch);
    });
    IRRewriter builder(&getContext());
    for (gfk::LaunchOp launch : launches) {
      auto relation = launch.getRowPtr().getDefiningOp<RelationOp>();
      if (!relation) {
        // The row pointer is normally a function argument. Find the relation
        // that binds the same pair and remains as schema metadata.
        launch->getParentOfType<func::FuncOp>().walk([&](RelationOp candidate) {
          if (candidate.getRowPtr() == launch.getRowPtr() &&
              candidate.getColIdx() == launch.getColIdx())
            relation = candidate;
        });
      }
      if (!relation || !relation.getMeshShapeAttr()) continue;
      ArrayAttr roles = launch.getInputRolesAttr();
      ArrayAttr names = launch.getInputNamesAttr();
      if (!roles || !names || roles.size() != launch.getInputs().size() ||
          names.size() != roles.size()) {
        launch.emitError(
            "distributed planning requires input_roles and input_names");
        signalPassFailure();
        return;
      }
      SmallVector<unsigned> remoteInputs;
      SmallVector<Attribute> haloFields;
      SmallVector<int64_t> fieldItemBytes;
      for (unsigned index = 0; index < roles.size(); ++index) {
        if (cast<StringAttr>(roles[index]).getValue() != "src") continue;
        remoteInputs.push_back(index);
        haloFields.push_back(names[index]);
        Type fieldType = launch.getInputs()[index].getType();
        int64_t bytes = 0;
        if (auto ranked = dyn_cast<RankedTensorType>(fieldType)) {
          Type element = ranked.getElementType();
          if (auto floating = dyn_cast<FloatType>(element))
            bytes = (floating.getWidth() + 7) / 8;
          else if (auto integer = dyn_cast<IntegerType>(element))
            bytes = (integer.getWidth() + 7) / 8;
          else if (auto complex = dyn_cast<ComplexType>(element))
            if (auto component = dyn_cast<FloatType>(complex.getElementType()))
              bytes = 2 * ((component.getWidth() + 7) / 8);
          for (int64_t extent : ranked.getShape().drop_front())
            if (!ShapedType::isDynamic(extent))
              bytes *= extent;
        }
        fieldItemBytes.push_back(bytes);
      }
      if (remoteInputs.empty()) continue;

      Location location = launch.getLoc();
      builder.setInsertionPoint(launch);
      ArrayRef<int64_t> mesh = *relation.getMeshShape();
      int64_t axis = relation.getMeshAxisAttr().getInt();
      int64_t owned = llvm::divideCeil(
          relation.getNumDstAttr().getInt(), mesh[axis]);

      OperationState partitionState(location, gft::PartitionOp::getOperationName());
      partitionState.addOperands(relation.getResult());
      partitionState.addTypes(gft::PartitionType::get(&getContext()));
      partitionState.addAttribute("scheme", builder.getStringAttr("destination"));
      partitionState.addAttribute("mesh_shape", relation.getMeshShapeAttr());
      partitionState.addAttribute("mesh_axis", relation.getMeshAxisAttr());
      partitionState.addAttribute("version", relation.getVersionAttr());
      partitionState.addAttribute("owned_entities", builder.getI64IntegerAttr(owned));
      partitionState.addAttribute("ghost_entities", builder.getI64IntegerAttr(-1));
      Value partition = builder.create(partitionState)->getResult(0);

      // The concrete owner/ghost map is derived from this immutable CSR
      // snapshot at runtime. Record an exact executable algorithm rather than
      // leaving ghost semantics implicit in a provider:
      //   owner(dst) = upper_bound(destination_offsets, dst) - 1
      //   ghost_ids  = unique(src of owned rows where owner(src) != rank)
      //   send_ids[p]= owned ids requested by peer p (all-to-all counts/ids)
      // This metadata is stable across transport plugins and is also used to
      // refine the conservative byte bound below after rank binding.
      partition.getDefiningOp()->setAttr(
          "owner_map_algorithm",
          builder.getStringAttr("contiguous-destination-prefix"));
      partition.getDefiningOp()->setAttr(
          "ghost_map_algorithm",
          builder.getStringAttr("csr-owned-row-remote-src-sort-unique"));
      partition.getDefiningOp()->setAttr(
          "neighbor_discovery",
          builder.getStringAttr("alltoall-counts-then-request-ids"));

      OperationState haloState(location, gft::HaloOp::getOperationName());
      haloState.addOperands(partition);
      haloState.addTypes(gft::HaloPlanType::get(&getContext()));
      haloState.addAttribute("depth", relation.getHaloDepthAttr());
      haloState.addAttribute("fields", builder.getArrayAttr(haloFields));
      haloState.addAttribute("strategy", builder.getStringAttr("auto"));
      Value halo = builder.create(haloState)->getResult(0);

      SmallVector<Value> unpackEvents;
      for (auto [remoteOrdinal, inputIndex] : llvm::enumerate(remoteInputs)) {
        Value field = launch.getInputs()[inputIndex];
        StringRef name = cast<StringAttr>(names[inputIndex]).getValue();
        int64_t elements = relation.getNumSrcAttr().getInt();
        int64_t world = mesh[axis];
        int64_t base = relation.getNumDstAttr().getInt() / world;
        int64_t remainder = relation.getNumDstAttr().getInt() % world;
        // Exact ghost cardinality depends on the concrete rank and owner map.
        // A deterministic worst-rank bound is executable and never silently
        // under-allocates; runtime refinement may shrink it after rank binding.
        int64_t maximumOwned = base + (remainder ? 1 : 0);
        int64_t maximumGhost = std::max<int64_t>(
            0, relation.getNumSrcAttr().getInt() - maximumOwned);
        int64_t fieldBytes = fieldItemBytes[remoteOrdinal];
        int64_t haloBytes = (
            fieldBytes > 0 && maximumGhost <= INT64_MAX / fieldBytes
                ? maximumGhost * fieldBytes
                : 0);
        OperationState regionState(location, gfs::RegionOp::getOperationName());
        regionState.addOperands(field);
        regionState.addTypes(gfs::RegionType::get(&getContext()));
        regionState.addAttribute("logical_id", builder.getStringAttr(name));
        regionState.addAttribute("version", relation.getVersionAttr());
        regionState.addAttribute("elements", builder.getI64IntegerAttr(elements));
        Value region = builder.create(regionState)->getResult(0);
        auto makeInstance = [&](StringRef memory, StringRef device) {
          OperationState state(location, gfs::InstanceOp::getOperationName());
          state.addOperands(region);
          state.addTypes(gfs::InstanceType::get(&getContext()));
          state.addAttribute("memory_space", builder.getStringAttr(memory));
          state.addAttribute("layout", builder.getStringAttr("partitioned"));
          state.addAttribute("device", builder.getStringAttr(device));
          state.addAttribute(
              "capacity_bytes",
              builder.getI64IntegerAttr(
                  device == "ghost" ? haloBytes : maximumOwned * fieldBytes));
          state.addAttribute("external", builder.getBoolAttr(true));
          return builder.create(state)->getResult(0);
        };
        Value ownedInstance = makeInstance("device", "local");
        Value ghostInstance = makeInstance("device", "ghost");

        OperationState packState(location, gft::HaloPackOp::getOperationName());
        packState.addOperands({halo, ownedInstance});
        packState.addTypes({gft::HaloBufferType::get(&getContext()),
                            gfs::EventType::get(&getContext())});
        packState.addAttribute("field", builder.getStringAttr(name));
        packState.addAttribute("bytes", builder.getI64IntegerAttr(haloBytes));
        packState.addAttribute("snapshot_version", relation.getVersionAttr());
        Operation *pack = builder.create(packState);

        OperationState exchangeState(location,
                                     gft::HaloExchangeOp::getOperationName());
        exchangeState.addOperands({halo, pack->getResult(0), pack->getResult(1)});
        exchangeState.addTypes({gft::HaloBufferType::get(&getContext()),
                                gfs::EventType::get(&getContext())});
        exchangeState.addAttribute("group", builder.getStringAttr("mesh"));
        exchangeState.addAttribute("bytes", builder.getI64IntegerAttr(haloBytes));
        exchangeState.addAttribute("snapshot_version", relation.getVersionAttr());
        Operation *exchange = builder.create(exchangeState);

        OperationState unpackState(location,
                                   gft::HaloUnpackOp::getOperationName());
        unpackState.addOperands(
            {halo, exchange->getResult(0), ghostInstance, exchange->getResult(1)});
        unpackState.addTypes(gfs::EventType::get(&getContext()));
        unpackState.addAttribute("field", builder.getStringAttr(name));
        unpackState.addAttribute("snapshot_version", relation.getVersionAttr());
        unpackEvents.push_back(builder.create(unpackState)->getResult(0));
      }

      auto makeComputeTask = [&](StringRef kind, ValueRange dependencies) {
        OperationState state(location, gft::LaunchOp::getOperationName());
        state.addOperands(partition);
        state.addOperands(dependencies);
        state.addTypes(gfs::EventType::get(&getContext()));
        state.addAttribute("operandSegmentSizes",
                           builder.getDenseI32ArrayAttr(
                               {1, static_cast<int32_t>(dependencies.size())}));
        state.addAttribute("task_kind", builder.getStringAttr(kind));
        auto function = launch->getParentOfType<func::FuncOp>();
        state.addAttribute("callee",
                           FlatSymbolRefAttr::get(&getContext(),
                                                  function.getSymName()));
        state.addAttribute("reads", builder.getArrayAttr(haloFields));
        state.addAttribute("writes",
                           builder.getArrayAttr({builder.getStringAttr("out")}));
        state.addAttribute("snapshot_version", relation.getVersionAttr());
        return builder.create(state)->getResult(0);
      };
      Value interior = makeComputeTask("interior", {});
      Value boundary = makeComputeTask("boundary", unpackEvents);
      OperationState joinState(location, gfs::JoinOp::getOperationName());
      joinState.addOperands({interior, boundary});
      joinState.addTypes(gfs::EventType::get(&getContext()));
      joinState.addAttribute("snapshot_version", relation.getVersionAttr());
      builder.create(joinState);
      launch->setAttr("gf_task.planned", builder.getUnitAttr());
    }
  }
};

} // namespace
} // namespace mlir::graphforge
