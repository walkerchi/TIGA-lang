#include "graphforge/Dialect/Task/TaskDialect.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/SmallSet.h"

#include <optional>

using namespace mlir;
using namespace mlir::graphforge::task;

static std::optional<int64_t> getTaskEventVersion(Value value) {
  Operation *producer = value.getDefiningOp();
  if (producer)
    if (auto version =
            producer->getAttrOfType<IntegerAttr>("snapshot_version"))
      return version.getInt();
  return std::nullopt;
}

static LogicalResult verifyDependencies(Operation *operation,
                                        ValueRange dependencies,
                                        int64_t version);

static FailureOr<int64_t> getHaloVersion(Value plan) {
  auto halo = plan.getDefiningOp<HaloOp>();
  if (!halo)
    return failure();
  auto partition = halo.getPartition().getDefiningOp<PartitionOp>();
  if (!partition)
    return failure();
  return partition.getVersionAttr().getInt();
}

static LogicalResult verifyHaloTask(Operation *operation, Value plan,
                                    ValueRange dependencies,
                                    int64_t snapshotVersion) {
  FailureOr<int64_t> haloVersion = getHaloVersion(plan);
  if (failed(haloVersion))
    return operation->emitOpError(
        "requires a halo plan derived from a visible partition");
  if (*haloVersion != snapshotVersion)
    return operation->emitOpError(
        "halo plan version does not match snapshot_version");
  return verifyDependencies(operation, dependencies, snapshotVersion);
}

static LogicalResult verifyDependencies(Operation *operation,
                                        ValueRange dependencies,
                                        int64_t version) {
  for (Value dependency : dependencies)
    if (auto dependencyVersion = getTaskEventVersion(dependency);
        dependencyVersion && *dependencyVersion != version)
      return operation->emitOpError(
          "dependency event belongs to another snapshot version");
  return success();
}

static LogicalResult verifyDegreeStorage(Operation *operation,
                                         Value relationValue,
                                         Value storageValue,
                                         ArrayRef<int64_t> upperBounds,
                                         int64_t rows, int64_t version) {
  if (version < 0 || rows < 0)
    return operation->emitOpError(
        "row count and snapshot version must be non-negative");
  if (upperBounds.empty())
    return operation->emitOpError("requires at least one degree upper bound");
  int64_t previous = -1;
  for (int64_t bound : upperBounds) {
    if (bound <= previous)
      return operation->emitOpError(
          "degree upper bounds must be strictly increasing");
    previous = bound;
  }
  auto relation = relationValue.getDefiningOp<graphforge::RelationOp>();
  if (relation) {
    if (relation.getVersionAttr().getInt() != version)
      return operation->emitOpError(
          "relation version does not match snapshot_version");
    if (relation.getNumDstAttr().getInt() >= 0 &&
        relation.getNumDstAttr().getInt() != rows)
      return operation->emitOpError(
          "row count does not match relation num_dst");
    if (IntegerAttr maximum = relation.getDegreeMaxAttr();
        maximum && previous < maximum.getInt())
      return operation->emitOpError(
          "last degree bound does not cover degree_max");
  }
  auto instance = storageValue.getDefiningOp<graphforge::storage::InstanceOp>();
  if (!instance)
    return operation->emitOpError(
        "requires a visible physical storage instance");
  if (instance.getLayout() != "degree-row-worklist")
    return operation->emitOpError(
        "storage layout must be 'degree-row-worklist'");
  if (instance.getExternal())
    return operation->emitOpError(
        "degree worklist storage must be compiler-owned");
  int64_t buckets = static_cast<int64_t>(upperBounds.size());
  if (buckets > (INT64_MAX - rows - 1) / 2)
    return operation->emitOpError("packed worklist size overflows i64");
  int64_t requiredElements = rows + 2 * buckets + 1;
  auto region =
      instance.getRegion().getDefiningOp<graphforge::storage::RegionOp>();
  if (region) {
    if (region.getVersionAttr().getInt() != version)
      return operation->emitOpError(
          "storage version does not match snapshot_version");
    if (region.getElementsAttr().getInt() < requiredElements)
      return operation->emitOpError(
          "storage region is too small for offsets, cursors and row ids");
  }
  if (requiredElements > INT64_MAX / 8 ||
      instance.getCapacityBytesAttr().getInt() < requiredElements * 8)
    return operation->emitOpError(
        "storage capacity is too small for packed i64 worklist");
  return success();
}

LogicalResult PartitionOp::verify() {
  if (getMeshShape().empty())
    return emitOpError("requires a non-empty device mesh");
  for (int64_t extent : getMeshShape())
    if (extent <= 0)
      return emitOpError("mesh extents must be positive");
  if (getMeshAxisAttr().getInt() < 0 ||
      getMeshAxisAttr().getInt() >= static_cast<int64_t>(getMeshShape().size()))
    return emitOpError("mesh_axis is outside the device mesh rank");
  if (getVersionAttr().getInt() < 0)
    return emitOpError("requires a non-negative snapshot version");
  if (getOwnedEntitiesAttr().getInt() < -1 ||
      getGhostEntitiesAttr().getInt() < -1)
    return emitOpError(
        "owned and ghost counts must be non-negative or -1 when dynamic");
  if (!llvm::StringSwitch<bool>(getScheme())
           .Cases("destination", "source", "edge", "spatial", "2d", true)
           .Default(false))
    return emitOpError("unknown partition scheme '") << getScheme() << "'";
  if (auto relation = getRelation().getDefiningOp<graphforge::RelationOp>();
      relation && relation.getVersionAttr().getInt() != getVersionAttr().getInt())
    return emitOpError("partition version does not match relation snapshot");
  return success();
}

LogicalResult HaloOp::verify() {
  // depth=-1 is the explicit spelling of compiler-derived `auto`.
  if (getDepthAttr().getInt() < -1)
    return emitOpError("depth must be non-negative or -1 for auto");
  if (getFields().empty())
    return emitOpError("requires at least one accessed Field");
  if (!llvm::StringSwitch<bool>(getStrategy())
           .Cases("pull", "push", "auto", true)
           .Default(false))
    return emitOpError("unknown halo strategy '") << getStrategy() << "'";
  return success();
}

LogicalResult HaloPackOp::verify() {
  if (getField().empty())
    return emitOpError("requires a Field name");
  if (getBytesAttr().getInt() < 0 || getSnapshotVersionAttr().getInt() < 0)
    return emitOpError("byte count and snapshot version must be non-negative");
  if (auto instance = getSource().getDefiningOp<storage::InstanceOp>()) {
    auto region = instance.getRegion().getDefiningOp<storage::RegionOp>();
    if (region && region.getVersionAttr().getInt() !=
                      getSnapshotVersionAttr().getInt())
      return emitOpError("source Field version does not match snapshot_version");
  }
  return verifyHaloTask(getOperation(), getPlan(), getDependsOn(),
                        getSnapshotVersionAttr().getInt());
}

LogicalResult HaloExchangeOp::verify() {
  if (getGroup().empty())
    return emitOpError("requires a communication group");
  if (getBytesAttr().getInt() < 0 || getSnapshotVersionAttr().getInt() < 0)
    return emitOpError("byte count and snapshot version must be non-negative");
  auto pack = getSend().getDefiningOp<HaloPackOp>();
  if (!pack || pack.getPlan() != getPlan())
    return emitOpError("send buffer must be packed by the same halo plan");
  return verifyHaloTask(getOperation(), getPlan(), getDependsOn(),
                        getSnapshotVersionAttr().getInt());
}

LogicalResult HaloUnpackOp::verify() {
  if (getField().empty() || getSnapshotVersionAttr().getInt() < 0)
    return emitOpError("requires a Field and non-negative snapshot version");
  auto exchange = getReceive().getDefiningOp<HaloExchangeOp>();
  if (!exchange || exchange.getPlan() != getPlan())
    return emitOpError("receive buffer must come from the same halo plan");
  if (auto instance = getDestination().getDefiningOp<storage::InstanceOp>()) {
    auto region = instance.getRegion().getDefiningOp<storage::RegionOp>();
    if (region && region.getVersionAttr().getInt() !=
                      getSnapshotVersionAttr().getInt())
      return emitOpError(
          "destination Field version does not match snapshot_version");
  }
  return verifyHaloTask(getOperation(), getPlan(), getDependsOn(),
                        getSnapshotVersionAttr().getInt());
}

LogicalResult DegreeWorklistOp::verify() {
  return verifyDegreeStorage(
      getOperation(), getRelation(), getStorage(), getUpperBounds(),
      getRowsAttr().getInt(), getSnapshotVersionAttr().getInt());
}

LogicalResult DegreeHistogramOp::verify() {
  if (failed(verifyDegreeStorage(
      getOperation(), getRelation(), getStorage(), getUpperBounds(),
      getRowsAttr().getInt(), getSnapshotVersionAttr().getInt())))
    return failure();
  auto reset = getDependsOn().getDefiningOp<DegreeResetOp>();
  if (!reset || reset.getStorage() != getStorage())
    return emitOpError("must depend on a reset over the same storage");
  return verifyDependencies(getOperation(), ValueRange{getDependsOn()},
                            getSnapshotVersionAttr().getInt());
}

LogicalResult DegreeResetOp::verify() {
  int64_t version = getSnapshotVersionAttr().getInt();
  if (getBucketsAttr().getInt() <= 0 || version < 0)
    return emitOpError(
        "bucket count must be positive and version non-negative");
  auto instance = getStorage().getDefiningOp<graphforge::storage::InstanceOp>();
  if (!instance || instance.getLayout() != "degree-row-worklist" ||
      instance.getExternal())
    return emitOpError("requires compiler-owned degree-row-worklist storage");
  auto region =
      instance.getRegion().getDefiningOp<graphforge::storage::RegionOp>();
  if (region && region.getVersionAttr().getInt() != version)
    return emitOpError("storage version does not match snapshot_version");
  return success();
}

LogicalResult DegreePrefixOp::verify() {
  int64_t version = getSnapshotVersionAttr().getInt();
  if (getBucketsAttr().getInt() <= 0 || version < 0)
    return emitOpError(
        "bucket count must be positive and version non-negative");
  auto histogram = getDependsOn().getDefiningOp<DegreeHistogramOp>();
  if (!histogram || histogram.getStorage() != getStorage())
    return emitOpError("must depend on a histogram over the same storage");
  return verifyDependencies(getOperation(), ValueRange{getDependsOn()}, version);
}

LogicalResult DegreeScatterOp::verify() {
  if (failed(verifyDegreeStorage(
          getOperation(), getRelation(), getStorage(), getUpperBounds(),
          getRowsAttr().getInt(), getSnapshotVersionAttr().getInt())))
    return failure();
  auto prefix = getDependsOn().getDefiningOp<DegreePrefixOp>();
  if (!prefix || prefix.getStorage() != getStorage())
    return emitOpError("must depend on a prefix over the same storage");
  return verifyDependencies(getOperation(), ValueRange{getDependsOn()},
                            getSnapshotVersionAttr().getInt());
}

LogicalResult DegreeBucketLaunchOp::verify() {
  int64_t version = getSnapshotVersionAttr().getInt();
  if (version < 0)
    return emitOpError("requires a non-negative snapshot version");
  auto logicalWorklist = getWorklist().getDefiningOp<DegreeWorklistOp>();
  auto scatterWorklist = getWorklist().getDefiningOp<DegreeScatterOp>();
  if (!logicalWorklist && !scatterWorklist)
    return emitOpError("requires a visible degree worklist");
  int64_t worklistVersion = logicalWorklist
      ? logicalWorklist.getSnapshotVersionAttr().getInt()
      : scatterWorklist.getSnapshotVersionAttr().getInt();
  ArrayRef<int64_t> bounds = logicalWorklist
      ? logicalWorklist.getUpperBounds()
      : scatterWorklist.getUpperBounds();
  Operation *materialization = logicalWorklist
      ? logicalWorklist.getOperation()
      : scatterWorklist.getOperation();
  if (worklistVersion != version)
    return emitOpError("worklist version does not match snapshot_version");
  int64_t ordinal = getBucketOrdinalAttr().getInt();
  if (ordinal < 0 || ordinal >= static_cast<int64_t>(bounds.size()))
    return emitOpError("bucket_ordinal is outside the worklist");
  int64_t expectedUpper = bounds[ordinal];
  int64_t expectedLower = ordinal == 0 ? -1 : bounds[ordinal - 1];
  if (getDegreeLowerExclusiveAttr().getInt() != expectedLower ||
      getDegreeUpperInclusiveAttr().getInt() != expectedUpper)
    return emitOpError("degree interval does not match worklist bucket");
  if (getRowsInBucketAttr().getInt() < 0)
    return emitOpError("rows_in_bucket must be non-negative");
  if (!llvm::StringSwitch<bool>(getRowMapping())
           .Cases("worklist", "direct-filter", "worklist-chunked", true)
           .Default(false))
    return emitOpError(
        "row_mapping must be 'worklist', 'direct-filter', or "
        "'worklist-chunked'");
  if (getPartitioning() != "degree-worklist" || getOutputPartition().empty())
    return emitOpError(
        "requires degree-worklist partitioning and an output partition id");
  llvm::SmallDenseSet<StringRef> parameters;
  llvm::SmallDenseSet<StringRef> resources;
  for (Attribute attribute : getArguments()) {
    auto binding = dyn_cast<DictionaryAttr>(attribute);
    auto parameter = binding ? binding.getAs<StringAttr>("parameter")
                             : StringAttr();
    auto resource = binding ? binding.getAs<StringAttr>("resource")
                            : StringAttr();
    if (!parameter || !resource || parameter.getValue().empty() ||
        resource.getValue().empty())
      return emitOpError(
          "arguments must be {parameter, resource} string dictionaries");
    if (!parameters.insert(parameter.getValue()).second)
      return emitOpError("argument parameters must be unique");
    resources.insert(resource.getValue());
  }
  if (!resources.contains("row_ptr") || !resources.contains("col_idx") ||
      !resources.contains("row_worklist") || !resources.contains("output"))
    return emitOpError(
        "arguments must bind row_ptr, col_idx, row_worklist and output");
  for (Attribute attribute : getReads())
    if (!resources.contains(cast<StringAttr>(attribute).getValue()))
      return emitOpError("every read resource must have an argument binding");
  for (Attribute attribute : getWrites())
    if (!resources.contains(cast<StringAttr>(attribute).getValue()))
      return emitOpError("every write resource must have an argument binding");
  bool dependsOnMaterialization = llvm::any_of(
      getDependsOn(), [&](Value dependency) {
        return dependency.getDefiningOp() == materialization;
      });
  if (!dependsOnMaterialization)
    return emitOpError("must depend on its worklist materialization event");
  return verifyDependencies(getOperation(), getDependsOn(), version);
}

static LogicalResult verifyNamedTaskABI(Operation *operation,
                                        ArrayAttr arguments, ArrayAttr reads,
                                        ArrayAttr writes) {
  llvm::SmallDenseSet<StringRef> parameters;
  llvm::SmallDenseSet<StringRef> resources;
  for (Attribute attribute : arguments) {
    auto binding = dyn_cast<DictionaryAttr>(attribute);
    auto parameter = binding ? binding.getAs<StringAttr>("parameter")
                             : StringAttr();
    auto resource = binding ? binding.getAs<StringAttr>("resource")
                            : StringAttr();
    if (!parameter || !resource || parameter.getValue().empty() ||
        resource.getValue().empty())
      return operation->emitOpError(
          "arguments must be {parameter, resource} string dictionaries");
    if (!parameters.insert(parameter.getValue()).second)
      return operation->emitOpError("argument parameters must be unique");
    resources.insert(resource.getValue());
  }
  auto verifyAccesses = [&](ArrayAttr accesses) -> LogicalResult {
    for (Attribute attribute : accesses) {
      auto name = dyn_cast<StringAttr>(attribute);
      if (!name || !resources.contains(name.getValue()))
        return operation->emitOpError(
            "every accessed resource must have an argument binding");
    }
    return success();
  };
  if (failed(verifyAccesses(reads)) || failed(verifyAccesses(writes)))
    return failure();
  return success();
}

static LogicalResult verifyPartialStorage(Operation *operation, Value storageValue,
                                          int64_t activeRows, int64_t partials,
                                          int64_t stateBytes,
                                          int64_t version) {
  if (activeRows <= 0 || partials <= 1 || stateBytes <= 0 || version < 0)
    return operation->emitOpError(
        "requires positive rows/state_bytes, multiple partials and a "
        "non-negative snapshot version");
  if (activeRows > INT64_MAX / partials ||
      activeRows * partials > INT64_MAX / stateBytes)
    return operation->emitOpError("partial-state size overflows i64");
  auto instance = storageValue.getDefiningOp<
      graphforge::storage::InstanceOp>();
  if (!instance || instance.getLayout() != "row-partial-f32" ||
      instance.getExternal())
    return operation->emitOpError(
        "requires compiler-owned row-partial-f32 storage");
  int64_t required = activeRows * partials * stateBytes;
  if (instance.getCapacityBytesAttr().getInt() < required)
    return operation->emitOpError(
        "partial-state instance capacity is too small");
  auto region = instance.getRegion().getDefiningOp<
      graphforge::storage::RegionOp>();
  if (region && region.getVersionAttr().getInt() != version)
    return operation->emitOpError(
        "partial-state storage version does not match snapshot_version");
  return success();
}

LogicalResult RowSplitPartialOp::verify() {
  int64_t version = getSnapshotVersionAttr().getInt();
  if (getChunkEdgesAttr().getInt() <= 0 || getRowsAttr().getInt() <= 0 ||
      getActiveRowsAttr().getInt() <= 0 ||
      getActiveRowsAttr().getInt() > getRowsAttr().getInt())
    return emitOpError("chunk_edges must be positive");
  if (failed(verifyPartialStorage(
          getOperation(), getStorage(), getActiveRowsAttr().getInt(),
          getPartialsPerRowAttr().getInt(), getStateBytesAttr().getInt(),
          version)))
    return failure();
  auto relation = getRelation().getDefiningOp<graphforge::RelationOp>();
  if (!relation || relation.getVersionAttr().getInt() != version ||
      (relation.getNumDstAttr().getInt() >= 0 &&
       relation.getNumDstAttr().getInt() != getRowsAttr().getInt()))
    return emitOpError(
        "requires a visible relation with matching rows and snapshot version");
  auto logicalWorklist = getWorklist().getDefiningOp<DegreeWorklistOp>();
  auto scatterWorklist = getWorklist().getDefiningOp<DegreeScatterOp>();
  if ((!logicalWorklist && !scatterWorklist) ||
      (logicalWorklist
           ? logicalWorklist.getSnapshotVersionAttr().getInt()
           : scatterWorklist.getSnapshotVersionAttr().getInt()) != version ||
      getBucketsAttr().getInt() != static_cast<int64_t>(
          logicalWorklist ? logicalWorklist.getUpperBounds().size()
                          : scatterWorklist.getUpperBounds().size()) ||
      getBucketOrdinalAttr().getInt() < 0 ||
      getBucketOrdinalAttr().getInt() >= getBucketsAttr().getInt())
    return emitOpError("requires a matching logical degree worklist bucket");
  Operation *materialization = logicalWorklist
                                   ? logicalWorklist.getOperation()
                                   : scatterWorklist.getOperation();
  if (getDependsOn().getDefiningOp() != materialization)
    return emitOpError("must depend on degree worklist materialization");
  llvm::SmallDenseSet<StringRef> resources;
  for (Attribute attribute : getWrites())
    resources.insert(cast<StringAttr>(attribute).getValue());
  if (!resources.contains("row_partials"))
    return emitOpError("must write the row_partials resource");
  return verifyNamedTaskABI(
      getOperation(), getArguments(), getReads(), getWrites());
}

LogicalResult RowSplitFinalizeOp::verify() {
  int64_t version = getSnapshotVersionAttr().getInt();
  if (getRowsAttr().getInt() <= 0 || getActiveRowsAttr().getInt() <= 0 ||
      getActiveRowsAttr().getInt() > getRowsAttr().getInt())
    return emitOpError("requires valid logical and active row counts");
  if (failed(verifyPartialStorage(
          getOperation(), getStorage(), getActiveRowsAttr().getInt(),
          getPartialsPerRowAttr().getInt(), getStateBytesAttr().getInt(),
          version)))
    return failure();
  auto partial = getDependsOn().getDefiningOp<RowSplitPartialOp>();
  if (!partial || partial.getStorage() != getStorage() ||
      partial.getRowsAttr() != getRowsAttr() ||
      partial.getActiveRowsAttr() != getActiveRowsAttr() ||
      partial.getWorklist() != getWorklist() ||
      partial.getBucketOrdinalAttr() != getBucketOrdinalAttr() ||
      partial.getBucketsAttr() != getBucketsAttr() ||
      partial.getPartialsPerRowAttr() != getPartialsPerRowAttr() ||
      partial.getStateBytesAttr() != getStateBytesAttr())
    return emitOpError(
        "must depend on a matching row_split_partial task");
  llvm::SmallDenseSet<StringRef> reads;
  llvm::SmallDenseSet<StringRef> writes;
  for (Attribute attribute : getReads())
    reads.insert(cast<StringAttr>(attribute).getValue());
  for (Attribute attribute : getWrites())
    writes.insert(cast<StringAttr>(attribute).getValue());
  if (!reads.contains("row_partials") || !writes.contains("output"))
    return emitOpError("must read row_partials and write output");
  if (!reads.contains("row_worklist") ||
      getPartitioning() != "degree-worklist" || getOutputPartition().empty())
    return emitOpError(
        "must read row_worklist and name its disjoint output partition");
  if (failed(verifyNamedTaskABI(
          getOperation(), getArguments(), getReads(), getWrites())))
    return failure();
  return verifyDependencies(
      getOperation(), ValueRange{getDependsOn()}, version);
}

LogicalResult LaunchOp::verify() {
  if (getSnapshotVersionAttr().getInt() < 0)
    return emitOpError("requires a non-negative snapshot version");
  if (!llvm::StringSwitch<bool>(getTaskKind())
           .Cases("local", "interior", "boundary", "halo-pack", true)
           .Cases("halo-unpack", "prefetch", "writeback", true)
           .Default(false))
    return emitOpError("unknown task_kind '") << getTaskKind() << "'";
  if ((getTaskKind() == "local" || getTaskKind() == "interior" ||
       getTaskKind() == "boundary") && !getCalleeAttr())
    return emitOpError("compute tasks require a kernel callee");
  if (Value partition = getPartition())
    if (auto defining = partition.getDefiningOp<PartitionOp>();
        defining && defining.getVersionAttr().getInt() !=
                        getSnapshotVersionAttr().getInt())
      return emitOpError("partition version does not match snapshot_version");
  return verifyDependencies(getOperation(), getDependsOn(),
                            getSnapshotVersionAttr().getInt());
}

LogicalResult CollectiveOp::verify() {
  if (getSnapshotVersionAttr().getInt() < 0 || getBytesAttr().getInt() < 0)
    return emitOpError("snapshot version and byte count must be non-negative");
  if (getGroup().empty())
    return emitOpError("requires a communication group");
  if (!llvm::StringSwitch<bool>(getKind())
           .Cases("all-reduce", "all-gather", "reduce-scatter", true)
           .Cases("all-to-all", "neighbor-exchange", true)
           .Default(false))
    return emitOpError("unknown collective kind '") << getKind() << "'";
  if (auto defining = getPartition().getDefiningOp<PartitionOp>();
      defining && defining.getVersionAttr().getInt() !=
                      getSnapshotVersionAttr().getInt())
    return emitOpError("partition version does not match snapshot_version");
  return verifyDependencies(getOperation(), getDependsOn(),
                            getSnapshotVersionAttr().getInt());
}

#define GET_OP_CLASSES
#include "graphforge/Dialect/Task/TaskOps.cpp.inc"
