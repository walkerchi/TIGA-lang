#include "graphforge/Dialect/Storage/StorageDialect.h"

#include "llvm/ADT/StringSwitch.h"

#include <optional>

using namespace mlir;
using namespace mlir::graphforge::storage;

static std::optional<int64_t> getInstanceVersion(Value value) {
  auto instance = value.getDefiningOp<InstanceOp>();
  if (!instance)
    return std::nullopt;
  auto region = instance.getRegion().getDefiningOp<RegionOp>();
  if (!region)
    return std::nullopt;
  return region.getVersionAttr().getInt();
}

static std::optional<int64_t> getEventVersion(Value value) {
  // Event is deliberately shared by storage and task dialects.  Keep this
  // verifier dependency-neutral by reading the common contract attribute
  // from any producer instead of enumerating operation classes.
  Operation *producer = value.getDefiningOp();
  if (producer)
    if (auto version =
            producer->getAttrOfType<IntegerAttr>("snapshot_version"))
      return version.getInt();
  return std::nullopt;
}

LogicalResult RegionOp::verify() {
  if (getLogicalId().empty())
    return emitOpError("requires a non-empty logical_id");
  if (getVersionAttr().getInt() < 0)
    return emitOpError("requires a non-negative version");
  if (getElementsAttr().getInt() < 0)
    return emitOpError("requires a non-negative element count");
  return success();
}

LogicalResult InstanceOp::verify() {
  if (getCapacityBytesAttr().getInt() < 0)
    return emitOpError("requires a non-negative capacity_bytes");
  if (!llvm::StringSwitch<bool>(getMemorySpace())
           .Cases("register", "local", "shared", "lds", "ub", true)
           .Cases("hbm", "device", "pinned_ram", "ram", true)
           .Cases("nvme", "peer_hbm", "remote", true)
           .Default(false))
    return emitOpError("unknown memory_space '") << getMemorySpace() << "'";
  if (getLayout().empty() || getDevice().empty())
    return emitOpError("requires non-empty layout and device");
  return success();
}

LogicalResult TransferOp::verify() {
  if (getBytesAttr().getInt() < 0)
    return emitOpError("requires a non-negative byte count");
  if (getEngine().empty())
    return emitOpError("requires a transfer engine");
  if (getSource() == getDestination())
    return emitOpError("source and destination instances must differ");
  int64_t version = getSnapshotVersionAttr().getInt();
  if (version < 0)
    return emitOpError("requires a non-negative snapshot version");
  for (Value instance : {getSource(), getDestination()})
    if (auto instanceVersion = getInstanceVersion(instance);
        instanceVersion && *instanceVersion != version)
      return emitOpError("instance version does not match snapshot_version");
  for (Value dependency : getDependsOn())
    if (auto dependencyVersion = getEventVersion(dependency);
        dependencyVersion && *dependencyVersion != version)
      return emitOpError("dependency event belongs to another snapshot version");
  return success();
}

LogicalResult ReleaseOp::verify() {
  if (getDependsOn().empty())
    return emitOpError("requires a dependency proving last use");
  int64_t version = getSnapshotVersionAttr().getInt();
  if (version < 0)
    return emitOpError("requires a non-negative snapshot version");
  if (auto instanceVersion = getInstanceVersion(getInstance());
      instanceVersion && *instanceVersion != version)
    return emitOpError("instance version does not match snapshot_version");
  for (Value dependency : getDependsOn())
    if (auto dependencyVersion = getEventVersion(dependency);
        dependencyVersion && *dependencyVersion != version)
      return emitOpError("dependency event belongs to another snapshot version");
  return success();
}

LogicalResult JoinOp::verify() {
  if (getEvents().empty())
    return emitOpError("requires at least one event");
  int64_t version = getSnapshotVersionAttr().getInt();
  if (version < 0)
    return emitOpError("requires a non-negative snapshot version");
  for (Value event : getEvents())
    if (auto eventVersion = getEventVersion(event);
        eventVersion && *eventVersion != version)
      return emitOpError("cannot join events from another snapshot version");
  return success();
}

#define GET_OP_CLASSES
#include "graphforge/Dialect/Storage/StorageOps.cpp.inc"
