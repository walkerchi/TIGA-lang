#include "tiga/Dialect/Iter/IterDialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace mlir::tiga::iter;

LogicalResult TraverseOp::verify() {
  StringRef hierarchy =
      (*this)->getAttrOfType<StringAttr>("coordinate_hierarchy").getValue();
  if (!llvm::StringSwitch<bool>(hierarchy)
           .Cases("compressed-row", "generated-neighborhood",
                  "cartesian-product", "ranked-pairs", true)
           .Default(false))
    return emitOpError("unknown coordinate hierarchy '") << hierarchy << "'";
  StringRef ordering =
      (*this)->getAttrOfType<StringAttr>("ordering").getValue();
  if (ordering != "destination-major")
    return emitOpError("unsupported ordering '") << ordering << "'";

  auto reducers = (*this)->getAttrOfType<ArrayAttr>("reducers");
  auto kinds = (*this)->getAttrOfType<DenseI64ArrayAttr>("region_kinds");
  auto segments =
      (*this)->getAttrOfType<DenseI64ArrayAttr>("input_segment_sizes");
  auto versions =
      (*this)->getAttrOfType<DenseI64ArrayAttr>("snapshot_versions");
  if (static_cast<size_t>(kinds.size()) != getNumRegions())
    return emitOpError("requires one kind for every iteration region");
  if (static_cast<size_t>(versions.size()) != getInputs().size())
    return emitOpError("requires one snapshot version per input");
  if (ArrayAttr roles = getInputRolesAttr()) {
    if (roles.size() != getInputs().size())
      return emitOpError("requires one input role per input");
    if (llvm::any_of(roles, [](Attribute attribute) {
          return !isa<StringAttr>(attribute);
        }))
      return emitOpError("input roles must be strings");
  }
  DenseI64ArrayAttr nodeIndices = getNodeInputIndicesAttr();
  DenseI64ArrayAttr nodeSegments = getNodeInputSegmentSizesAttr();
  if (static_cast<bool>(nodeIndices) != static_cast<bool>(nodeSegments))
    return emitOpError("requires node_input_indices and "
                       "node_input_segment_sizes together");
  if (nodeIndices) {
    if (static_cast<size_t>(nodeSegments.size()) != reducers.size())
      return emitOpError(
          "requires one node input segment size per reducer");
    int64_t flattened = 0;
    size_t regionIndex = 0;
    for (int64_t count : nodeSegments.asArrayRef()) {
      if (count < 0)
        return emitOpError("node input segment sizes must be non-negative");
      bool hasNode = regionIndex + 1 < static_cast<size_t>(kinds.size()) &&
                     kinds[regionIndex + 1] == 1;
      if (!hasNode && count != 0)
        return emitOpError("node input segment is non-empty without a node region");
      if (hasNode) {
        Region &nodeRegion = getRegions()[regionIndex + 1];
        if (!llvm::hasSingleElement(nodeRegion) ||
            nodeRegion.front().getNumArguments() !=
                static_cast<unsigned>(count + 1))
          return emitOpError("node region arguments must be indexed inputs "
                             "followed by the reducer result");
      }
      flattened += count;
      regionIndex += hasNode ? 2 : 1;
    }
    if (flattened != static_cast<int64_t>(nodeIndices.size()))
      return emitOpError("node input segments do not cover node_input_indices");
    for (int64_t index : nodeIndices.asArrayRef())
      if (index < 0 || index >= static_cast<int64_t>(getInputs().size()))
        return emitOpError("node input index is outside flattened inputs");
  }
  if (static_cast<size_t>(segments.size()) != reducers.size() ||
      reducers.size() != getNumResults())
    return emitOpError("requires one reducer/input segment per result");

  int64_t totalInputs = 0;
  for (int64_t size : segments.asArrayRef()) {
    if (size < 0)
      return emitOpError("input segment sizes must be non-negative");
    totalInputs += size;
  }
  if (totalInputs != static_cast<int64_t>(getInputs().size()))
    return emitOpError("input segment sizes do not cover all inputs");
  if (llvm::any_of(versions.asArrayRef(),
                   [](int64_t version) { return version < 0; }))
    return emitOpError("snapshot versions must be non-negative");
  for (Attribute reducer : reducers)
    if (!isa<StringAttr, FlatSymbolRefAttr>(reducer))
      return emitOpError(
          "reducers must be legacy names or gf.reducer symbol references");
  auto effects = (*this)->getAttrOfType<ArrayAttr>("effects");
  for (Attribute attribute : effects) {
    auto effect = dyn_cast<StringAttr>(attribute);
    if (!effect)
      return emitOpError("effects must be string attributes");
    if (!llvm::StringSwitch<bool>(effect.getValue())
             .Cases("read", "write", "reduce", "atomic", true)
             .Cases("io", "random", true)
             .Default(false))
      return emitOpError("unknown effect '") << effect.getValue() << "'";
  }

  int64_t edgeRegions = 0;
  bool previousWasEdge = false;
  for (int64_t kind : kinds.asArrayRef()) {
    if (kind == 0) {
      ++edgeRegions;
      previousWasEdge = true;
    } else if (kind == 1 && previousWasEdge) {
      previousWasEdge = false;
    } else {
      return emitOpError("region kinds must be edge(0), optional node(1)");
    }
  }
  if (edgeRegions != static_cast<int64_t>(getNumResults()))
    return emitOpError("requires one edge region per result");
  for (Region &region : getRegions()) {
    if (!llvm::hasSingleElement(region) ||
        !isa<YieldOp>(region.front().getTerminator()))
      return emitOpError(
          "each iteration region must be single-block and end in gf_iter.yield");
  }
  return success();
}

LogicalResult YieldOp::verify() {
  if (!isa<TraverseOp>((*this)->getParentOp()))
    return emitOpError("must terminate a region owned by gf_iter.traverse");
  return success();
}

#define GET_OP_CLASSES
#include "tiga/Dialect/Iter/IterOps.cpp.inc"
