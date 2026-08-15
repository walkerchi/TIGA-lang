#include "graphforge/Dialect/Domain/DomainDialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"

#include <cmath>

using namespace mlir;
using namespace mlir::graphforge;

LogicalResult RelationOp::verify() {
  auto origin = (*this)->getAttrOfType<StringAttr>("origin").getValue();
  if (origin != "external" && origin != "procedural")
    return emitOpError("origin must be 'external' or 'procedural'");

  auto lifecycle =
      (*this)->getAttrOfType<StringAttr>("lifecycle").getValue();
  if (lifecycle != "frozen" && lifecycle != "rebuildable")
    return emitOpError("lifecycle must be 'frozen' or 'rebuildable'");

  auto realization =
      (*this)->getAttrOfType<StringAttr>("realization").getValue();
  if (realization != "materialized" && realization != "paged" &&
      realization != "generated")
    return emitOpError(
        "realization must be 'materialized', 'paged', or 'generated'");

  auto version = (*this)->getAttrOfType<IntegerAttr>("version").getInt();
  auto numSrc = (*this)->getAttrOfType<IntegerAttr>("num_src").getInt();
  auto numDst = (*this)->getAttrOfType<IntegerAttr>("num_dst").getInt();
  if (version < 0)
    return emitOpError("version must be non-negative");
  if (numSrc < 0 || numDst < 0)
    return emitOpError("endpoint cardinalities must be non-negative");
  auto degreeMin = getDegreeMinAttr();
  auto degreeMax = getDegreeMaxAttr();
  auto degreeSum = getDegreeSumAttr();
  if (degreeMin && degreeMin.getInt() < 0)
    return emitOpError("degree_min must be non-negative");
  if (degreeMax && degreeMax.getInt() < 0)
    return emitOpError("degree_max must be non-negative");
  if (degreeMin && degreeMax && degreeMin.getInt() > degreeMax.getInt())
    return emitOpError("requires degree_min <= degree_max");
  if (degreeSum) {
    if (!(degreeMin && degreeMax))
      return emitOpError("degree_sum requires degree_min and degree_max");
    int64_t sum = degreeSum.getInt();
    int64_t floorMean = numDst > 0 ? sum / numDst : 0;
    int64_t ceilMean = numDst > 0
                           ? floorMean + static_cast<int64_t>(sum % numDst != 0)
                           : 0;
    if (sum < 0 || (numDst == 0 && sum != 0) ||
        (numDst > 0 && (floorMean < degreeMin.getInt() ||
                        ceilMean > degreeMax.getInt())))
      return emitOpError("degree_sum is inconsistent with row count and bounds");
  }
  if (DenseI64ArrayAttr histogram = getDegreeHistogramAttr()) {
    if (!(degreeMin && degreeMax && degreeSum))
      return emitOpError("degree_histogram requires min/max/sum statistics");
    if (histogram.size() != degreeMax.getInt() + 1)
      return emitOpError("degree_histogram extent must be degree_max + 1");
    int64_t countedRows = 0;
    int64_t countedEdges = 0;
    for (auto [degree, count] : llvm::enumerate(histogram.asArrayRef())) {
      if (count < 0 || count > INT64_MAX - countedRows ||
          (degree > 0 && count >
              (INT64_MAX - countedEdges) / static_cast<int64_t>(degree)))
        return emitOpError("degree_histogram counts overflow or are negative");
      countedRows += count;
      countedEdges += static_cast<int64_t>(degree) * count;
    }
    if (countedRows != numDst || countedEdges != degreeSum.getInt())
      return emitOpError("degree_histogram disagrees with row/edge counts");
    if (numDst > 0 &&
        (histogram[degreeMin.getInt()] == 0 ||
         histogram[histogram.size() - 1] == 0))
      return emitOpError("degree_histogram disagrees with degree bounds");
  }
  if (FloatAttr ratio = getSourceIndexSpanRatioAttr()) {
    double value = ratio.getValueAsDouble();
    if (!std::isfinite(value) || value < 0.0 || value > 1.0)
      return emitOpError("source_index_span_ratio must be finite in [0, 1]");
  }
  auto mesh = getMeshShapeAttr();
  auto axis = getMeshAxisAttr();
  auto depth = getHaloDepthAttr();
  auto balance = getPartitionBalanceAttr();
  bool hasPlacement = mesh || axis || depth || balance;
  if (hasPlacement && !(mesh && axis && depth && balance))
    return emitOpError("distributed placement attributes must be complete");
  if (mesh) {
    if (mesh.empty() || llvm::any_of(mesh.asArrayRef(),
                                     [](int64_t value) { return value <= 0; }))
      return emitOpError("mesh_shape must contain positive extents");
    if (axis.getInt() < 0 || axis.getInt() >= static_cast<int64_t>(mesh.size()))
      return emitOpError("mesh_axis is outside mesh_shape");
    if (depth.getInt() < -1)
      return emitOpError("halo_depth must be non-negative or -1 for auto");
    if (!llvm::StringSwitch<bool>(balance.getValue())
             .Cases("edges", "entities", "auto", true)
             .Default(false))
      return emitOpError("unknown partition balance policy");
  }
  return success();
}

LogicalResult GeneratedRadiusOp::verify() {
  if (getCutoff().convertToDouble() <= 0.0)
    return emitOpError("requires a positive cutoff");
  if (getDimensions() < 1 || getDimensions() > 3)
    return emitOpError("supports one, two, or three dimensions");
  if (getNumEntitiesAttr().getInt() < 0 || getVersionAttr().getInt() < 0)
    return emitOpError("entity count and version must be non-negative");
  return success();
}

LogicalResult CartesianOp::verify() {
  if (getNumSrcAttr().getInt() < 0 || getNumDstAttr().getInt() < 0 ||
      getVersionAttr().getInt() < 0)
    return emitOpError("endpoint counts and version must be non-negative");
  if (auto boundary = (*this)->getAttrOfType<StringAttr>("boundary")) {
    if (boundary.getValue() != "full" &&
        boundary.getValue() != "lower_inclusive")
      return emitOpError("unknown Cartesian boundary policy");
    if (boundary.getValue() != "full" && getNumSrc() != getNumDst())
      return emitOpError("triangular Cartesian boundary requires a square relation");
  }
  return success();
}

LogicalResult RankedRelationOp::verify() {
  int64_t numQueries = getNumQueriesAttr().getInt();
  int64_t numCandidates = getNumCandidatesAttr().getInt();
  int64_t dimensions = getDimensionsAttr().getInt();
  int64_t k = getKAttr().getInt();
  if (numQueries < 0 || numCandidates < 0 || getVersionAttr().getInt() < 0)
    return emitOpError("endpoint counts and version must be non-negative");
  if (dimensions <= 0)
    return emitOpError("dimensions must be positive");
  int64_t available =
      numCandidates - static_cast<int64_t>(getExcludeSelf());
  if (k <= 0 || k > available)
    return emitOpError("k exceeds the candidates available to each query");
  if (getExcludeSelf() && !getSameEntityDomain())
    return emitOpError("exclude_self requires one shared entity domain");
  if (getSameEntityDomain() && numQueries != numCandidates)
    return emitOpError("a shared entity domain requires equal endpoint counts");
  if (getMetric() != "squared_euclidean")
    return emitOpError("unsupported ranked metric '") << getMetric() << "'";
  if (getSelection() != "smallest")
    return emitOpError("unsupported ranked selection '") << getSelection()
                                                           << "'";
  if (getTieBreak() != "source_index")
    return emitOpError("unsupported ranked tie_break '") << getTieBreak()
                                                           << "'";
  if (!getExact())
    return emitOpError("M0 ranked relations require exact selection");
  auto query = dyn_cast<RankedTensorType>(getQueryPositions().getType());
  auto candidate =
      dyn_cast<RankedTensorType>(getCandidatePositions().getType());
  if (!query || !candidate || query.getRank() != 2 || candidate.getRank() != 2)
    return emitOpError("position operands must be rank-two tensors");
  if (!query.getElementType().isF32() || !candidate.getElementType().isF32())
    return emitOpError("M0 ranked positions must use f32 elements");
  auto agrees = [](int64_t dynamicOrStatic, int64_t expected) {
    return ShapedType::isDynamic(dynamicOrStatic) ||
           dynamicOrStatic == expected;
  };
  if (!agrees(query.getDimSize(0), numQueries) ||
      !agrees(candidate.getDimSize(0), numCandidates) ||
      !agrees(query.getDimSize(1), dimensions) ||
      !agrees(candidate.getDimSize(1), dimensions))
    return emitOpError("position tensor shape disagrees with relation attributes");
  if (getSameEntityDomain() &&
      getQueryPositions() != getCandidatePositions())
    return emitOpError(
        "a shared entity domain must reuse the same position SSA value");
  return success();
}

LogicalResult ReducerOp::verify() {
  auto readTypes = [&](ArrayAttr attributes, StringRef name,
                       SmallVectorImpl<Type> &types) -> LogicalResult {
    for (Attribute attribute : attributes) {
      auto type = dyn_cast<TypeAttr>(attribute);
      if (!type)
        return emitOpError() << name << " must contain only type attributes";
      types.push_back(type.getValue());
    }
    if (types.empty())
      return emitOpError() << name << " must not be empty";
    return success();
  };
  SmallVector<Type> messageTypes, stateTypes, resultTypes;
  if (failed(readTypes(getMessageTypes(), "message_types", messageTypes)) ||
      failed(readTypes(getStateTypes(), "state_types", stateTypes)) ||
      failed(readTypes(getResultTypes(), "result_types", resultTypes)))
    return failure();
  for (Region *region : {&getIdentity(), &getLift(), &getCombine(),
                         &getFinalize()})
    if (!llvm::hasSingleElement(*region) ||
        !isa<ReducerYieldOp>(region->front().getTerminator()))
      return emitOpError(
          "identity/lift/combine/finalize must be single-block and terminate "
          "with gf.reducer_yield");
  if (!getAssociative())
    return emitOpError("parallel reducer definitions must declare associativity");

  auto verifyRegion = [&](Region &region, ArrayRef<Type> arguments,
                          ArrayRef<Type> results,
                          StringRef name) -> LogicalResult {
    Block &block = region.front();
    if (block.getArgumentTypes() != arguments)
      return emitOpError() << name << " argument types must match its contract";
    auto yield = cast<ReducerYieldOp>(block.getTerminator());
    if (yield.getOperandTypes() != results)
      return emitOpError() << name << " yielded types must match its contract";
    return success();
  };
  SmallVector<Type> combineTypes(stateTypes);
  llvm::append_range(combineTypes, stateTypes);
  if (failed(verifyRegion(getIdentity(), {}, stateTypes, "identity")) ||
      failed(verifyRegion(getLift(), messageTypes, stateTypes, "lift")) ||
      failed(verifyRegion(getCombine(), combineTypes, stateTypes, "combine")) ||
      failed(verifyRegion(getFinalize(), stateTypes, resultTypes, "finalize")))
    return failure();
  return success();
}

LogicalResult ReducerYieldOp::verify() {
  if (!isa<ReducerOp>((*this)->getParentOp()))
    return emitOpError("must terminate a region owned by gf.reducer");
  return success();
}

LogicalResult ApplyOp::verify() {
  auto reducers = (*this)->getAttrOfType<ArrayAttr>("reducers");
  auto regionKinds =
      (*this)->getAttrOfType<DenseI64ArrayAttr>("region_kinds");
  auto segments =
      (*this)->getAttrOfType<DenseI64ArrayAttr>("input_segment_sizes");
  auto versions =
      (*this)->getAttrOfType<DenseI64ArrayAttr>("snapshot_versions");
  auto effects = (*this)->getAttrOfType<ArrayAttr>("effects");

  if (static_cast<size_t>(regionKinds.size()) != getNumRegions())
    return emitOpError("requires one kind for every region");
  int64_t edgeRegions = 0;
  bool previousWasEdge = false;
  for (int64_t kind : regionKinds.asArrayRef()) {
    if (kind == 0) {
      ++edgeRegions;
      previousWasEdge = true;
      continue;
    }
    if (kind != 1)
      return emitOpError("region kind must be 0 (edge) or 1 (node)");
    if (!previousWasEdge)
      return emitOpError("a node region must immediately follow an edge region");
    previousWasEdge = false;
  }
  if (edgeRegions != static_cast<int64_t>(getNumResults()))
    return emitOpError("requires one edge region per output result");
  if (static_cast<int64_t>(reducers.size()) != edgeRegions)
    return emitOpError("requires one reducer per edge region");
  if (static_cast<int64_t>(segments.size()) != edgeRegions)
    return emitOpError("requires one input segment size per edge region");
  if (static_cast<size_t>(versions.size()) != getInputs().size())
    return emitOpError("requires one snapshot version per input operand");
  if (ArrayAttr roles = getInputRolesAttr()) {
    if (roles.size() != getInputs().size())
      return emitOpError("requires one input role per input operand");
    for (Attribute attribute : roles) {
      auto role = dyn_cast<StringAttr>(attribute);
      if (!role || !llvm::StringSwitch<bool>(role.getValue())
                        .Cases("src", "dst", "edge", "param", true)
                        .Default(false))
        return emitOpError("input roles must be src, dst, edge, or param");
    }
  }
  if (ArrayAttr names = getInputNamesAttr()) {
    if (names.size() != getInputs().size())
      return emitOpError("requires one input name per input operand");
    if (llvm::any_of(names, [](Attribute attribute) {
          auto name = dyn_cast<StringAttr>(attribute);
          return !name || name.getValue().empty();
        }))
      return emitOpError("input names must be non-empty strings");
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
    size_t nodeRegion = 0;
    for (int64_t count : nodeSegments.asArrayRef()) {
      if (count < 0)
        return emitOpError("node input segment sizes must be non-negative");
      bool hasNode = nodeRegion + 1 < static_cast<size_t>(regionKinds.size()) &&
                     regionKinds[nodeRegion + 1] == 1;
      if (!hasNode && count != 0)
        return emitOpError("node input segment is non-empty without a node region");
      if (hasNode) {
        Region &nodeRegionValue = getRegions()[nodeRegion + 1];
        if (!llvm::hasSingleElement(nodeRegionValue))
          return emitOpError("node region must contain exactly one block");
        Block &node = nodeRegionValue.front();
        if (node.getNumArguments() != static_cast<unsigned>(count + 1))
          return emitOpError("node region arguments must be indexed inputs "
                             "followed by the reducer result");
      }
      flattened += count;
      nodeRegion += hasNode ? 2 : 1;
    }
    if (flattened != static_cast<int64_t>(nodeIndices.size()))
      return emitOpError("node input segments do not cover node_input_indices");
    for (int64_t index : nodeIndices.asArrayRef())
      if (index < 0 || index >= static_cast<int64_t>(getInputs().size()))
        return emitOpError("node input index is outside flattened inputs");
  }
  if (DenseI64ArrayAttr lanes = getIterationLanesAttr())
    if (llvm::any_of(lanes.asArrayRef(),
                     [](int64_t extent) { return extent <= 0; }))
      return emitOpError("iteration lane extents must be positive");

  int64_t totalInputs = 0;
  for (int64_t size : segments.asArrayRef()) {
    if (size < 0)
      return emitOpError("input segment sizes must be non-negative");
    totalInputs += size;
  }
  if (totalInputs != static_cast<int64_t>(getInputs().size()))
    return emitOpError("input segment sizes do not cover all inputs");

  ArrayRef<int64_t> versionValues = versions.asArrayRef();
  if (llvm::any_of(versionValues, [](int64_t version) { return version < 0; }))
    return emitOpError("snapshot versions must be non-negative");
  for (auto [leftIndex, left] : llvm::enumerate(getInputs())) {
    for (auto [rightIndex, right] : llvm::enumerate(getInputs())) {
      if (leftIndex < rightIndex && left == right &&
          versionValues[leftIndex] != versionValues[rightIndex])
        return emitOpError(
            "the same input SSA value cannot have two snapshot versions");
    }
  }

  for (Attribute attribute : reducers) {
    if (isa<StringAttr>(attribute))
      continue; // Temporary parser compatibility for pre-reducer-SSA lit.
    auto reference = dyn_cast<FlatSymbolRefAttr>(attribute);
    if (!reference ||
        !SymbolTable::lookupNearestSymbolFrom<ReducerOp>(*this, reference))
      return emitOpError(
          "reducers must reference a visible gf.reducer definition");
  }

  // A reducer symbol is the ABI of its associated edge/optional-node pair.
  // Validate that regions actually implement that ABI before scheduling or
  // cloning them into lower dialects.
  size_t regionIndex = 0;
  for (Attribute attribute : reducers) {
    auto reference = dyn_cast<FlatSymbolRefAttr>(attribute);
    if (!reference) {
      ++regionIndex;
      if (regionIndex < static_cast<size_t>(regionKinds.size()) &&
          regionKinds[regionIndex] == 1)
        ++regionIndex;
      continue;
    }
    auto definition =
        SymbolTable::lookupNearestSymbolFrom<ReducerOp>(*this, reference);
    if (!definition)
      continue; // Diagnosed above.
    auto readTypes = [](ArrayAttr attributes) {
      SmallVector<Type> types;
      for (Attribute attribute : attributes)
        types.push_back(cast<TypeAttr>(attribute).getValue());
      return types;
    };
    SmallVector<Type> messageTypes = readTypes(definition.getMessageTypes());
    SmallVector<Type> resultTypes = readTypes(definition.getResultTypes());
    auto edgeYield = cast<YieldOp>(
        getRegions()[regionIndex].front().getTerminator());
    if (edgeYield.getOperandTypes() != ArrayRef<Type>(messageTypes))
      return emitOpError(
          "edge region yielded types do not match reducer message_types");
    ++regionIndex;
    if (regionIndex >= static_cast<size_t>(regionKinds.size()) ||
        regionKinds[regionIndex] != 1)
      continue;
    if (resultTypes.size() != 1)
      return emitOpError(
          "optional node currently requires one finalized reducer result");
    Block &node = getRegions()[regionIndex].front();
    if (node.getNumArguments() == 0 ||
        node.getArgument(node.getNumArguments() - 1).getType() != resultTypes[0])
      return emitOpError(
          "node region must receive the finalized reducer result as its last argument");
    auto nodeYield = cast<YieldOp>(node.getTerminator());
    if (nodeYield.getMessages().size() != 1 ||
        nodeYield.getMessages()[0].getType() != resultTypes[0])
      return emitOpError(
          "node region must yield one value matching the reducer result type");
    ++regionIndex;
  }

  for (Attribute attribute : effects) {
    auto effect = dyn_cast<StringAttr>(attribute);
    if (!effect)
      return emitOpError("effects must be string attributes");
    bool known = llvm::StringSwitch<bool>(effect.getValue())
                     .Cases("read", "write", "reduce", "atomic", true)
                     .Cases("io", "random", true)
                     .Default(false);
    if (!known)
      return emitOpError("unknown effect '") << effect.getValue() << "'";
  }

  for (Region &region : getRegions()) {
    if (!llvm::hasSingleElement(region))
      return emitOpError("each edge region must contain exactly one block");
    auto yield = dyn_cast<YieldOp>(region.front().getTerminator());
    if (!yield)
      return emitOpError("each edge region must terminate with gf.yield");
  }
  return success();
}

LogicalResult YieldOp::verify() {
  if (!isa<ApplyOp>((*this)->getParentOp()))
    return emitOpError("must terminate a region directly owned by gf.apply");
  return success();
}

#define GET_OP_CLASSES
#include "graphforge/Dialect/Domain/DomainOps.cpp.inc"
