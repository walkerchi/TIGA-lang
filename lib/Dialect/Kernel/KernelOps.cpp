#include "graphforge/Dialect/Kernel/KernelDialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"

#include <cmath>

using namespace mlir;
using namespace mlir::graphforge::kernel;

static LogicalResult verifyVocabulary(Operation *operation, StringRef name,
                                      ArrayAttr values,
                                      ArrayRef<StringRef> allowed) {
  if (!values || values.empty())
    return operation->emitOpError() << name << " must be a non-empty array";
  llvm::StringSet<> seen;
  for (Attribute attribute : values) {
    auto value = dyn_cast<StringAttr>(attribute);
    if (!value)
      return operation->emitOpError() << name << " entries must be strings";
    if (!llvm::is_contained(allowed, value.getValue()))
      return operation->emitOpError() << "unknown " << name << " entry '"
                                      << value.getValue() << "'";
    if (!seen.insert(value.getValue()).second)
      return operation->emitOpError() << "duplicate " << name << " entry '"
                                      << value.getValue() << "'";
  }
  return success();
}

template <typename LaunchOp>
static LogicalResult verifyScheduleABI(LaunchOp launch) {
  SmallVector<Attribute> attributes = {
      launch.getScheduleKindAttr(), launch.getBlockRowsAttr(),
      launch.getBlockNeighborsAttr(), launch.getNumWarpsAttr(),
      launch.getPipelineStagesAttr(), launch.getTargetContractAttr(),
      launch.getScheduleResourcesAttr(), launch.getExecutionRolesAttr(),
      launch.getScheduleHandoffsAttr(), launch.getInstructionContractsAttr()};
  size_t present = llvm::count_if(attributes,
                                  [](Attribute value) { return bool(value); });
  if (present == 0) return success();
  if (present != attributes.size())
    return launch.emitOpError(
        "machine schedule contract must be either absent or complete");
  StringRef scheduleKind = launch.getScheduleKindAttr().getValue();
  if (!llvm::StringSwitch<bool>(scheduleKind)
           .Cases({"fixed-row-neighbor-feature",
                   "bounded-ragged-row-neighbor-feature", "provider-deferred",
                   "fixed-row-neighbor", "bounded-ragged-row-neighbor",
                   "scalar-row-loop", "generated-cell-neighbor",
                   "dense-query-key-tile", "ranked-candidate-select"}, true)
           .Default(false))
    return launch.emitOpError("unknown schedule_kind '")
           << scheduleKind << "'";
  if (launch.getBlockRowsAttr().getInt() <= 0 ||
      launch.getBlockNeighborsAttr().getInt() <= 0 ||
      launch.getNumWarpsAttr().getInt() <= 0 ||
      launch.getNumWarpsAttr().getInt() > 64 ||
      launch.getPipelineStagesAttr().getInt() <= 0)
    return launch.emitOpError(
        "machine schedule dimensions must be positive and num_warps <= 64");
  StringRef targetContract = launch.getTargetContractAttr().getValue();
  if (targetContract != "provider-neutral-v1")
    return launch.emitOpError("unknown target_contract '")
           << targetContract << "'";
  if (failed(verifyVocabulary(
          launch, "schedule_resources", launch.getScheduleResourcesAttr(),
          {"input.global.read", "relation.global.read",
           "directory.global.read", "state.register.private",
           "selection.register.private",
           "output.global.write"})))
    return failure();
  if (failed(verifyVocabulary(
          launch, "execution_roles", launch.getExecutionRolesAttr(),
          {"workgroup.destination-rows", "workgroup.destination-queries",
           "subgroup.neighbor-reduction", "subgroup.key-reduction",
           "subgroup.ranked-selection",
           "lane.feature", "lane.scalar"})))
    return failure();
  if (failed(verifyVocabulary(
          launch, "schedule_handoffs", launch.getScheduleHandoffsAttr(),
          {"load.global-to-register", "reduce.subgroup",
           "store.register-to-global"})))
    return failure();
  return verifyVocabulary(
      launch, "instruction_contracts", launch.getInstructionContractsAttr(),
      {"masked-memory", "distance-filter", "vector-contraction",
       "ordered-reduce", "associative-reduce", "scalar-control-flow",
       "hierarchical-topk", "stable-index-tie-break",
       "provider-selection-required"});
}

template <typename LaunchOp>
static LogicalResult verifyNodeInputABI(LaunchOp launch) {
  DenseI64ArrayAttr indices = launch.getNodeInputIndicesAttr();
  DenseI64ArrayAttr segments = launch.getNodeInputSegmentSizesAttr();
  if (static_cast<bool>(indices) != static_cast<bool>(segments))
    return launch.emitOpError("requires node_input_indices and "
                              "node_input_segment_sizes together");
  if (!indices) return success();
  if (static_cast<size_t>(segments.size()) != launch.getReducers().size())
    return launch.emitOpError(
        "requires one node input segment size per reducer");
  int64_t flattened = 0;
  size_t regionIndex = 0;
  auto kinds = launch.getRegionKinds();
  for (int64_t count : segments.asArrayRef()) {
    if (count < 0)
      return launch.emitOpError(
          "node input segment sizes must be non-negative");
    bool hasNode = regionIndex + 1 < static_cast<size_t>(kinds.size()) &&
                   regionIndex + 1 < launch.getNumRegions() &&
                   kinds[regionIndex + 1] == 1;
    if (!hasNode && count != 0)
      return launch.emitOpError(
          "node input segment is non-empty without a node region");
    if (hasNode) {
      Region &nodeRegion = launch.getRegions()[regionIndex + 1];
      if (!llvm::hasSingleElement(nodeRegion) ||
          nodeRegion.front().getNumArguments() !=
              static_cast<unsigned>(count + 1))
        return launch.emitOpError("node region arguments must be indexed "
                                  "inputs followed by the reducer result");
    }
    flattened += count;
    regionIndex += hasNode ? 2 : 1;
  }
  if (flattened != static_cast<int64_t>(indices.size()))
    return launch.emitOpError(
        "node input segments do not cover node_input_indices");
  for (int64_t index : indices.asArrayRef())
    if (index < 0 || index >= static_cast<int64_t>(launch.getInputs().size()))
      return launch.emitOpError(
          "node input index is outside flattened inputs");
  return success();
}

LogicalResult LaunchOp::verify() {
  if (getNumRowsAttr().getInt() < 0)
    return emitOpError("requires a non-negative physical row count");
  if (getDegreeMinAttr() && getDegreeMinAttr().getInt() < 0)
    return emitOpError("requires a non-negative minimum degree");
  if (getDegreeMaxAttr() && getDegreeMaxAttr().getInt() < 0)
    return emitOpError("requires a non-negative maximum degree");
  if (getDegreeMinAttr() && getDegreeMaxAttr() &&
      getDegreeMinAttr().getInt() > getDegreeMaxAttr().getInt())
    return emitOpError("requires degree_min <= degree_max");
  if (getDegreeSumAttr()) {
    if (!(getDegreeMinAttr() && getDegreeMaxAttr()))
      return emitOpError("degree_sum requires degree_min and degree_max");
    int64_t rows = getNumRowsAttr().getInt();
    int64_t sum = getDegreeSumAttr().getInt();
    int64_t floorMean = rows > 0 ? sum / rows : 0;
    int64_t ceilMean = rows > 0
                           ? floorMean + static_cast<int64_t>(sum % rows != 0)
                           : 0;
    if (sum < 0 || (rows == 0 && sum != 0) ||
        (rows > 0 && (floorMean < getDegreeMinAttr().getInt() ||
                      ceilMean > getDegreeMaxAttr().getInt())))
      return emitOpError("degree_sum is inconsistent with row count and bounds");
  }
  if (DenseI64ArrayAttr histogram = getDegreeHistogramAttr()) {
    if (!(getDegreeMinAttr() && getDegreeMaxAttr() && getDegreeSumAttr()))
      return emitOpError("degree_histogram requires min/max/sum statistics");
    if (histogram.size() != getDegreeMaxAttr().getInt() + 1)
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
    if (countedRows != getNumRowsAttr().getInt() ||
        countedEdges != getDegreeSumAttr().getInt())
      return emitOpError("degree_histogram disagrees with row/edge counts");
  }
  if (FloatAttr ratio = getSourceIndexSpanRatioAttr()) {
    double value = ratio.getValueAsDouble();
    if (!std::isfinite(value) || value < 0.0 || value > 1.0)
      return emitOpError("source_index_span_ratio must be finite in [0, 1]");
  }
  auto traversal = (*this)->getAttrOfType<StringAttr>("traversal").getValue();
  if (!llvm::StringSwitch<bool>(traversal)
           .Cases("csr-row", "generated-tile", true)
           .Default(false))
    return emitOpError("unknown target-independent traversal '")
           << traversal << "'";

  auto reducers = (*this)->getAttrOfType<ArrayAttr>("reducers");
  auto kinds = (*this)->getAttrOfType<DenseI64ArrayAttr>("region_kinds");
  auto segments =
      (*this)->getAttrOfType<DenseI64ArrayAttr>("input_segment_sizes");
  auto versions =
      (*this)->getAttrOfType<DenseI64ArrayAttr>("snapshot_versions");
  if (static_cast<size_t>(kinds.size()) != getNumRegions())
    return emitOpError("requires one kind for every local region");
  if (static_cast<size_t>(versions.size()) != getInputs().size())
    return emitOpError("requires one snapshot version per input");
  if (failed(verifyNodeInputABI(*this))) return failure();
  if (ArrayAttr roles = getInputRolesAttr())
    if (roles.size() != getInputs().size())
      return emitOpError("requires one input role per input");
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
          "each local region must be single-block and end in gf_kernel.yield");
  }
  if (failed(verifyScheduleABI(*this))) return failure();
  return success();
}

LogicalResult GeneratedLaunchOp::verify() {
  if (getNumRowsAttr().getInt() < 0)
    return emitOpError("requires a non-negative physical row count");
  if (getCutoffAttr().getValueAsDouble() <= 0.0)
    return emitOpError("requires a positive cutoff");
  if (getDimensionsAttr().getInt() < 1 || getDimensionsAttr().getInt() > 3)
    return emitOpError("supports one, two, or three dimensions");
  if (getTraversal() != "generated-tile")
    return emitOpError("requires traversal 'generated-tile'");
  if (getSnapshotVersions().size() != getInputs().size())
    return emitOpError("requires one snapshot version per input");
  if (failed(verifyNodeInputABI(*this))) return failure();
  if (ArrayAttr roles = getInputRolesAttr())
    if (roles.size() != getInputs().size())
      return emitOpError("requires one input role per input");
  if (getInputSegmentSizes().size() != getReducers().size() ||
      getReducers().size() != getNumResults())
    return emitOpError("requires one reducer/input segment per result");
  int64_t totalInputs = 0;
  for (int64_t size : getInputSegmentSizes()) {
    if (size < 0)
      return emitOpError("input segment sizes must be non-negative");
    totalInputs += size;
  }
  if (totalInputs != static_cast<int64_t>(getInputs().size()))
    return emitOpError("input segment sizes do not cover all inputs");
  if (static_cast<size_t>(getRegionKinds().size()) != getNumRegions())
    return emitOpError("requires one kind for every local region");
  int64_t edgeRegions = 0;
  bool previousWasEdge = false;
  for (int64_t kind : getRegionKinds()) {
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
  for (Region &region : getRegions())
    if (!llvm::hasSingleElement(region) ||
        !isa<YieldOp>(region.front().getTerminator()))
      return emitOpError(
          "each local region must be single-block and end in gf_kernel.yield");
  if (failed(verifyScheduleABI(*this))) return failure();
  return success();
}

LogicalResult DenseLaunchOp::verify() {
  if (getNumSrcAttr().getInt() < 0 || getNumDstAttr().getInt() < 0)
    return emitOpError("endpoint counts must be non-negative");
  if (getTraversal() != "dense-tile")
    return emitOpError("requires traversal 'dense-tile'");
  if (auto boundary = (*this)->getAttrOfType<StringAttr>("boundary")) {
    if (boundary.getValue() != "full" &&
        boundary.getValue() != "lower_inclusive")
      return emitOpError("unknown dense boundary policy");
    if (boundary.getValue() != "full" && getNumSrc() != getNumDst())
      return emitOpError("triangular dense boundary requires a square relation");
  }
  if (getSnapshotVersions().size() != getInputs().size())
    return emitOpError("requires one snapshot version per input");
  if (failed(verifyNodeInputABI(*this))) return failure();
  if (ArrayAttr roles = getInputRolesAttr())
    if (roles.size() != getInputs().size())
      return emitOpError("requires one input role per input");
  if (getInputSegmentSizes().size() != getReducers().size() ||
      getReducers().size() != getNumResults())
    return emitOpError("requires one reducer/input segment per result");
  int64_t totalInputs = 0;
  for (int64_t size : getInputSegmentSizes())
    totalInputs += size;
  if (totalInputs != static_cast<int64_t>(getInputs().size()))
    return emitOpError("input segment sizes do not cover all inputs");
  if (static_cast<size_t>(getRegionKinds().size()) != getNumRegions())
    return emitOpError("requires one kind for every local region");
  for (Region &region : getRegions())
    if (!llvm::hasSingleElement(region) ||
        !isa<YieldOp>(region.front().getTerminator()))
      return emitOpError(
          "each local region must be single-block and end in gf_kernel.yield");
  if (failed(verifyScheduleABI(*this))) return failure();
  return success();
}

LogicalResult RankedLaunchOp::verify() {
  int64_t numQueries = getNumQueriesAttr().getInt();
  int64_t numCandidates = getNumCandidatesAttr().getInt();
  int64_t dimensions = getDimensionsAttr().getInt();
  int64_t k = getKAttr().getInt();
  if (numQueries < 0 || numCandidates < 0 || dimensions <= 0)
    return emitOpError("endpoint counts must be non-negative and dimensions positive");
  int64_t available =
      numCandidates - static_cast<int64_t>(getExcludeSelf());
  if (k <= 0 || k > available)
    return emitOpError("k exceeds the candidates available to each query");
  if (getExcludeSelf() && !getSameEntityDomain())
    return emitOpError("exclude_self requires one shared entity domain");
  if (getSameEntityDomain() && numQueries != numCandidates)
    return emitOpError("a shared entity domain requires equal endpoint counts");
  if (getMetric() != "squared_euclidean" || getSelection() != "smallest" ||
      getTieBreak() != "source_index" || !getExact())
    return emitOpError(
        "unsupported ranked selection contract; expected exact squared_euclidean/smallest/source_index");
  if (getTraversal() != "ranked-candidate-tile")
    return emitOpError("requires traversal 'ranked-candidate-tile'");
  if (getSnapshotVersions().size() != getInputs().size())
    return emitOpError("requires one snapshot version per input");
  if (failed(verifyNodeInputABI(*this))) return failure();
  if (ArrayAttr roles = getInputRolesAttr())
    if (roles.size() != getInputs().size())
      return emitOpError("requires one input role per input");
  if (getInputSegmentSizes().size() != getReducers().size() ||
      getReducers().size() != getNumResults())
    return emitOpError("requires one reducer/input segment per result");
  int64_t totalInputs = 0;
  for (int64_t size : getInputSegmentSizes()) {
    if (size < 0)
      return emitOpError("input segment sizes must be non-negative");
    totalInputs += size;
  }
  if (totalInputs != static_cast<int64_t>(getInputs().size()))
    return emitOpError("input segment sizes do not cover all inputs");
  if (static_cast<size_t>(getRegionKinds().size()) != getNumRegions())
    return emitOpError("requires one kind for every local region");
  int64_t edgeRegions = 0;
  bool previousWasEdge = false;
  for (int64_t kind : getRegionKinds()) {
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
  for (Region &region : getRegions())
    if (!llvm::hasSingleElement(region) ||
        !isa<YieldOp>(region.front().getTerminator()))
      return emitOpError(
          "each local region must be single-block and end in gf_kernel.yield");
  if (static_cast<bool>(getCandidateTileAttr()) !=
      static_cast<bool>(getMergeFanInAttr()))
    return emitOpError(
        "candidate_tile and merge_fan_in must be absent or present together");
  if (getCandidateTileAttr() &&
      (getCandidateTileAttr().getInt() < k ||
       getMergeFanInAttr().getInt() < 2))
    return emitOpError("candidate_tile must cover k and merge_fan_in must be >= 2");
  if (getMergeFanInAttr() && getMergeFanInAttr().getInt() != 2)
    return emitOpError("M0 ranked launch requires pairwise merge_fan_in = 2");
  return verifyScheduleABI(*this);
}

LogicalResult YieldOp::verify() {
  if (!isa<LaunchOp, GeneratedLaunchOp, DenseLaunchOp, RankedLaunchOp>(
          (*this)->getParentOp()))
    return emitOpError(
        "must terminate a region owned by a gf_kernel launch");
  return success();
}

#define GET_OP_CLASSES
#include "graphforge/Dialect/Kernel/KernelOps.cpp.inc"
