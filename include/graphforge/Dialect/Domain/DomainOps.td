#ifndef GRAPHFORGE_DOMAIN_OPS_TD
#define GRAPHFORGE_DOMAIN_OPS_TD

include "graphforge/Dialect/Domain/DomainTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

def GraphForge_RelationOp : GraphForge_Op<"relation", [Pure]> {
  let summary = "Bind runtime index buffers to a logical relation snapshot";
  let description = [{
    `gf.relation` separates the logical relation schema from its runtime
    realization.  M0 accepts two index buffers for materialized CSR.  The
    attributes deliberately keep origin, lifecycle, realization, logical ID,
    and version independent so future generated/paged relations do not require
    a new public Graph type.
  }];

  let arguments = (ins
    AnyType:$row_ptr,
    AnyType:$col_idx,
    StrAttr:$origin,
    StrAttr:$lifecycle,
    StrAttr:$realization,
    StrAttr:$relation_id,
    I64Attr:$version,
    I64Attr:$num_src,
    I64Attr:$num_dst,
    OptionalAttr<I64Attr>:$degree_min,
    OptionalAttr<I64Attr>:$degree_max,
    OptionalAttr<I64Attr>:$degree_sum,
    OptionalAttr<DenseI64ArrayAttr>:$degree_histogram,
    OptionalAttr<F64Attr>:$source_index_span_ratio,
    OptionalAttr<DenseI64ArrayAttr>:$mesh_shape,
    OptionalAttr<I64Attr>:$mesh_axis,
    OptionalAttr<I64Attr>:$halo_depth,
    OptionalAttr<StrAttr>:$partition_balance
  );
  let results = (outs GraphForge_RelationType:$result);
  let hasVerifier = 1;
}

def GraphForge_GeneratedRadiusOp
    : GraphForge_Op<"generated_radius", [Pure]> {
  let summary = "Bind a compact cell directory to a generated radius relation";
  let description = [{
    The physical operands form an O(N + cells) dense cell directory. During
    traversal the relation synthesizes endpoint geometry instead of storing
    edges. For the M0 Euclidean contract, an edge region may append one `f32`
    block argument after its explicit input projections; that argument is the
    generated endpoint distance and is not counted by `input_segment_sizes`.
  }];
  let arguments = (ins
    AnyType:$positions,
    AnyType:$cell_ptr,
    AnyType:$particle_order,
    AnyType:$cell_coordinates,
    AnyType:$extents,
    AnyType:$strides,
    AnyType:$neighbor_offsets,
    AnyType:$lattice,
    AnyType:$inverse_lattice,
    F64Attr:$cutoff,
    I64Attr:$dimensions,
    BoolAttr:$periodic,
    DefaultValuedAttr<BoolAttr, "false">:$hash_grid,
    I64Attr:$num_entities,
    I64Attr:$version
  );
  let results = (outs GraphForge_RelationType:$result);
  let hasVerifier = 1;
}

def GraphForge_CartesianOp : GraphForge_Op<"cartesian", [Pure]> {
  let summary = "Represent an implicit complete bipartite relation";
  let description = [{
    Contains every destination/source pair without adjacency storage.  It is
    the domain relation used by tiled dense consumers such as attention.
  }];
  let arguments = (ins I64Attr:$num_src, I64Attr:$num_dst, I64Attr:$version);
  let results = (outs GraphForge_RelationType:$result);
  let hasVerifier = 1;
}

def GraphForge_RankedRelationOp
    : GraphForge_Op<"ranked_relation", [Pure]> {
  let summary = "Select an exact fixed-size ranked neighborhood per query";
  let description = [{
    `gf.ranked_relation` is a provider-neutral relation, not a materialized
    adjacency list.  It ranks the Cartesian query/candidate pairs using the
    declared metric and retains exactly `k` candidates for every query.
    Physical candidate tiling and hierarchical selection are introduced only
    by later iteration, kernel, and task passes.

    M0 deliberately specifies squared Euclidean ranking with smallest-first
    selection and source-index tie breaking.  Keeping tie breaking explicit is
    required for exact differential tests and deterministic recompilation.
  }];
  let arguments = (ins
    AnyType:$query_positions,
    AnyType:$candidate_positions,
    I64Attr:$num_queries,
    I64Attr:$num_candidates,
    I64Attr:$dimensions,
    I64Attr:$k,
    StrAttr:$metric,
    StrAttr:$selection,
    StrAttr:$tie_break,
    BoolAttr:$exclude_self,
    BoolAttr:$same_entity_domain,
    BoolAttr:$exact,
    I64Attr:$version
  );
  let results = (outs GraphForge_RelationType:$result);
  let hasVerifier = 1;
}

def GraphForge_ReducerOp : GraphForge_Op<"reducer", [Symbol]> {
  let summary = "Define reducer identity, lift, combine, and finalize regions";
  let arguments = (ins
    SymbolNameAttr:$sym_name, StrAttr:$kind,
    ArrayAttr:$message_types, ArrayAttr:$state_types, ArrayAttr:$result_types,
    BoolAttr:$associative, BoolAttr:$commutative
  );
  let regions = (region AnyRegion:$identity, AnyRegion:$lift,
                         AnyRegion:$combine, AnyRegion:$finalize);
  let hasVerifier = 1;
}

def GraphForge_ReducerYieldOp : GraphForge_Op<"reducer_yield", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let hasVerifier = 1;
}

def GraphForge_ApplyOp : GraphForge_Op<"apply"> {
  let summary = "Apply edge regions and reducers over one relation snapshot";
  let description = [{
    Every output has one edge region followed by zero or one node region.
    `region_kinds` encodes this as `0` (edge) and `1` (node), avoiding two
    adjacent variadic region groups. Horizontal fusion concatenates complete
    edge/optional-node pairs and reducers while retaining separate SSA results.
    Every region has one block and terminates with `gf.yield`.

    `input_segment_sizes[i]` records how many flattened input operands belong
    to region `i`; `snapshot_versions` records the logical version of every
    flattened input.  These attributes are required for fusion legality and
    are not backend scheduling hints.
  }];

  let arguments = (ins
    GraphForge_RelationType:$relation,
    Variadic<AnyType>:$inputs,
    ArrayAttr:$reducers,
    DenseI64ArrayAttr:$region_kinds,
    DenseI64ArrayAttr:$input_segment_sizes,
    DenseI64ArrayAttr:$snapshot_versions,
    ArrayAttr:$effects,
    BoolAttr:$deterministic,
    OptionalAttr<ArrayAttr>:$input_roles,
    OptionalAttr<ArrayAttr>:$input_names,
    OptionalAttr<DenseI64ArrayAttr>:$node_input_indices,
    OptionalAttr<DenseI64ArrayAttr>:$node_input_segment_sizes,
    OptionalAttr<DenseI64ArrayAttr>:$iteration_lanes,
    OptionalAttr<I64Attr>:$fusion_group,
    OptionalAttr<I64Attr>:$fusion_count
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region VariadicRegion<AnyRegion>:$regions);
  let hasVerifier = 1;
}

def GraphForge_YieldOp : GraphForge_Op<"yield", [Terminator]> {
  let summary = "Yield a structured message or one node result";
  let arguments = (ins Variadic<AnyType>:$messages);
  let hasVerifier = 1;
}

#endif
