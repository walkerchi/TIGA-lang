#ifndef GRAPHFORGE_KERNEL_OPS_TD
#define GRAPHFORGE_KERNEL_OPS_TD

include "graphforge/Dialect/Kernel/KernelDialect.td"

def GraphForge_LaunchOp : GraphForge_KernelOp<"launch"> {
  let summary = "Execute captured local regions with a selected traversal";
  let description = [{
    `gf_kernel.launch` is provider-neutral. `traversal` selects a proven
    skeleton such as `csr-row` or `generated-tile`; vendor layout, subgroup,
    shared-memory encoding and assembly remain provider decisions. Regions
    preserve edge/node scalar algebra and terminate with `gf_kernel.yield`.
  }];

  let arguments = (ins
    AnyType:$row_ptr,
    AnyType:$col_idx,
    Variadic<AnyType>:$inputs,
    I64Attr:$num_rows,
    OptionalAttr<I64Attr>:$degree_min,
    OptionalAttr<I64Attr>:$degree_max,
    OptionalAttr<I64Attr>:$degree_sum,
    OptionalAttr<DenseI64ArrayAttr>:$degree_histogram,
    OptionalAttr<F64Attr>:$source_index_span_ratio,
    StrAttr:$traversal,
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
    OptionalAttr<StrAttr>:$schedule_kind,
    OptionalAttr<I64Attr>:$block_rows,
    OptionalAttr<I64Attr>:$block_neighbors,
    OptionalAttr<I64Attr>:$num_warps,
    OptionalAttr<I64Attr>:$pipeline_stages,
    OptionalAttr<StrAttr>:$target_contract,
    OptionalAttr<ArrayAttr>:$schedule_resources,
    OptionalAttr<ArrayAttr>:$execution_roles,
    OptionalAttr<ArrayAttr>:$schedule_handoffs,
    OptionalAttr<ArrayAttr>:$instruction_contracts
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region VariadicRegion<AnyRegion>:$regions);
  let hasVerifier = 1;
}

def GraphForge_GeneratedLaunchOp
    : GraphForge_KernelOp<"generated_launch"> {
  let summary = "Generate and consume relation tiles from a cell directory";
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
    Variadic<AnyType>:$inputs,
    I64Attr:$num_rows,
    F64Attr:$cutoff,
    I64Attr:$dimensions,
    BoolAttr:$periodic,
    DefaultValuedAttr<BoolAttr, "false">:$hash_grid,
    StrAttr:$traversal,
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
    OptionalAttr<StrAttr>:$schedule_kind,
    OptionalAttr<I64Attr>:$block_rows,
    OptionalAttr<I64Attr>:$block_neighbors,
    OptionalAttr<I64Attr>:$num_warps,
    OptionalAttr<I64Attr>:$pipeline_stages,
    OptionalAttr<StrAttr>:$target_contract,
    OptionalAttr<ArrayAttr>:$schedule_resources,
    OptionalAttr<ArrayAttr>:$execution_roles,
    OptionalAttr<ArrayAttr>:$schedule_handoffs,
    OptionalAttr<ArrayAttr>:$instruction_contracts
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region VariadicRegion<AnyRegion>:$regions);
  let hasVerifier = 1;
}

def GraphForge_DenseLaunchOp : GraphForge_KernelOp<"dense_launch"> {
  let summary = "Traverse an implicit Cartesian relation in dense tiles";
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    I64Attr:$num_src,
    I64Attr:$num_dst,
    StrAttr:$traversal,
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
    OptionalAttr<StrAttr>:$schedule_kind,
    OptionalAttr<I64Attr>:$block_rows,
    OptionalAttr<I64Attr>:$block_neighbors,
    OptionalAttr<I64Attr>:$num_warps,
    OptionalAttr<I64Attr>:$pipeline_stages,
    OptionalAttr<StrAttr>:$target_contract,
    OptionalAttr<ArrayAttr>:$schedule_resources,
    OptionalAttr<ArrayAttr>:$execution_roles,
    OptionalAttr<ArrayAttr>:$schedule_handoffs,
    OptionalAttr<ArrayAttr>:$instruction_contracts
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region VariadicRegion<AnyRegion>:$regions);
  let hasVerifier = 1;
}

def GraphForge_RankedLaunchOp : GraphForge_KernelOp<"ranked_launch"> {
  let summary = "Build and consume a ranked relation without materializing CSR";
  let description = [{
    A ranked launch preserves exact selection semantics while exposing the
    physical candidate tile and merge fan-in selected by compiler passes.
    Selected source indices are internal values: consumers run on the retained
    pairs and no global adjacency allocation is part of this ABI.
  }];
  let arguments = (ins
    AnyType:$query_positions,
    AnyType:$candidate_positions,
    Variadic<AnyType>:$inputs,
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
    StrAttr:$traversal,
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
    OptionalAttr<StrAttr>:$schedule_kind,
    OptionalAttr<I64Attr>:$block_rows,
    OptionalAttr<I64Attr>:$block_neighbors,
    OptionalAttr<I64Attr>:$num_warps,
    OptionalAttr<I64Attr>:$pipeline_stages,
    OptionalAttr<StrAttr>:$target_contract,
    OptionalAttr<ArrayAttr>:$schedule_resources,
    OptionalAttr<ArrayAttr>:$execution_roles,
    OptionalAttr<ArrayAttr>:$schedule_handoffs,
    OptionalAttr<ArrayAttr>:$instruction_contracts,
    OptionalAttr<I64Attr>:$candidate_tile,
    OptionalAttr<I64Attr>:$merge_fan_in
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region VariadicRegion<AnyRegion>:$regions);
  let hasVerifier = 1;
}

def GraphForge_YieldOp
    : GraphForge_KernelOp<"yield", [Terminator]> {
  let summary = "Yield a structured local message or one node result";
  let arguments = (ins Variadic<AnyType>:$values);
  let hasVerifier = 1;
}

#endif
