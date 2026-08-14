// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel -gf-select-kernel-schedule | FileCheck %s --check-prefix=KERNEL
// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel -gf-select-kernel-schedule | gf-translate -gf-kernel-to-ttir | FileCheck %s --check-prefix=TTIR

func.func @ragged_vector_weighted_sum(
    %row: tensor<?xi64>, %col: tensor<?xi64>,
    %x: tensor<?x16xf32>, %weight: tensor<?xf32>) -> tensor<?x16xf32> {
  %r = "gf.relation"(%row, %col) {
    origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "ragged-vector",
    version = 0 : i64, num_src = 128 : i64, num_dst = 128 : i64,
    degree_min = 0 : i64, degree_max = 16 : i64
  } : (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %0 = "gf.apply"(%r, %x, %weight) ({
  ^bb0(%src: vector<16xf32>, %edge_weight: f32):
    %weight_vector = vector.broadcast %edge_weight : f32 to vector<16xf32>
    %message = arith.mulf %src, %weight_vector : vector<16xf32>
    "gf.yield"(%message) : (vector<16xf32>) -> ()
  }) {
    reducers = ["sum"], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 2>,
    snapshot_versions = array<i64: 0, 0>, effects = ["read", "read"],
    deterministic = false
  } : (!gf.relation, tensor<?x16xf32>, tensor<?xf32>) -> tensor<?x16xf32>
  return %0 : tensor<?x16xf32>
}

// KERNEL: "gf_kernel.launch"
// KERNEL-SAME: block_neighbors = 16 : i64
// KERNEL-SAME: block_rows = 32 : i64
// KERNEL-SAME: execution_roles = ["workgroup.destination-rows", "subgroup.neighbor-reduction", "lane.feature"]
// KERNEL-SAME: schedule_kind = "bounded-ragged-row-neighbor-feature"
// KERNEL: {block_features = 16 : i64}

// TTIR: graphforge.launch entry=gf_csr_weighted_sum block_rows=32 num_warps=4
// TTIR: %starts = tt.load %start_ptr
// TTIR: %edge_limit = arith.cmpi slt, %edges, %ends_b
// TTIR: %edge_mask = arith.andi %row_mask, %edge_limit
// TTIR: %message = arith.mulf %x_value, %weight_b : tensor<32x16x16xf32>
// TTIR: }) : (tensor<32x16x16xf32>) -> tensor<32x16xf32>
