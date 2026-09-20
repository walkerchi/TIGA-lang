// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel -gf-select-kernel-schedule | gf-translate -gf-kernel-to-ttir | FileCheck %s

func.func @ragged_weighted_sum(%row: tensor<?xi64>, %col: tensor<?xi64>,
                               %x: tensor<?xf32>,
                               %weight: tensor<?xf32>) -> tensor<?xf32> {
  %r = "gf.relation"(%row, %col) {
    origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "ragged",
    version = 0 : i64, num_src = 128 : i64, num_dst = 128 : i64,
    degree_min = 0 : i64, degree_max = 37 : i64
  } : (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %0 = "gf.apply"(%r, %x, %weight) ({
  ^bb0(%src: f32, %edge_weight: f32):
    %message = arith.mulf %src, %edge_weight : f32
    "gf.yield"(%message) : (f32) -> ()
  }) {
    reducers = ["sum"], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 2>,
    snapshot_versions = array<i64: 0, 0>, effects = ["read", "read"],
    deterministic = false
  } : (!gf.relation, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK: tiga.launch entry=gf_csr_weighted_sum block_rows=32 num_warps=1
// CHECK: %starts = tt.load
// CHECK: %edges = arith.addi {{.*}} : tensor<32x64xi64>
// CHECK: %mask = arith.andi {{.*}} : tensor<32x64xi1>
// CHECK: %sum = "tt.reduce"
// CHECK: }) : (tensor<32x64xf32>) -> tensor<32xf32>
