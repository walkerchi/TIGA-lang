// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel | gf-translate -gf-kernel-to-ttir | FileCheck %s

func.func @weighted_sum(%row: tensor<?xi64>, %col: tensor<?xi64>,
                        %x: tensor<?xf32>,
                        %weight: tensor<?xf32>) -> tensor<?xf32> {
  %r = "gf.relation"(%row, %col) {
    origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "r0",
    version = 0 : i64, num_src = 16 : i64, num_dst = 16 : i64
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

// CHECK: tt.func public @gf_csr_weighted_sum
// CHECK-SAME: %row_ptr: !tt.ptr<i64>
// CHECK-SAME: %col_idx: !tt.ptr<i64>
// CHECK-SAME: %x: !tt.ptr<f32>
// CHECK-SAME: %weight: !tt.ptr<f32>
// CHECK: %sum = scf.for
// CHECK: %message = arith.mulf %x_value, %w_value : f32
// CHECK: tt.store %out_ptr, %sum
