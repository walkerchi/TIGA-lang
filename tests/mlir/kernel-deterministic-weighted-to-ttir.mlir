// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel -gf-select-kernel-schedule | gf-translate -gf-kernel-to-ttir | FileCheck %s

func.func @strict(%row: tensor<?xi64>, %col: tensor<?xi64>,
                  %x: tensor<?xf32>, %w: tensor<?xf32>) -> tensor<?xf32> {
  %r = "gf.relation"(%row, %col) {origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "r", version = 0 : i64,
    num_src = 8 : i64, num_dst = 8 : i64,
    degree_min = 4 : i64, degree_max = 4 : i64} :
    (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %out = "gf.apply"(%r, %x, %w) ({
  ^bb0(%value: f32, %weight: f32):
    %message = arith.mulf %value, %weight : f32
    "gf.yield"(%message) : (f32) -> ()
  }) {reducers = ["sum"], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 2>, snapshot_versions = array<i64: 0, 0>,
    effects = ["read", "read"], deterministic = true} :
    (!gf.relation, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %out : tensor<?xf32>
}

// CHECK: tiga.launch entry=gf_csr_weighted_sum block_rows=1
// CHECK: %sum = scf.for
// CHECK-NOT: "tt.reduce"
