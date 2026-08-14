// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel -gf-select-kernel-schedule | FileCheck %s

func.func @skewed(%row: tensor<?xi64>, %col: tensor<?xi64>,
                  %x: tensor<?xf32>) -> tensor<?xf32> {
  %r = "gf.relation"(%row, %col) {origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "skewed", version = 0 : i64,
    num_src = 128 : i64, num_dst = 128 : i64,
    degree_min = 8 : i64, degree_max = 64 : i64,
    degree_sum = 2048 : i64} :
    (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %out = "gf.apply"(%r, %x) ({
  ^bb0(%value: f32):
    "gf.yield"(%value) : (f32) -> ()
  }) {reducers = ["sum"], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 1>, snapshot_versions = array<i64: 0>,
    effects = ["read"], deterministic = false} :
    (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  return %out : tensor<?xf32>
}

// CHECK: degree_sum = 2048 : i64
// CHECK: load_balance_plan = "degree-bucketing-required"
// CHECK-SAME: padding_utilization_milli = 250 : i64

// A planning annotation is not permission for provider translation to invent
// a private row worklist.  A realization pass must materialize that ABI first.
