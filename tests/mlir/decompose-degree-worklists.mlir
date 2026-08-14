// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel -gf-select-kernel-schedule -gf-plan-degree-buckets -gf-decompose-degree-worklists | FileCheck %s

func.func @skewed(%row: tensor<?xi64>, %col: tensor<?xi64>,
                  %x: tensor<?xf32>) -> tensor<?xf32> {
  %r = "gf.relation"(%row, %col) {origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "skewed", version = 2 : i64,
    num_src = 64 : i64, num_dst = 64 : i64, degree_min = 0 : i64,
    degree_max = 64 : i64, degree_sum = 512 : i64} :
    (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %out = "gf.apply"(%r, %x) ({
  ^bb0(%value: f32):
    "gf.yield"(%value) : (f32) -> ()
  }) {reducers = ["sum"], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 1>, snapshot_versions = array<i64: 2>,
    effects = ["read"], deterministic = false} :
    (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  return %out : tensor<?xf32>
}

// CHECK-NOT: "gf_task.degree_worklist"
// CHECK: %[[RESET:.*]] = "gf_task.degree_reset"
// CHECK: %[[HIST:.*]] = "gf_task.degree_histogram"({{.*}}, {{.*}}, %[[RESET]])
// CHECK: %[[PREFIX:.*]] = "gf_task.degree_prefix"({{.*}}, %[[HIST]])
// CHECK: %[[ROWS:.*]], %[[SCATTER:.*]] = "gf_task.degree_scatter"({{.*}}, {{.*}}, %[[PREFIX]])
// CHECK-COUNT-4: "gf_task.degree_bucket_launch"(%[[ROWS]], %[[SCATTER]])
