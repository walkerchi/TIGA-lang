// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel -gf-select-kernel-schedule -gf-plan-degree-buckets -gf-decompose-degree-worklists | gf-translate -gf-task-to-bundle | FileCheck %s

func.func @skewed(%row: tensor<?xi64>, %col: tensor<?xi64>,
                  %x: tensor<?xf32>) -> tensor<?xf32> {
  %r = "gf.relation"(%row, %col) {origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "skewed", version = 3 : i64,
    num_src = 128 : i64, num_dst = 128 : i64,
    degree_min = 0 : i64, degree_max = 64 : i64,
    degree_sum = 1024 : i64} :
    (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %out = "gf.apply"(%r, %x) ({
  ^bb0(%value: f32):
    "gf.yield"(%value) : (f32) -> ()
  }) {reducers = ["sum"], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 1>, snapshot_versions = array<i64: 3>,
    effects = ["read"], deterministic = false} :
    (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  return %out : tensor<?xf32>
}

// CHECK: "executable_symbol": "__gf_degree_histogram"
// CHECK: "name": "degree-histogram:0"
// CHECK: "executable_symbol": "__gf_degree_reset"
// CHECK: "name": "degree-reset:0"
// CHECK: "executable_symbol": "__gf_degree_prefix"
// CHECK: "name": "degree-prefix:0"
// CHECK: "executable_symbol": "__gf_degree_scatter"
// CHECK: "name": "degree-scatter:0"
// CHECK: "partitioning": "degree-worklist"
// CHECK: "task_kind": "degree-bucket"
// CHECK-COUNT-3: "task_kind": "degree-bucket"
// CHECK: "depends_on": [
// CHECK-NEXT: {{ *}}"degree-bucket:0",
// CHECK-NEXT: {{ *}}"degree-bucket:1",
// CHECK-NEXT: {{ *}}"degree-bucket:2",
// CHECK-NEXT: {{ *}}"degree-bucket:3"
// CHECK: "task_kind": "join"
// CHECK: "resources": [
// CHECK: "capacity_bytes": 1096
// CHECK: "layout": "degree-row-worklist"
// CHECK: "name": "row_worklist"
// CHECK: "schema": "tiga.executable-bundle-plan.v1"
// CHECK: "terminals": [
// CHECK-NEXT: {{ *}}"join:0"
