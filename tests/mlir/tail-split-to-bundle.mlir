// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel -gf-select-kernel-schedule -gf-plan-degree-buckets -gf-decompose-degree-worklists | gf-translate -gf-task-to-bundle | FileCheck %s

// The <=32 partition contains only 486 rows, but its direct-filter kernel
// traverses the original 512-row domain with blockM=8.  Its grid must therefore
// be 64, not ceil(486/8)=61.
func.func @tail_split(%row: tensor<?xi64>, %col: tensor<?xi64>,
                      %x: tensor<?xf32>) -> tensor<?xf32> {
  %r = "gf.relation"(%row, %col) {origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "tail", version = 7 : i64,
    num_src = 512 : i64, num_dst = 512 : i64,
    degree_min = 8 : i64, degree_max = 64 : i64, degree_sum = 8000 : i64,
    degree_histogram = array<i64:
      0, 0, 0, 0, 0, 0, 0, 0, 384,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 102,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26>} :
    (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %out = "gf.apply"(%r, %x) ({
  ^bb0(%value: f32):
    "gf.yield"(%value) : (f32) -> ()
  }) {reducers = ["sum"], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 1>, snapshot_versions = array<i64: 7>,
    effects = ["read"], deterministic = false} :
    (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  return %out : tensor<?xf32>
}

// CHECK: "degree_upper_inclusive": 32
// CHECK: "grid_x": 64
// CHECK: "row_mapping": "direct-filter"
// CHECK: "rows_in_bucket": 486
// CHECK: "degree_upper_inclusive": 64
// CHECK: "grid_x": 7
// CHECK: "row_mapping": "worklist"
// CHECK: "rows_in_bucket": 26
