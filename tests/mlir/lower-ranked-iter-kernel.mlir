// RUN: gf-opt %s -gf-lower-domain-to-iter | FileCheck %s --check-prefix=ITER
// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel -gf-select-kernel-schedule | FileCheck %s --check-prefix=KERNEL

func.func @ranked_distance_sum(%query: tensor<?x3xf32>,
                               %candidate: tensor<?x3xf32>,
                               %x: tensor<?xf32>) -> tensor<?xf32> {
  %relation = "gf.ranked_relation"(%query, %candidate) {
    num_queries = 64 : i64, num_candidates = 128 : i64,
    dimensions = 3 : i64, k = 16 : i64,
    metric = "squared_euclidean", selection = "smallest",
    tie_break = "source_index", exclude_self = false,
    same_entity_domain = false, exact = true, version = 0 : i64
  } : (tensor<?x3xf32>, tensor<?x3xf32>) -> !gf.relation
  %out = "gf.apply"(%relation, %x) ({
  ^bb0(%src: f32, %distance: f32):
    %message = arith.mulf %src, %distance : f32
    "gf.yield"(%message) : (f32) -> ()
  }) {
    reducers = ["sum"], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 1>,
    snapshot_versions = array<i64: 0>, effects = ["read"],
    deterministic = false
  } : (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  return %out : tensor<?xf32>
}

// ITER: "gf_iter.traverse"
// ITER-SAME: coordinate_hierarchy = "ranked-pairs"
// KERNEL: "gf_kernel.ranked_launch"
// KERNEL-SAME: candidate_tile = 256 : i64
// KERNEL-SAME: instruction_contracts = ["masked-memory", "hierarchical-topk", "stable-index-tie-break"]
// KERNEL-SAME: merge_fan_in = 2 : i64
// KERNEL-SAME: schedule_kind = "ranked-candidate-select"
// KERNEL-SAME: traversal = "ranked-candidate-tile"
