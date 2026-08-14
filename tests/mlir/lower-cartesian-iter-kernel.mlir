// RUN: gf-opt %s -gf-lower-domain-to-iter | FileCheck %s --check-prefix=ITER
// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel -gf-select-kernel-schedule | FileCheck %s --check-prefix=KERNEL

func.func @dense_sum(%x: tensor<?xf32>) -> tensor<?xf32> {
  %relation = "gf.cartesian"() {
    num_src = 128 : i64, num_dst = 64 : i64, version = 0 : i64
  } : () -> !gf.relation
  %out = "gf.apply"(%relation, %x) ({
  ^bb0(%src: f32):
    "gf.yield"(%src) : (f32) -> ()
  }) {
    reducers = ["sum"], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 1>,
    snapshot_versions = array<i64: 0>, effects = ["read"],
    deterministic = false
  } : (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  return %out : tensor<?xf32>
}

// ITER: "gf_iter.traverse"
// ITER-SAME: coordinate_hierarchy = "cartesian-product"
// KERNEL: "gf_kernel.dense_launch"
// KERNEL-SAME: block_neighbors = 64 : i64
// KERNEL-SAME: block_rows = 128 : i64
// KERNEL-SAME: execution_roles = ["workgroup.destination-queries", "subgroup.key-reduction", "lane.feature"]
// KERNEL-SAME: instruction_contracts = ["masked-memory", "vector-contraction", "associative-reduce"]
// KERNEL-SAME: schedule_kind = "dense-query-key-tile"
// KERNEL-SAME: traversal = "dense-tile"
