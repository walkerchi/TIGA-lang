// RUN: gf-opt %s -gf-form-apply-fusion-groups | FileCheck %s --check-prefix=GROUP
// RUN: gf-opt %s -gf-form-apply-fusion-groups -gf-form-apply-fusion-groups | FileCheck %s --check-prefix=GROUP
// RUN: gf-opt %s -gf-fuse-compatible-applies | FileCheck %s --check-prefix=FUSE
// RUN: gf-opt %s -gf-fuse-compatible-applies -gf-fuse-compatible-applies | FileCheck %s --check-prefix=FUSE
// RUN: gf-opt %s -gf-fuse-compatible-applies -gf-lower-domain-to-iter -gf-lower-iter-to-kernel | FileCheck %s --check-prefix=KERNEL

func.func @horizontal(%row: tensor<?xi64>, %col: tensor<?xi64>,
                      %x: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
  %r = "gf.relation"(%row, %col) {
    origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "r0",
    version = 0 : i64, num_src = 16 : i64, num_dst = 16 : i64
  } : (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %sum = "gf.apply"(%r, %x) ({
  ^bb0(%src: f32, %dst: f32):
    %message = arith.subf %src, %dst : f32
    "gf.yield"(%message) : (f32) -> ()
  }) {
    reducers = ["sum"], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 1>,
    snapshot_versions = array<i64: 0>, effects = ["read"],
    deterministic = false
  } : (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  %max = "gf.apply"(%r, %x) ({
  ^bb0(%src: f32, %dst: f32):
    %delta = arith.subf %src, %dst : f32
    %message = math.absf %delta : f32
    "gf.yield"(%message) : (f32) -> ()
  }) {
    reducers = ["max"], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 1>,
    snapshot_versions = array<i64: 0>, effects = ["read"],
    deterministic = false
  } : (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  return %sum, %max : tensor<?xf32>, tensor<?xf32>
}

// GROUP-COUNT-2: fusion_group = 0 : i64
// FUSE-COUNT-1: "gf.apply"
// FUSE: fusion_count = 2 : i64
// FUSE: reducers = ["sum", "max"]
// KERNEL-COUNT-1: "gf_kernel.launch"
// KERNEL: reducers = ["sum", "max"]
// KERNEL: region_kinds = array<i64: 0, 0>
// KERNEL: traversal = "csr-row"
// KERNEL-COUNT-2: "gf_kernel.yield"
