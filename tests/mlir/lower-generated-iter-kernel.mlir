// RUN: gf-opt %s -gf-lower-domain-to-iter | FileCheck %s --check-prefix=ITER
// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel | FileCheck %s --check-prefix=KERNEL

func.func @generated_with_node(%positions: tensor<?x3xf32>,
                               %cell_ptr: tensor<?xi64>,
                               %order: tensor<?xi64>,
                               %coordinates: tensor<?x3xi64>,
                               %extents: tensor<3xi64>,
                               %strides: tensor<3xi64>,
                               %offsets: tensor<27x3xi64>,
                               %lattice: tensor<3x3xf32>,
                               %inverse: tensor<3x3xf32>,
                               %x: tensor<?xf32>) -> tensor<?xf32> {
  %r = "gf.generated_radius"(%positions, %cell_ptr, %order, %coordinates,
                              %extents, %strides, %offsets, %lattice, %inverse) {
    cutoff = 1.250000e-01 : f64, dimensions = 3 : i64,
    periodic = false, num_entities = 16 : i64, version = 3 : i64
  } : (tensor<?x3xf32>, tensor<?xi64>, tensor<?xi64>, tensor<?x3xi64>,
       tensor<3xi64>, tensor<3xi64>, tensor<27x3xi64>, tensor<3x3xf32>,
       tensor<3x3xf32>) -> !gf.relation
  %0 = "gf.apply"(%r, %x, %x) ({
  ^bb0(%src: f32, %dst: f32):
    %message = arith.subf %src, %dst : f32
    "gf.yield"(%message) : (f32) -> ()
  }, {
  ^bb0(%old: f32, %aggregate: f32):
    %updated = arith.addf %old, %aggregate : f32
    "gf.yield"(%updated) : (f32) -> ()
  }) {
    reducers = ["sum"], region_kinds = array<i64: 0, 1>,
    input_segment_sizes = array<i64: 2>,
    snapshot_versions = array<i64: 3, 3>, effects = ["read"],
    deterministic = false
  } : (!gf.relation, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// ITER: "gf_iter.traverse"
// ITER-SAME: coordinate_hierarchy = "generated-neighborhood"
// ITER-SAME: region_kinds = array<i64: 0, 1>
// ITER-COUNT-2: "gf_iter.yield"

// KERNEL: "gf_kernel.generated_launch"
// KERNEL-SAME: region_kinds = array<i64: 0, 1>
// KERNEL-SAME: traversal = "generated-tile"
// KERNEL-COUNT-2: "gf_kernel.yield"
