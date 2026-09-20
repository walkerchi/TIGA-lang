// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel | gf-translate -gf-kernel-to-ttir | FileCheck %s

func.func @periodic_distance_sum(
    %positions: tensor<?x2xf32>, %cell_ptr: tensor<?xi64>,
    %order: tensor<?xi64>, %coordinates: tensor<?x2xi64>,
    %extents: tensor<2xi64>, %strides: tensor<2xi64>,
    %offsets: tensor<9x2xi64>, %lattice: tensor<2x2xf32>,
    %inverse: tensor<2x2xf32>, %x: tensor<?xf32>) -> tensor<?xf32> {
  %r = "gf.generated_radius"(%positions, %cell_ptr, %order, %coordinates,
      %extents, %strides, %offsets, %lattice, %inverse) {
    cutoff = 2.000000e-01 : f64, dimensions = 2 : i64,
    periodic = true, num_entities = 64 : i64, version = 5 : i64
  } : (tensor<?x2xf32>, tensor<?xi64>, tensor<?xi64>, tensor<?x2xi64>,
       tensor<2xi64>, tensor<2xi64>, tensor<9x2xi64>, tensor<2x2xf32>,
       tensor<2x2xf32>) -> !gf.relation
  %0 = "gf.apply"(%r, %x) ({
  ^bb0(%src: f32, %distance: f32):
    %message = arith.mulf %distance, %src : f32
    "gf.yield"(%message) : (f32) -> ()
  }) {
    reducers = ["sum"], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 1>, snapshot_versions = array<i64: 5>,
    effects = ["read"], deterministic = false
  } : (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK: tiga.launch entry=gf_generated_radius_distance_sum
// CHECK-SAME: neighbor_offsets,lattice,inverse_lattice,positions,x,out
// CHECK: %neighbor_coord_shifted0 = arith.addi
// CHECK: %neighbor_coord0 = arith.remui
// CHECK: %inverse_ptr0_0 = tt.addptr %inverse_lattice
// CHECK: %fractional_round0 = math.floor
// CHECK: %lattice_ptr0_0 = tt.addptr %lattice
// CHECK: %distance = math.sqrt
