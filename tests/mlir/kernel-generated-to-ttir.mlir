// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel | gf-translate -gf-kernel-to-ttir | FileCheck %s

func.func @generated_distance_sum(
    %positions: tensor<?x3xf32>, %cell_ptr: tensor<?xi64>,
    %order: tensor<?xi64>, %coordinates: tensor<?x3xi64>,
    %extents: tensor<3xi64>, %strides: tensor<3xi64>,
    %offsets: tensor<27x3xi64>, %lattice: tensor<3x3xf32>,
    %inverse: tensor<3x3xf32>, %x: tensor<?xf32>) -> tensor<?xf32> {
  %r = "gf.generated_radius"(%positions, %cell_ptr, %order, %coordinates,
                              %extents, %strides, %offsets, %lattice, %inverse) {
    cutoff = 1.250000e-01 : f64, dimensions = 3 : i64,
    periodic = false, num_entities = 16 : i64, version = 3 : i64
  } : (tensor<?x3xf32>, tensor<?xi64>, tensor<?xi64>, tensor<?x3xi64>,
       tensor<3xi64>, tensor<3xi64>, tensor<27x3xi64>, tensor<3x3xf32>,
       tensor<3x3xf32>) -> !gf.relation
  %0 = "gf.apply"(%r, %x) ({
  ^bb0(%src: f32, %distance: f32):
    %message = arith.mulf %distance, %src : f32
    "gf.yield"(%message) : (f32) -> ()
  }) {
    reducers = ["sum"], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 1>, snapshot_versions = array<i64: 3>,
    effects = ["read"], deterministic = false
  } : (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK: graphforge.launch entry=gf_generated_radius_distance_sum
// CHECK-SAME: block_rows=1 num_warps=1
// CHECK-SAME: abi=cell_ptr,particle_order,cell_coordinates,extents,strides,neighbor_offsets,lattice,inverse_lattice,positions,x,out
// CHECK: tt.func public @gf_generated_radius_distance_sum
// CHECK: %cell_sum = scf.for %neighbor
// CHECK: %slot_sum = scf.for %tile
// CHECK: %slots = arith.addi %tile_v, %lane : tensor<32xi64>
// CHECK: %distance = math.sqrt {{.*}} : tensor<32xf32>
// CHECK: %tile_sum = "tt.reduce"
// CHECK: tt.store %out_ptr, %cell_sum
