// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel | gf-translate -gf-kernel-to-ttir | FileCheck %s

func.func @hash_radius(
    %p: tensor<?x3xf32>, %ptr: tensor<?xi64>, %order: tensor<?xi64>,
    %coords: tensor<?x3xi64>, %extents: tensor<3xi64>, %strides: tensor<3xi64>,
    %offsets: tensor<27x3xi64>, %lattice: tensor<3x3xf32>,
    %inverse: tensor<3x3xf32>, %x: tensor<?xf32>) -> tensor<?xf32> {
  %r = "gf.generated_radius"(%p, %ptr, %order, %coords, %extents, %strides,
                              %offsets, %lattice, %inverse) {
    cutoff = 0.25 : f64, dimensions = 3 : i64, periodic = false,
    hash_grid = true, num_entities = 16 : i64, version = 0 : i64
  } : (tensor<?x3xf32>, tensor<?xi64>, tensor<?xi64>, tensor<?x3xi64>,
       tensor<3xi64>, tensor<3xi64>, tensor<27x3xi64>, tensor<3x3xf32>,
       tensor<3x3xf32>) -> !gf.relation
  %0 = "gf.apply"(%r, %x) ({
  ^bb0(%src: f32, %distance: f32):
    %message = arith.mulf %distance, %src : f32
    "gf.yield"(%message) : (f32) -> ()
  }) {
    reducers = ["sum"], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 1>, snapshot_versions = array<i64: 0>,
    effects = ["read"], deterministic = false
  } : (!gf.relation, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// Queries are scheduled in bucket order, but outputs retain original node IDs.
// CHECK: %row_order_ptr = tt.addptr %particle_order
// CHECK: %row = tt.load %row_order_ptr
// Hash addressing wraps bucket coordinates only, not Euclidean displacement.
// CHECK: %neighbor_coord0 = arith.andi
// CHECK-NOT: %fractional0_0
// CHECK: %distance = math.sqrt
// CHECK: tt.store %out_ptr, %cell_sum
