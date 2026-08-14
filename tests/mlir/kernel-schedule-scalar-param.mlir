// RUN: gf-opt %s -gf-select-kernel-schedule | FileCheck %s

module {
  func.func @scalar_param(%row: tensor<?xi64>, %col: tensor<?xi64>,
                          %x: tensor<?xf32>, %scale: f32) -> tensor<?xf32> {
    %out = "gf_kernel.launch"(%row, %col, %x, %scale) ({
    ^bb0(%value: f32, %factor: f32):
      %message = arith.mulf %value, %factor : f32
      "gf_kernel.yield"(%message) : (f32) -> ()
    }) {num_rows = 32 : i64, degree_min = 16 : i64, degree_max = 16 : i64,
        traversal = "csr-row", reducers = ["sum"],
        region_kinds = array<i64: 0>, input_segment_sizes = array<i64: 2>,
        snapshot_versions = array<i64: 0, 0>, effects = ["read", "read"],
        deterministic = false, input_roles = ["src", "param"]}
        : (tensor<?xi64>, tensor<?xi64>, tensor<?xf32>, f32) -> tensor<?xf32>
    return %out : tensor<?xf32>
  }
}

// CHECK: block_neighbors = 16 : i64
// CHECK-SAME: schedule_kind = "fixed-row-neighbor"
// CHECK-NOT: provider-deferred
