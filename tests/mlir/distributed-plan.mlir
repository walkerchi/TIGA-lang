// RUN: gf-opt %s -gf-lower-domain-to-iter -gf-lower-iter-to-kernel \
// RUN:   -gf-select-kernel-schedule -gf-plan-distributed-tasks | FileCheck %s

module {
  "gf.reducer"() ({
    %zero = arith.constant 0.0 : f32
    "gf.reducer_yield"(%zero) : (f32) -> ()
  }, {
  ^bb0(%message: f32):
    "gf.reducer_yield"(%message) : (f32) -> ()
  }, {
  ^bb0(%left: f32, %right: f32):
    %sum = arith.addf %left, %right : f32
    "gf.reducer_yield"(%sum) : (f32) -> ()
  }, {
  ^bb0(%state: f32):
    "gf.reducer_yield"(%state) : (f32) -> ()
  }) {sym_name = "sum", kind = "algebraic", message_types = [f32],
      state_types = [f32], result_types = [f32], associative = true,
      commutative = true} : () -> ()

  func.func @distributed(%row: tensor<?xi64>, %col: tensor<?xi64>,
                         %x: tensor<?xf32>, %weight: tensor<?xf32>)
      -> tensor<?xf32> {
    %relation = "gf.relation"(%row, %col) {
      origin = "external", lifecycle = "frozen", realization = "materialized",
      relation_id = "mesh", version = 3 : i64,
      num_src = 1024 : i64, num_dst = 1024 : i64,
      degree_min = 4 : i64, degree_max = 16 : i64,
      mesh_shape = array<i64: 2, 4>, mesh_axis = 1 : i64,
      halo_depth = -1 : i64, partition_balance = "edges"
    } : (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
    %out = "gf.apply"(%relation, %x, %weight) ({
    ^bb0(%source: f32, %edge: f32):
      %message = arith.mulf %source, %edge : f32
      "gf.yield"(%message) : (f32) -> ()
    }) {reducers = [@sum], region_kinds = array<i64: 0>,
        input_segment_sizes = array<i64: 2>, snapshot_versions = array<i64: 3, 3>,
        effects = ["read", "read"], deterministic = false,
        input_roles = ["src", "edge"], input_names = ["x", "weight"]
    } : (!gf.relation, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    return %out : tensor<?xf32>
  }
}

// CHECK: %[[PARTITION:.*]] = "gf_task.partition"
// CHECK-SAME: ghost_map_algorithm = "csr-owned-row-remote-src-sort-unique"
// CHECK-SAME: neighbor_discovery = "alltoall-counts-then-request-ids"
// CHECK-SAME: owner_map_algorithm = "contiguous-destination-prefix"
// CHECK: %[[HALO:.*]] = "gf_task.halo"(%[[PARTITION]])
// CHECK: %[[SEND:.*]], %[[PACK:.*]] = "gf_task.halo_pack"(%[[HALO]]
// CHECK-SAME: bytes = 3072 : i64
// CHECK: %[[RECV:.*]], %[[EXCHANGE:.*]] = "gf_task.halo_exchange"(%[[HALO]], %[[SEND]], %[[PACK]])
// CHECK-SAME: bytes = 3072 : i64
// CHECK: %[[UNPACK:.*]] = "gf_task.halo_unpack"(%[[HALO]], %[[RECV]]
// CHECK: %[[INTERIOR:.*]] = "gf_task.launch"(%[[PARTITION]])
// CHECK-SAME: task_kind = "interior"
// CHECK: %[[BOUNDARY:.*]] = "gf_task.launch"(%[[PARTITION]], %[[UNPACK]])
// CHECK-SAME: task_kind = "boundary"
// CHECK: "gf_storage.join"(%[[INTERIOR]], %[[BOUNDARY]])
// CHECK: "gf_kernel.launch"
// CHECK: gf_task.planned
