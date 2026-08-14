// RUN: gf-opt %s -split-input-file -verify-diagnostics

func.func @unknown_traversal(%row: tensor<?xi64>, %col: tensor<?xi64>,
                             %x: tensor<?xf32>) -> tensor<?xf32> {
  %r = "gf.relation"(%row, %col) {
    origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "r0",
    version = 0 : i64, num_src = 16 : i64, num_dst = 16 : i64
  } : (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  // expected-error @+1 {{unknown target-independent traversal 'warp32'}}
  %0 = "gf_kernel.launch"(%row, %col, %x) ({
  ^bb0(%src: f32, %dst: f32):
    %message = arith.mulf %src, %dst : f32
    "gf_kernel.yield"(%message) : (f32) -> ()
  }) {
    num_rows = 16 : i64, traversal = "warp32", reducers = ["sum"],
    region_kinds = array<i64: 0>, input_segment_sizes = array<i64: 1>,
    snapshot_versions = array<i64: 0>, effects = ["read"],
    deterministic = false
  } : (tensor<?xi64>, tensor<?xi64>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}


// -----

func.func @incomplete_schedule(%row: tensor<?xi64>, %col: tensor<?xi64>,
                               %x: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{machine schedule contract must be either absent or complete}}
  %0 = "gf_kernel.launch"(%row, %col, %x) ({
  ^bb0(%src: f32):
    "gf_kernel.yield"(%src) : (f32) -> ()
  }) {
    num_rows = 16 : i64, traversal = "csr-row", reducers = ["sum"],
    region_kinds = array<i64: 0>, input_segment_sizes = array<i64: 1>,
    snapshot_versions = array<i64: 0>, effects = ["read"],
    deterministic = false, schedule_kind = "scalar-row-loop"
  } : (tensor<?xi64>, tensor<?xi64>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @invalid_schedule_resource(%row: tensor<?xi64>, %col: tensor<?xi64>,
                                     %x: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{unknown schedule_resources entry 'tensor-core.secret'}}
  %0 = "gf_kernel.launch"(%row, %col, %x) ({
  ^bb0(%src: f32):
    "gf_kernel.yield"(%src) : (f32) -> ()
  }) {
    num_rows = 16 : i64, traversal = "csr-row", reducers = ["sum"],
    region_kinds = array<i64: 0>, input_segment_sizes = array<i64: 1>,
    snapshot_versions = array<i64: 0>, effects = ["read"],
    deterministic = false,
    schedule_kind = "scalar-row-loop", block_rows = 1 : i64,
    block_neighbors = 1 : i64, num_warps = 1 : i64,
    pipeline_stages = 1 : i64, target_contract = "provider-neutral-v1",
    schedule_resources = ["tensor-core.secret"],
    execution_roles = ["workgroup.destination-rows", "lane.scalar"],
    schedule_handoffs = ["load.global-to-register", "store.register-to-global"],
    instruction_contracts = ["scalar-control-flow", "associative-reduce"]
  } : (tensor<?xi64>, tensor<?xi64>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
