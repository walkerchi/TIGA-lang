// RUN: not gf-opt %s -split-input-file 2>&1 | FileCheck %s

func.func @bad_partition_version(%row: tensor<?xi64>, %col: tensor<?xi64>) {
  %relation = "gf.relation"(%row, %col) {
    origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "mesh",
    version = 3 : i64, num_src = 16 : i64, num_dst = 16 : i64
  } : (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  // expected-error @+1 {{partition version does not match relation snapshot}}
  %partition = "gf_task.partition"(%relation) {
    scheme = "destination", mesh_shape = array<i64: 2>, mesh_axis = 0 : i64,
    version = 4 : i64, owned_entities = 8 : i64, ghost_entities = 2 : i64
  } : (!gf.relation) -> !gf_task.partition
  return
}

// CHECK: partition version does not match relation snapshot

// -----

func.func @bad_transfer_version(%host: tensor<?xf32>, %device: tensor<?xf32>) {
  %r0 = "gf_storage.region"(%host) {
    logical_id = "u", version = 2 : i64, elements = 16 : i64
  } : (tensor<?xf32>) -> !gf_storage.region
  %r1 = "gf_storage.region"(%device) {
    logical_id = "u", version = 2 : i64, elements = 16 : i64
  } : (tensor<?xf32>) -> !gf_storage.region
  %ram = "gf_storage.instance"(%r0) {
    memory_space = "ram", layout = "contiguous", device = "host:0",
    capacity_bytes = 64 : i64, external = true
  } : (!gf_storage.region) -> !gf_storage.instance
  %hbm = "gf_storage.instance"(%r1) {
    memory_space = "hbm", layout = "contiguous", device = "cuda:0",
    capacity_bytes = 64 : i64, external = true
  } : (!gf_storage.region) -> !gf_storage.instance
  // expected-error @+1 {{instance version does not match snapshot_version}}
  %ready = "gf_storage.transfer"(%ram, %hbm) {
    bytes = 64 : i64, engine = "dma", snapshot_version = 3 : i64
  } : (!gf_storage.instance, !gf_storage.instance) -> !gf_storage.event
  return
}

// CHECK: instance version does not match snapshot_version

// -----

func.func @bad_halo_depth(%row: tensor<?xi64>, %col: tensor<?xi64>) {
  %relation = "gf.relation"(%row, %col) {
    origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "mesh",
    version = 0 : i64, num_src = 16 : i64, num_dst = 16 : i64
  } : (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %partition = "gf_task.partition"(%relation) {
    scheme = "destination", mesh_shape = array<i64: 2>, mesh_axis = 0 : i64,
    version = 0 : i64, owned_entities = 8 : i64, ghost_entities = 2 : i64
  } : (!gf.relation) -> !gf_task.partition
  // expected-error @+1 {{depth must be non-negative or -1 for auto}}
  %halo = "gf_task.halo"(%partition) {
    depth = -2 : i64, fields = ["u"], strategy = "auto"
  } : (!gf_task.partition) -> !gf_task.halo_plan
  return
}

// CHECK: depth must be non-negative or -1 for auto

// -----

func.func private @boundary()

func.func @bad_event_version() {
  %pack = "gf_task.launch"() {
    task_kind = "halo-pack", reads = ["owned:u"], writes = ["send:u"],
    snapshot_version = 5 : i64, operandSegmentSizes = array<i32: 0, 0>
  } : () -> !gf_storage.event
  // expected-error @+1 {{dependency event belongs to another snapshot version}}
  %boundary = "gf_task.launch"(%pack) {
    task_kind = "boundary", callee = @boundary,
    reads = ["ghost:u"], writes = ["owned:out"],
    snapshot_version = 6 : i64, operandSegmentSizes = array<i32: 0, 1>
  } : (!gf_storage.event) -> !gf_storage.event
  return
}

// CHECK: dependency event belongs to another snapshot version

// -----

func.func @bad_cross_dialect_join_version() {
  %task = "gf_task.launch"() {
    task_kind = "prefetch", reads = [], writes = ["owned:u"],
    snapshot_version = 5 : i64, operandSegmentSizes = array<i32: 0, 0>
  } : () -> !gf_storage.event
  // expected-error @+1 {{cannot join events from another snapshot version}}
  %joined = "gf_storage.join"(%task) {
    snapshot_version = 6 : i64
  } : (!gf_storage.event) -> !gf_storage.event
  return
}

// CHECK: cannot join events from another snapshot version
