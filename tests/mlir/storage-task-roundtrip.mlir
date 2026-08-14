// RUN: gf-opt %s | FileCheck %s
// RUN: gf-translate --gf-task-to-bundle %s | FileCheck %s --check-prefix=BUNDLE

module {
  func.func private @interior()
  func.func private @boundary()

  func.func @halo_pipeline(%row: tensor<?xi64>, %col: tensor<?xi64>,
                           %host: tensor<?xf32>, %device: tensor<?xf32>) {
    %relation = "gf.relation"(%row, %col) {
      origin = "external", lifecycle = "frozen",
      realization = "materialized", relation_id = "mesh",
      version = 7 : i64, num_src = 1024 : i64, num_dst = 1024 : i64
    } : (tensor<?xi64>, tensor<?xi64>) -> !gf.relation

    %host_region = "gf_storage.region"(%host) {
      logical_id = "u", version = 7 : i64, elements = 1024 : i64
    } : (tensor<?xf32>) -> !gf_storage.region
    %device_region = "gf_storage.region"(%device) {
      logical_id = "u", version = 7 : i64, elements = 1024 : i64
    } : (tensor<?xf32>) -> !gf_storage.region
    %ram = "gf_storage.instance"(%host_region) {
      memory_space = "pinned_ram", layout = "contiguous", device = "host:0",
      capacity_bytes = 4096 : i64, external = true
    } : (!gf_storage.region) -> !gf_storage.instance
    %hbm = "gf_storage.instance"(%device_region) {
      memory_space = "hbm", layout = "contiguous", device = "cuda:0",
      capacity_bytes = 4096 : i64, external = true
    } : (!gf_storage.region) -> !gf_storage.instance
    %ready = "gf_storage.transfer"(%ram, %hbm) {
      bytes = 4096 : i64, engine = "dma", snapshot_version = 7 : i64
    } : (!gf_storage.instance, !gf_storage.instance) -> !gf_storage.event

    %partition = "gf_task.partition"(%relation) {
      scheme = "destination", mesh_shape = array<i64: 2, 4>,
      mesh_axis = 1 : i64, version = 7 : i64,
      owned_entities = 128 : i64, ghost_entities = 32 : i64
    } : (!gf.relation) -> !gf_task.partition
    %halo = "gf_task.halo"(%partition) {
      depth = -1 : i64, fields = ["u"], strategy = "auto"
    } : (!gf_task.partition) -> !gf_task.halo_plan

    %send, %pack = "gf_task.halo_pack"(%halo, %hbm, %ready) {
      field = "u", bytes = 512 : i64, snapshot_version = 7 : i64
    } : (!gf_task.halo_plan, !gf_storage.instance, !gf_storage.event)
        -> (!gf_task.halo_buffer, !gf_storage.event)
    %receive, %exchange = "gf_task.halo_exchange"(%halo, %send, %pack) {
      group = "mesh", bytes = 512 : i64, snapshot_version = 7 : i64
    } : (!gf_task.halo_plan, !gf_task.halo_buffer, !gf_storage.event)
        -> (!gf_task.halo_buffer, !gf_storage.event)
    %unpack = "gf_task.halo_unpack"(%halo, %receive, %hbm, %exchange) {
      field = "u", snapshot_version = 7 : i64
    } : (!gf_task.halo_plan, !gf_task.halo_buffer,
         !gf_storage.instance, !gf_storage.event) -> !gf_storage.event

    // Interior work depends only on local residency, so it can overlap exchange.
    %interior = "gf_task.launch"(%partition, %ready) {
      task_kind = "interior", callee = @interior,
      reads = ["owned:u"], writes = ["owned:out"],
      snapshot_version = 7 : i64, operandSegmentSizes = array<i32: 1, 1>
    } : (!gf_task.partition, !gf_storage.event) -> !gf_storage.event
    %boundary = "gf_task.launch"(%partition, %unpack) {
      task_kind = "boundary", callee = @boundary,
      reads = ["owned:u", "ghost:u"], writes = ["owned:out"],
      snapshot_version = 7 : i64, operandSegmentSizes = array<i32: 1, 1>
    } : (!gf_task.partition, !gf_storage.event) -> !gf_storage.event
    %done = "gf_storage.join"(%interior, %boundary) {
      snapshot_version = 7 : i64
    } : (!gf_storage.event, !gf_storage.event) -> !gf_storage.event
    %released = "gf_storage.release"(%hbm, %done) {
      snapshot_version = 7 : i64
    } : (!gf_storage.instance, !gf_storage.event) -> !gf_storage.event
    return
  }
}

// CHECK: %[[READY:.*]] = "gf_storage.transfer"
// CHECK: %[[PARTITION:.*]] = "gf_task.partition"
// CHECK: "gf_task.halo"(%[[PARTITION]])
// CHECK: %[[SEND:.*]], %[[PACK:.*]] = "gf_task.halo_pack"
// CHECK: %[[RECEIVE:.*]], %[[EXCHANGE:.*]] = "gf_task.halo_exchange"
// CHECK: %[[UNPACK:.*]] = "gf_task.halo_unpack"
// CHECK: %[[INTERIOR:.*]] = "gf_task.launch"(%[[PARTITION]], %[[READY]])
// CHECK: %[[BOUNDARY:.*]] = "gf_task.launch"(%[[PARTITION]], %[[UNPACK]])
// CHECK: %[[DONE:.*]] = "gf_storage.join"(%[[INTERIOR]], %[[BOUNDARY]])
// CHECK: "gf_storage.release"({{.*}}, %[[DONE]])

// BUNDLE: "task_kind": "storage-transfer"
// BUNDLE: "task_kind": "halo-pack"
// BUNDLE: "task_kind": "halo-exchange"
// BUNDLE: "task_kind": "halo-unpack"
// BUNDLE: "resource": "u@hbm:cuda:0"
// BUNDLE: "name": "halo-send:0"
// BUNDLE: "name": "halo-receive:0"
