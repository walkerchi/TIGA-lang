// RUN: gf-opt %s -split-input-file -verify-diagnostics

func.func @bad_bounds(%row: tensor<?xi64>, %col: tensor<?xi64>) {
  %r = "gf.relation"(%row, %col) {origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "r", version = 0 : i64,
    num_src = 16 : i64, num_dst = 16 : i64, degree_max = 16 : i64} :
    (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %region = "gf_storage.region"(%r) {logical_id = "rows", version = 0 : i64,
    elements = 23 : i64} : (!gf.relation) -> !gf_storage.region
  %instance = "gf_storage.instance"(%region) {memory_space = "device",
    layout = "degree-row-worklist", device = "provider:0",
    capacity_bytes = 184 : i64, external = false} :
    (!gf_storage.region) -> !gf_storage.instance
  // expected-error@+1 {{degree upper bounds must be strictly increasing}}
  %worklist, %ready = "gf_task.degree_worklist"(%r, %instance) {
    upper_bounds = array<i64: 8, 8, 16>, rows = 16 : i64,
    snapshot_version = 0 : i64} : (!gf.relation, !gf_storage.instance) ->
    (!gf_task.degree_worklist, !gf_storage.event)
  return
}

// -----

func.func @external_storage(%row: tensor<?xi64>, %col: tensor<?xi64>) {
  %r = "gf.relation"(%row, %col) {origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "r", version = 0 : i64,
    num_src = 16 : i64, num_dst = 16 : i64, degree_max = 16 : i64} :
    (tensor<?xi64>, tensor<?xi64>) -> !gf.relation
  %region = "gf_storage.region"(%r) {logical_id = "rows", version = 0 : i64,
    elements = 21 : i64} : (!gf.relation) -> !gf_storage.region
  %instance = "gf_storage.instance"(%region) {memory_space = "device",
    layout = "degree-row-worklist", device = "provider:0",
    capacity_bytes = 168 : i64, external = true} :
    (!gf_storage.region) -> !gf_storage.instance
  // expected-error@+1 {{degree worklist storage must be compiler-owned}}
  %worklist, %ready = "gf_task.degree_worklist"(%r, %instance) {
    upper_bounds = array<i64: 8, 16>, rows = 16 : i64,
    snapshot_version = 0 : i64} : (!gf.relation, !gf_storage.instance) ->
    (!gf_task.degree_worklist, !gf_storage.event)
  return
}
