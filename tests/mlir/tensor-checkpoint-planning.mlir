// RUN: gf-opt --gf-plan-tensor-checkpoints='memory-budget-bytes=128' %s | FileCheck %s --check-prefix=ONE
// RUN: gf-opt --gf-plan-tensor-checkpoints='memory-budget-bytes=0' %s | FileCheck %s --check-prefix=NONE
// RUN: gf-opt --gf-plan-tensor-checkpoints='memory-budget-bytes=128 spill-budget-bytes=128' %s | FileCheck %s --check-prefix=SPILL

// ONE: module attributes {
// ONE-SAME: gf_tensor.checkpoint_candidate_count = 2 : i64
// ONE-SAME: gf_tensor.checkpoint_decisions = array<i64: 1, 0>
// ONE-SAME: gf_tensor.checkpoint_memory_budget_bytes = 128 : i64
// ONE-SAME: gf_tensor.checkpoint_peak_live_bytes = 128 : i64
// ONE-SAME: gf_tensor.checkpoint_recompute_costs = array<i64: 256, 256>
// ONE-SAME: gf_tensor.checkpoint_saved_bytes = 128 : i64
// ONE-SAME: gf_tensor.checkpoint_tiers = ["device", "recompute"]
// ONE-COUNT-1: "gf_tensor.checkpoint"
// ONE-NOT: "gf_tensor.checkpoint_candidate"

// NONE: module attributes {
// NONE-SAME: gf_tensor.checkpoint_candidate_count = 2 : i64
// NONE-SAME: gf_tensor.checkpoint_decisions = array<i64: 0, 0>
// NONE-SAME: gf_tensor.checkpoint_memory_budget_bytes = 0 : i64
// NONE-SAME: gf_tensor.checkpoint_saved_bytes = 0 : i64
// NONE-NOT: "gf_tensor.checkpoint"

// SPILL: module attributes {
// SPILL-SAME: gf_tensor.checkpoint_decisions = array<i64: 1, 2>
// SPILL-SAME: gf_tensor.checkpoint_peak_live_bytes = 128 : i64
// SPILL-SAME: gf_tensor.checkpoint_peak_spill_bytes = 128 : i64
// SPILL-SAME: gf_tensor.checkpoint_saved_bytes = 128 : i64
// SPILL-SAME: gf_tensor.checkpoint_spilled_bytes = 128 : i64
// SPILL-SAME: gf_tensor.checkpoint_tiers = ["device", "host-pinned"]
// SPILL-COUNT-1: storage_tier = "device"
// SPILL-COUNT-1: storage_tier = "host-pinned"

module {
  func.func @plan(%arg0: tensor<16x4xf32>, %arg1: tensor<8xi64>)
      -> tensor<8x4xf32> {
    %x = "gf_tensor.input"(%arg0) {
      offset = 0 : i64, strides = array<i64: 4, 1>
    } : (tensor<16x4xf32>) -> tensor<16x4xf32>
    %index = "gf_tensor.input"(%arg1) {
      offset = 0 : i64, strides = array<i64: 1>
    } : (tensor<8xi64>) -> tensor<8xi64>
    %first = "gf_tensor.gather"(%x, %index)
      : (tensor<16x4xf32>, tensor<8xi64>) -> tensor<8x4xf32>
    %first_candidate = "gf_tensor.checkpoint_candidate"(%first)
      : (tensor<8x4xf32>) -> tensor<8x4xf32>
    %second = "gf_tensor.gather"(%x, %index)
      : (tensor<16x4xf32>, tensor<8xi64>) -> tensor<8x4xf32>
    %second_candidate = "gf_tensor.checkpoint_candidate"(%second)
      : (tensor<8x4xf32>) -> tensor<8x4xf32>
    %result = "gf_tensor.add"(%first_candidate, %second_candidate)
      : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>
    return %result : tensor<8x4xf32>
  }
}
