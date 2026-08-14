// RUN: gf-opt %s | FileCheck %s --check-prefix=DOMAIN
// RUN: gf-translate --gf-tensor-to-ttir %s | FileCheck %s --check-prefix=TTIR

module {
  func.func @axis_zero(%input_abi: tensor<5x3xf32>) -> tensor<1x3xf32> {
    %input = "gf_tensor.input"(%input_abi) {
      offset = 0 : i64, strides = array<i64: 3, 1>
    } : (tensor<5x3xf32>) -> tensor<5x3xf32>
    %result = "gf_tensor.reduce_sum"(%input) {
      axes = array<i64: 0>, keep_dims = true
    } : (tensor<5x3xf32>) -> tensor<1x3xf32>
    return %result : tensor<1x3xf32>
  }
}

// DOMAIN: "gf_tensor.reduce_sum"
// TTIR: graphforge.tensor entry=gf_tensor_fused_reduce block_rows=1 block_elements=8
// TTIR: %row = arith.extsi %row_i32
// TTIR: %row_product{{[0-9]+}} = arith.muli %row
// TTIR: %column_offset{{[0-9]+}} = arith.muli %lane
// TTIR: "tt.reduce"
// TTIR: tt.store
