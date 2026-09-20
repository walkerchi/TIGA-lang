// RUN: gf-opt %s | FileCheck %s --check-prefix=DOMAIN
// RUN: gf-translate --gf-tensor-to-ttir %s | FileCheck %s --check-prefix=TTIR

module {
  func.func @segment_vector(%values_abi: tensor<5x3xf32>,
                            %index_abi: tensor<5xi64>) -> tensor<4x3xf32> {
    %values = "gf_tensor.input"(%values_abi) {
      offset = 0 : i64, strides = array<i64: 3, 1>
    } : (tensor<5x3xf32>) -> tensor<5x3xf32>
    %index = "gf_tensor.input"(%index_abi) {
      offset = 0 : i64, strides = array<i64: 1>
    } : (tensor<5xi64>) -> tensor<5xi64>
    %result = "gf_tensor.segment_sum"(%values, %index) {
      num_segments = 4 : i64
    } : (tensor<5x3xf32>, tensor<5xi64>) -> tensor<4x3xf32>
    return %result : tensor<4x3xf32>
  }
}

// DOMAIN: "gf_tensor.segment_sum"
// TTIR: tiga.tensor entry=gf_tensor_segment_sum block_rows=1 block_elements=8
// TTIR: arith.divui
// TTIR: arith.remui
// TTIR: %input_feature_offset0 = arith.muli
// TTIR: "tt.reduce"
// TTIR: %out_index = arith.addi
// TTIR: tt.store
