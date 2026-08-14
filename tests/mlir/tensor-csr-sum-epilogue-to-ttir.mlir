// RUN: gf-opt %s | FileCheck %s --check-prefix=DOMAIN
// RUN: gf-translate --gf-tensor-to-ttir %s | FileCheck %s --check-prefix=TTIR

module {
  func.func @step(%rank_abi: tensor<8xf32>,
                  %column_abi: tensor<32xi64>,
                  %weight_abi: tensor<32xf32>,
                  %row_abi: tensor<9xi64>,
                  %damping_abi: tensor<f32>,
                  %base_abi: tensor<f32>) -> tensor<8xf32> {
    %rank = "gf_tensor.input"(%rank_abi) {
      offset = 0 : i64, strides = array<i64: 1>
    } : (tensor<8xf32>) -> tensor<8xf32>
    %column = "gf_tensor.input"(%column_abi) {
      offset = 0 : i64, strides = array<i64: 1>
    } : (tensor<32xi64>) -> tensor<32xi64>
    %weight = "gf_tensor.input"(%weight_abi) {
      offset = 0 : i64, strides = array<i64: 1>
    } : (tensor<32xf32>) -> tensor<32xf32>
    %row = "gf_tensor.input"(%row_abi) {
      offset = 0 : i64, strides = array<i64: 1>
    } : (tensor<9xi64>) -> tensor<9xi64>
    %damping = "gf_tensor.input"(%damping_abi) {
      offset = 0 : i64, strides = array<i64>
    } : (tensor<f32>) -> tensor<f32>
    %base = "gf_tensor.input"(%base_abi) {
      offset = 0 : i64, strides = array<i64>
    } : (tensor<f32>) -> tensor<f32>
    %source = "gf_tensor.gather"(%rank, %column) :
      (tensor<8xf32>, tensor<32xi64>) -> tensor<32xf32>
    %message = "gf_tensor.mul"(%source, %weight) :
      (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %sum = "gf_tensor.csr_segment_sum"(%message, %row) {
      num_rows = 8 : i64, degree_min = 4 : i64, degree_max = 4 : i64
    } : (tensor<32xf32>, tensor<9xi64>) -> tensor<8xf32>
    %scaled = "gf_tensor.mul"(%sum, %damping) :
      (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
    %result = "gf_tensor.add"(%scaled, %base) :
      (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}

// DOMAIN: "gf_tensor.csr_segment_sum"
// TTIR: graphforge.tensor entry=gf_tensor_csr_sum_epilogue block_rows=16 block_elements=4 num_warps=1
// TTIR: tensor<16x4xf32>
// TTIR: "tt.reduce"
// TTIR: arith.mulf %gf_sum
// TTIR: arith.addf
// TTIR: tt.store
