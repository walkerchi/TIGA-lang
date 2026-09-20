// RUN: gf-translate --gf-tensor-to-ttir %s | FileCheck %s --check-prefix=TTIR

module {
  func.func @product(%message_abi: tensor<5xf32>,
                     %row_abi: tensor<4xi64>,
                     %destination_abi: tensor<5xi64>) -> tensor<3xf32> {
    %message = "gf_tensor.input"(%message_abi) <{
      offset = 0 : i64, strides = array<i64: 1>
    }> : (tensor<5xf32>) -> tensor<5xf32>
    %row = "gf_tensor.input"(%row_abi) <{
      offset = 0 : i64, strides = array<i64: 1>
    }> : (tensor<4xi64>) -> tensor<4xi64>
    %destination = "gf_tensor.input"(%destination_abi) <{
      offset = 0 : i64, strides = array<i64: 1>
    }> : (tensor<5xi64>) -> tensor<5xi64>
    %result = "gf_tensor.csr_segment_product"(
        %message, %row, %destination) <{
      max_degree = 3 : i64, num_rows = 3 : i64, uniform_degree = false
    }> : (tensor<5xf32>, tensor<4xi64>, tensor<5xi64>) -> tensor<3xf32>
    return %result : tensor<3xf32>
  }
}

// TTIR: tiga.tensor entry=gf_tensor_csr_product block_rows=1 block_elements=4
// TTIR: tt.func public @gf_tensor_csr_product
// TTIR: %product = "tt.reduce"(%values)
// TTIR: %next = arith.mulf %a, %b : f32
