// RUN: gf-opt --gf-tensor-vjp %s | FileCheck %s --check-prefix=VJP
// RUN: gf-opt --gf-tensor-vjp --canonicalize %s | gf-translate --gf-tensor-to-ttir | FileCheck %s --check-prefix=TTIR

module {
  func.func @product_vjp(%message_abi: tensor<5xf32>,
                         %row_abi: tensor<4xi64>,
                         %destination_abi: tensor<5xi64>,
                         %cotangent_abi: tensor<3xf32>) -> tensor<5xf32> {
    %message = "gf_tensor.input"(%message_abi) <{
      offset = 0 : i64, strides = array<i64: 1>
    }> : (tensor<5xf32>) -> tensor<5xf32>
    %row = "gf_tensor.input"(%row_abi) <{
      offset = 0 : i64, strides = array<i64: 1>
    }> : (tensor<4xi64>) -> tensor<4xi64>
    %destination = "gf_tensor.input"(%destination_abi) <{
      offset = 0 : i64, strides = array<i64: 1>
    }> : (tensor<5xi64>) -> tensor<5xi64>
    %cotangent = "gf_tensor.input"(%cotangent_abi) <{
      offset = 0 : i64, strides = array<i64: 1>
    }> : (tensor<3xf32>) -> tensor<3xf32>
    %forward = "gf_tensor.csr_segment_product"(
        %message, %row, %destination) <{
      max_degree = 3 : i64, num_rows = 3 : i64, uniform_degree = false
    }> : (tensor<5xf32>, tensor<4xi64>, tensor<5xi64>) -> tensor<3xf32>
    %gradient = "gf_tensor.grad"(%forward, %message, %cotangent)
      : (tensor<3xf32>, tensor<5xf32>, tensor<3xf32>) -> tensor<5xf32>
    return %gradient : tensor<5xf32>
  }
}

// VJP: "gf_tensor.csr_segment_product_vjp"
// VJP-NOT: "gf_tensor.grad"
// TTIR: graphforge.tensor entry=gf_tensor_csr_product_vjp block_rows=1 block_elements=4
// TTIR: tt.func public @gf_tensor_csr_product_vjp
// TTIR: %not_self = arith.cmpi ne
// TTIR: %excluded = "tt.reduce"(%values)
// TTIR: %gradient = arith.mulf %excluded, %cotangent : f32
