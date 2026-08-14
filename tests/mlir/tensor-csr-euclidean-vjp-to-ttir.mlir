// RUN: gf-opt %s | FileCheck %s --check-prefix=DOMAIN
// RUN: gf-translate --gf-tensor-to-ttir %s | FileCheck %s --check-prefix=TTIR

module {
  func.func @euclidean_vjp(
      %positions_abi: tensor<4x3xf32>,
      %source_value_abi: tensor<4xf32>,
      %destination_abi: tensor<5xi64>,
      %source_index_abi: tensor<5xi64>,
      %upstream_abi: tensor<4xf32>,
      %lattice_abi: tensor<3x3xf32>,
      %inverse_abi: tensor<3x3xf32>) -> tensor<4x4xf32> {
    %positions = "gf_tensor.input"(%positions_abi) <{
      offset = 0 : i64, strides = array<i64: 3, 1>
    }> : (tensor<4x3xf32>) -> tensor<4x3xf32>
    %source_value = "gf_tensor.input"(%source_value_abi) <{
      offset = 0 : i64, strides = array<i64: 1>
    }> : (tensor<4xf32>) -> tensor<4xf32>
    %destination = "gf_tensor.input"(%destination_abi) <{
      offset = 0 : i64, strides = array<i64: 1>
    }> : (tensor<5xi64>) -> tensor<5xi64>
    %source_index = "gf_tensor.input"(%source_index_abi) <{
      offset = 0 : i64, strides = array<i64: 1>
    }> : (tensor<5xi64>) -> tensor<5xi64>
    %upstream = "gf_tensor.input"(%upstream_abi) <{
      offset = 0 : i64, strides = array<i64: 1>
    }> : (tensor<4xf32>) -> tensor<4xf32>
    %lattice = "gf_tensor.input"(%lattice_abi) <{
      offset = 0 : i64, strides = array<i64: 3, 1>
    }> : (tensor<3x3xf32>) -> tensor<3x3xf32>
    %inverse = "gf_tensor.input"(%inverse_abi) <{
      offset = 0 : i64, strides = array<i64: 3, 1>
    }> : (tensor<3x3xf32>) -> tensor<3x3xf32>
    %result = "gf_tensor.csr_euclidean_distance_sum_vjp"(
        %positions, %source_value, %destination, %source_index, %upstream,
        %lattice, %inverse) <{dimensions = 3 : i64, periodic = true}>
      : (tensor<4x3xf32>, tensor<4xf32>, tensor<5xi64>, tensor<5xi64>,
         tensor<4xf32>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<4x4xf32>
    return %result : tensor<4x4xf32>
  }
}

// DOMAIN: "gf_tensor.csr_euclidean_distance_sum_vjp"
// DOMAIN-SAME: dimensions = 3 : i64
// DOMAIN-SAME: periodic = true
// TTIR: graphforge.tensor entry=gf_tensor_csr_euclidean_distance_sum_vjp block_rows=256 block_elements=256 num_warps=8
// TTIR: tt.func public @gf_tensor_csr_euclidean_distance_sum_vjp
// TTIR: math.floor
// TTIR: tt.atomic_rmw fadd
// TTIR: %source_gradient_value = arith.mulf
