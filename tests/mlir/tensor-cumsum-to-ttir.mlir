// RUN: gf-translate --gf-tensor-to-ttir %s | FileCheck %s

// CHECK: tiga.tensor entry=gf_tensor_cumsum
// CHECK: tt.func public @gf_tensor_cumsum
// CHECK: scf.for
// CHECK: %logical = arith.subi
// CHECK: tt.store
func.func @reverse_cumsum(%input: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = "gf_tensor.input"(%input) {
    offset = 0 : i64, strides = array<i64: 4, 1>
  } : (tensor<3x4xf32>) -> tensor<3x4xf32>
  %1 = "gf_tensor.cumsum"(%0) <{axis = 1 : i64, reverse = true}>
    : (tensor<3x4xf32>) -> tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}
