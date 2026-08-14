// RUN: gf-translate --gf-tensor-to-ttir %s | FileCheck %s

// CHECK: graphforge.tensor entry=gf_tensor_pointwise
// CHECK: %neg{{[0-9]+}} = arith.subf %zero_float, %input{{[0-9]+}}
// CHECK-NOT: arith.negf
// CHECK: tt.store
func.func @neg_pointwise(%input: tensor<1024xf32>) -> tensor<1024xf32> {
  %0 = "gf_tensor.input"(%input) {
    offset = 0 : i64, strides = array<i64: 1>
  } : (tensor<1024xf32>) -> tensor<1024xf32>
  %1 = "gf_tensor.neg"(%0)
    : (tensor<1024xf32>) -> tensor<1024xf32>
  return %1 : tensor<1024xf32>
}
