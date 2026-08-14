// RUN: gf-opt --verify-diagnostics %s | FileCheck %s
// RUN: gf-translate --gf-tensor-to-ttir %s | FileCheck %s --check-prefix=TTIR

// CHECK: "gf_tensor.checkpoint"
// TTIR: graphforge.tensor entry=gf_tensor_pointwise
// TTIR: tt.load
// TTIR: tt.store
func.func @checkpoint_pointwise(%arg0: tensor<1024xf32>,
                                %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
  %x = "gf_tensor.input"(%arg0) {
    offset = 0 : i64, strides = array<i64: 1>
  } : (tensor<1024xf32>) -> tensor<1024xf32>
  %y = "gf_tensor.input"(%arg1) {
    offset = 0 : i64, strides = array<i64: 1>
  } : (tensor<1024xf32>) -> tensor<1024xf32>
  %saved = "gf_tensor.checkpoint"(%x)
    : (tensor<1024xf32>) -> tensor<1024xf32>
  %result = "gf_tensor.mul"(%saved, %y)
    : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
  return %result : tensor<1024xf32>
}
