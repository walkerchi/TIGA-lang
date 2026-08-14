// RUN: gf-opt --verify-diagnostics %s | FileCheck %s
// RUN: gf-translate --gf-tensor-to-ttir %s | FileCheck %s --check-prefix=TTIR

// CHECK: "gf_tensor.gather"
// CHECK: "gf_tensor.reduce_sum"
// TTIR: graphforge.tensor entry=gf_tensor_fused_reduce
// TTIR-SAME: abi=arg0,arg1,arg2,out
// TTIR: %[[INDEX:.+]] = tt.load {{.+}} : !tt.ptr<i64>
// TTIR: %[[VALUES:.+]] = tt.load {{.+}} : tensor<16x!tt.ptr<f32>>
// TTIR: %[[PRODUCT:.+]] = arith.mulf
// TTIR: %[[SUM:.+]] = "tt.reduce"(%[[PRODUCT]])
// TTIR: tt.store
func.func @edge_dot(%arg0: tensor<128x16xf32>,
                    %arg1: tensor<1024xi64>,
                    %arg2: tensor<1024x16xf32>) -> tensor<1024x1xf32> {
  %dy = "gf_tensor.input"(%arg0) {
    offset = 0 : i64, strides = array<i64: 16, 1>
  } : (tensor<128x16xf32>) -> tensor<128x16xf32>
  %dst = "gf_tensor.input"(%arg1) {
    offset = 0 : i64, strides = array<i64: 1>
  } : (tensor<1024xi64>) -> tensor<1024xi64>
  %saved = "gf_tensor.input"(%arg2) {
    offset = 0 : i64, strides = array<i64: 16, 1>
  } : (tensor<1024x16xf32>) -> tensor<1024x16xf32>
  %expanded = "gf_tensor.gather"(%dy, %dst)
    : (tensor<128x16xf32>, tensor<1024xi64>) -> tensor<1024x16xf32>
  %product = "gf_tensor.mul"(%expanded, %saved)
    : (tensor<1024x16xf32>, tensor<1024x16xf32>) -> tensor<1024x16xf32>
  %result = "gf_tensor.reduce_sum"(%product) {
    axes = array<i64: 1>, keep_dims = true
  } : (tensor<1024x16xf32>) -> tensor<1024x1xf32>
  return %result : tensor<1024x1xf32>
}
