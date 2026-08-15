// RUN: gf-opt --gf-lower-tensor-to-cpu %s | FileCheck %s

// CHECK-LABEL: func.func @bounded_multi
// CHECK-SAME: graphforge.cpu.bounded_while
// CHECK-SAME: graphforge.cpu.loop_buffers = 4
// CHECK-SAME: graphforge.cpu.max_iterations = 12
// CHECK: scf.while
// CHECK: arith.cmpf ogt
// CHECK: arith.cmpi ult
// CHECK: scf.condition
// CHECK: scf.yield
func.func @bounded_multi(%vector: tensor<32xf32>, %residual: tensor<f32>,
                         %threshold: tensor<f32>) -> tensor<32xf32> {
  %0 = "gf_tensor.input"(%vector) <{offset = 0 : i64,
      strides = array<i64: 1>}> : (tensor<32xf32>) -> tensor<32xf32>
  %1 = "gf_tensor.input"(%residual) <{offset = 0 : i64,
      strides = array<i64>}> : (tensor<f32>) -> tensor<f32>
  %2 = "gf_tensor.input"(%threshold) <{offset = 0 : i64,
      strides = array<i64>}> : (tensor<f32>) -> tensor<f32>
  %3:2 = "gf_control.while"(%0, %1, %2)
      <{max_iterations = 12 : i64, num_carried = 2 : i64}> ({
  ^bb0(%value: tensor<32xf32>, %rr: tensor<f32>, %limit: tensor<f32>):
    %4 = "gf_tensor.compare"(%rr, %limit) <{predicate = "gt"}>
        : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "gf_control.condition"(%4) : (tensor<i1>) -> ()
  }, {
  ^bb0(%value: tensor<32xf32>, %rr: tensor<f32>, %limit: tensor<f32>):
    %4 = "gf_tensor.add"(%value, %rr)
        : (tensor<32xf32>, tensor<f32>) -> tensor<32xf32>
    %5 = "gf_tensor.mul"(%rr, %limit)
        : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "gf_control.yield"(%4, %5)
        : (tensor<32xf32>, tensor<f32>) -> ()
  }) : (tensor<32xf32>, tensor<f32>, tensor<f32>)
      -> (tensor<32xf32>, tensor<f32>)
  return %3#0 : tensor<32xf32>
}
