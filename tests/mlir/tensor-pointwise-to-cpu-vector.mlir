// RUN: gf-opt --gf-lower-tensor-to-cpu %s | FileCheck %s

// CHECK-LABEL: func.func @pointwise
// CHECK-SAME: %[[BEGIN:[a-zA-Z0-9_]+]]: index, %[[END:[a-zA-Z0-9_]+]]: index)
// CHECK-SAME: graphforge.cpu.vector_width = 16
// CHECK: %[[COUNT:.*]] = arith.subi %[[END]], %[[BEGIN]] : index
// CHECK: %[[STEP:.*]] = arith.constant 16 : index
// CHECK: %[[BLOCKS:.*]] = arith.divui %[[COUNT]], %[[STEP]] : index
// CHECK: %[[VCOUNT:.*]] = arith.muli %[[BLOCKS]], %{{.*}} : index
// CHECK: %[[VEND:.*]] = arith.addi %[[BEGIN]], %[[VCOUNT]] : index
// CHECK: scf.for %{{.*}} = %[[BEGIN]] to %[[VEND]] step %{{.*}}
// CHECK: vector.load
// CHECK: arith.mulf {{.*}} : vector<16xf32>
// CHECK: vector.store
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[END]] step
// CHECK: memref.store
func.func @pointwise(%lhs: tensor<33xf32>, %rhs: tensor<33xf32>)
    -> tensor<33xf32> {
  %0 = "gf_tensor.input"(%lhs) <{offset = 0 : i64,
      strides = array<i64: 1>}> : (tensor<33xf32>) -> tensor<33xf32>
  %1 = "gf_tensor.input"(%rhs) <{offset = 0 : i64,
      strides = array<i64: 1>}> : (tensor<33xf32>) -> tensor<33xf32>
  %2 = "gf_tensor.mul"(%0, %1)
      : (tensor<33xf32>, tensor<33xf32>) -> tensor<33xf32>
  %3 = "gf_tensor.add"(%2, %0)
      : (tensor<33xf32>, tensor<33xf32>) -> tensor<33xf32>
  return %3 : tensor<33xf32>
}
