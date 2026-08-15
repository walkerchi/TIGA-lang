// RUN: gf-opt %s -split-input-file -verify-diagnostics

func.func @nonscalar_condition(%state: tensor<4xf32>, %flag: tensor<1xi1>)
    -> tensor<4xf32> {
  // expected-error@+1 {{condition must yield a rank-zero tensor<i1>}}
  %0 = "gf_control.while"(%state, %flag)
      <{max_iterations = 2 : i64, num_carried = 1 : i64}> ({
  ^bb0(%value: tensor<4xf32>, %condition: tensor<1xi1>):
    "gf_control.condition"(%condition) : (tensor<1xi1>) -> ()
  }, {
  ^bb0(%value: tensor<4xf32>, %condition: tensor<1xi1>):
    "gf_control.yield"(%value) : (tensor<4xf32>) -> ()
  }) : (tensor<4xf32>, tensor<1xi1>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @wrong_body_type(%state: tensor<4xf32>, %flag: tensor<i1>,
                           %other: tensor<5xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{yielded tensor types must match loop results}}
  %0 = "gf_control.while"(%state, %flag, %other)
      <{max_iterations = 2 : i64, num_carried = 1 : i64}> ({
  ^bb0(%value: tensor<4xf32>, %condition: tensor<i1>, %next: tensor<5xf32>):
    "gf_control.condition"(%condition) : (tensor<i1>) -> ()
  }, {
  ^bb0(%value: tensor<4xf32>, %condition: tensor<i1>, %next: tensor<5xf32>):
    "gf_control.yield"(%next) : (tensor<5xf32>) -> ()
  }) : (tensor<4xf32>, tensor<i1>, tensor<5xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
