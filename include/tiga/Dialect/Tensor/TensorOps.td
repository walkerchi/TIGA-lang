#ifndef TIGA_TENSOR_OPS_TD
#define TIGA_TENSOR_OPS_TD

include "tiga/Dialect/Tensor/TensorDialect.td"

def Tiga_InputOp : Tiga_TensorOp<"input"> {
  let summary = "Bind an ABI buffer to a logical strided tensor value";
  let arguments = (ins AnyRankedTensor:$buffer, I64Attr:$offset,
                       DenseI64ArrayAttr:$strides);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_AddOp : Tiga_TensorOp<"add"> {
  let summary = "Elementwise addition with NumPy-style broadcasting";
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_MulOp : Tiga_TensorOp<"mul"> {
  let summary = "Elementwise multiplication with NumPy-style broadcasting";
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_MatmulOp : Tiga_TensorOp<"matmul"> {
  let summary = "Rank-two matrix contraction";
  let description = [{
    Preserves matrix multiplication as a semantic contraction so providers
    may select reduction loops, vector instructions, or tensor-core tiles.
  }];
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_DivOp : Tiga_TensorOp<"div"> {
  let summary = "Elementwise floating division with NumPy-style broadcasting";
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_CompareOp : Tiga_TensorOp<"compare"> {
  let summary = "Elementwise ordered comparison with broadcasting";
  let description = [{
    `predicate` is one of `eq`, `ne`, `lt`, `le`, `gt`, or `ge`. Operands
    have identical real/integer element types; the result has the broadcast
    shape and i1 elements. Keeping comparison semantic is required by bounded
    device-side control flow and avoids host scalar synchronization.
  }];
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs,
                       StrAttr:$predicate);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_NegOp : Tiga_TensorOp<"neg"> {
  let summary = "Elementwise arithmetic negation";
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_ExpOp : Tiga_TensorOp<"exp"> {
  let summary = "Elementwise floating-point exponential";
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_SqrtOp : Tiga_TensorOp<"sqrt"> {
  let summary = "Elementwise floating-point square root";
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_ConjOp : Tiga_TensorOp<"conj"> {
  let summary = "Elementwise complex conjugate";
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_ReshapeOp : Tiga_TensorOp<"reshape"> {
  let summary = "Logical row-major reshape";
  let arguments = (ins AnyRankedTensor:$input, DenseI64ArrayAttr:$shape);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_PermuteOp : Tiga_TensorOp<"permute"> {
  let summary = "Permute tensor dimensions";
  let arguments = (ins AnyRankedTensor:$input, DenseI64ArrayAttr:$axes);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_BroadcastOp : Tiga_TensorOp<"broadcast"> {
  let summary = "Broadcast singleton and leading dimensions";
  let arguments = (ins AnyRankedTensor:$input, DenseI64ArrayAttr:$shape);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_CheckpointOp : Tiga_TensorOp<"checkpoint"> {
  let summary = "Mark a primal value as an explicit backward saved tensor";
  let description = [{
    This operation is semantically the identity. It preserves the compiler's
    save-versus-recompute decision until physical memory planning. Backends
    may materialize its operand and bind that allocation as a backward ABI
    input instead of recomputing the operand expression.
  }];
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_CheckpointCandidateOp
    : Tiga_TensorOp<"checkpoint_candidate"> {
  let summary = "Offer a primal value to compiler save/recompute planning";
  let description = [{
    This operation is a semantic identity before planning. The checkpoint
    planner uses the static result size, producer structure and deployment
    memory budget to rewrite it either to `gf_tensor.checkpoint` or directly
    to its input. Provider lowering must not make a hidden independent choice.
  }];
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_ReduceSumOp : Tiga_TensorOp<"reduce_sum"> {
  let summary = "Sum over a set of tensor dimensions";
  let arguments = (ins AnyRankedTensor:$input, DenseI64ArrayAttr:$axes,
                       BoolAttr:$keep_dims);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_CumsumOp : Tiga_TensorOp<"cumsum"> {
  let summary = "Inclusive additive scan along one tensor dimension";
  let description = [{
    Preserves an inclusive prefix/suffix sum as a semantic scan. `reverse`
    selects suffix order without changing the physical result layout. Keeping
    this operation intact lets a provider choose serial, warp, block or
    hierarchical scans instead of expanding every prefix into reductions.
  }];
  let arguments = (ins AnyRankedTensor:$input, I64Attr:$axis,
                       BoolAttr:$reverse);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_GatherOp : Tiga_TensorOp<"gather"> {
  let summary = "Gather entity rows through an explicit relation index";
  let description = [{
    Gathers the leading dimension of `input`; trailing feature dimensions are
    preserved. The index is integer and deliberately non-differentiable.
  }];
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$index);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_ScatterRowsOp : Tiga_TensorOp<"scatter_rows"> {
  let summary = "Place disjoint input rows into a larger row domain";
  let description = [{
    `destination` maps every input row to one unique result row. `inverse`
    maps each result row back to its input row, or contains -1 for an empty
    result row. Keeping both maps makes forward placement and reverse gather
    linear without atomics or a quadratic generic segment reduction. The
    distributed partition planner produces and proves these maps.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$destination,
                       AnyRankedTensor:$inverse,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$num_rows);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_SegmentSumOp : Tiga_TensorOp<"segment_sum"> {
  let summary = "Reduce messages by destination segment";
  let description = [{
    Sums leading-dimension messages into `num_segments` destinations. This is
    the canonical sum-reducer primitive used by differentiable relation UDFs.
  }];
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$index,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$num_segments);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_CSRExpandRowsOp : Tiga_TensorOp<"csr_expand_rows"> {
  let summary = "Expand destination node values over CSR row edges";
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$row_ptr,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$num_edges);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_CSRSegmentSumOp : Tiga_TensorOp<"csr_segment_sum"> {
  let summary = "Sum edge messages into CSR destination rows";
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$row_ptr,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$num_rows,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$degree_min,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$degree_max);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_CSRSegmentProductOp : Tiga_TensorOp<"csr_segment_product"> {
  let summary = "Multiply edge messages within each CSR destination row";
  let description = [{
    This is the canonical lowering target for a captured reducer whose
    identity/lift/combine/finalize regions prove the product monoid.  The
    destination index is compiler-derived topology metadata retained for the
    reverse program; the forward reduction is driven by `row_ptr`.
  }];
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$row_ptr,
                       AnyRankedTensor:$destination,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$num_rows,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$max_degree,
                       BoolAttr:$uniform_degree);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_CSRSegmentProductVJPOp
    : Tiga_TensorOp<"csr_segment_product_vjp"> {
  let summary = "Zero-safe edge VJP of a CSR product reduction";
  let description = [{
    For every edge and feature lane, multiplies all other messages in the same
    row and scales by the row cotangent.  Excluding the current edge instead
    of dividing the total product makes the rule exact in the presence of
    zero-valued messages.
  }];
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$row_ptr,
                       AnyRankedTensor:$destination,
                       AnyRankedTensor:$upstream,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$num_rows,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$max_degree,
                       BoolAttr:$uniform_degree);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_CSRSegmentMaxStopGradientOp
    : Tiga_TensorOp<"csr_segment_max_stop_gradient"> {
  let summary = "Detached CSR row maximum used as a stable reducer shift";
  let description = [{
    Computes an exact row maximum, but carries an explicit stop-gradient
    contract.  This is an internal numerical shift for translation-invariant
    reducers such as online softmax, not the public semantics of a
    differentiable maximum reduction.
  }];
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$row_ptr,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$num_rows);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_CSREuclideanDistanceSumVJPOp
    : Tiga_TensorOp<"csr_euclidean_distance_sum_vjp"> {
  let summary = "Fixed-snapshot VJP of an Euclidean distance-weighted CSR sum";
  let description = [{
    Given the selected CSR edge endpoints, positions, scalar source values and
    row cotangents, computes both endpoint-position and source-value adjoints.
    The packed result has shape `[N,D+1]`: its first D columns are position
    gradients and its final column is the scalar source gradient. Relation
    membership is discrete and intentionally non-differentiable.
  }];
  let arguments = (ins AnyRankedTensor:$positions,
                       AnyRankedTensor:$source_value,
                       AnyRankedTensor:$destination,
                       AnyRankedTensor:$source_index,
                       AnyRankedTensor:$upstream,
                       AnyRankedTensor:$lattice,
                       AnyRankedTensor:$inverse_lattice,
                       I64Attr:$dimensions,
                       BoolAttr:$periodic);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def Tiga_GradOp : Tiga_TensorOp<"grad"> {
  let summary = "Request a vector-Jacobian product for one captured input";
  let description = [{
    `gf_tensor.grad` requests d(output)/d(wrt) multiplied by `cotangent`.
    `gf-tensor-vjp` replaces the request with ordinary `gf_tensor` operations;
    it is deliberately an IR operation rather than Python-only autograd state.
  }];
  let arguments = (ins AnyRankedTensor:$output, AnyRankedTensor:$wrt,
                       AnyRankedTensor:$cotangent);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

#endif
