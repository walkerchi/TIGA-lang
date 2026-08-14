#ifndef GRAPHFORGE_TARGET_TRITON_TRANSLATE_H
#define GRAPHFORGE_TARGET_TRITON_TRANSLATE_H

namespace mlir::graphforge {

/// Register the provider-boundary translation from a verified gf_kernel
/// module to serialized Triton IR.  The result is text on purpose: GraphForge
/// and a vendor Triton distribution never share an LLVM/MLIR process ABI.
void registerKernelToTritonTranslation();

/// Register the first canonical gf_tensor provider translation.  The initial
/// executable slice covers fused FP32 elementwise expressions followed by a
/// contiguous innermost reduction; unsupported programs fail explicitly.
void registerTensorToTritonTranslation();

/// Register provider-neutral serialization of verified gf_task/gf_storage IR
/// into the runtime executable-bundle plan schema. This contains scheduling
/// and resource metadata only; executable objects remain provider-owned.
void registerTaskToBundleTranslation();

} // namespace mlir::graphforge

#endif
