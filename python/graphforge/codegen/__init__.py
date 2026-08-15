"""GraphForge code-generation providers.

The stable compiler boundary is ``gf.kernel``.  Modules in this package adapt
that IR (or, during bootstrap, an equivalent captured plan) to versioned
provider inputs such as Triton IR.  Hardware runtime backends live separately.
"""

from .provider import ProviderIdentity, triton_provider_identity
from .registry import (
    CodegenProvider, ProviderCapabilities, discover_providers, get_provider,
    provider_conformance, register_provider,
)
from .ttir import (
    TTIRCompileResult,
    TTIRRankedPlan,
    TTIRGeneratedRadiusPlan,
    TTIRDenseStreamingPlan,
    TTIRDenseScalarPlan,
    TTIRCSRScalarPlan,
    TTIRCSRProductPlan,
    TTIRTaskPrimitivePlan,
    TTIRWeightedSumPlan,
    compile_ttir,
    prepare_ttir_weighted_sum,
    prepare_ttir_ranked,
    prepare_ttir_generated_radius,
    prepare_ttir_dense_streaming,
    prepare_ttir_dense_scalar,
    prepare_ttir_csr_scalar,
    prepare_ttir_csr_product,
    prepare_ttir_task_primitive,
)

__all__ = [
    "ProviderIdentity",
    "CodegenProvider",
    "ProviderCapabilities",
    "TTIRCompileResult",
    "TTIRRankedPlan",
    "TTIRGeneratedRadiusPlan",
    "TTIRDenseStreamingPlan",
    "TTIRDenseScalarPlan",
    "TTIRCSRScalarPlan",
    "TTIRCSRProductPlan",
    "TTIRTaskPrimitivePlan",
    "TTIRWeightedSumPlan",
    "compile_ttir",
    "discover_providers",
    "get_provider",
    "prepare_ttir_weighted_sum",
    "prepare_ttir_ranked",
    "prepare_ttir_generated_radius",
    "prepare_ttir_dense_streaming",
    "prepare_ttir_dense_scalar",
    "prepare_ttir_csr_scalar",
    "prepare_ttir_csr_product",
    "prepare_ttir_task_primitive",
    "provider_conformance",
    "register_provider",
    "triton_provider_identity",
]
