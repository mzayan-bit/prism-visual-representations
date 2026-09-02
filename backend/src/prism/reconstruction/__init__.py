"""Generative and reconstruction-based visual representation learning package."""

from prism.reconstruction.batch import (
    ReconstructionBatch,
    prepare_denoising_batch,
    prepare_masked_patch_batch,
)
from prism.reconstruction.context import MaskingContext
from prism.reconstruction.decoders import (
    PatchReconstructionDecoder,
    SpatialReconstructionDecoder,
)
from prism.reconstruction.diagnostics import (
    ReconstructionDiagnosticsReport,
    compute_reconstruction_diagnostics,
)
from prism.reconstruction.engine import ReconstructionTrainingEngine
from prism.reconstruction.enums import (
    ReconstructionFailureCategory,
    ReconstructionMethod,
)
from prism.reconstruction.loss import MaskedMSELoss
from prism.reconstruction.mask import PatchMask, generate_patch_mask
from prism.reconstruction.reports import (
    ReconstructionLearningReport,
    SupervisedVsSSLVsReconstructionSummary,
)
from prism.reconstruction.specification import ReconstructionLearningSpecification
from prism.reconstruction.tokens import LearnableMaskToken

__all__ = [
    "LearnableMaskToken",
    "MaskedMSELoss",
    "MaskingContext",
    "PatchMask",
    "PatchReconstructionDecoder",
    "ReconstructionBatch",
    "ReconstructionDiagnosticsReport",
    "ReconstructionFailureCategory",
    "ReconstructionLearningReport",
    "ReconstructionLearningSpecification",
    "ReconstructionMethod",
    "ReconstructionTrainingEngine",
    "SpatialReconstructionDecoder",
    "SupervisedVsSSLVsReconstructionSummary",
    "compute_reconstruction_diagnostics",
    "generate_patch_mask",
    "prepare_denoising_batch",
    "prepare_masked_patch_batch",
]
