"""Multimodal Vision-Language Representation Alignment package for PRISM."""

from prism.multimodal.contracts import (
    ClassPrompt,
    CrossModalCentroidAlignment,
    CrossModalRetrievalSummary,
    MultimodalCollapseSummary,
    RetrievalResult,
    TokenizedText,
    VisionLanguageBatch,
    VisionLanguageSample,
    ZeroShotClassificationSummary,
)
from prism.multimodal.embeddings import (
    MaskedMeanPooling,
    MultimodalProjectionHead,
    TextEncoder,
    TextProjectionHead,
    TokenEmbeddingTable,
    VisualProjectionHead,
)
from prism.multimodal.enums import (
    MultimodalFailureType,
    MultimodalTaskType,
    MultimodalTransferStrategy,
    PretrainingObjective,
    RetrievalDirection,
    SpecialToken,
)
from prism.multimodal.loss import SymmetricContrastiveLoss
from prism.multimodal.synthetic import (
    COLOR_MAP,
    COLORS,
    POSITIONS,
    PROMPT_TEMPLATES,
    SHAPES,
    SIZES,
    build_synthetic_vocabulary,
    generate_synthetic_multimodal_dataset,
    render_synthetic_image,
)
from prism.multimodal.tokenizer import SimpleTokenizer, Vocabulary

__all__ = [
    "COLORS",
    "COLOR_MAP",
    "POSITIONS",
    "PROMPT_TEMPLATES",
    "SHAPES",
    "SIZES",
    "ClassPrompt",
    "CrossModalCentroidAlignment",
    "CrossModalRetrievalSummary",
    "MaskedMeanPooling",
    "MultimodalCollapseSummary",
    "MultimodalFailureType",
    "MultimodalProjectionHead",
    "MultimodalTaskType",
    "MultimodalTransferStrategy",
    "PretrainingObjective",
    "RetrievalDirection",
    "RetrievalResult",
    "SimpleTokenizer",
    "SpecialToken",
    "SymmetricContrastiveLoss",
    "TextEncoder",
    "TextProjectionHead",
    "TokenEmbeddingTable",
    "TokenizedText",
    "VisionLanguageBatch",
    "VisionLanguageSample",
    "VisualProjectionHead",
    "Vocabulary",
    "ZeroShotClassificationSummary",
    "build_synthetic_vocabulary",
    "generate_synthetic_multimodal_dataset",
    "render_synthetic_image",
]
