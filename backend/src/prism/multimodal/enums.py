"""Enumerations for Multimodal Vision-Language Representation Alignment."""

from enum import Enum


class MultimodalTaskType(str, Enum):
    """Supported multimodal task types."""

    VISION_LANGUAGE_ALIGNMENT = "vision_language_alignment"
    CROSS_MODAL_RETRIEVAL = "cross_modal_retrieval"
    ZERO_SHOT_CLASSIFICATION = "zero_shot_classification"
    PROMPT_SENSITIVITY_ANALYSIS = "prompt_sensitivity_analysis"


class RetrievalDirection(str, Enum):
    """Direction for cross-modal retrieval queries."""

    IMAGE_TO_TEXT = "image_to_text"
    TEXT_TO_IMAGE = "text_to_image"


class PretrainingObjective(str, Enum):
    """Pretraining paradigms compared in PRISM."""

    SUPERVISED = "supervised"
    SIMCLR = "simclr"
    RECONSTRUCTION = "reconstruction"
    VISION_LANGUAGE = "vision_language"
    SCRATCH = "scratch"


class MultimodalTransferStrategy(str, Enum):
    """Transfer learning strategies for vision-language models."""

    FROZEN_VISION_PROBE = "frozen_vision_probe"
    PARTIAL_FINE_TUNE = "partial_fine_tune"
    FULL_FINE_TUNE = "full_fine_tune"


class MultimodalFailureType(str, Enum):
    """Descriptive taxonomy of multimodal alignment diagnostic failures."""

    PAIR_MISALIGNMENT = "pair_misalignment"
    IMAGE_TO_TEXT_RETRIEVAL_FAILURE = "image_to_text_retrieval_failure"
    TEXT_TO_IMAGE_RETRIEVAL_FAILURE = "text_to_image_retrieval_failure"
    ZERO_SHOT_CLASS_MISMATCH = "zero_shot_class_mismatch"
    PROMPT_SENSITIVITY = "prompt_sensitivity"
    VISUAL_CORRUPTION_ALIGNMENT_FAILURE = "visual_corruption_alignment_failure"
    MODALITY_COLLAPSE = "modality_collapse"


class SpecialToken(str, Enum):
    """Special vocabulary tokens for tokenizer."""

    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"
