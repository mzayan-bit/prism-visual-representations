"""Unit tests for ModelSpecification."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from prism.core.enums import InitializationStrategy, ModelFamily, TaskType
from prism.core.errors import ValidationError
from prism.models.specifications import ModelSpecification


@pytest.mark.unit
def test_valid_model_specification() -> None:
    """Verify construction of valid model specifications across architectures."""
    resnet = ModelSpecification(
        model_id="model-resnet18-scratch",
        name="ResNet-18",
        family=ModelFamily.RESNET,
        architecture="resnet18",
        compatible_tasks=[TaskType.CLASSIFICATION, TaskType.ROBUSTNESS],
        initialization=InitializationStrategy.SCRATCH,
        input_shape=(3, 224, 224),
        num_classes=1000,
        hyperparameters={"zero_init_residual": True},
    )
    assert resnet.model_id == "model-resnet18-scratch"
    assert resnet.family == ModelFamily.RESNET
    assert resnet.num_classes == 1000

    vit = ModelSpecification(
        model_id="model-vit-tiny",
        name="ViT-Tiny/16",
        family=ModelFamily.VISION_TRANSFORMER,
        architecture="vit_tiny_patch16_224",
        compatible_tasks=[TaskType.CLASSIFICATION],
        initialization=InitializationStrategy.RANDOM,
        input_shape=(3, 224, 224),
        hyperparameters={
            "patch_size": 16,
            "embed_dim": 192,
            "depth": 12,
            "num_heads": 3,
        },
    )
    assert vit.family == ModelFamily.VISION_TRANSFORMER
    assert vit.hyperparameters["embed_dim"] == 192


@pytest.mark.unit
def test_pretrained_requires_source() -> None:
    """Verify that pretrained initialization strategy requires pretrained_source."""
    with pytest.raises(ValidationError, match="requires a 'pretrained_source'"):
        ModelSpecification(
            model_id="model-vit-pretrained-invalid",
            name="ViT Pretrained Invalid",
            family=ModelFamily.VISION_TRANSFORMER,
            architecture="vit_base_patch16_224",
            initialization=InitializationStrategy.PRETRAINED,
            pretrained_source=None,  # missing required source
        )


@pytest.mark.unit
def test_valid_pretrained_specification() -> None:
    """Verify that pretrained specification succeeds when source is provided."""
    model = ModelSpecification(
        model_id="model-vit-dino",
        name="ViT-Base DINO",
        family=ModelFamily.SELF_SUPERVISED_ENCODER,
        architecture="vit_base_patch16_224",
        initialization=InitializationStrategy.PRETRAINED,
        pretrained_source="facebookresearch/dino-vitb16",
        backbone_freeze=True,
        probe_head="linear",
    )
    assert model.pretrained_source == "facebookresearch/dino-vitb16"
    assert model.backbone_freeze is True
    assert model.probe_head == "linear"


@pytest.mark.unit
def test_invalid_input_shape_rejected() -> None:
    """Verify that non-positive input dimensions are rejected."""
    with pytest.raises((PydanticValidationError, ValueError)):
        ModelSpecification(
            model_id="model-invalid-shape",
            name="Invalid Shape Model",
            family=ModelFamily.CNN,
            architecture="custom_cnn",
            input_shape=(3, 0, 224),  # 0 is invalid
        )
