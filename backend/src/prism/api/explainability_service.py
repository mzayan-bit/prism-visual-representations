"""Service layer for explainability, visual attribution, and lab data."""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.core.enums import ModelFamily
from prism.core.errors import SerializationError
from prism.explainability.attention_attribution import (
    compute_vit_attention_attribution,
)
from prism.explainability.attribution import (
    AttributionMethod,
    AttributionResult,
    TargetClassMode,
    ViTAttentionHeadPolicy,
)
from prism.explainability.comparison import (
    AttributionComparisonReport,
    compare_attributions,
)
from prism.explainability.drift import (
    AttributionDriftSummary,
    compute_attribution_drift,
)
from prism.explainability.failures import (
    ExplanationFailureFlag,
    flag_explanation_failures,
)
from prism.explainability.grad_cam import compute_grad_cam
from prism.explainability.gradients import (
    compute_gradient_x_input,
    compute_input_gradient_saliency,
)
from prism.explainability.occlusion import compute_occlusion_sensitivity
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.models.transformer import VisionTransformer
from prism.robustness.corruptions import (
    apply_gaussian_noise,
)


class ExplainabilityExperimentMeta(BaseModel):
    """Metadata describing models, methods, and configs in the laboratory."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(description="Experiment identifier")
    name: str = Field(description="Display title of experiment")
    architectures: list[str] = Field(description="Available architecture keys")
    supported_methods: dict[str, list[str]] = Field(
        description="Supported attribution methods keyed by architecture"
    )
    layers: dict[str, list[str]] = Field(
        description="Available spatial layers keyed by architecture"
    )
    num_classes: int = Field(description="Number of classification categories")
    class_names: list[str] = Field(description="Class labels list")
    sample_ids: list[str] = Field(description="Sample identifiers list")


class ExplainabilitySamplePayload(BaseModel):
    """Full self-contained explainability payload for a single sample."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(description="Sample identifier")
    true_class: int = Field(description="Ground truth category index")
    class_name: str = Field(description="Ground truth category name")
    image_tensor: list[list[list[float]]] = Field(
        description="Source 3D image tensor [C, H, W]"
    )
    corrupted_image_tensor: list[list[list[float]]] | None = Field(
        default=None, description="Corrupted 3D image tensor [C, H, W]"
    )
    corruption_name: str | None = Field(
        default=None, description="Applied corruption name"
    )
    corruption_severity: float | None = Field(
        default=None, description="Applied corruption severity"
    )
    predictions: dict[str, dict[str, Any]] = Field(
        description="Predictions keyed by architecture name"
    )
    corrupted_predictions: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Predictions on corrupted input keyed by architecture",
    )
    attributions: dict[str, dict[str, AttributionResult]] = Field(
        description="AttributionResult indexed by architecture and method key"
    )
    corrupted_attributions: dict[str, dict[str, AttributionResult]] = Field(
        default_factory=dict,
        description="AttributionResult on corrupted input by arch and method",
    )
    comparison_reports: dict[str, AttributionComparisonReport] = Field(
        description="Cross-method comparison reports keyed by architecture"
    )
    drift_summaries: dict[str, dict[str, AttributionDriftSummary]] = Field(
        default_factory=dict,
        description="Clean vs corrupted drift keyed by arch and method",
    )
    failure_flags: dict[str, list[ExplanationFailureFlag]] = Field(
        default_factory=dict,
        description="Identified failure taxonomy flags keyed by architecture",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert payload to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Convert payload to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExplainabilitySamplePayload:
        """Construct payload from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize ExplainabilitySamplePayload: {exc}"
            ) from exc


class ExplainabilityDemoPayload(BaseModel):
    """Complete multi-sample explainability laboratory dataset."""

    model_config = ConfigDict(extra="forbid")

    metadata: ExplainabilityExperimentMeta = Field(description="Experiment metadata")
    samples: list[ExplainabilitySamplePayload] = Field(
        description="List of evaluated sample payloads"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert demo payload to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Convert demo payload to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExplainabilityDemoPayload:
        """Construct demo payload from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize ExplainabilityDemoPayload: {exc}"
            ) from exc


class ExplainabilityService:
    """Service managing explainability pipelines, method resolution, and reports."""

    def __init__(self) -> None:
        self._reports_cache: dict[str, AttributionComparisonReport] = {}
        self._drift_cache: dict[str, AttributionDriftSummary] = {}
        self._demo_payload: ExplainabilityDemoPayload | None = None

    def register_demo_payload(self, payload: ExplainabilityDemoPayload) -> None:
        """Register and cache full demo payload."""
        self._demo_payload = payload

    def get_metadata(self) -> ExplainabilityExperimentMeta | None:
        """Retrieve experiment metadata from cached payload."""
        if self._demo_payload is not None:
            return self._demo_payload.metadata
        return None

    def get_all_samples(self) -> list[ExplainabilitySamplePayload]:
        """Retrieve all sample payloads from cached payload."""
        if self._demo_payload is not None:
            return self._demo_payload.samples
        return []

    def get_sample(self, sample_id: str) -> ExplainabilitySamplePayload | None:
        """Retrieve specific sample payload by ID."""
        if self._demo_payload is not None:
            for s in self._demo_payload.samples:
                if s.sample_id == sample_id:
                    return s
        return None

    @staticmethod
    def get_supported_methods(architecture: str) -> list[AttributionMethod]:
        """Return valid attribution methods for a given model architecture."""
        arch_norm = architecture.strip().lower()
        if arch_norm in ("cnn", "convolutional") or arch_norm in (
            "resnet",
            "residual",
        ):
            return [
                AttributionMethod.INPUT_GRADIENT,
                AttributionMethod.GRADIENT_X_INPUT,
                AttributionMethod.OCCLUSION_SENSITIVITY,
                AttributionMethod.GRAD_CAM,
            ]
        elif arch_norm in ("vit", "transformer"):
            return [
                AttributionMethod.INPUT_GRADIENT,
                AttributionMethod.GRADIENT_X_INPUT,
                AttributionMethod.OCCLUSION_SENSITIVITY,
                AttributionMethod.VIT_ATTENTION,
            ]
        else:
            return [
                AttributionMethod.INPUT_GRADIENT,
                AttributionMethod.GRADIENT_X_INPUT,
                AttributionMethod.OCCLUSION_SENSITIVITY,
            ]

    @staticmethod
    def get_available_layers(architecture: str) -> list[str]:
        """Return available spatial layer options for a given model architecture."""
        arch_norm = architecture.strip().lower()
        if arch_norm in ("cnn", "convolutional"):
            return ["final_conv", "conv_0", "conv_1", "conv_2"]
        elif arch_norm in ("resnet", "residual"):
            return ["final_stage", "stage_0", "stage_1", "stage_2", "stem"]
        elif arch_norm in ("vit", "transformer"):
            return ["last_block", "block_0", "block_1", "block_2"]
        return ["final_hidden"]


def generate_explainability_demo_data() -> ExplainabilityDemoPayload:
    """Generate precomputed explainability demo dataset for CNN, ResNet, and ViT."""
    classes = ["airplane", "automobile", "bird", "cat", "deer"]
    num_classes = len(classes)
    h, w, c = 8, 8, 3

    # 1. Instantiate 3 models
    cnn_spec = ModelSpecification(
        model_id="cnn_cifar_base",
        name="CNN Base",
        family=ModelFamily.CNN,
        architecture="cnn_2layer",
        input_shape=(c, h, w),
        num_classes=num_classes,
        hyperparameters={
            "conv_channels": [8, 16],
            "kernel_sizes": [3, 3],
            "activation": "relu",
            "use_batch_norm": False,
            "pooling": "none",
            "hidden_dims": [16],
        },
    )
    cnn_model = ConvolutionalNeuralNetwork(cnn_spec, seed=42)
    cnn_model.eval()

    resnet_spec = ModelSpecification(
        model_id="resnet_cifar_base",
        name="ResNet Base",
        family=ModelFamily.RESNET,
        architecture="resnet_2stage",
        input_shape=(c, h, w),
        num_classes=num_classes,
        hyperparameters={
            "stem_channels": 8,
            "stage_channels": [8, 16],
            "stage_blocks": [1, 1],
            "activation": "relu",
            "use_batch_norm": False,
            "hidden_dims": [16],
        },
    )
    resnet_model = ResidualNeuralNetwork(resnet_spec, seed=43)
    resnet_model.eval()

    vit_spec = ModelSpecification(
        model_id="vit_cifar_base",
        name="ViT Base",
        family=ModelFamily.VISION_TRANSFORMER,
        architecture="vit_tiny_p2",
        input_shape=(c, h, w),
        num_classes=num_classes,
        hyperparameters={
            "patch_size": 2,
            "embed_dim": 16,
            "depth": 2,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "activation": "gelu",
        },
    )
    vit_model = VisionTransformer(vit_spec, seed=44)
    vit_model.eval()

    models_dict = {
        "cnn": cnn_model,
        "resnet": resnet_model,
        "vit": vit_model,
    }

    # 2. Synthesize 5 structured samples with geometric shapes
    samples_data: list[tuple[str, int, list[list[list[float]]]]] = []

    # Sample 1: Airplane (Class 0) - horizontal bar and cross wing
    s1_img = [[[0.1 for _ in range(w)] for _ in range(h)] for _ in range(c)]
    for col in range(1, 7):
        s1_img[0][3][col] = 0.9
        s1_img[1][3][col] = 0.8
        s1_img[2][3][col] = 0.7
    for r in range(1, 7):
        s1_img[0][r][3] = 0.85
        s1_img[1][r][3] = 0.75
        s1_img[2][r][3] = 0.65
    samples_data.append(("sample_001_airplane", 0, s1_img))

    # Sample 2: Automobile (Class 1) - bottom rectangle chassis and wheel dots
    s2_img = [[[0.05 for _ in range(w)] for _ in range(h)] for _ in range(c)]
    for r in range(4, 7):
        for col in range(1, 7):
            s2_img[0][r][col] = 0.95
            s2_img[1][r][col] = 0.2
            s2_img[2][r][col] = 0.2
    s2_img[0][7][2] = 0.1
    s2_img[0][7][5] = 0.1
    samples_data.append(("sample_002_automobile", 1, s2_img))

    # Sample 3: Bird (Class 2) - diagonal stroke and top beak
    s3_img = [[[0.15 for _ in range(w)] for _ in range(h)] for _ in range(c)]
    for i in range(2, 6):
        s3_img[0][i][i] = 0.9
        s3_img[1][i][i] = 0.9
        s3_img[2][i][i] = 0.1
    s3_img[0][1][5] = 0.95
    s3_img[1][1][5] = 0.6
    samples_data.append(("sample_003_bird", 2, s3_img))

    # Sample 4: Cat (Class 3) - two ear peaks and central face
    s4_img = [[[0.08 for _ in range(w)] for _ in range(h)] for _ in range(c)]
    s4_img[0][1][2] = 0.85
    s4_img[0][1][5] = 0.85
    for r in range(3, 7):
        for col in range(2, 6):
            s4_img[0][r][col] = 0.75
            s4_img[1][r][col] = 0.65
            s4_img[2][r][col] = 0.45
    samples_data.append(("sample_004_cat", 3, s4_img))

    # Sample 5: Deer (Class 4) - tall vertical legs and antler spikes
    s5_img = [[[0.12 for _ in range(w)] for _ in range(h)] for _ in range(c)]
    s5_img[1][0][1] = 0.8
    s5_img[1][0][6] = 0.8
    for r in range(2, 8):
        s5_img[0][r][2] = 0.7
        s5_img[1][r][2] = 0.5
        s5_img[0][r][5] = 0.7
        s5_img[1][r][5] = 0.5
    samples_data.append(("sample_005_deer", 4, s5_img))

    # 3. Compute all attributions and comparisons for each sample
    sample_payloads: list[ExplainabilitySamplePayload] = []

    for s_id, t_class, img in samples_data:
        # Create corrupted image
        corr_img = apply_gaussian_noise(img, sigma=0.15, seed=42)

        predictions: dict[str, dict[str, Any]] = {}
        corr_predictions: dict[str, dict[str, Any]] = {}
        attributions: dict[str, dict[str, AttributionResult]] = {}
        corr_attributions: dict[str, dict[str, AttributionResult]] = {}
        comparison_reports: dict[str, AttributionComparisonReport] = {}
        drift_summaries: dict[str, dict[str, AttributionDriftSummary]] = {}
        failure_flags: dict[str, list[ExplanationFailureFlag]] = {}

        for arch_name, model in models_dict.items():
            # Clean predictions
            clean_logits = model.forward([img])[0]
            clean_pred = max(range(num_classes), key=lambda i: clean_logits[i])
            # Softmax
            clean_exp = [math.exp(v) for v in clean_logits]
            clean_sum = sum(clean_exp) if sum(clean_exp) > 0 else 1.0
            clean_probs = [v / clean_sum for v in clean_exp]

            predictions[arch_name] = {
                "predicted_class": clean_pred,
                "predicted_name": classes[clean_pred],
                "score": float(clean_logits[clean_pred]),
                "confidence": float(clean_probs[clean_pred]),
                "probabilities": [float(p) for p in clean_probs],
            }

            # Corrupted predictions
            corr_logits = model.forward([corr_img])[0]
            corr_pred = max(range(num_classes), key=lambda i: corr_logits[i])
            corr_exp = [math.exp(v) for v in corr_logits]
            corr_sum = sum(corr_exp) if sum(corr_exp) > 0 else 1.0
            corr_probs = [v / corr_sum for v in corr_exp]

            corr_predictions[arch_name] = {
                "predicted_class": corr_pred,
                "predicted_name": classes[corr_pred],
                "score": float(corr_logits[corr_pred]),
                "confidence": float(corr_probs[corr_pred]),
                "probabilities": [float(p) for p in corr_probs],
            }

            # Methods for this model
            arch_results: list[AttributionResult] = []
            arch_corr_results: list[AttributionResult] = []
            drift_dict: dict[str, AttributionDriftSummary] = {}

            # A. Input Gradient
            ig_res = compute_input_gradient_saliency(
                model=model,
                image=img,
                target_mode=TargetClassMode.PREDICTED_CLASS,
                true_class=t_class,
                sample_id=s_id,
            )
            ig_corr = compute_input_gradient_saliency(
                model=model,
                image=corr_img,
                target_mode=TargetClassMode.PREDICTED_CLASS,
                true_class=t_class,
                sample_id=s_id,
            )
            arch_results.append(ig_res)
            arch_corr_results.append(ig_corr)
            drift_dict[AttributionMethod.INPUT_GRADIENT.value] = (
                compute_attribution_drift(
                    ig_res,
                    ig_corr,
                    corruption_type="gaussian_noise",
                    corruption_severity=2.0,
                )
            )

            # B. Gradient x Input
            gxi_res = compute_gradient_x_input(
                model=model,
                image=img,
                target_mode=TargetClassMode.PREDICTED_CLASS,
                true_class=t_class,
                sample_id=s_id,
            )
            gxi_corr = compute_gradient_x_input(
                model=model,
                image=corr_img,
                target_mode=TargetClassMode.PREDICTED_CLASS,
                true_class=t_class,
                sample_id=s_id,
            )
            arch_results.append(gxi_res)
            arch_corr_results.append(gxi_corr)
            drift_dict[AttributionMethod.GRADIENT_X_INPUT.value] = (
                compute_attribution_drift(
                    gxi_res,
                    gxi_corr,
                    corruption_type="gaussian_noise",
                    corruption_severity=2.0,
                )
            )

            # C. Occlusion Sensitivity
            occ_res = compute_occlusion_sensitivity(
                model=model,
                image=img,
                window_size=(2, 2),
                stride=(1, 1),
                target_mode=TargetClassMode.PREDICTED_CLASS,
                true_class=t_class,
                sample_id=s_id,
            )
            occ_corr = compute_occlusion_sensitivity(
                model=model,
                image=corr_img,
                window_size=(2, 2),
                stride=(1, 1),
                target_mode=TargetClassMode.PREDICTED_CLASS,
                true_class=t_class,
                sample_id=s_id,
            )
            arch_results.append(occ_res)
            arch_corr_results.append(occ_corr)
            drift_dict[AttributionMethod.OCCLUSION_SENSITIVITY.value] = (
                compute_attribution_drift(
                    occ_res,
                    occ_corr,
                    corruption_type="gaussian_noise",
                    corruption_severity=2.0,
                )
            )

            # D. Grad-CAM (CNN/ResNet) or ViT Attention (ViT)
            if arch_name in ("cnn", "resnet"):
                layer_target = "final_conv" if arch_name == "cnn" else "final_stage"
                cam_res = compute_grad_cam(
                    model=model,
                    image=img,
                    layer_name=layer_target,
                    target_mode=TargetClassMode.PREDICTED_CLASS,
                    true_class=t_class,
                    sample_id=s_id,
                )
                cam_corr = compute_grad_cam(
                    model=model,
                    image=corr_img,
                    layer_name=layer_target,
                    target_mode=TargetClassMode.PREDICTED_CLASS,
                    true_class=t_class,
                    sample_id=s_id,
                )
                arch_results.append(cam_res)
                arch_corr_results.append(cam_corr)
                drift_dict[AttributionMethod.GRAD_CAM.value] = (
                    compute_attribution_drift(
                        cam_res,
                        cam_corr,
                        corruption_type="gaussian_noise",
                        corruption_severity=2.0,
                    )
                )

            elif arch_name == "vit":
                attn_res = compute_vit_attention_attribution(
                    model=model,
                    image=img,
                    head_policy=ViTAttentionHeadPolicy.MEAN_HEADS,
                    layer_index=-1,
                    target_mode=TargetClassMode.PREDICTED_CLASS,
                    true_class=t_class,
                    sample_id=s_id,
                )
                attn_corr = compute_vit_attention_attribution(
                    model=model,
                    image=corr_img,
                    head_policy=ViTAttentionHeadPolicy.MEAN_HEADS,
                    layer_index=-1,
                    target_mode=TargetClassMode.PREDICTED_CLASS,
                    true_class=t_class,
                    sample_id=s_id,
                )
                arch_results.append(attn_res)
                arch_corr_results.append(attn_corr)
                drift_dict[AttributionMethod.VIT_ATTENTION.value] = (
                    compute_attribution_drift(
                        attn_res,
                        attn_corr,
                        corruption_type="gaussian_noise",
                        corruption_severity=2.0,
                    )
                )

            # Store result maps
            attributions[arch_name] = {r.method.value: r for r in arch_results}
            corr_attributions[arch_name] = {
                r.method.value: r for r in arch_corr_results
            }

            # Generate comparison report
            comp_report = compare_attributions(arch_results)
            comparison_reports[arch_name] = comp_report
            drift_summaries[arch_name] = drift_dict

            # Check failure flags
            flags = flag_explanation_failures(
                attribution_result=arch_results[0],
                comparison_report=comp_report,
                drift_summary=drift_dict.get(AttributionMethod.INPUT_GRADIENT.value),
            )
            failure_flags[arch_name] = flags

        payload = ExplainabilitySamplePayload(
            sample_id=s_id,
            true_class=t_class,
            class_name=classes[t_class],
            image_tensor=img,
            corrupted_image_tensor=corr_img,
            corruption_name="gaussian_noise",
            corruption_severity=2.0,
            predictions=predictions,
            corrupted_predictions=corr_predictions,
            attributions=attributions,
            corrupted_attributions=corr_attributions,
            comparison_reports=comparison_reports,
            drift_summaries=drift_summaries,
            failure_flags=failure_flags,
        )
        sample_payloads.append(payload)

    metadata = ExplainabilityExperimentMeta(
        experiment_id="exp_phase16_explainability_suite",
        name="Phase 16 — Explainability & Visual Attribution Laboratory",
        architectures=["cnn", "resnet", "vit"],
        supported_methods={
            "cnn": [
                AttributionMethod.INPUT_GRADIENT.value,
                AttributionMethod.GRADIENT_X_INPUT.value,
                AttributionMethod.OCCLUSION_SENSITIVITY.value,
                AttributionMethod.GRAD_CAM.value,
            ],
            "resnet": [
                AttributionMethod.INPUT_GRADIENT.value,
                AttributionMethod.GRADIENT_X_INPUT.value,
                AttributionMethod.OCCLUSION_SENSITIVITY.value,
                AttributionMethod.GRAD_CAM.value,
            ],
            "vit": [
                AttributionMethod.INPUT_GRADIENT.value,
                AttributionMethod.GRADIENT_X_INPUT.value,
                AttributionMethod.OCCLUSION_SENSITIVITY.value,
                AttributionMethod.VIT_ATTENTION.value,
            ],
        },
        layers={
            "cnn": ["final_conv", "conv_0", "conv_1"],
            "resnet": ["final_stage", "stage_0", "stage_1", "stem"],
            "vit": ["last_block", "block_0", "block_1"],
        },
        num_classes=num_classes,
        class_names=classes,
        sample_ids=[s[0] for s in samples_data],
    )

    return ExplainabilityDemoPayload(
        metadata=metadata,
        samples=sample_payloads,
    )
