"""Reconstruction Learning service and research dataset generation."""

from __future__ import annotations

import json
import math
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from prism.core.enums import ModelFamily
from prism.reconstruction.enums import (
    ReconstructionFailureCategory,
    ReconstructionMethod,
)


class ReconstructionMetadata(BaseModel):
    """Metadata for the Reconstruction Learning benchmark suite."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(..., description="Reconstruction benchmark identifier")
    title: str = Field(..., description="Benchmark suite title")
    description: str = Field(..., description="Scientific objective and description")
    methods: list[str] = Field(..., description="Evaluated reconstruction methods")
    architectures: list[str] = Field(..., description="Evaluated encoder architectures")
    mask_ratios: list[float] = Field(..., description="Evaluated masking ratios")
    dataset_id: str = Field(..., description="Source image dataset")
    created_at_utc: str = Field(..., description="Timestamp")


class VisualTripletSample(BaseModel):
    """Visual triplet: Original, Corrupted/Masked, and Reconstructed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str = Field(..., description="Unique sample identifier")
    class_name: str = Field(..., description="Post-hoc class label name")
    method: str = Field(..., description="Reconstruction method applied")
    original_image: list[list[list[float]]] = Field(
        ..., description="Clean target image [C x H x W]"
    )
    corrupted_or_masked_image: list[list[list[float]]] = Field(
        ..., description="Model input image [C x H x W]"
    )
    reconstructed_image: list[list[list[float]]] = Field(
        ..., description="Reconstructed output image [C x H x W]"
    )
    error_map: list[list[float]] = Field(
        ..., description="2D spatial heatmap of pixel reconstruction errors [H x W]"
    )
    masked_patch_indices: list[int] = Field(
        default_factory=list, description="Patch indices that were masked"
    )
    sample_mse: float = Field(
        ..., description="Sample mean squared reconstruction error"
    )
    failure_category: str | None = Field(
        default=None, description="Assigned failure taxonomy category if failed"
    )


class MaskingRatioPoint(BaseModel):
    """Masking ratio scaling evaluation point."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mask_ratio: float = Field(..., description="Evaluated masking ratio")
    mask_ratio_percent: str = Field(..., description="Human-readable ratio label")
    reconstruction_mse: float = Field(
        ..., description="Reconstruction error on masked patches"
    )
    linear_probe_accuracy: float = Field(
        ..., description="Downstream linear probe accuracy"
    )
    latent_std: float = Field(
        ..., description="Mean latent representation standard deviation"
    )


class ThreeWayComparisonEntry(BaseModel):
    """Side-by-side benchmark among Supervised, SimCLR, and Reconstruction learning."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    architecture: str = Field(..., description="Encoder architecture name")
    supervised_accuracy: float = Field(..., description="Supervised test accuracy")
    simclr_accuracy: float = Field(..., description="SimCLR linear probe accuracy")
    reconstruction_accuracy: float = Field(
        ..., description="Reconstruction linear probe accuracy"
    )
    supervised_latent_std: float = Field(..., description="Supervised latent std")
    simclr_latent_std: float = Field(..., description="SimCLR latent std")
    reconstruction_latent_std: float = Field(
        ..., description="Reconstruction latent std"
    )
    supervised_compactness: float = Field(
        ..., description="Supervised intra-class compactness"
    )
    simclr_compactness: float = Field(..., description="SimCLR intra-class compactness")
    reconstruction_compactness: float = Field(
        ..., description="Reconstruction compactness"
    )
    supervised_separation: float = Field(
        ..., description="Supervised inter-class separation"
    )
    simclr_separation: float = Field(..., description="SimCLR inter-class separation")
    reconstruction_separation: float = Field(
        ..., description="Reconstruction separation"
    )


class ReconstructionLayerProbeEntry(BaseModel):
    """Layer-wise probe accuracy across architectural depth."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    layer_id: str = Field(..., description="Layer identifier")
    depth_index: int = Field(..., description="Zero-indexed depth")
    supervised_accuracy: float = Field(..., description="Supervised probe accuracy")
    simclr_accuracy: float = Field(..., description="SimCLR probe accuracy")
    reconstruction_accuracy: float = Field(
        ..., description="Reconstruction probe accuracy"
    )


class ReconstructionFailureCase(BaseModel):
    """Specific diagnosed failure case in reconstruction learning."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str = Field(..., description="Sample ID")
    category: str = Field(..., description="Failure taxonomy category")
    reconstruction_mse: float = Field(..., description="Reconstruction MSE")
    description: str = Field(..., description="Factual description of failure")
    patch_index: int | None = Field(
        default=None, description="Worst patch index if applicable"
    )


def _generate_synthetic_image(seed: int, pattern: str) -> list[list[list[float]]]:
    """Generate deterministic 3x8x8 synthetic image pattern."""
    c, h, w = 3, 8, 8
    img: list[list[list[float]]] = []
    for ch in range(c):
        plane: list[list[float]] = []
        for r in range(h):
            row: list[float] = []
            for col in range(w):
                if pattern == "diagonal":
                    val = math.sin((r + col + seed + ch * 2) * 0.7) * 0.4 + 0.5
                elif pattern == "block":
                    val = 0.8 if (r < 4 and col < 4) or (r >= 4 and col >= 4) else 0.2
                    val = (val + ch * 0.1) % 1.0
                elif pattern == "circle":
                    dist = math.sqrt((r - 3.5) ** 2 + (col - 3.5) ** 2)
                    val = max(0.0, 1.0 - dist / 4.0) * 0.7 + ch * 0.1
                else:
                    val = math.cos((r * 0.8 + seed) + col * 0.4) * 0.3 + 0.5
                row.append(round(min(1.0, max(0.0, val)), 4))
            plane.append(row)
        img.append(plane)
    return img


def generate_reconstruction_demo_data() -> dict[str, Any]:
    """Synthesize complete PRISM Reconstruction benchmark research dataset."""
    metadata = ReconstructionMetadata(
        experiment_id="recon_benchmark_v1",
        title="PRISM Reconstruction & Masked Representation Learning Laboratory",
        description=(
            "Investigation of visual representations emerging from masked patch "
            "reconstruction (MIM) and denoising autoencoders (DAE) across CNN, "
            "ResNet, and ViT encoders compared against supervised and SimCLR baselines."
        ),
        methods=[
            ReconstructionMethod.MASKED_PATCH_RECONSTRUCTION.value,
            ReconstructionMethod.DENOISING_AUTOENCODER.value,
        ],
        architectures=[
            ModelFamily.VISION_TRANSFORMER.value,
            ModelFamily.RESNET.value,
            ModelFamily.CNN.value,
        ],
        mask_ratios=[0.25, 0.50, 0.75],
        dataset_id="cifar10_reconstruction_curated",
        created_at_utc="2026-09-03T00:00:00Z",
    )

    # 1. Visual triplets for ViT Masked Patch Reconstruction
    triplets_vit: list[VisualTripletSample] = []
    patterns = ["diagonal", "block", "circle", "wave", "diagonal", "block"]
    class_names = ["plane", "car", "bird", "cat", "deer", "ship"]

    for i in range(6):
        orig = _generate_synthetic_image(seed=i * 11 + 7, pattern=patterns[i])
        # Mask 2 out of 4 patches (total 4 patches in 8x8 image with 4x4 patches)
        masked_patches = [1, 2] if i % 2 == 0 else [0, 3]

        # Create masked image
        masked_img = json.loads(json.dumps(orig))
        for p_idx in masked_patches:
            pr = (p_idx // 2) * 4
            pc = (p_idx % 2) * 4
            for ch in range(3):
                for r in range(4):
                    for col in range(4):
                        masked_img[ch][pr + r][pc + col] = (
                            0.05  # gray mask token representation
                        )

        # Reconstructed image: visible patches preserved,
        # masked patches predicted
        rec_img = json.loads(json.dumps(orig))
        for p_idx in masked_patches:
            pr = (p_idx // 2) * 4
            pc = (p_idx % 2) * 4
            blur_factor = (
                0.12 if i != 3 else 0.45
            )  # sample 3 will be a localized failure
            for ch in range(3):
                for r in range(4):
                    for col in range(4):
                        rec_img[ch][pr + r][pc + col] = round(
                            orig[ch][pr + r][pc + col] * (1.0 - blur_factor)
                            + 0.3 * blur_factor,
                            4,
                        )

        # Compute error map [8 x 8]
        error_map: list[list[float]] = []
        sample_sq_sum = 0.0
        for r in range(8):
            err_row: list[float] = []
            for col in range(8):
                pix_err = (
                    sum(
                        (rec_img[ch][r][col] - orig[ch][r][col]) ** 2 for ch in range(3)
                    )
                    / 3.0
                )
                sample_sq_sum += pix_err
                err_row.append(round(pix_err, 4))
            error_map.append(err_row)

        sample_mse = sample_sq_sum / 64.0
        failure_cat = (
            ReconstructionFailureCategory.LOCALIZED_PATCH_FAILURE.value
            if i == 3
            else (
                ReconstructionFailureCategory.HIGH_RECONSTRUCTION_ERROR.value
                if i == 5
                else None
            )
        )

        triplets_vit.append(
            VisualTripletSample(
                sample_id=f"vit_mim_sample_{i + 1}",
                class_name=class_names[i],
                method="masked_patch_reconstruction",
                original_image=orig,
                corrupted_or_masked_image=masked_img,
                reconstructed_image=rec_img,
                error_map=error_map,
                masked_patch_indices=masked_patches,
                sample_mse=round(sample_mse, 4),
                failure_category=failure_cat,
            )
        )

    # 2. Visual triplets for Denoising Autoencoder (CNN / ResNet)
    triplets_dae: list[VisualTripletSample] = []
    for i in range(4):
        orig = _generate_synthetic_image(
            seed=i * 23 + 3, pattern=patterns[i % len(patterns)]
        )
        # Corrupted input: Gaussian noise addition
        corr_img = json.loads(json.dumps(orig))
        for ch in range(3):
            for r in range(8):
                for col in range(8):
                    noise = math.sin(i * 17 + ch * 5 + r * 3 + col * 2) * 0.25
                    corr_img[ch][r][col] = round(
                        min(1.0, max(0.0, orig[ch][r][col] + noise)), 4
                    )

        # Denoised reconstruction
        rec_img = json.loads(json.dumps(orig))
        for ch in range(3):
            for r in range(8):
                for col in range(8):
                    rec_img[ch][r][col] = round(
                        orig[ch][r][col] * 0.85 + corr_img[ch][r][col] * 0.15, 4
                    )

        error_map = []
        sample_sq_sum = 0.0
        for r in range(8):
            err_row = []
            for col in range(8):
                pix_err = (
                    sum(
                        (rec_img[ch][r][col] - orig[ch][r][col]) ** 2 for ch in range(3)
                    )
                    / 3.0
                )
                sample_sq_sum += pix_err
                err_row.append(round(pix_err, 4))
            error_map.append(err_row)

        triplets_dae.append(
            VisualTripletSample(
                sample_id=f"dae_sample_{i + 1}",
                class_name=class_names[i],
                method="denoising_autoencoder",
                original_image=orig,
                corrupted_or_masked_image=corr_img,
                reconstructed_image=rec_img,
                error_map=error_map,
                masked_patch_indices=[],
                sample_mse=round(sample_sq_sum / 64.0, 4),
                failure_category=(
                    ReconstructionFailureCategory.CORRUPTION_RECOVERY_FAILURE.value
                    if i == 2
                    else None
                ),
            )
        )

    # 3. Training dynamics history (20 epochs)
    dynamics = {
        "epochs": list(range(1, 21)),
        "total_loss": [round(0.24 * (0.88**ep) + 0.018, 4) for ep in range(20)],
        "masked_mse": [round(0.29 * (0.86**ep) + 0.022, 4) for ep in range(20)],
        "latent_std": [round(0.18 + 0.24 * (1.0 - (0.90**ep)), 4) for ep in range(20)],
        "learning_rate": [
            round(0.05 * 0.5 * (1.0 + math.cos(math.pi * ep / 20.0)), 5)
            for ep in range(20)
        ],
    }

    # 4. Masking ratio study
    masking_ratio_study = [
        MaskingRatioPoint(
            mask_ratio=0.25,
            mask_ratio_percent="25%",
            reconstruction_mse=0.0195,
            linear_probe_accuracy=0.684,
            latent_std=0.382,
        ),
        MaskingRatioPoint(
            mask_ratio=0.50,
            mask_ratio_percent="50%",
            reconstruction_mse=0.0342,
            linear_probe_accuracy=0.748,
            latent_std=0.426,
        ),
        MaskingRatioPoint(
            mask_ratio=0.75,
            mask_ratio_percent="75%",
            reconstruction_mse=0.0781,
            linear_probe_accuracy=0.712,
            latent_std=0.408,
        ),
    ]

    # 5. Three-way comparison (Supervised vs SimCLR vs Reconstruction)
    three_way_comparison = [
        ThreeWayComparisonEntry(
            architecture="Vision Transformer (ViT)",
            supervised_accuracy=0.792,
            simclr_accuracy=0.764,
            reconstruction_accuracy=0.748,
            supervised_latent_std=0.512,
            simclr_latent_std=0.468,
            reconstruction_latent_std=0.426,
            supervised_compactness=0.184,
            simclr_compactness=0.221,
            reconstruction_compactness=0.289,
            supervised_separation=1.842,
            simclr_separation=1.624,
            reconstruction_separation=1.285,
        ),
        ThreeWayComparisonEntry(
            architecture="Residual Network (ResNet)",
            supervised_accuracy=0.814,
            simclr_accuracy=0.782,
            reconstruction_accuracy=0.729,
            supervised_latent_std=0.495,
            simclr_latent_std=0.452,
            reconstruction_latent_std=0.402,
            supervised_compactness=0.172,
            simclr_compactness=0.215,
            reconstruction_compactness=0.312,
            supervised_separation=1.921,
            simclr_separation=1.689,
            reconstruction_separation=1.210,
        ),
        ThreeWayComparisonEntry(
            architecture="Convolutional Network (CNN)",
            supervised_accuracy=0.756,
            simclr_accuracy=0.718,
            reconstruction_accuracy=0.682,
            supervised_latent_std=0.442,
            simclr_latent_std=0.418,
            reconstruction_latent_std=0.374,
            supervised_compactness=0.210,
            simclr_compactness=0.248,
            reconstruction_compactness=0.334,
            supervised_separation=1.642,
            simclr_separation=1.450,
            reconstruction_separation=1.095,
        ),
    ]

    # 6. Layer probes across depth
    layer_probes = [
        ReconstructionLayerProbeEntry(
            layer_id="patch_embed",
            depth_index=0,
            supervised_accuracy=0.354,
            simclr_accuracy=0.382,
            reconstruction_accuracy=0.468,
        ),
        ReconstructionLayerProbeEntry(
            layer_id="encoder_stage_1",
            depth_index=1,
            supervised_accuracy=0.528,
            simclr_accuracy=0.564,
            reconstruction_accuracy=0.612,
        ),
        ReconstructionLayerProbeEntry(
            layer_id="encoder_stage_2",
            depth_index=2,
            supervised_accuracy=0.672,
            simclr_accuracy=0.689,
            reconstruction_accuracy=0.695,
        ),
        ReconstructionLayerProbeEntry(
            layer_id="final_representation",
            depth_index=3,
            supervised_accuracy=0.792,
            simclr_accuracy=0.764,
            reconstruction_accuracy=0.748,
        ),
    ]

    # 7. Diagnosed Failure Cases
    failure_cases = [
        ReconstructionFailureCase(
            sample_id="vit_mim_sample_3",
            category=ReconstructionFailureCategory.LOCALIZED_PATCH_FAILURE.value,
            reconstruction_mse=0.0784,
            description="Localized patch failure at edge boundary (patch 1).",
            patch_index=1,
        ),
        ReconstructionFailureCase(
            sample_id="vit_mim_sample_5",
            category=ReconstructionFailureCategory.HIGH_RECONSTRUCTION_ERROR.value,
            reconstruction_mse=0.0921,
            description="High overall reconstruction error across all masked regions.",
            patch_index=None,
        ),
        ReconstructionFailureCase(
            sample_id="dae_sample_2",
            category=ReconstructionFailureCategory.CORRUPTION_RECOVERY_FAILURE.value,
            reconstruction_mse=0.0652,
            description="Model failed to denoise structured occlusion corruptions.",
            patch_index=None,
        ),
        ReconstructionFailureCase(
            sample_id="dae_sample_4",
            category=ReconstructionFailureCategory.OVER_SMOOTH_RECONSTRUCTION.value,
            reconstruction_mse=0.0412,
            description="Reconstructed output blurred high-frequency texture details.",
            patch_index=None,
        ),
    ]

    return {
        "metadata": metadata.model_dump(),
        "triplets_masked_patch": [t.model_dump() for t in triplets_vit],
        "triplets_denoising": [t.model_dump() for t in triplets_dae],
        "dynamics": dynamics,
        "masking_ratio_study": [p.model_dump() for p in masking_ratio_study],
        "three_way_comparison": [c.model_dump() for c in three_way_comparison],
        "layer_probes": [lp.model_dump() for lp in layer_probes],
        "failure_cases": [f.model_dump() for f in failure_cases],
    }


def save_reconstruction_demo_data(
    filepath: str = "frontend/app/data/reconstructionDataset.json",
) -> str:
    """Generate and write reconstruction dataset to JSON."""
    data = generate_reconstruction_demo_data()
    serialized = json.dumps(data, indent=2)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(serialized)
    return filepath


class ReconstructionService:
    """Service interface for querying reconstruction experiments and benchmark data."""

    def __init__(self, data_filepath: str | None = None) -> None:
        self.data_filepath = data_filepath
        self._data: dict[str, Any] | None = None

    def _get_data(self) -> dict[str, Any]:
        if self._data is None:
            if self.data_filepath:
                with open(self.data_filepath, encoding="utf-8") as f:
                    self._data = json.load(f)
            else:
                self._data = generate_reconstruction_demo_data()
        return self._data

    def get_metadata(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._get_data()["metadata"])

    def get_triplets(
        self, method: str = "masked_patch_reconstruction"
    ) -> list[dict[str, Any]]:
        data = self._get_data()
        if method == "denoising_autoencoder":
            return cast(list[dict[str, Any]], data["triplets_denoising"])
        return cast(list[dict[str, Any]], data["triplets_masked_patch"])

    def get_dynamics(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._get_data()["dynamics"])

    def get_masking_ratio_study(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self._get_data()["masking_ratio_study"])

    def get_three_way_comparison(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self._get_data()["three_way_comparison"])

    def get_layer_probes(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self._get_data()["layer_probes"])

    def get_failure_cases(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self._get_data()["failure_cases"])
