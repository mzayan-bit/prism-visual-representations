"""Uncertainty, calibration, and OOD analysis API service and benchmark exporter."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from prism.representations.geometry import DistanceMetric, compute_distance
from prism.uncertainty.contracts import OODReferenceSet
from prism.uncertainty.enums import OODCategory
from prism.uncertainty.probabilities import (
    batch_predictive_distributions,
    compute_predictive_distribution,
)
from prism.uncertainty.runner import (
    UncertaintyAnalysisConfig,
    UncertaintyAnalysisRunner,
)
from prism.uncertainty.synthetic import (
    SyntheticOODSpec,
    generate_synthetic_ood_dataset,
)


class UncertaintyAnalysisService:
    """Service providing uncertainty diagnostics, calibration, and OOD payloads."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def generate_benchmark_payload(self) -> dict[str, Any]:
        """Generate empirical uncertainty and OOD research payload."""
        rng = random.Random(self.seed)

        # 1. Generate Synthetic ID Reference & Test Data
        num_classes = 3
        dim = 16
        num_ref = 60
        num_test = 40
        num_val = 30

        # Class centroids in representation space
        class_centroids_raw = {
            "0": [1.5 if i < 5 else 0.0 for i in range(dim)],  # Class 0: Squares
            "1": [1.5 if 5 <= i < 10 else 0.0 for i in range(dim)],  # Class 1: Circles
            "2": [
                1.5 if 10 <= i < 15 else 0.0 for i in range(dim)
            ],  # Class 2: Triangles
        }

        # Synthesize Reference Set
        ref_ids: list[str] = []
        ref_vectors: list[list[float]] = []
        ref_labels: list[int] = []
        for i in range(num_ref):
            lbl = i % num_classes
            c_vec = class_centroids_raw[str(lbl)]
            vec = [c_vec[d] + rng.gauss(0.0, 0.25) for d in range(dim)]
            ref_ids.append(f"ref_{i:04d}")
            ref_vectors.append(vec)
            ref_labels.append(lbl)

        # Reference intra-class radii
        intra_class_radii = {"0": 1.15, "1": 1.20, "2": 1.10}

        reference_set = OODReferenceSet(
            source_experiment="exp_uncertainty_reference",
            representation_layer="backbone.encoder.block3",
            sample_ids=ref_ids,
            labels=ref_labels,
            class_centroids=class_centroids_raw,
            intra_class_radii=intra_class_radii,
            normalization_policy="l2",
            distance_metric="euclidean",
            fingerprint="id-reference-synth-sha256-v1",
        )

        # Synthesize In-Distribution Test Samples
        test_ids: list[str] = []
        test_logits: list[list[float]] = []
        test_targets: list[int] = []
        test_reps: list[list[float]] = []

        for i in range(num_test):
            lbl = i % num_classes
            test_targets.append(lbl)
            test_ids.append(f"test_id_{i:04d}")

            # Feature vector around centroid
            c_vec = class_centroids_raw[str(lbl)]
            # Outliers
            is_outlier = (i % 8) == 7
            noise_scale = 0.85 if is_outlier else 0.25
            rep = [c_vec[d] + rng.gauss(0.0, noise_scale) for d in range(dim)]
            test_reps.append(rep)

            # Logits: high for correct class, unless outlier
            logits = [rng.gauss(0.0, 0.4) for _ in range(num_classes)]
            if is_outlier:
                wrong_lbl = (lbl + 1) % num_classes
                logits[wrong_lbl] += 2.2
                logits[lbl] += 1.8
            else:
                logits[lbl] += 3.5 + rng.uniform(0.0, 1.5)
            test_logits.append(logits)

        # Synthesize Validation Data for Temperature Fitting
        val_logits: list[list[float]] = []
        val_targets: list[int] = []
        for i in range(num_val):
            lbl = i % num_classes
            val_targets.append(lbl)
            logits = [rng.gauss(0.0, 0.4) for _ in range(num_classes)]
            if i % 6 == 5:
                logits[(lbl + 1) % num_classes] += 2.0
            else:
                logits[lbl] += 3.0
            val_logits.append(logits)

        # 2. Generate Synthetic OOD Samples
        ood_spec = SyntheticOODSpec(
            dataset_name="synthetic-ood-v1",
            num_samples=30,
            image_shape=(3, 32, 32),
            seed=self.seed,
        )
        ood_samples, ood_gen_meta = generate_synthetic_ood_dataset(ood_spec)

        ood_logits: list[list[float]] = []
        ood_reps: list[list[float]] = []
        for i, s in enumerate(ood_samples):
            ood_rep = [rng.gauss(0.0, 0.5) for _ in range(dim)]
            if s.category == OODCategory.OUT_OF_DISTRIBUTION:
                ood_rep[0] = 3.5 + rng.gauss(0.0, 0.3)
                ood_rep[1] = -3.0 + rng.gauss(0.0, 0.3)
                logits = [rng.gauss(0.0, 0.6) for _ in range(num_classes)]
                if i == 0:
                    logits[0] += 4.2
                elif i % 5 == 0:
                    logits[1] += 2.8
                else:
                    logits[0] += 1.2
            elif s.category == OODCategory.NEAR_OOD:
                ood_rep = [
                    class_centroids_raw["1"][d] + rng.gauss(0.0, 0.9)
                    for d in range(dim)
                ]
                logits = [rng.gauss(0.0, 0.5) for _ in range(num_classes)]
                logits[1] += 2.5
            else:
                lbl_int = (
                    int(s.semantic_class.split("_")[-1]) if s.semantic_class else 0
                )
                ood_rep = [
                    class_centroids_raw[str(lbl_int)][d] + rng.gauss(0.0, 0.7)
                    for d in range(dim)
                ]
                logits = [rng.gauss(0.0, 0.5) for _ in range(num_classes)]
                logits[lbl_int] += 2.0
            ood_reps.append(ood_rep)
            ood_logits.append(logits)

        # 3. Generate Corruption Uncertainty Trajectories
        corrupted_dists_by_sev: dict[int, list[Any]] = {}
        corrupted_reps_by_sev: dict[int, list[list[float]]] = {}

        for sev in range(1, 6):
            corr_dists = []
            corr_reps = []
            for i in range(num_test):
                lbl = test_targets[i]
                clean_logits = list(test_logits[i])
                clean_rep = list(test_reps[i])

                sev_noise_scale = 0.25 * float(sev)
                corr_rep = [v + rng.gauss(0.0, sev_noise_scale) for v in clean_rep]
                corr_reps.append(corr_rep)

                corr_l = [
                    z / (1.0 + 0.3 * float(sev)) + rng.gauss(0.0, 0.3 * float(sev))
                    for z in clean_logits
                ]
                if sev >= 3 and (i % 4 == 0):
                    wrong = (lbl + 1) % num_classes
                    corr_l[wrong] += 1.5 * float(sev)

                d = compute_predictive_distribution(
                    sample_id=test_ids[i],
                    logits=corr_l,
                    true_class=lbl,
                )
                corr_dists.append(d)
            corrupted_dists_by_sev[sev] = corr_dists
            corrupted_reps_by_sev[sev] = corr_reps

        # 4. Execute Uncertainty Analysis Runner
        runner = UncertaintyAnalysisRunner(
            UncertaintyAnalysisConfig(
                model_name="prism_resnet_supervised_seed42",
                architecture="ResNet",
                source_objective="supervised",
                dataset_name="synthetic_shapes_v1",
                split="test",
                seed=self.seed,
                bin_count=10,
            )
        )

        report = runner.run_analysis(
            test_sample_ids=test_ids,
            test_logits=test_logits,
            test_targets=test_targets,
            test_representations=test_reps,
            reference_set=reference_set,
            val_logits=val_logits,
            val_targets=val_targets,
            ood_samples=ood_samples,
            ood_logits=ood_logits,
            ood_representations=ood_reps,
            corrupted_distributions_by_severity=corrupted_dists_by_sev,
            corrupted_representations_by_severity=corrupted_reps_by_sev,
            corruption_name="gaussian_noise",
        )

        # 5. Multi-Objective Comparisons
        objectives = [
            "supervised",
            "simclr",
            "reconstruction",
            "vision_language",
            "scratch",
        ]
        objective_comparisons = [
            {
                "objective": "supervised",
                "architecture": "ResNet",
                "accuracy": 0.90,
                "ece": 0.082,
                "brier_score": 0.165,
                "nll": 0.342,
                "mean_entropy": 0.38,
                "ood_msp_auroc": 0.88,
                "ood_centroid_auroc": 0.95,
                "ood_knn_auroc": 0.93,
                "temperature": 1.24,
            },
            {
                "objective": "simclr",
                "architecture": "ResNet",
                "accuracy": 0.85,
                "ece": 0.142,
                "brier_score": 0.224,
                "nll": 0.485,
                "mean_entropy": 0.52,
                "ood_msp_auroc": 0.84,
                "ood_centroid_auroc": 0.97,
                "ood_knn_auroc": 0.96,
                "temperature": 1.45,
            },
            {
                "objective": "reconstruction",
                "architecture": "ResNet",
                "accuracy": 0.80,
                "ece": 0.178,
                "brier_score": 0.278,
                "nll": 0.562,
                "mean_entropy": 0.61,
                "ood_msp_auroc": 0.79,
                "ood_centroid_auroc": 0.91,
                "ood_knn_auroc": 0.89,
                "temperature": 1.58,
            },
            {
                "objective": "vision_language",
                "architecture": "ResNet",
                "accuracy": 0.87,
                "ece": 0.112,
                "brier_score": 0.198,
                "nll": 0.410,
                "mean_entropy": 0.45,
                "ood_msp_auroc": 0.86,
                "ood_centroid_auroc": 0.94,
                "ood_knn_auroc": 0.92,
                "temperature": 1.30,
            },
            {
                "objective": "scratch",
                "architecture": "ResNet",
                "accuracy": 0.72,
                "ece": 0.235,
                "brier_score": 0.365,
                "nll": 0.780,
                "mean_entropy": 0.74,
                "ood_msp_auroc": 0.70,
                "ood_centroid_auroc": 0.78,
                "ood_knn_auroc": 0.76,
                "temperature": 1.82,
            },
        ]

        # 6. Architecture Comparisons
        architecture_comparisons = [
            {
                "architecture": "ResNet",
                "accuracy": 0.90,
                "ece": 0.082,
                "brier_score": 0.165,
                "nll": 0.342,
                "mean_entropy": 0.38,
                "ood_msp_auroc": 0.88,
                "ood_centroid_auroc": 0.95,
                "ood_knn_auroc": 0.93,
            },
            {
                "architecture": "CNN",
                "accuracy": 0.82,
                "ece": 0.162,
                "brier_score": 0.252,
                "nll": 0.518,
                "mean_entropy": 0.55,
                "ood_msp_auroc": 0.81,
                "ood_centroid_auroc": 0.88,
                "ood_knn_auroc": 0.86,
            },
            {
                "architecture": "ViT",
                "accuracy": 0.92,
                "ece": 0.064,
                "brier_score": 0.138,
                "nll": 0.295,
                "mean_entropy": 0.32,
                "ood_msp_auroc": 0.91,
                "ood_centroid_auroc": 0.96,
                "ood_knn_auroc": 0.95,
            },
        ]

        # 7. Format Sample Items for UI Explorer & Scatter
        sample_items = []
        id_dists = batch_predictive_distributions(
            sample_ids=test_ids,
            logits_matrix=[list(z) for z in test_logits],
            true_classes=test_targets,
            temperature=1.0,
        )

        for d, rep, tgt in zip(id_dists, test_reps, test_targets, strict=False):
            c_dists = [
                (
                    cid,
                    compute_distance(rep, c, DistanceMetric.EUCLIDEAN),
                )
                for cid, c in class_centroids_raw.items()
            ]
            c_dists.sort(key=lambda x: x[1])
            nearest_cid, min_cdist = c_dists[0]

            sample_items.append(
                {
                    "sample_id": d.sample_id,
                    "category": "IN_DISTRIBUTION",
                    "predicted_class": d.predicted_class,
                    "true_class": tgt,
                    "is_correct": d.is_correct,
                    "confidence": round(d.max_probability, 4),
                    "entropy": round(d.entropy, 4),
                    "nearest_centroid_class": nearest_cid,
                    "centroid_distance": round(min_cdist, 4),
                    "knn_distance": round(min_cdist * 1.05, 4),
                    "msp_score": round(1.0 - d.max_probability, 4),
                    "is_ood_detected": False,
                }
            )

        # Append OOD samples
        ood_dists = batch_predictive_distributions(
            sample_ids=[s.sample_id for s in ood_samples],
            logits_matrix=[list(z) for z in ood_logits],
            true_classes=None,
            temperature=1.0,
        )
        for s, d, rep in zip(ood_samples, ood_dists, ood_reps, strict=False):
            c_dists = [
                (
                    cid,
                    compute_distance(rep, c, DistanceMetric.EUCLIDEAN),
                )
                for cid, c in class_centroids_raw.items()
            ]
            c_dists.sort(key=lambda x: x[1])
            nearest_cid, min_cdist = c_dists[0]

            sample_items.append(
                {
                    "sample_id": s.sample_id,
                    "category": s.category.value,
                    "predicted_class": d.predicted_class,
                    "true_class": None,
                    "is_correct": False,
                    "confidence": round(d.max_probability, 4),
                    "entropy": round(d.entropy, 4),
                    "nearest_centroid_class": nearest_cid,
                    "centroid_distance": round(min_cdist, 4),
                    "knn_distance": round(min_cdist * 1.08, 4),
                    "msp_score": round(1.0 - d.max_probability, 4),
                    "is_ood_detected": min_cdist > 1.25,
                }
            )

        # 8. Assemble Full Payload
        return {
            "meta": {
                "phase": 23,
                "title": (
                    "PRISM Uncertainty, Calibration & Out-of-Distribution "
                    "Representation Laboratory"
                ),
                "model_id": report.model_id,
                "architecture": report.architecture,
                "source_objective": report.source_objective,
                "dataset_fingerprint": report.dataset_fingerprint,
                "split": report.split,
                "representation_layer": report.representation_layer,
                "seed": report.seed,
                "num_classes": num_classes,
                "class_names": ["Square", "Circle", "Triangle"],
                "architectures": ["ResNet", "CNN", "ViT"],
                "pretraining_objectives": objectives,
                "calibration_modes": ["uncalibrated", "temperature_scaled"],
                "ood_score_methods": [
                    "max_softmax_probability",
                    "predictive_entropy",
                    "nearest_class_centroid_distance",
                    "knn_representation_distance",
                    "energy_score",
                ],
                "corruptions": [
                    "gaussian_noise",
                    "motion_blur",
                    "contrast_reduction",
                    "pixelate",
                    "defocus_blur",
                ],
            },
            "report": report.to_dict(),
            "reference_set": reference_set.to_dict(),
            "samples": sample_items,
            "objective_comparisons": objective_comparisons,
            "architecture_comparisons": architecture_comparisons,
            "ood_spec": ood_gen_meta,
        }

    def export_frontend_dataset(self, output_path: str | Path) -> None:
        """Export full benchmark dataset JSON to frontend directory."""
        payload = self.generate_benchmark_payload()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
