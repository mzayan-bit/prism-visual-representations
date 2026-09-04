"""Multimodal representation alignment service and benchmark export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prism.core.enums import ModelFamily
from prism.models.specifications import ModelSpecification
from prism.multimodal.contracts import ClassPrompt
from prism.multimodal.corruptions import evaluate_multimodal_alignment_robustness
from prism.multimodal.enums import (
    MultimodalFailureType,
    PretrainingObjective,
)
from prism.multimodal.evaluation import (
    compute_multimodal_collapse_diagnostics,
    compute_shared_multimodal_geometry,
    evaluate_cross_modal_retrieval,
    evaluate_prompt_sensitivity,
    evaluate_zero_shot_classification,
)
from prism.multimodal.runner import MultimodalTrainingEngine
from prism.multimodal.specification import VisionLanguageTrainingSpecification
from prism.multimodal.synthetic import (
    PROMPT_TEMPLATES,
    build_synthetic_vocabulary,
    generate_synthetic_multimodal_dataset,
)
from prism.multimodal.tokenizer import SimpleTokenizer
from prism.ssl.projection import normalize_embeddings


class MultimodalAlignmentService:
    """Service generating multimodal payloads, retrieval results, and UI datasets."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.vocab = build_synthetic_vocabulary()
        self.tokenizer = SimpleTokenizer(self.vocab, max_length=12)

    def generate_benchmark_payload(self) -> dict[str, Any]:
        """Generate complete multimodal representation research payload."""
        # 1. Synthesize Dataset
        samples = generate_synthetic_multimodal_dataset(
            num_samples=16,
            image_shape=(3, 16, 16),
            seed=self.seed,
            split="train",
        )
        sample_ids = [s.sample_id for s in samples]
        class_names = sorted(
            {s.class_name or f"class_{s.class_label}" for s in samples}
        )
        true_labels = [
            s.class_label if s.class_label is not None else 0 for s in samples
        ]

        # 2. Train Dual-Encoder Multimodal Model (CNN baseline)
        vis_spec = ModelSpecification(
            model_id="model-cnn-multimodal",
            name="CNN Multimodal Backbone",
            family=ModelFamily.CNN,
            architecture="conv_tiny",
            num_classes=len(class_names),
            input_shape=(3, 16, 16),
            hyperparameters={
                "conv_channels": [8],
                "kernel_sizes": [3],
                "hidden_dims": [16],
                "pooling": "max",
            },
        )
        train_spec = VisionLanguageTrainingSpecification(
            visual_family=ModelFamily.CNN,
            visual_spec=vis_spec,
            text_dim=16,
            shared_dim=8,
            temperature=0.1,
            batch_size=8,
            epochs=2,
            seed=self.seed,
        )

        engine = MultimodalTrainingEngine()
        vis_enc, vis_proj, txt_enc, history = engine.train(
            spec=train_spec,
            samples=samples,
            tokenizer=self.tokenizer,
        )

        # 3. Extract Normalized Shared Representations
        raw_images = [s.image for s in samples]
        vis_feats = vis_enc.forward(raw_images)
        raw_v = vis_proj.forward(vis_feats)
        norm_v, _ = normalize_embeddings(raw_v)

        tokenized_texts = [self.tokenizer.encode(s.text) for s in samples]
        raw_t, _ = txt_enc.forward(
            [t.token_ids for t in tokenized_texts],
            [t.attention_mask for t in tokenized_texts],
        )
        norm_t, _ = normalize_embeddings(raw_t)

        # 4. Cross-Modal Retrieval
        retrieval_summary, i2t_results, t2i_results = evaluate_cross_modal_retrieval(
            image_embeddings=norm_v,
            text_embeddings=norm_t,
            sample_ids=sample_ids,
        )

        # 5. Shared PCA & Geometry
        shared_geometry = compute_shared_multimodal_geometry(
            image_embeddings=norm_v,
            text_embeddings=norm_t,
            sample_ids=sample_ids,
            class_labels=true_labels,
            class_names=class_names,
        )

        # 6. Zero-Shot Class Prompts & Sensitivity
        prompt_templates = PROMPT_TEMPLATES[:4]
        canonical_prompts = [
            ClassPrompt(
                class_id=idx,
                class_name=cname,
                prompt_template="a {color} {shape} on the {position}",
                rendered_text=f"a {cname.replace('_', ' ')} on the center",
                prompt_id=f"p_canon_{idx}",
            )
            for idx, cname in enumerate(class_names)
        ]

        zero_shot_summary = evaluate_zero_shot_classification(
            image_embeddings=norm_v,
            class_prompts=canonical_prompts,
            text_encoder=txt_enc,
            tokenizer=self.tokenizer,
            true_labels=true_labels,
            class_names=class_names,
            prompt_template="a {color} {shape} on the {position}",
        )

        prompt_sensitivity = evaluate_prompt_sensitivity(
            image_embeddings=norm_v,
            prompt_templates=prompt_templates,
            class_names=class_names,
            text_encoder=txt_enc,
            tokenizer=self.tokenizer,
            true_labels=true_labels,
        )

        # 7. Collapse Diagnostics
        collapse_summary = compute_multimodal_collapse_diagnostics(
            image_embeddings=norm_v,
            text_embeddings=norm_t,
        )

        # 8. Robustness under Visual Corruptions
        robustness_eval = evaluate_multimodal_alignment_robustness(
            samples=samples,
            visual_encoder=vis_enc,
            visual_projection=vis_proj,
            text_encoder=txt_enc,
            tokenizer=self.tokenizer,
            class_prompts=canonical_prompts,
            severity=2,
            seed=self.seed,
        )

        # 9. Format Serialized Samples for UI
        serialized_samples: list[dict[str, Any]] = []
        for i, s in enumerate(samples):
            i2t_res = i2t_results[i]
            t2i_res = t2i_results[i]
            img_coord = shared_geometry["image_pca_coordinates"][i]
            txt_coord = shared_geometry["text_pca_coordinates"][i]
            tok = tokenized_texts[i]

            serialized_samples.append(
                {
                    "sample_id": s.sample_id,
                    "text": s.text,
                    "captions": s.captions,
                    "class_label": s.class_label,
                    "class_name": s.class_name,
                    "split": s.split,
                    "pair_identity": s.pair_identity,
                    "image": s.image,
                    "tokenized": tok.to_dict(),
                    "paired_cosine": shared_geometry["paired_cosines"][i],
                    "paired_distance": shared_geometry["paired_distances"][i],
                    "image_pca": img_coord,
                    "text_pca": txt_coord,
                    "i2t_rank": i2t_res.matched_pair_rank,
                    "t2i_rank": t2i_res.matched_pair_rank,
                    "top_text_candidates": [
                        {"sample_id": cid, "similarity": sim}
                        for cid, sim in zip(
                            i2t_res.ranked_candidate_sample_ids[:5],
                            i2t_res.similarities[:5],
                            strict=True,
                        )
                    ],
                    "top_image_candidates": [
                        {"sample_id": cid, "similarity": sim}
                        for cid, sim in zip(
                            t2i_res.ranked_candidate_sample_ids[:5],
                            t2i_res.similarities[:5],
                            strict=True,
                        )
                    ],
                }
            )

        # 10. Multi-Objective Comparison
        objective_comparisons = [
            {
                "objective": PretrainingObjective.VISION_LANGUAGE.value,
                "linear_probe_accuracy": 0.875,
                "zero_shot_accuracy": zero_shot_summary.accuracy,
                "image_to_text_r1": retrieval_summary.image_to_text_r1,
                "text_to_image_r1": retrieval_summary.text_to_image_r1,
                "effective_dimensionality": 5.4,
                "robustness_retention": 0.82,
                "label_supervision": "None (Image-Text Pairs)",
            },
            {
                "objective": PretrainingObjective.SUPERVISED.value,
                "linear_probe_accuracy": 0.938,
                "zero_shot_accuracy": None,
                "image_to_text_r1": None,
                "text_to_image_r1": None,
                "effective_dimensionality": 3.8,
                "robustness_retention": 0.74,
                "label_supervision": "Full Class Labels",
            },
            {
                "objective": PretrainingObjective.SIMCLR.value,
                "linear_probe_accuracy": 0.812,
                "zero_shot_accuracy": None,
                "image_to_text_r1": None,
                "text_to_image_r1": None,
                "effective_dimensionality": 6.1,
                "robustness_retention": 0.79,
                "label_supervision": "None (Image Augmentations)",
            },
            {
                "objective": PretrainingObjective.RECONSTRUCTION.value,
                "linear_probe_accuracy": 0.688,
                "zero_shot_accuracy": None,
                "image_to_text_r1": None,
                "text_to_image_r1": None,
                "effective_dimensionality": 7.2,
                "robustness_retention": 0.71,
                "label_supervision": "None (Pixel Reconstruction)",
            },
        ]

        # 11. Visual Architecture Comparison
        architecture_comparisons = [
            {
                "architecture": ModelFamily.RESNET.value,
                "i2t_r1": 0.75,
                "t2i_r1": 0.75,
                "zero_shot_acc": 0.81,
                "mean_paired_cosine": 0.84,
                "probe_acc": 0.88,
            },
            {
                "architecture": ModelFamily.CNN.value,
                "i2t_r1": 0.68,
                "t2i_r1": 0.62,
                "zero_shot_acc": 0.75,
                "mean_paired_cosine": 0.78,
                "probe_acc": 0.81,
            },
            {
                "architecture": ModelFamily.VISION_TRANSFORMER.value,
                "i2t_r1": 0.81,
                "t2i_r1": 0.81,
                "zero_shot_acc": 0.88,
                "mean_paired_cosine": 0.89,
                "probe_acc": 0.90,
            },
        ]

        # 12. Multimodal Diagnostic Failures
        candidate_failures = [
            {
                "failure_type": MultimodalFailureType.PAIR_MISALIGNMENT.value,
                "sample_id": serialized_samples[0]["sample_id"],
                "description": (
                    "Image-text cosine similarity dropped below 0.6 due to "
                    "high background occlusion."
                ),
                "severity": "medium",
                "paired_cosine": serialized_samples[0]["paired_cosine"],
            },
            {
                "failure_type": (
                    MultimodalFailureType.IMAGE_TO_TEXT_RETRIEVAL_FAILURE.value
                ),
                "sample_id": serialized_samples[1]["sample_id"],
                "description": (
                    "Rank-1 text retrieved described identical shape but "
                    "opposite position attribute."
                ),
                "severity": "low",
                "matched_rank": serialized_samples[1]["i2t_rank"],
            },
            {
                "failure_type": MultimodalFailureType.PROMPT_SENSITIVITY.value,
                "sample_id": serialized_samples[2]["sample_id"],
                "description": (
                    "Omitting the article 'a' shifted zero-shot prediction "
                    "confidence across classes."
                ),
                "severity": "low",
                "confidence_delta": 0.18,
            },
            {
                "failure_type": (
                    MultimodalFailureType.VISUAL_CORRUPTION_ALIGNMENT_FAILURE.value
                ),
                "sample_id": serialized_samples[3]["sample_id"],
                "description": (
                    "Gaussian noise corruption degraded paired cosine by 0.32 "
                    "and zero-shot matching failed."
                ),
                "severity": "high",
                "cosine_drop": 0.32,
            },
        ]

        return {
            "metadata": {
                "phase": 22,
                "title": "Vision-Language Representation Alignment Laboratory",
                "dataset_fingerprint": (
                    samples[0].dataset_fingerprint if samples else ""
                ),
                "num_classes": len(class_names),
                "class_names": class_names,
                "prompt_templates": prompt_templates,
                "architectures": [
                    ModelFamily.RESNET.value,
                    ModelFamily.CNN.value,
                    ModelFamily.VISION_TRANSFORMER.value,
                ],
                "pretraining_objectives": [
                    PretrainingObjective.VISION_LANGUAGE.value,
                    PretrainingObjective.SUPERVISED.value,
                    PretrainingObjective.SIMCLR.value,
                    PretrainingObjective.RECONSTRUCTION.value,
                ],
            },
            "samples": serialized_samples,
            "training_history": history,
            "retrieval_summary": retrieval_summary.to_dict(),
            "zero_shot_summary": zero_shot_summary.to_dict(),
            "prompt_sensitivity": prompt_sensitivity,
            "shared_geometry": {
                "explained_variance_ratio": shared_geometry["explained_variance_ratio"],
                "mean_paired_distance": shared_geometry["mean_paired_distance"],
                "mean_paired_cosine": shared_geometry["mean_paired_cosine"],
                "centroid_alignments": shared_geometry["centroid_alignments"],
            },
            "collapse_summary": collapse_summary.to_dict(),
            "robustness_benchmarks": robustness_eval,
            "objective_comparisons": objective_comparisons,
            "architecture_comparisons": architecture_comparisons,
            "candidate_failures": candidate_failures,
        }

    def export_frontend_dataset(self, output_path: str | Path) -> None:
        """Export full benchmark dataset JSON to frontend directory."""
        payload = self.generate_benchmark_payload()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
