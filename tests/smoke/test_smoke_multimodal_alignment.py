"""End-to-End Smoke Test for Multimodal Vision-Language Representation Alignment."""

from __future__ import annotations

import json

from prism.core.enums import ModelFamily
from prism.models.specifications import ModelSpecification
from prism.multimodal.contracts import ClassPrompt
from prism.multimodal.corruptions import evaluate_multimodal_alignment_robustness
from prism.multimodal.evaluation import (
    compute_multimodal_collapse_diagnostics,
    compute_shared_multimodal_geometry,
    evaluate_cross_modal_retrieval,
    evaluate_prompt_sensitivity,
    evaluate_zero_shot_classification,
)
from prism.multimodal.reports import VisionLanguageRepresentationReport
from prism.multimodal.runner import MultimodalTrainingEngine
from prism.multimodal.specification import VisionLanguageTrainingSpecification
from prism.multimodal.synthetic import (
    PROMPT_TEMPLATES,
    build_synthetic_vocabulary,
    generate_synthetic_multimodal_dataset,
)
from prism.multimodal.tokenizer import SimpleTokenizer
from prism.ssl.projection import normalize_embeddings


def test_smoke_multimodal_alignment_pipeline() -> None:
    """Execute complete Phase 22 multimodal alignment pipeline end-to-end."""
    seed = 42

    # 1. Deterministic Synthetic Dataset
    samples = generate_synthetic_multimodal_dataset(
        num_samples=8,
        image_shape=(3, 8, 8),
        seed=seed,
        split="train",
    )
    sample_ids = [s.sample_id for s in samples]
    class_names = sorted({s.class_name or f"class_{s.class_label}" for s in samples})
    true_labels = [s.class_label if s.class_label is not None else 0 for s in samples]

    # 2. Tokenizer and Vocabulary
    vocab = build_synthetic_vocabulary()
    tokenizer = SimpleTokenizer(vocab, max_length=10)

    # 3. Model Specifications & Training
    vis_spec = ModelSpecification(
        model_id="smoke-cnn-multimodal",
        name="Smoke CNN Backbone",
        family=ModelFamily.CNN,
        architecture="conv_tiny",
        num_classes=len(class_names),
        input_shape=(3, 8, 8),
        hyperparameters={
            "conv_channels": [4],
            "kernel_sizes": [3],
            "hidden_dims": [8],
            "pooling": "max",
        },
    )

    spec = VisionLanguageTrainingSpecification(
        visual_family=ModelFamily.CNN,
        visual_spec=vis_spec,
        text_dim=8,
        shared_dim=4,
        temperature=0.1,
        learning_rate=0.02,
        batch_size=4,
        epochs=2,
        seed=seed,
    )

    engine = MultimodalTrainingEngine()
    vis_enc, vis_proj, txt_enc, history = engine.train(
        spec=spec,
        samples=samples,
        tokenizer=tokenizer,
    )

    assert len(history) == 2

    # 4. Extract Shared Normalized Embeddings
    raw_images = [s.image for s in samples]
    vis_feats = vis_enc.forward(raw_images)
    raw_v = vis_proj.forward(vis_feats)
    norm_v, _ = normalize_embeddings(raw_v)

    tokenized = [tokenizer.encode(s.text) for s in samples]
    raw_t, _ = txt_enc.forward(
        [t.token_ids for t in tokenized],
        [t.attention_mask for t in tokenized],
    )
    norm_t, _ = normalize_embeddings(raw_t)

    # 5. Cross-Modal Retrieval
    retrieval_summary, i2t_res, t2i_res = evaluate_cross_modal_retrieval(
        image_embeddings=norm_v,
        text_embeddings=norm_t,
        sample_ids=sample_ids,
    )
    assert retrieval_summary.sample_count == 8
    assert len(i2t_res) == 8
    assert len(t2i_res) == 8

    # 6. Zero-Shot Class Prompts & Classification
    prompts = [
        ClassPrompt(
            class_id=idx,
            class_name=cname,
            prompt_template="a {color} {shape} on the {position}",
            rendered_text=f"a {cname.replace('_', ' ')} on the center",
            prompt_id=f"p_{idx}",
        )
        for idx, cname in enumerate(class_names)
    ]

    zero_shot_summary = evaluate_zero_shot_classification(
        image_embeddings=norm_v,
        class_prompts=prompts,
        text_encoder=txt_enc,
        tokenizer=tokenizer,
        true_labels=true_labels,
        class_names=class_names,
    )
    assert 0.0 <= zero_shot_summary.accuracy <= 1.0

    # 7. Prompt Sensitivity
    sensitivity = evaluate_prompt_sensitivity(
        image_embeddings=norm_v,
        prompt_templates=PROMPT_TEMPLATES[:2],
        class_names=class_names,
        text_encoder=txt_enc,
        tokenizer=tokenizer,
        true_labels=true_labels,
    )
    assert len(sensitivity["templates"]) == 2

    # 8. Shared Geometry & Joint PCA
    shared_geometry = compute_shared_multimodal_geometry(
        image_embeddings=norm_v,
        text_embeddings=norm_t,
        sample_ids=sample_ids,
        class_labels=true_labels,
        class_names=class_names,
    )
    assert len(shared_geometry["image_pca_coordinates"]) == 8

    # 9. Collapse Diagnostics
    collapse = compute_multimodal_collapse_diagnostics(norm_v, norm_t)
    assert isinstance(collapse.is_collapsed, bool)

    # 10. Robustness Under Visual Corruption
    robustness = evaluate_multimodal_alignment_robustness(
        samples=samples,
        visual_encoder=vis_enc,
        visual_projection=vis_proj,
        text_encoder=txt_enc,
        tokenizer=tokenizer,
        class_prompts=prompts,
        corruptions=["clean", "gaussian_noise", "blur"],
        severity=2,
        seed=seed,
    )
    assert len(robustness["results"]) == 3

    # 11. Complete Report Generation and Serialization
    report = VisionLanguageRepresentationReport(
        experiment_id="smoke_vl_exp_01",
        visual_family=ModelFamily.CNN.value,
        visual_architecture="conv_tiny",
        text_dim=8,
        shared_dim=4,
        temperature=0.1,
        seed=seed,
        dataset_fingerprint=samples[0].dataset_fingerprint,
        final_loss=history[-1]["loss"],
        image_to_text_loss=history[-1]["image_to_text_loss"],
        text_to_image_loss=history[-1]["text_to_image_loss"],
        matched_similarity=history[-1]["matched_similarity"],
        unmatched_similarity=history[-1]["unmatched_similarity"],
        similarity_gap=history[-1]["similarity_gap"],
        training_history=history,
        retrieval_summary=retrieval_summary,
        zero_shot_summary=zero_shot_summary,
        prompt_sensitivity=sensitivity,
        explained_variance_ratio=shared_geometry["explained_variance_ratio"],
        mean_paired_distance=shared_geometry["mean_paired_distance"],
        mean_paired_cosine=shared_geometry["mean_paired_cosine"],
        centroid_alignments=[],
        collapse_summary=collapse,
        robustness_summary=robustness,
        failure_types=[],
        warnings=["Synthetic paired dataset for representation research."],
    )

    report_json = report.to_json()
    parsed_report = json.loads(report_json)
    assert parsed_report["experiment_id"] == "smoke_vl_exp_01"

    restored_report = VisionLanguageRepresentationReport.from_dict(parsed_report)
    assert restored_report.experiment_id == report.experiment_id
    assert restored_report.retrieval_summary.sample_count == 8
