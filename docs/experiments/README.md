# PRISM Experiments Guide

## Controlled architecture comparisons

Phase 13 provides a Python API for controlled CNN, residual CNN, and Vision
Transformer studies:

```python
from prism.experiments import ComparisonMode, ExperimentSuiteRunner
from prism.experiments import create_architecture_comparison_suite

suite = create_architecture_comparison_suite(
    "suite-cnn-resnet-vit",
    "Architecture comparison",
    "How do visual models behave under shared conditions?",
    [cnn_experiment, resnet_experiment, vit_experiment],
    comparison_mode=ComparisonMode.STRICT_CONTROLLED,
)
audit = suite.validate_factors()
report = ExperimentSuiteRunner().run(suite, execute_experiment)
serialized = report.to_json(indent=2)
```

The callback should reuse `ExperimentExecutionHarness`, `DataPreparer`,
`TrainingEngine`, and `EvaluationEngine`, returning an
`ArchitectureRunResult`. Reports include real metric records, aligned curves,
convergence descriptors, exact parameter counts, gradient-flow and final
representation summaries, ViT-only attention profiles, pairwise deltas, and
honest partial-failure status. `SampleEfficiencyPlan` preserves nested
data-budget identity, while repeated-seed contracts avoid fabricating runs.

Phase 13 does not run large benchmarks, download datasets, perform
hyperparameter search, compute confidence intervals, or build the Observatory
dashboard.

## Purpose
This directory stores documentation, interactive notebooks, and synthesized research findings for PRISM experimental campaigns.

---

## Experiment Domain Concepts

PRISM separates the declarative research specification from physical execution attempts through strongly typed domain contracts:

```
ExperimentDefinition (Immutable scientific intent)
        │
        ├── Bound ControlledDataReference (Canonical universe + Partition + Nested subset)
        ├── Validate Definition & Compute SHA-256 Fingerprint
        │
        ▼
ExperimentExecutionHarness.prepare(experiment)
        │
        ├── Probe Host Hardware (CPU, CUDA, MPS)
        ├── Capture Environment Snapshot & Git Revision
        ├── Initialize Multi-Backend RNG (Python, NumPy, PyTorch)
        └── Output Immutable PreparedExecution Context
        │
        ▼
DataPreparer.prepare(prepared_execution, ...)
        │
        ├── Resolve Exact Canonical Sample Identities
        ├── Execute Deterministic Preprocessing
        ├── Bind Deterministic Ordering (Sequential / Fixed Shuffle / Epoch-Aware)
        └── Output MaterializedDataset + DeterministicBatchLoader + DataRuntimeContext
        │
        ▼
TrainingEngine.train(...)
        │
        ├── Execute Deterministic Epoch Loops (Forward -> Loss -> Backward -> SGD)
        ├── Route Analytical Gradients through Residual Skip Branches (dF + dS)
        ├── Update BatchNorm Running Statistics during training
        ├── Step Learning Rate Scheduler (Constant / Step / Cosine Annealing)
        ├── Record Real-time MetricRecords (Loss, Accuracy, Learning Rate) into ExperimentRun
        ├── Evaluate Test Partition via EvaluationEngine in Evaluation Mode
        └── Transition Run Lifecycle (RUNNING -> COMPLETED)
        │
        ▼
TrainingResult (Consolidated execution metrics and evaluation summaries)
        │
        ▼
compute_gradient_flow_summary & extract_representations (Gradient tracking & feature analysis)
```

---

## Defining, Preparing, and Training a Residual CNN Experiment

```python
from prism.core.enums import (
    TaskType,
    ModelFamily,
    MetricDirection,
    PrecisionMode,
    OrderingStrategy,
)
from prism.data.adapters import CIFAR10Adapter
from prism.data.manifests import ControlledDataReference, DatasetManifest
from prism.data.preparer import DataPreparer
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.training.configuration import (
    TrainingConfiguration,
    OptimizerSpecification,
    SchedulerSpecification,
)
from prism.training.engine import TrainingEngine
from prism.training.gradient_flow import (
    compute_gradient_flow_summary,
    compare_gradient_flow_summaries,
)
from prism.evaluation.configuration import EvaluationConfiguration, MetricSpecification
from prism.experiments.definitions import ExperimentDefinition
from prism.experiments.reproducibility import ReproducibilityConfiguration
from prism.experiments.harness import ExperimentExecutionHarness
from prism.experiments.comparisons import create_residual_comparison

# 1. Obtain standardized CIFAR-10 manifests & 25% nested subset
adapter = CIFAR10Adapter()
canonical = adapter.get_canonical_manifest()
partition = adapter.get_default_partition(seed=42)
subsets = adapter.get_nested_subsets(seed=42)

subset_25pct = subsets[0.25]
controlled_ref = ControlledDataReference(
    canonical_manifest_fingerprint=canonical.compute_fingerprint(),
    partition_manifest_fingerprint=partition.compute_fingerprint(),
    subset_manifest_fingerprint=subset_25pct.compute_fingerprint(),
    partition_id=partition.partition_id,
    subset_id=subset_25pct.subset_id,
    budget_ratio=0.25,
)

dataset = DatasetManifest(
    **adapter.get_dataset_manifest().model_dump(exclude={"controlled_data"}),
    controlled_data=controlled_ref,
)

# 2. Declare Model Architecture (Multi-Stage ResNet with BatchNorm)
model = ModelSpecification(
    model_id="model-cifar10-resnet-bn",
    name="CIFAR-10 ResNet with Skip Connections",
    family=ModelFamily.RESNET,
    architecture="resnet",
    compatible_tasks=[TaskType.CLASSIFICATION],
    input_shape=(3, 32, 32),
    num_classes=10,
    hyperparameters={
        "stem_channels": 16,
        "stage_widths": [16, 32, 64],
        "blocks_per_stage": [2, 2, 2],
        "strides": [1, 2, 2],
        "activation": "relu",
        "normalization": "batch_norm",
        "norm_eps": 1e-5,
        "norm_momentum": 0.1,
        "norm_affine": True,
        "dropout": 0.0,
    },
)

# 3. Declare Training & Evaluation Budget with Cosine Annealing Schedule
training = TrainingConfiguration(
    epochs=50,
    batch_size=64,
    optimizer=OptimizerSpecification(
        type="sgd", lr=0.05, momentum=0.9, weight_decay=1e-4
    ),
    scheduler=SchedulerSpecification(type="cosine", min_lr=0.001, warmup_epochs=5),
    precision=PrecisionMode.FP32,
)

evaluation = EvaluationConfiguration(
    target_splits=["test"],
    metrics=[
        MetricSpecification(name="top1_accuracy", direction=MetricDirection.MAXIMIZE),
        MetricSpecification(name="loss", direction=MetricDirection.MINIMIZE),
    ],
)

# 4. Construct Immutable Experiment Definition
experiment = ExperimentDefinition(
    experiment_id="exp-cifar10-resnet-25pct",
    name="CIFAR-10 Residual CNN 25% Data Efficiency",
    task_type=TaskType.CLASSIFICATION,
    dataset=dataset,
    model=model,
    training=training,
    evaluation=evaluation,
    reproducibility=ReproducibilityConfiguration(seed=42, deterministic=True),
)

# 5. Prepare Execution Context via Harness
harness = ExperimentExecutionHarness()
run, prepared_context = harness.prepare(experiment)

# 6. Materialize Train and Test Datasets
preparer = DataPreparer()
train_dataset, train_loader, _ = preparer.prepare(
    adapter=adapter,
    canonical_manifest=canonical,
    partition_manifest=partition,
    subset_manifest=subset_25pct,
    batch_size=64,
    ordering_strategy=OrderingStrategy.EPOCH_AWARE_SHUFFLE,
    seed=42,
    prepared_execution=prepared_context,
)

test_dataset, test_loader, _ = preparer.prepare(
    adapter=adapter,
    canonical_manifest=canonical,
    partition_manifest=partition,
    split_name="test",
    batch_size=100,
    ordering_strategy=OrderingStrategy.SEQUENTIAL,
    seed=42,
    prepared_execution=prepared_context,
)

# 7. Execute Training and Evaluation Loop
engine = TrainingEngine()
result = engine.train(
    experiment=experiment,
    prepared_execution=prepared_context,
    train_dataset=train_dataset,
    train_loader=train_loader,
    test_dataset=test_dataset,
    test_loader=test_loader,
    run=run,
)

# 8. Compute Gradient Flow Summaries across Model Depth
res_model = ResidualNeuralNetwork(spec=model, seed=42)
test_batch = [test_dataset[i].data for i in range(10)]
_ = res_model.forward(test_batch)
res_model.backward(
    [
        [1.0 / 10 if j == test_dataset[i].target else 0.0 for j in range(10)]
        for i in range(10)
    ]
)

grad_summary = compute_gradient_flow_summary(res_model)
print(f"Global Gradient L2 Norm: {grad_summary.global_grad_norm_l2:.6f}")
for param_s in grad_summary.parameter_summaries[:5]:
    print(
        f"Layer {param_s.parameter_name} ({param_s.logical_stage}): Norm={param_s.norm_l2:.6f}"
    )
```

---

## Controlled Learning Rate Schedule Comparison

```python
from prism.experiments.comparisons import create_scheduler_comparison

# Declaratively isolate learning rate schedule effect (Constant vs Cosine Annealing + Warmup)
comparison = create_scheduler_comparison(
    comparison_id="comp-lr-constant-vs-warmup-cosine",
    name="Constant vs Warmup-Cosine Schedule on ResNet",
    baseline_experiment_id="exp-cifar10-resnet-constant-lr",
    candidate_experiment_id="exp-cifar10-resnet-warmup-cosine-lr",
    baseline_scheduler_type="constant",
    candidate_scheduler_type="cosine",
    baseline_scheduler_params={"min_lr": 0.0},
    candidate_scheduler_params={"warmup_epochs": 5, "min_lr": 0.001},
    dataset_fingerprint=dataset.compute_fingerprint(),
    seed=42,
    description="Controlled study of warmup and cosine annealing on convergence speed and representation geometry.",
)

print(f"Comparison Fingerprint: {comparison.compute_fingerprint()}")
```

---

## Vision Transformer Foundations & Attention Analysis

```python
from prism.models.patches import (
    PatchExtractor,
    PatchEmbedding,
    ClassToken,
    PositionalEmbedding,
)
from prism.models.attention import MultiHeadSelfAttention
from prism.representations.attention import summarize_attention_weights

# 1. Extract 4x4 non-overlapping patches from 32x32 image (64 patches of dim 48)
patch_ext = PatchExtractor(patch_size=4)
patches = patch_ext.forward(image_batch)  # [N, 64, 48]

# 2. Linear projection to embedding dimension (48 -> 128)
patch_emb = PatchEmbedding(in_features=48, embed_dim=128, seed=42)
tokens = patch_emb.forward(patches)  # [N, 64, 128]

# 3. Prepend learnable classification token [1, 1, 128] -> 65 tokens
cls_token = ClassToken(embed_dim=128, seed=42)
tokens_with_cls = cls_token.forward(tokens)  # [N, 65, 128]

# 4. Add learnable 1D position embeddings
pos_emb = PositionalEmbedding(num_positions=65, embed_dim=128, seed=42)
embedded_tokens = pos_emb.forward(tokens_with_cls)  # [N, 65, 128]

# 5. Multi-Head Self-Attention (128 dim, 4 heads -> 32 dim per head)
mhsa = MultiHeadSelfAttention(embed_dim=128, num_heads=4, seed=42)
contextualized_tokens = mhsa.forward(embedded_tokens)  # [N, 65, 128]

# 6. Audit attention distributions across heads
attn_summary = summarize_attention_weights(mhsa.last_attention_weights)
print(f"Mean Attention Entropy: {attn_summary.mean_entropy:.4f} nats")
print(f"Row Normalized: {attn_summary.is_row_normalized}")
for head in attn_summary.head_summaries:
    print(
        f"Head {head.head_index}: entropy={head.entropy:.4f}, min={head.min_value:.4f}, max={head.max_value:.4f}"
    )
```

---

## 7. Representation Geometry Analysis (`prism.representations`)

Phase 14 provides programmatic APIs for evaluating the geometry of learned representations:

```python
from prism.representations import (
    RepresentationDataset,
    SpatialVectorizationPolicy,
    VectorNormalizationPolicy,
    DistanceMetric,
    analyze_representation_geometry,
    analyze_layer_geometry_profile,
    compare_architecture_geometries,
)

# 1. Construct RepresentationDataset with spatial pooling and L2 normalization
dataset = RepresentationDataset.from_raw_representations(
    raw_embeddings=model.extract_representations(test_images, layer="final_hidden"),
    sample_ids=test_sample_ids,
    labels=test_labels,
    experiment_id="exp-geom-demo",
    model_id=model.model_id,
    layer_name="final_hidden",
    spatial_policy=SpatialVectorizationPolicy.GLOBAL_AVERAGE_POOL,
    norm_policy=VectorNormalizationPolicy.NONE,
)

# 2. Complete Geometric Analysis (Centroids, k-NN Consistency, PCA 2D)
report = analyze_representation_geometry(
    dataset=dataset,
    k=5,
    metric=DistanceMetric.EUCLIDEAN,
    n_pca_components=2,
)

print(
    f"Intra-Class Compactness: {report.centroid_geometry.mean_intra_class_distance:.4f}"
)
print(
    f"Inter-Class Separation: {report.centroid_geometry.mean_inter_class_centroid_distance:.4f}"
)
print(
    f"Separation/Compactness Ratio: {report.centroid_geometry.separation_to_compactness_ratio:.2f}x"
)
print(f"5-NN Consistency: {report.neighborhood_geometry.mean_label_consistency:.2%}")
print(f"Candidate Failures Flagged: {len(report.candidate_failures)}")

# 3. Layer Evolution Profile across network depth
profile = analyze_layer_geometry_profile(
    model=model,
    inputs=test_images,
    sample_ids=test_sample_ids,
    labels=test_labels,
    layers=["conv_0", "conv_1", "final_hidden"],
)

# 4. Cross-Architecture Geometry Benchmark (CNN vs ResNet vs ViT)
comparison = compare_architecture_geometries(
    models={"cnn": cnn_model, "resnet": resnet_model, "vit": vit_model},
    inputs=test_images,
    sample_ids=test_sample_ids,
    labels=test_labels,
)
```

---

## 8. Robustness & Distribution Shift Experiments (`prism.robustness`)

Phase 15 provides programmatic APIs for evaluating visual model and representation resilience under controlled input corruptions and distribution shifts:

```python
from prism.robustness import (
    CorruptionType,
    CorruptionSpecification,
    CorruptionSuite,
    CorruptedDatasetView,
    RobustnessSuiteRunner,
    compare_architecture_robustness,
    compute_representation_drift,
    compute_geometry_drift,
    compute_vit_attention_drift,
)

# 1. Define Declarative Robustness Suite
suite = CorruptionSuite(
    suite_id="suite-standard-robustness",
    name="Standard Robustness Evaluation Suite",
    corruption_types=[
        CorruptionType.GAUSSIAN_NOISE,
        CorruptionType.BLUR,
        CorruptionType.BRIGHTNESS,
        CorruptionType.CONTRAST,
        CorruptionType.OCCLUSION,
        CorruptionType.RESOLUTION_DEGRADATION,
    ],
    severities=[1, 2, 3, 4, 5],
    eval_split="test",
    layer_name="final_hidden",
    seed=42,
    k_neighbors=5,
    pca_components=2,
)

# 2. Run Robustness Suite on Frozen Model
runner = RobustnessSuiteRunner()
report = runner.run_suite(
    model=model,
    clean_dataset=materialized_test_dataset,
    suite=suite,
    experiment_id="exp-robustness-resnet",
)

# 3. Inspect Clean vs Corrupted Summary
eval_summary = report.evaluations["gaussian_noise::sev3"]
print(f"Clean Accuracy: {eval_summary.clean_accuracy:.2%}")
print(f"Corrupted Accuracy: {eval_summary.corrupted_accuracy:.2%}")
print(f"Accuracy Drop: -{eval_summary.absolute_accuracy_drop:.2%}")
print(
    f"Mean Euclidean Representation Drift: {eval_summary.representation_drift.mean_euclidean_drift:.4f}"
)
print(
    f"Cosine Similarity: {eval_summary.representation_drift.mean_cosine_similarity:.4f}"
)
print(
    f"5-NN Retention Overlap: {eval_summary.geometry_drift.neighborhood_drift.mean_neighbor_overlap_ratio:.2%}"
)

# 4. Severity Degradation Curves & AUC
curve = report.severity_curves[CorruptionType.GAUSSIAN_NOISE.value]
print(f"Area Under Curve (AUC): {curve.area_under_curve:.2%}")
print(f"Total Accuracy Drop (Sev 1->5): -{curve.total_accuracy_drop:.2%}")

# 5. Cross-Architecture Robustness Benchmark
comp_report = compare_architecture_robustness(
    models={"cnn": cnn_model, "resnet": resnet_model, "vit": vit_model},
    clean_dataset=materialized_test_dataset,
    suite=suite,
    comparison_id="comp-arch-robustness-eval",
)
```

---

## Phase 16 Explainability & Visual Attribution Experiments

PRISM provides a unified, mathematically rigorous interface for computing, comparing, and evaluating attribution signals across CNNs, ResNets, and Vision Transformers:

```python
from prism.explainability import (
    AttributionMethod,
    AttributionSpecification,
    TargetClassMode,
    compute_input_gradient_saliency,
    compute_gradient_x_input,
    compute_occlusion_sensitivity,
    compute_grad_cam,
    compute_vit_attention_attribution,
    compare_attributions,
    compute_attribution_drift,
    flag_explanation_failures,
)

# 1. Compute Input Saliency and Gradient x Input
res_ig = compute_input_gradient_saliency(
    model=resnet_model,
    image=sample_image_3d,
    target_mode=TargetClassMode.PREDICTED_CLASS,
)

res_gxi = compute_gradient_x_input(
    model=resnet_model,
    image=sample_image_3d,
    target_mode=TargetClassMode.PREDICTED_CLASS,
)

# 2. Compute Sliding-Window Occlusion Sensitivity
res_occ = compute_occlusion_sensitivity(
    model=resnet_model,
    image=sample_image_3d,
    window_size=(2, 2),
    stride=(1, 1),
)

# 3. Compute Grad-CAM (CNN and ResNet)
res_cam = compute_grad_cam(
    model=resnet_model,
    image=sample_image_3d,
    layer_name="final_stage",
)

# 4. ViT CLS-to-Patch Attention Attribution (ViT only)
res_vit = compute_vit_attention_attribution(
    model=vit_model,
    image=sample_image_3d,
)

# 5. Cross-Method Spatial Agreement Analysis
report = compare_attributions([res_ig, res_gxi, res_occ, res_cam])
print(f"Mean Pairwise Agreement: {report.mean_cross_method_agreement:.3f}")

# 6. Attribution Drift under Input Corruption
drift = compute_attribution_drift(
    clean_result=res_ig,
    corrupted_result=res_ig_corrupted,
    corruption_type="gaussian_noise",
    corruption_severity=0.15,
)
print(f"Attribution Cosine Similarity: {drift.attribution_cosine_similarity:.3f}")
print(f"Prediction Preserved: {drift.prediction_preserved}")

# 7. Diagnostic Failure Taxonomy
flags = flag_explanation_failures(
    attribution_result=res_ig,
    comparison_report=report,
    drift_summary=drift,
)
```

---

## Running Transfer Learning & Representation Reuse Experiments

```python
from prism.transfer import (
    TransferStrategy,
    NormalizationTransferPolicy,
    TransferLearningSpecification,
    create_model_state_snapshot,
    create_freeze_plan,
    replace_classifier_head,
    TransferTrainingRunner,
    probe_all_layers_transferability,
    compute_representation_retention,
    compute_transfer_shared_pca,
)

# 1. Snapshot trained source model
snapshot = create_model_state_snapshot(
    source_model, source_experiment_id="exp_cifar_source"
)

# 2. Configure Transfer Learning Specification
spec = TransferLearningSpecification(
    transfer_id="trans_resnet_linear_probe",
    source_model_id=source_model.model_id,
    target_dataset_id="target_task_train",
    target_num_classes=5,
    strategy=TransferStrategy.LINEAR_PROBE,
    normalization_policy=NormalizationTransferPolicy.FREEZE_SOURCE_STATS,
    target_epochs=10,
    target_learning_rate=0.01,
)

# 3. Execute Transfer Experiment Runner
runner = TransferTrainingRunner()
report = runner.run_transfer(
    specification=spec,
    source_snapshot=snapshot,
    target_train_dataset=target_train_ds,
    target_train_loader=train_loader,
    target_val_dataset=target_val_ds,
    target_val_loader=val_loader,
    evaluate_layer_probes=True,
    probed_layers=["stem", "stage_0", "stage_1", "final_hidden"],
    evaluate_retention=True,
    compare_with_scratch=True,
)

print(f"Transfer Strategy: {report.strategy.value}")
print(f"Target Accuracy: {report.val_accuracy * 100:.1f}%")
if report.scratch_comparison:
    print(f"Scratch Baseline: {report.scratch_comparison.scratch_accuracy * 100:.1f}%")
    print(
        f"Linear Probe Gain: {report.scratch_comparison.linear_probe_gain * 100:+.1f}% Δ"
    )

# 4. Layer Transferability Probes Across Depth
for probe in report.layer_probes:
    print(
        f"Layer {probe.layer_name}: {probe.val_accuracy * 100:.1f}% (dim={probe.representation_dim})"
    )

# 5. Representation Retention & Shared PCA Drift
if report.representation_drift:
    print(
        f"Mean Cosine Similarity: {report.representation_drift.mean_cosine_similarity:.4f}"
    )
    print(
        f"Mean Euclidean Drift: {report.representation_drift.mean_euclidean_drift:.4f}"
    )
```

---

## Self-Supervised Representation Learning & Contrastive Pretraining

Phase 18 introduces self-supervised representation learning via SimCLR contrastive pretraining, non-linear projection heads, analytical NT-Xent gradient updates, and dimensional collapse diagnostics:

```python
from prism.core.enums import ModelFamily
from prism.ssl.specification import SelfSupervisedTrainingSpecification
from prism.ssl.engine import SelfSupervisedTrainingEngine

# 1. Configure Self-Supervised Pretraining Specification (No Class Labels)
ssl_spec = SelfSupervisedTrainingSpecification(
    ssl_id="ssl_resnet_simclr_cifar",
    encoder_family=ModelFamily.RESNET,
    encoder_spec=resnet_model_spec,
    dataset_id="cifar10_spatial_unlabeled",
    projection_hidden_dim=128,
    projection_out_dim=64,
    temperature=0.5,
    epochs=20,
    batch_size=32,
    learning_rate=0.05,
    seed=42,
)

# 2. Train Self-Supervised Encoder
engine = SelfSupervisedTrainingEngine()
encoder, snapshot, ssl_report = engine.train_ssl(
    specification=ssl_spec,
    dataset=unlabeled_train_dataset,
)

# 3. Inspect Collapse Diagnostics
collapse = ssl_report.collapse_summary
print(f"Mean Feature Std: {collapse.mean_feature_std:.4f}")
print(
    f"Active Channels: {collapse.total_dimensions - collapse.near_zero_variance_dimensions}/{collapse.total_dimensions}"
)
print(f"Angular Spread: {collapse.distinct_sample_cosine_spread:.4f}")
print(f"Collapse Status: {'COLLAPSED' if collapse.is_collapsed else 'HEALTHY'}")

# 4. Downstream Linear Probe Transfer Evaluation
# Discard projection head, freeze SSL backbone snapshot, attach linear classifier head
```

---

## Generative & Reconstruction-Based Representation Learning

Phase 19 introduces generative and reconstruction-based representation learning via Masked Image Modeling (MIM) on Vision Transformers and spatial Denoising Autoencoders (DAE) on CNNs / ResNets:

```python
from prism.core.enums import ModelFamily
from prism.reconstruction.enums import ReconstructionMethod
from prism.reconstruction.specification import ReconstructionLearningSpecification
from prism.reconstruction.engine import ReconstructionTrainingEngine

# 1. Configure Masked Image Modeling Specification (Label Independent)
recon_spec = ReconstructionLearningSpecification(
    reconstruction_id="recon_vit_mim_cifar",
    method=ReconstructionMethod.MASKED_PATCH_RECONSTRUCTION,
    encoder_family=ModelFamily.VISION_TRANSFORMER,
    encoder_spec=vit_model_spec,
    input_shape=(3, 8, 8),
    patch_size=4,
    mask_ratio=0.5,
    epochs=20,
    batch_size=32,
    learning_rate=0.05,
    seed=42,
    dataset_id="cifar10_unlabeled",
)

# 2. Train Reconstruction Encoder & Decoders
engine = ReconstructionTrainingEngine()
report = engine.train(
    dataset=unlabeled_train_dataset,
    spec=recon_spec,
    downstream_target_dataset=labeled_target_dataset,
)

# 3. Inspect Reconstruction Diagnostics & Failure Taxonomy
diagnostics = report.diagnostics
print(f"Mean Reconstruction Error: {diagnostics.mean_reconstruction_error:.4f}")
print(f"Latent Representation Std: {diagnostics.latent_std:.4f}")
print(
    f"Near-Zero Variance Fraction: {diagnostics.near_zero_variance_fraction * 100:.1f}%"
)
print(
    f"Failure Categories Flagged: {[f.value for f in diagnostics.failure_categories]}"
)

# 4. Downstream Linear Probe Performance
if report.downstream_linear_probe_accuracy is not None:
    print(
        f"Linear Probe Test Accuracy: {report.downstream_linear_probe_accuracy * 100:.1f}%"
    )
```

---

## Detection & Segmentation Representation Transfer

Phase 20 enables evaluating visual representations transferred to spatial downstream localization (lightweight object detection) and dense prediction (semantic segmentation) across Supervised, SimCLR, Reconstruction, and Scratch pretraining objectives:

```python
from prism.spatial.enums import (
    PretrainingObjective,
    SpatialTaskType,
    SpatialTransferStrategy,
)
from prism.spatial.specification import SpatialTransferSpecification
from prism.spatial.runner import SpatialTransferRunner
from prism.spatial.synthetic import generate_synthetic_spatial_dataset
from prism.api.spatial_service import get_default_model_spec, SpatialTransferService

# 1. Generate Deterministic Synthetic Spatial Benchmark Data
det_train, seg_train = generate_synthetic_spatial_dataset(
    num_samples=16, image_shape=(3, 16, 16), num_classes=3, seed=42
)
det_val, seg_val = generate_synthetic_spatial_dataset(
    num_samples=8, image_shape=(3, 16, 16), num_classes=3, seed=84
)

# 2. Configure and Run Frozen Spatial Probe on Object Detection
model_spec = get_default_model_spec("vit")
det_spec = SpatialTransferSpecification.create(
    source_objective=PretrainingObjective.RECONSTRUCTION,
    source_experiment_id="recon_vit_mim",
    model_spec=model_spec,
    task_type=SpatialTaskType.OBJECT_DETECTION,
    spatial_layer="final_spatial",
    transfer_strategy=SpatialTransferStrategy.FROZEN_SPATIAL_PROBE,
    num_classes=3,
    epochs=5,
    learning_rate=0.01,
)

runner = SpatialTransferRunner(det_spec)
report = runner.train_and_evaluate(train_samples=det_train, eval_samples=det_val)

if report.detection_metrics:
    print(f"Detection Mean IoU: {report.detection_metrics.mean_iou:.4f}")
    print(f"Detection Precision: {report.detection_metrics.precision:.4f}")
    print(f"Detection Recall: {report.detection_metrics.recall:.4f}")

# 3. Cross-Objective & Layer Transferability Benchmarking via API Service
service = SpatialTransferService(seed=42)
comparison = service.generate_objective_comparison(
    architecture="cnn",
    task_type=SpatialTaskType.SEMANTIC_SEGMENTATION,
)
for obj_name, rep in comparison.reports_by_objective.items():
    if rep.segmentation_metrics:
        print(
            f"[{obj_name.upper()}] mIoU: {rep.segmentation_metrics.mean_iou:.4f}, "
            f"Pixel Acc: {rep.segmentation_metrics.pixel_accuracy:.4f}"
        )
```

---

## Video & Temporal Representation Learning

Phase 21 extends PRISM from static-image representations into short-video and temporal sequence learning, comparing frame-independent baselines, temporal pooling methods, and SimpleRNN models across pretraining objectives (Supervised, SimCLR, Reconstruction, Scratch):

```python
from prism.temporal.enums import (
    PretrainingObjective,
    TemporalAggregationType,
    TemporalCorruptionType,
    TemporalTransferStrategy,
)
from prism.temporal.specification import TemporalTransferSpecification
from prism.temporal.synthetic import SyntheticVideoGenerator
from prism.temporal.runner import TemporalTrainingRunner
from prism.temporal.corruptions import apply_temporal_corruption
from prism.api.temporal_service import TemporalRepresentationService
from prism.core.enums import ModelFamily, SplitName

# 1. Generate Deterministic Synthetic Video Dataset
gen = SyntheticVideoGenerator(num_frames=4, height=16, width=16, seed=42)
train_videos = gen.generate_dataset(num_samples=8, split=SplitName.TRAIN)
val_videos = gen.generate_dataset(num_samples=4, split=SplitName.VAL)

# 2. Configure and Run Temporal Representation Transfer
spec = TemporalTransferSpecification(
    source_objective=PretrainingObjective.SUPERVISED,
    architecture=ModelFamily.CNN,
    selected_layer="final_hidden",
    temporal_aggregator=TemporalAggregationType.SIMPLE_RNN,
    transfer_strategy=TemporalTransferStrategy.FROZEN_FRAME_ENCODER,
    rnn_hidden_dim=16,
    epochs=5,
    learning_rate=0.05,
    seed=42,
)

service = TemporalRepresentationService(seed=42)
runner = TemporalTrainingRunner(
    spec=spec,
    model=service._instantiate_backbone(ModelFamily.CNN),
    train_samples=train_videos,
    val_samples=val_videos,
)
report = runner.run_transfer()

print(f"Video Accuracy: {report.video_accuracy * 100:.1f}%")
print(f"Frame Baseline Accuracy: {report.frame_baseline_accuracy * 100:.1f}%")
print(
    f"Temporal Consistency (Mean Adjacent Dist): {report.temporal_consistency.mean_adjacent_distance:.4f}"
)
print(
    f"Temporal Consistency (Mean Cosine Sim): {report.temporal_consistency.mean_adjacent_cosine_similarity:.4f}"
)

# 3. Evaluate Temporal Robustness under Corruptions
drop_sample, drop_meta = apply_temporal_corruption(
    val_videos[0],
    TemporalCorruptionType.FRAME_DROP,
    drop_fraction=0.5,
)
print(
    f"Clean Frames: {val_videos[0].frame_count} -> Perturbed Frames: {drop_sample.frame_count}"
)
print(f"Dropped Frame IDs: {drop_meta.get('dropped_frame_ids')}")

# 4. Cross-Objective Benchmarks & Layer Transferability via Service
benchmarks = service.get_precomputed_benchmarks()
print("Objective Comparisons:", list(benchmarks["objective_comparisons"].keys()))
print("Layer Transferability Models:", list(benchmarks["layer_transferability"].keys()))
```



