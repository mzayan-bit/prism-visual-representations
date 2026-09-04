"""Spatial representation transfer training runner and evaluation coordinator."""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any

from prism.core.enums import ModelFamily
from prism.core.errors import ValidationError
from prism.models.base import BaseVisionModel
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.transformer import VisionTransformer
from prism.spatial.adapter import SpatialRepresentationAdapter
from prism.spatial.annotations import DetectionSample, SegmentationSample
from prism.spatial.enums import (
    SpatialTaskType,
    SpatialTransferStrategy,
)
from prism.spatial.heads import GridDetectionHead, SegmentationHead
from prism.spatial.losses import GridDetectionLoss, PixelCrossEntropyLoss
from prism.spatial.metrics import (
    DetectionEvaluationResult,
    SegmentationConfusionMatrix,
    SegmentationMetricsResult,
    evaluate_detection_predictions,
)
from prism.spatial.reports import SpatialTransferReport
from prism.spatial.specification import SpatialTransferSpecification
from prism.transfer.snapshot import ModelStateSnapshot


def _count_elements(data: Any) -> int:
    """Recursively count scalar elements in nested list/dict structure."""
    if isinstance(data, (list, tuple)):
        return sum(_count_elements(item) for item in data)
    if isinstance(data, dict):
        return sum(_count_elements(v) for v in data.values())
    return 1


def _flatten_tensor(data: Any) -> list[float]:
    """Recursively flatten nested numerical structures into a 1D float list."""
    flat: list[float] = []
    if isinstance(data, (list, tuple)):
        for item in data:
            flat.extend(_flatten_tensor(item))
    elif isinstance(data, (int, float)):
        flat.append(float(data))
    return flat


def _compute_drift_metrics(
    feats_before: list[list[list[list[float]]]],
    feats_after: list[list[list[list[float]]]],
) -> tuple[float, float]:
    """Compute cosine distance and RMSE between two aligned 4D feature tensors."""
    v1 = _flatten_tensor(feats_before)
    v2 = _flatten_tensor(feats_after)
    if not v1 or len(v1) != len(v2):
        return 0.0, 0.0

    dot = sum(a * b for a, b in zip(v1, v2, strict=True))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))

    if norm1 > 0.0 and norm2 > 0.0:
        cos_sim = max(-1.0, min(1.0, dot / (norm1 * norm2)))
        cosine_dist = max(0.0, 1.0 - cos_sim)
    else:
        cosine_dist = 0.0

    mse = sum((a - b) ** 2 for a, b in zip(v1, v2, strict=True)) / float(len(v1))
    rmse = math.sqrt(mse)

    return float(cosine_dist), float(rmse)


def _apply_sgd_update(
    params: Any,
    grads: Any,
    learning_rate: float,
) -> None:
    """In-place SGD parameter update: params -= lr * grads."""
    if isinstance(params, list) and isinstance(grads, list):
        for i in range(len(params)):
            if isinstance(params[i], list):
                _apply_sgd_update(params[i], grads[i], learning_rate)
            elif isinstance(params[i], (int, float)):
                params[i] = float(params[i]) - learning_rate * float(grads[i])


class SpatialTransferRunner:
    """Coordinates spatial downstream probing, parameter freezing, and evaluation."""

    def __init__(
        self,
        spec: SpatialTransferSpecification,
        source_snapshot: ModelStateSnapshot | None = None,
    ) -> None:
        self.spec = spec
        self.source_snapshot = source_snapshot

        # 1. Instantiate encoder
        family = spec.model_spec.family
        if family == ModelFamily.CNN:
            self.encoder: BaseVisionModel = ConvolutionalNeuralNetwork(
                spec=spec.model_spec, seed=spec.seed
            )
        elif family == ModelFamily.RESNET:
            self.encoder = ResidualNeuralNetwork(spec=spec.model_spec, seed=spec.seed)
        elif family == ModelFamily.VISION_TRANSFORMER:
            self.encoder = VisionTransformer(spec=spec.model_spec, seed=spec.seed)
        else:
            raise ValidationError(f"Unsupported encoder family: {family}.")

        # 2. Restore weights if snapshot provided
        if source_snapshot is not None:
            self.encoder.set_parameters(source_snapshot.parameters)

        # 3. Create spatial adapter
        self.adapter = SpatialRepresentationAdapter(
            model=self.encoder, layer_name=spec.spatial_layer
        )

        # 4. Determine feature dimensions
        c_in = spec.model_spec.input_shape[0]
        h_in = spec.model_spec.input_shape[1]
        w_in = spec.model_spec.input_shape[2]
        self.feature_shape = self.adapter.compute_feature_shape((c_in, h_in, w_in))
        c_feat = self.feature_shape[0]

        # 5. Create spatial task head and loss function
        if spec.task_type == SpatialTaskType.OBJECT_DETECTION:
            self.detection_head: GridDetectionHead | None = GridDetectionHead(
                in_channels=c_feat,
                num_classes=spec.num_classes,
                seed=spec.seed,
            )
            self.segmentation_head: SegmentationHead | None = None
            self.det_loss_fn: GridDetectionLoss | None = GridDetectionLoss(
                lambda_obj=spec.lambda_obj,
                lambda_cls=spec.lambda_cls,
                lambda_box=spec.lambda_box,
            )
            self.seg_loss_fn: PixelCrossEntropyLoss | None = None

        elif spec.task_type == SpatialTaskType.SEMANTIC_SEGMENTATION:
            self.detection_head = None
            self.segmentation_head = SegmentationHead(
                in_channels=c_feat,
                num_classes=spec.num_classes,
                target_spatial_shape=(h_in, w_in),
                resize_policy=spec.resize_policy,
                seed=spec.seed,
            )
            self.det_loss_fn = None
            self.seg_loss_fn = PixelCrossEntropyLoss()
        else:
            raise ValidationError(f"Unsupported spatial task: {spec.task_type}.")

    def _get_parameter_counts(self) -> tuple[int, int, int, int, float]:
        """Compute total, frozen, trainable, head parameters and trainable fraction."""
        encoder_params = self.encoder.get_parameters()
        encoder_total = sum(_count_elements(v) for v in encoder_params.values())

        if self.detection_head is not None:
            head_params = self.detection_head.get_parameters()
        elif self.segmentation_head is not None:
            head_params = self.segmentation_head.get_parameters()
        else:
            head_params = {}

        head_total = sum(_count_elements(v) for v in head_params.values())

        strategy = self.spec.transfer_strategy
        if strategy == SpatialTransferStrategy.FROZEN_SPATIAL_PROBE:
            frozen_count = encoder_total
            trainable_count = head_total
        elif strategy == SpatialTransferStrategy.PARTIAL_FINE_TUNE:
            frozen_count = encoder_total // 2
            trainable_count = (encoder_total - frozen_count) + head_total
        elif strategy == SpatialTransferStrategy.FULL_FINE_TUNE:
            frozen_count = 0
            trainable_count = encoder_total + head_total
        else:
            frozen_count = encoder_total
            trainable_count = head_total

        total_count = frozen_count + trainable_count
        fraction = (
            float(trainable_count) / float(total_count) if total_count > 0 else 1.0
        )

        return total_count, frozen_count, trainable_count, head_total, fraction

    def train_and_evaluate(
        self,
        train_samples: list[DetectionSample] | list[SegmentationSample],
        eval_samples: list[DetectionSample] | list[SegmentationSample],
    ) -> SpatialTransferReport:
        """Execute transfer training workflow and return structured report."""
        if not train_samples:
            raise ValidationError("train_samples cannot be empty.")
        if not eval_samples:
            raise ValidationError("eval_samples cannot be empty.")

        budget_n = max(
            1, math.ceil(len(train_samples) * self.spec.data_budget_fraction)
        )
        active_train = train_samples[:budget_n]

        probe_images = [s.image for s in active_train[: min(2, len(active_train))]]
        init_probe_feats = self.adapter.extract_spatial_features(probe_images)

        total_p, frozen_p, train_p, head_p, frac_p = self._get_parameter_counts()

        loss_trajectory: list[float] = []
        lr = self.spec.learning_rate
        b_size = self.spec.batch_size
        strategy = self.spec.transfer_strategy

        cached_train_features: list[list[list[list[float]]]] | None = None
        if strategy == SpatialTransferStrategy.FROZEN_SPATIAL_PROBE:
            cached_train_features = self.adapter.extract_spatial_features(
                [s.image for s in active_train]
            )

        for _epoch in range(self.spec.epochs):
            epoch_loss_sum = 0.0
            num_batches = 0

            for b_idx in range(0, len(active_train), b_size):
                batch = active_train[b_idx : b_idx + b_size]
                num_batches += 1

                if cached_train_features is not None:
                    spatial_feats = cached_train_features[b_idx : b_idx + b_size]
                else:
                    batch_images = [s.image for s in batch]
                    spatial_feats = self.adapter.extract_spatial_features(batch_images)

                if self.spec.task_type == SpatialTaskType.OBJECT_DETECTION:
                    assert self.detection_head is not None
                    assert self.det_loss_fn is not None
                    det_batch = [s for s in batch if isinstance(s, DetectionSample)]

                    head_out = self.detection_head.forward(spatial_feats)
                    loss_metrics, d_out = self.det_loss_fn.compute_loss_and_gradients(
                        head_out, det_batch, self.spec.num_classes
                    )
                    batch_loss = loss_metrics["loss"]
                    epoch_loss_sum += batch_loss

                    self.detection_head.backward(d_out)

                    head_params = self.detection_head.conv.weights
                    head_grads = self.detection_head.conv.grad_weights
                    _apply_sgd_update(head_params, head_grads, lr)

                    if self.detection_head.conv.use_bias:
                        bias_p = self.detection_head.conv.bias_weights
                        bias_g = self.detection_head.conv.grad_bias_weights
                        _apply_sgd_update(bias_p, bias_g, lr)

                    self.detection_head.zero_grad()

                elif self.spec.task_type == SpatialTaskType.SEMANTIC_SEGMENTATION:
                    assert self.segmentation_head is not None
                    assert self.seg_loss_fn is not None
                    seg_batch = [s for s in batch if isinstance(s, SegmentationSample)]

                    logits = self.segmentation_head.forward(spatial_feats)
                    loss_metrics, d_logits = (
                        self.seg_loss_fn.compute_loss_and_gradients(logits, seg_batch)
                    )
                    batch_loss = loss_metrics["loss"]
                    epoch_loss_sum += batch_loss

                    self.segmentation_head.backward(d_logits)

                    head_params = self.segmentation_head.conv.weights
                    head_grads = self.segmentation_head.conv.grad_weights
                    _apply_sgd_update(head_params, head_grads, lr)

                    if self.segmentation_head.conv.use_bias:
                        bias_p = self.segmentation_head.conv.bias_weights
                        bias_g = self.segmentation_head.conv.grad_bias_weights
                        _apply_sgd_update(bias_p, bias_g, lr)

                    self.segmentation_head.zero_grad()

                if strategy in (
                    SpatialTransferStrategy.PARTIAL_FINE_TUNE,
                    SpatialTransferStrategy.FULL_FINE_TUNE,
                ):
                    enc_params = self.encoder.get_parameters()
                    for p_key, p_val in enc_params.items():
                        if strategy == SpatialTransferStrategy.PARTIAL_FINE_TUNE and (
                            "stem" in p_key or "conv_0" in p_key or "block_0" in p_key
                        ):
                            continue
                        if isinstance(p_val, list):
                            _apply_sgd_update(p_val, copy.deepcopy(p_val), lr * 0.01)
                    self.encoder.set_parameters(enc_params)

            mean_epoch_loss = epoch_loss_sum / float(max(1, num_batches))
            loss_trajectory.append(float(mean_epoch_loss))

        final_probe_feats = self.adapter.extract_spatial_features(probe_images)
        cos_drift, rmse_drift = _compute_drift_metrics(
            init_probe_feats, final_probe_feats
        )

        det_metrics: DetectionEvaluationResult | None = None
        seg_metrics: SegmentationMetricsResult | None = None

        if self.spec.task_type == SpatialTaskType.OBJECT_DETECTION:
            assert self.detection_head is not None
            eval_det = [s for s in eval_samples if isinstance(s, DetectionSample)]
            eval_images = [s.image for s in eval_det]
            feats = self.adapter.extract_spatial_features(eval_images)
            outputs = self.detection_head.forward(feats)
            predictions = self.detection_head.decode_predictions(
                outputs,
                objectness_threshold=0.2,
                sample_ids=[s.sample_id for s in eval_det],
            )
            det_metrics = evaluate_detection_predictions(
                predictions, eval_det, iou_threshold=0.5
            )

        elif self.spec.task_type == SpatialTaskType.SEMANTIC_SEGMENTATION:
            assert self.segmentation_head is not None
            eval_seg = [s for s in eval_samples if isinstance(s, SegmentationSample)]
            eval_images = [s.image for s in eval_seg]
            feats = self.adapter.extract_spatial_features(eval_images)
            logits = self.segmentation_head.forward(feats)
            pred_masks = self.segmentation_head.predict_masks(logits)
            gt_masks = [s.mask for s in eval_seg]

            cm = SegmentationConfusionMatrix(num_classes=self.spec.num_classes)
            cm.update(pred_masks, gt_masks)
            seg_metrics = cm.compute_metrics()

        warnings: list[str] = []
        if self.feature_shape[1] <= 4:
            warnings.append(
                f"Low resolution ({self.feature_shape[1]}x{self.feature_shape[2]})."
            )
        if self.spec.data_budget_fraction < 1.0:
            warnings.append(
                f"Trained on {int(self.spec.data_budget_fraction * 100)}% budget."
            )

        report_hasher = hashlib.sha256()
        report_hasher.update(
            f"{self.spec.specification_id}_{loss_trajectory}_{det_metrics}_{seg_metrics}".encode()
        )
        report_id = f"spatial_rep_{report_hasher.hexdigest()[:12]}"

        return SpatialTransferReport(
            report_id=report_id,
            specification=self.spec,
            total_parameters=total_p,
            frozen_parameters=frozen_p,
            trainable_parameters=train_p,
            head_parameters=head_p,
            trainable_fraction=frac_p,
            feature_shape=self.feature_shape,
            feature_resolution=f"{self.feature_shape[1]}x{self.feature_shape[2]}",
            training_loss_trajectory=loss_trajectory,
            epochs_completed=self.spec.epochs,
            detection_metrics=det_metrics,
            segmentation_metrics=seg_metrics,
            spatial_representation_drift_cosine=cos_drift,
            spatial_representation_drift_rmse=rmse_drift,
            warnings=warnings,
        )
