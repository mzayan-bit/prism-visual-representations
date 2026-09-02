"""Reconstruction engine executing masked modeling and denoising."""

from __future__ import annotations

import copy
import math
from typing import Any

from prism.core.enums import ModelFamily
from prism.data.materialized import MaterializedDataset
from prism.models.base import BaseVisionModel
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.patches import PatchGeometry
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.transformer import VisionTransformer
from prism.reconstruction.batch import (
    ReconstructionBatch,
    prepare_denoising_batch,
    prepare_masked_patch_batch,
)
from prism.reconstruction.decoders import (
    PatchReconstructionDecoder,
    SpatialReconstructionDecoder,
)
from prism.reconstruction.diagnostics import compute_reconstruction_diagnostics
from prism.reconstruction.enums import ReconstructionMethod
from prism.reconstruction.loss import MaskedMSELoss
from prism.reconstruction.reports import ReconstructionLearningReport
from prism.reconstruction.specification import ReconstructionLearningSpecification
from prism.reconstruction.tokens import LearnableMaskToken
from prism.robustness.corruptions import CorruptionSpecification, CorruptionType
from prism.ssl.adapter import RepresentationEncoder
from prism.transfer.probes import probe_layer_transferability
from prism.transfer.snapshot import create_model_state_snapshot


class ReconstructionTrainingEngine:
    """Orchestrates generative and reconstruction-based representation learning."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def _zeros_like(v: Any) -> Any:
        if isinstance(v, list):
            return [ReconstructionTrainingEngine._zeros_like(x) for x in v]
        return 0.0

    @staticmethod
    def _recursive_sgd_step(
        p: Any,
        g: Any,
        vel: Any,
        lr: float,
        momentum: float,
        weight_decay: float,
    ) -> tuple[Any, Any]:
        if isinstance(p, list):
            new_p = []
            new_vel = []
            for p_elem, g_elem, vel_elem in zip(p, g, vel, strict=True):
                np, nv = ReconstructionTrainingEngine._recursive_sgd_step(
                    p_elem, g_elem, vel_elem, lr, momentum, weight_decay
                )
                new_p.append(np)
                new_vel.append(nv)
            return new_p, new_vel
        elif isinstance(p, (int, float)):
            grad_val = float(g) + weight_decay * float(p)
            v = momentum * float(vel) + grad_val
            updated_p = float(p) - lr * v
            return updated_p, v
        else:
            return copy.deepcopy(p), copy.deepcopy(vel)

    def train(
        self,
        dataset: MaterializedDataset,
        spec: ReconstructionLearningSpecification,
        downstream_target_dataset: MaterializedDataset | None = None,
    ) -> ReconstructionLearningReport:
        """Execute reconstruction pretraining and downstream evaluation."""
        fam = spec.encoder_family
        is_vit_masked = (
            fam == ModelFamily.VISION_TRANSFORMER
            and spec.method == ReconstructionMethod.MASKED_PATCH_RECONSTRUCTION
        )

        loss_fn = MaskedMSELoss(reduction="mean")
        loss_history: list[float] = []
        masked_mse_history: list[float] = []
        lr_history: list[float] = []

        # 1. Instantiate encoder backbone
        encoder_backbone: BaseVisionModel
        if fam == ModelFamily.CNN:
            encoder_backbone = ConvolutionalNeuralNetwork(
                spec.encoder_spec, seed=spec.seed
            )
        elif fam == ModelFamily.RESNET:
            encoder_backbone = ResidualNeuralNetwork(spec.encoder_spec, seed=spec.seed)
        elif fam == ModelFamily.VISION_TRANSFORMER:
            encoder_backbone = VisionTransformer(spec.encoder_spec, seed=spec.seed)
        else:
            raise ValueError(f"Unsupported reconstruction encoder family: {fam}")

        # 2. Instantiate decoder and masking adapters
        patch_decoder: PatchReconstructionDecoder | None = None
        spatial_decoder: SpatialReconstructionDecoder | None = None
        mask_token: LearnableMaskToken | None = None
        geometry: PatchGeometry | None = None
        rep_encoder: RepresentationEncoder | None = None

        if is_vit_masked:
            assert isinstance(encoder_backbone, VisionTransformer)
            geometry = encoder_backbone.geometry
            embed_dim = encoder_backbone.embed_dim
            patch_dim = geometry.flattened_patch_dimension
            mask_token = LearnableMaskToken(embed_dim=embed_dim, seed=spec.seed)
            patch_decoder = PatchReconstructionDecoder(
                in_features=embed_dim, patch_dim=patch_dim, bias=True, seed=spec.seed
            )
        else:
            rep_encoder = RepresentationEncoder(
                backbone=encoder_backbone, seed=spec.seed
            )
            spatial_decoder = SpatialReconstructionDecoder(
                in_features=rep_encoder.representation_dim,
                output_shape=spec.input_shape,
                bias=True,
                seed=spec.seed,
            )

        # 3. Setup optimizer velocities
        velocities_encoder: dict[str, Any] = {
            k: self._zeros_like(v)
            for k, v in (
                encoder_backbone.get_parameters().items()
                if is_vit_masked
                else rep_encoder.backbone.get_parameters().items()  # type: ignore[union-attr]
            )
        }
        dec_params_init = (
            patch_decoder.get_parameters()  # type: ignore[union-attr]
            if is_vit_masked
            else spatial_decoder.get_parameters()  # type: ignore[union-attr]
        )
        velocities_decoder: dict[str, Any] = {
            k: self._zeros_like(v) for k, v in dec_params_init.items()
        }
        velocities_mask_token: dict[str, Any] = (
            {k: self._zeros_like(v) for k, v in mask_token.get_parameters().items()}
            if mask_token
            else {}
        )

        samples = [dataset[i] for i in range(len(dataset))]

        # 4. Training loop
        for epoch in range(spec.epochs):
            lr = (
                spec.learning_rate
                * 0.5
                * (1.0 + math.cos(math.pi * epoch / float(spec.epochs)))
            )
            epoch_loss = 0.0
            epoch_masked_mse = 0.0
            step_count = 0

            # Batching
            for start_idx in range(0, len(samples), spec.batch_size):
                batch_samples = samples[start_idx : start_idx + spec.batch_size]
                if not batch_samples:
                    continue

                batch: ReconstructionBatch
                if is_vit_masked:
                    assert geometry is not None
                    batch = prepare_masked_patch_batch(
                        samples=batch_samples,
                        geometry=geometry,
                        epoch=epoch,
                        mask_ratio=spec.mask_ratio,
                        seed=spec.seed,
                    )
                else:
                    corr_spec = CorruptionSpecification(
                        corruption_type=spec.corruption_type
                        or CorruptionType.GAUSSIAN_NOISE,
                        severity=spec.corruption_severity,
                    )
                    batch = prepare_denoising_batch(
                        samples=batch_samples,
                        corruption_spec=corr_spec,
                        epoch=epoch,
                        seed=spec.seed,
                    )

                # Forward & Backward
                if is_vit_masked:
                    assert isinstance(encoder_backbone, VisionTransformer)
                    assert patch_decoder is not None
                    assert mask_token is not None
                    assert batch.masks is not None

                    encoder_backbone.zero_grad()
                    patch_decoder.zero_grad()
                    mask_token.zero_grad()

                    # ViT patch processing with mask tokens
                    # batch.inputs is [N x T x D_patch]
                    patch_embeds = encoder_backbone.patch_embed.forward(batch.inputs)
                    n_batch = len(batch_samples)

                    masked_embeds: list[list[list[float]]] = []
                    for b_idx in range(n_batch):
                        m = batch.masks[b_idx]
                        sub_emb = mask_token.replace_masked_patches(
                            patch_embeds[b_idx], m.masked_indices
                        )
                        masked_embeds.append(sub_emb)

                    tokens_with_cls = encoder_backbone.cls_token.forward(masked_embeds)
                    tokens_with_pos = encoder_backbone.pos_embed.forward(
                        tokens_with_cls
                    )
                    encoder_out = encoder_backbone.encoder.forward(tokens_with_pos)
                    norm_out = encoder_backbone.norm.forward(encoder_out)

                    # Extract patch tokens (indices 1 to T+1)
                    patch_latents: list[list[list[float]]] = [
                        [list(norm_out[b][t]) for t in range(1, len(norm_out[b]))]
                        for b in range(n_batch)
                    ]

                    # Decoder reconstructs patch pixels [N x T x D_patch]
                    pred_patches = patch_decoder.forward(patch_latents)

                    loss_val, d_preds, metrics = loss_fn(
                        pred_patches, batch.targets, masks=batch.masks
                    )

                    # Backward pass
                    d_patch_latents = patch_decoder.backward(d_preds)

                    # Embed into sequence gradient [N x T+1 x D_model]
                    d_norm_out: list[list[list[float]]] = []
                    for b in range(n_batch):
                        cls_grad = [0.0] * encoder_backbone.embed_dim
                        d_norm_out.append([cls_grad] + d_patch_latents[b])

                    d_encoder_out = encoder_backbone.norm.backward(d_norm_out)
                    d_pos = encoder_backbone.encoder.backward(d_encoder_out)
                    d_cls_prepended = encoder_backbone.pos_embed.backward(d_pos)
                    d_embeddings = encoder_backbone.cls_token.backward(d_cls_prepended)

                    # Route through mask token
                    d_patch_embeds: list[list[list[float]]] = []
                    for b in range(n_batch):
                        d_sub = mask_token.backward_masked_tokens(
                            d_embeddings[b], batch.masks[b].masked_indices
                        )
                        d_patch_embeds.append(d_sub)

                    encoder_backbone.patch_embed.backward(d_patch_embeds)

                    # Optimizer step: update encoder, decoder, mask token
                    enc_params = encoder_backbone.get_parameters()
                    enc_grads = encoder_backbone.get_gradients()
                    for k in enc_params:
                        if k in enc_grads:
                            new_p, new_v = self._recursive_sgd_step(
                                enc_params[k],
                                enc_grads[k],
                                velocities_encoder[k],
                                lr,
                                spec.momentum,
                                spec.weight_decay,
                            )
                            enc_params[k] = new_p
                            velocities_encoder[k] = new_v
                    encoder_backbone.set_parameters(enc_params)

                    dec_params = patch_decoder.get_parameters()
                    dec_grads = patch_decoder.get_gradients()
                    for k in dec_params:
                        new_p, new_v = self._recursive_sgd_step(
                            dec_params[k],
                            dec_grads[k],
                            velocities_decoder[k],
                            lr,
                            spec.momentum,
                            spec.weight_decay,
                        )
                        dec_params[k] = new_p
                        velocities_decoder[k] = new_v
                    patch_decoder.set_parameters(dec_params)

                    tok_params = mask_token.get_parameters()
                    tok_grads = mask_token.get_gradients()
                    for k in tok_params:
                        new_p, new_v = self._recursive_sgd_step(
                            tok_params[k],
                            tok_grads[k],
                            velocities_mask_token[k],
                            lr,
                            spec.momentum,
                            spec.weight_decay,
                        )
                        tok_params[k] = new_p
                        velocities_mask_token[k] = new_v
                    mask_token.set_parameters(tok_params)

                    epoch_loss += loss_val
                    epoch_masked_mse += metrics["masked_mse"]
                    step_count += 1
                else:
                    # Spatial Denoising Autoencoder (CNN / ResNet)
                    assert rep_encoder is not None
                    assert spatial_decoder is not None

                    rep_encoder.backbone.zero_grad()
                    spatial_decoder.zero_grad()

                    latents = rep_encoder.forward(batch.inputs)
                    rec_images = spatial_decoder.forward(latents)

                    loss_val, d_preds_img, metrics = loss_fn.compute_image_mse(
                        rec_images, batch.targets
                    )

                    d_latents = spatial_decoder.backward(d_preds_img)
                    rep_encoder.backward(d_latents)

                    enc_params = rep_encoder.backbone.get_parameters()
                    enc_grads = rep_encoder.backbone.get_gradients()
                    for k in enc_params:
                        if k in enc_grads:
                            new_p, new_v = self._recursive_sgd_step(
                                enc_params[k],
                                enc_grads[k],
                                velocities_encoder[k],
                                lr,
                                spec.momentum,
                                spec.weight_decay,
                            )
                            enc_params[k] = new_p
                            velocities_encoder[k] = new_v
                    rep_encoder.backbone.set_parameters(enc_params)

                    dec_params = spatial_decoder.get_parameters()
                    dec_grads = spatial_decoder.get_gradients()
                    for k in dec_params:
                        new_p, new_v = self._recursive_sgd_step(
                            dec_params[k],
                            dec_grads[k],
                            velocities_decoder[k],
                            lr,
                            spec.momentum,
                            spec.weight_decay,
                        )
                        dec_params[k] = new_p
                        velocities_decoder[k] = new_v
                    spatial_decoder.set_parameters(dec_params)

                    epoch_loss += loss_val
                    epoch_masked_mse += metrics["full_mse"]
                    step_count += 1

            loss_history.append(epoch_loss / float(max(1, step_count)))
            masked_mse_history.append(epoch_masked_mse / float(max(1, step_count)))
            lr_history.append(lr)

        # 5. Compute final diagnostics over full dataset
        final_preds: list[Any] = []
        final_targets: list[Any] = []
        final_latents: list[list[float]] = []
        final_corrupted: list[Any] = []

        all_imgs = [s.data for s in samples]

        if is_vit_masked:
            assert isinstance(encoder_backbone, VisionTransformer)
            assert patch_decoder is not None
            assert geometry is not None
            extractor = encoder_backbone.patch_extractor
            patches = extractor.forward(all_imgs)
            final_targets = patches

            # Use fixed 0.5 mask for final evaluation
            eval_ctx = [
                prepare_masked_patch_batch(
                    [s], geometry=geometry, epoch=0, mask_ratio=0.5, seed=spec.seed
                )
                for s in samples
            ]
            for _i, s_batch in enumerate(eval_ctx):
                assert s_batch.masks is not None
                p_emb = encoder_backbone.patch_embed.forward(s_batch.inputs)
                m_emb = mask_token.replace_masked_patches(  # type: ignore[union-attr]
                    p_emb[0], s_batch.masks[0].masked_indices
                )
                t_cls = encoder_backbone.cls_token.forward([m_emb])
                t_pos = encoder_backbone.pos_embed.forward(t_cls)
                enc_out = encoder_backbone.encoder.forward(t_pos)
                norm_out = encoder_backbone.norm.forward(enc_out)
                p_latents = [norm_out[0][t] for t in range(1, len(norm_out[0]))]
                rec_p = patch_decoder.forward([p_latents])[0]
                final_preds.append(rec_p)
                # CLS token as representation
                final_latents.append(list(norm_out[0][0]))
        else:
            assert rep_encoder is not None
            assert spatial_decoder is not None
            corr_op = CorruptionSpecification(
                corruption_type=spec.corruption_type or CorruptionType.GAUSSIAN_NOISE,
                severity=spec.corruption_severity,
            )
            denoise_batch = prepare_denoising_batch(
                samples=samples,
                corruption_spec=corr_op,
                epoch=0,
                seed=spec.seed,
            )
            final_targets = denoise_batch.targets
            final_corrupted = denoise_batch.inputs
            final_latents = rep_encoder.forward(denoise_batch.inputs)
            final_preds = spatial_decoder.forward(final_latents)

        diagnostics = compute_reconstruction_diagnostics(
            predictions=final_preds,
            targets=final_targets,
            latents=final_latents,
            is_patch_based=is_vit_masked,
            corrupted_inputs=final_corrupted if not is_vit_masked else None,
        )

        # 6. Save encoder snapshot
        trained_backbone = encoder_backbone if is_vit_masked else rep_encoder.backbone  # type: ignore[union-attr]
        snapshot = create_model_state_snapshot(
            model=trained_backbone,
            source_experiment_id=spec.reconstruction_id,
        )

        # 7. Optional downstream linear probe evaluation
        probe_acc: float | None = None
        if downstream_target_dataset is not None:
            layer_name = "cls_representation" if is_vit_masked else "final_hidden"
            probe_result = probe_layer_transferability(
                model=trained_backbone,
                train_dataset=downstream_target_dataset,
                layer=layer_name,
                target_num_classes=2,
                epochs=5,
                seed=spec.seed,
            )
            probe_acc = probe_result.train_accuracy

        return ReconstructionLearningReport(
            reconstruction_id=spec.reconstruction_id,
            method=spec.method,
            encoder_family=spec.encoder_family,
            dataset_id=spec.dataset_id,
            mask_ratio=spec.mask_ratio if is_vit_masked else None,
            corruption_type=spec.corruption_type if not is_vit_masked else None,
            corruption_severity=spec.corruption_severity if not is_vit_masked else None,
            epochs_trained=spec.epochs,
            loss_history=loss_history,
            masked_mse_history=masked_mse_history,
            learning_rate_history=lr_history,
            diagnostics=diagnostics,
            downstream_linear_probe_accuracy=probe_acc,
            encoder_snapshot_id=snapshot.source_experiment_id,
            parameter_checksum=snapshot.parameter_checksum,
        )
