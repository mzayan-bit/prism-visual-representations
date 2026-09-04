"""Dual-Encoder Vision-Language Multimodal Training Engine."""

from __future__ import annotations

from typing import Any

from prism.core.enums import ModelFamily
from prism.core.errors import ValidationError
from prism.models.base import BaseVisionModel
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.transformer import VisionTransformer
from prism.multimodal.contracts import VisionLanguageSample
from prism.multimodal.embeddings import (
    TextEncoder,
    VisualProjectionHead,
)
from prism.multimodal.loss import SymmetricContrastiveLoss
from prism.multimodal.specification import VisionLanguageTrainingSpecification
from prism.multimodal.tokenizer import SimpleTokenizer
from prism.ssl.adapter import RepresentationEncoder
from prism.ssl.projection import (
    backward_normalize_embeddings,
    normalize_embeddings,
)


class MultimodalTrainingEngine:
    """Orchestrates CLIP-style dual-encoder symmetric contrastive pretraining."""

    def __init__(self) -> None:
        pass

    def _instantiate_visual_backbone(
        self, spec: VisionLanguageTrainingSpecification
    ) -> RepresentationEncoder:
        """Instantiate visual encoder wrapped with RepresentationEncoder."""
        fam = spec.visual_family
        backbone: BaseVisionModel
        if fam == ModelFamily.CNN:
            backbone = ConvolutionalNeuralNetwork(spec.visual_spec, seed=spec.seed)
        elif fam == ModelFamily.RESNET:
            backbone = ResidualNeuralNetwork(spec.visual_spec, seed=spec.seed)
        elif fam == ModelFamily.VISION_TRANSFORMER:
            backbone = VisionTransformer(spec.visual_spec, seed=spec.seed)
        else:
            raise ValidationError(f"Unsupported visual encoder family: {fam}")

        return RepresentationEncoder(backbone=backbone, seed=spec.seed)

    @staticmethod
    def _zeros_like(v: Any) -> Any:
        if isinstance(v, list):
            return [MultimodalTrainingEngine._zeros_like(x) for x in v]
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
                np, nv = MultimodalTrainingEngine._recursive_sgd_step(
                    p_elem, g_elem, vel_elem, lr, momentum, weight_decay
                )
                new_p.append(np)
                new_vel.append(nv)
            return new_p, new_vel
        else:
            val_p = float(p)
            val_g = float(g) + weight_decay * val_p
            new_v = momentum * float(vel) + val_g
            updated_p = val_p - lr * new_v
            return updated_p, new_v

    def _update_parameters(
        self,
        params: dict[str, Any],
        grads: dict[str, Any],
        velocities: dict[str, Any],
        lr: float,
        momentum: float,
        weight_decay: float,
    ) -> None:
        """Apply recursive SGD step with momentum and weight decay."""
        for k, v in params.items():
            g = grads.get(k)
            if g is None:
                g = grads.get(f"grad_{k}")
            if g is None:
                continue
            if k not in velocities:
                velocities[k] = self._zeros_like(v)
            new_p, new_vel = self._recursive_sgd_step(
                v, g, velocities[k], lr, momentum, weight_decay
            )
            params[k] = new_p
            velocities[k] = new_vel

    def train(
        self,
        spec: VisionLanguageTrainingSpecification,
        samples: list[VisionLanguageSample],
        tokenizer: SimpleTokenizer,
    ) -> tuple[
        RepresentationEncoder,
        VisualProjectionHead,
        TextEncoder,
        list[dict[str, float]],
    ]:
        """Train dual encoder on paired vision-language dataset.

        Returns:
            tuple of (visual_encoder, visual_projection, text_encoder, history)
        """
        if not samples:
            raise ValidationError("Cannot train on empty multimodal samples list.")

        # 1. Instantiate Models
        visual_encoder = self._instantiate_visual_backbone(spec)
        visual_dim = visual_encoder.representation_dim

        visual_projection = VisualProjectionHead(
            in_dim=visual_dim,
            out_dim=spec.shared_dim,
            use_mlp=spec.use_mlp_projection,
            seed=spec.seed + 10,
        )

        text_encoder = TextEncoder(
            vocab_size=tokenizer.vocab.size,
            text_dim=spec.text_dim,
            shared_dim=spec.shared_dim,
            use_mlp=spec.use_mlp_projection,
            seed=spec.seed + 20,
        )

        loss_fn = SymmetricContrastiveLoss(temperature=spec.temperature)

        # 2. Initialize SGD Velocities
        vis_params = visual_encoder.get_parameters()
        vis_vel = {k: self._zeros_like(v) for k, v in vis_params.items()}

        vis_proj_params = visual_projection.get_parameters()
        vis_proj_vel = {k: self._zeros_like(v) for k, v in vis_proj_params.items()}

        txt_params = text_encoder.get_parameters()
        txt_vel = {k: self._zeros_like(v) for k, v in txt_params.items()}

        history: list[dict[str, float]] = []
        n_samples = len(samples)
        batch_size = min(spec.batch_size, n_samples)

        # 3. Training Epochs
        for epoch in range(spec.epochs):
            visual_encoder.train(True)
            epoch_loss = 0.0
            epoch_v2t_loss = 0.0
            epoch_t2v_loss = 0.0
            epoch_pos_sim = 0.0
            epoch_neg_sim = 0.0
            steps = 0

            # Batch iteration
            for start_idx in range(0, n_samples, batch_size):
                batch_samples = samples[start_idx : start_idx + batch_size]
                if len(batch_samples) < 2:
                    continue

                # Zero gradients
                visual_encoder.zero_grad()
                visual_projection.zero_grad()
                text_encoder.zero_grad()

                # Visual forward pass
                images = [s.image for s in batch_samples]
                vis_feats = visual_encoder.forward(images)  # (B, D_vis)
                vis_proj = visual_projection.forward(vis_feats)  # (B, D_shared)
                norm_v, v_norms = normalize_embeddings(vis_proj)

                # Text forward pass
                tokenized = [tokenizer.encode(s.text) for s in batch_samples]
                token_ids = [t.token_ids for t in tokenized]
                masks = [t.attention_mask for t in tokenized]
                txt_proj, _ = text_encoder.forward(token_ids, masks)  # (B, D_shared)
                norm_t, t_norms = normalize_embeddings(txt_proj)

                # Multimodal symmetric loss (Strictly label-independent)
                loss_val, d_norm_v, d_norm_t, metrics = loss_fn(norm_v, norm_t)

                # Backward pass through L2 normalization
                d_vis_proj = backward_normalize_embeddings(d_norm_v, vis_proj, v_norms)
                d_txt_proj = backward_normalize_embeddings(d_norm_t, txt_proj, t_norms)

                # Backprop visual branch
                d_vis_feats = visual_projection.backward(d_vis_proj)
                visual_encoder.backward(d_vis_feats)

                # Backprop text branch
                text_encoder.backward(d_txt_proj)

                # SGD Update: Visual Encoder
                cur_v_params = visual_encoder.get_parameters()
                cur_v_grads = visual_encoder.get_gradients()
                self._update_parameters(
                    cur_v_params,
                    cur_v_grads,
                    vis_vel,
                    lr=spec.learning_rate,
                    momentum=spec.momentum,
                    weight_decay=spec.weight_decay,
                )
                visual_encoder.set_parameters(cur_v_params)

                # SGD Update: Visual Projection Head
                cur_vp_params = visual_projection.get_parameters()
                cur_vp_grads = visual_projection.get_gradients()
                self._update_parameters(
                    cur_vp_params,
                    cur_vp_grads,
                    vis_proj_vel,
                    lr=spec.learning_rate,
                    momentum=spec.momentum,
                    weight_decay=spec.weight_decay,
                )
                visual_projection.set_parameters(cur_vp_params)

                # SGD Update: Text Encoder
                cur_t_params = text_encoder.get_parameters()
                cur_t_grads = text_encoder.get_gradients()
                self._update_parameters(
                    cur_t_params,
                    cur_t_grads,
                    txt_vel,
                    lr=spec.learning_rate,
                    momentum=spec.momentum,
                    weight_decay=spec.weight_decay,
                )
                text_encoder.set_parameters(cur_t_params)

                # Accumulate metrics
                epoch_loss += loss_val
                epoch_v2t_loss += metrics["image_to_text_loss"]
                epoch_t2v_loss += metrics["text_to_image_loss"]
                epoch_pos_sim += metrics["matched_similarity"]
                epoch_neg_sim += metrics["unmatched_similarity"]
                steps += 1

            if steps > 0:
                history.append(
                    {
                        "epoch": float(epoch + 1),
                        "loss": epoch_loss / float(steps),
                        "image_to_text_loss": epoch_v2t_loss / float(steps),
                        "text_to_image_loss": epoch_t2v_loss / float(steps),
                        "matched_similarity": epoch_pos_sim / float(steps),
                        "unmatched_similarity": epoch_neg_sim / float(steps),
                        "similarity_gap": (epoch_pos_sim - epoch_neg_sim)
                        / float(steps),
                        "learning_rate": spec.learning_rate,
                    }
                )

        visual_encoder.eval()
        return visual_encoder, visual_projection, text_encoder, history
