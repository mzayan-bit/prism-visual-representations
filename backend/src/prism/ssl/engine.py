"""Self-supervised contrastive training engine for SimCLR pretraining."""

from __future__ import annotations

from typing import Any

from prism.core.enums import ModelFamily
from prism.data.materialized import MaterializedDataset, MaterializedSample
from prism.models.base import BaseVisionModel
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.transformer import VisionTransformer
from prism.ssl.adapter import RepresentationEncoder
from prism.ssl.diagnostics import compute_collapse_diagnostics
from prism.ssl.loss import ContrastiveNTXentLoss
from prism.ssl.projection import (
    SimCLRProjectionHead,
    backward_normalize_embeddings,
    normalize_embeddings,
)
from prism.ssl.reports import SelfSupervisedLearningReport
from prism.ssl.specification import SelfSupervisedTrainingSpecification
from prism.ssl.views import ContrastiveBatchLoader
from prism.transfer.snapshot import ModelStateSnapshot, create_model_state_snapshot


class SelfSupervisedTrainingEngine:
    """Orchestrates SimCLR contrastive pretraining loop."""

    def __init__(self) -> None:
        pass

    def _instantiate_model(
        self, spec: SelfSupervisedTrainingSpecification
    ) -> RepresentationEncoder:
        """Instantiate underlying backbone and wrap with RepresentationEncoder."""
        fam = spec.encoder_family
        backbone: BaseVisionModel
        if fam == ModelFamily.CNN:
            backbone = ConvolutionalNeuralNetwork(spec.encoder_spec, seed=spec.seed)
        elif fam == ModelFamily.RESNET:
            backbone = ResidualNeuralNetwork(spec.encoder_spec, seed=spec.seed)
        elif fam == ModelFamily.VISION_TRANSFORMER:
            backbone = VisionTransformer(spec.encoder_spec, seed=spec.seed)
        else:
            raise ValueError(f"Unsupported SSL encoder family: {fam}")

        return RepresentationEncoder(backbone=backbone, seed=spec.seed)

    def _update_parameters(
        self,
        params: dict[str, Any],
        grads: dict[str, Any],
        velocities: dict[str, Any],
        lr: float,
        momentum: float,
        weight_decay: float,
    ) -> None:
        """Apply SGD update with momentum and weight decay."""
        for k, v in params.items():
            if isinstance(v, list) and v and isinstance(v[0], list):
                # 2D Matrix (weights)
                if k not in velocities:
                    velocities[k] = [
                        [0.0 for _ in range(len(v[0]))] for _ in range(len(v))
                    ]
                g_mat = grads.get(k)
                if g_mat is None:
                    continue
                for r in range(len(v)):
                    for c in range(len(v[0])):
                        g = g_mat[r][c] + weight_decay * v[r][c]
                        vel = momentum * velocities[k][r][c] + g
                        velocities[k][r][c] = vel
                        v[r][c] -= lr * vel
            elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
                # 1D Vector (bias)
                if k not in velocities:
                    velocities[k] = [0.0 for _ in range(len(v))]
                g_vec = grads.get(k)
                if g_vec is None:
                    continue
                for i in range(len(v)):
                    g = float(g_vec[i])
                    vel = momentum * float(velocities[k][i]) + g
                    velocities[k][i] = vel
                    v[i] -= lr * vel

    def train_ssl(
        self,
        specification: SelfSupervisedTrainingSpecification,
        dataset: MaterializedDataset,
        reference_samples: list[MaterializedSample] | None = None,
    ) -> tuple[RepresentationEncoder, ModelStateSnapshot, SelfSupervisedLearningReport]:
        """Execute full SimCLR contrastive pretraining loop."""
        encoder = self._instantiate_model(specification)
        encoder.train(True)

        rep_dim = encoder.representation_dim
        head = SimCLRProjectionHead(
            in_dim=rep_dim,
            hidden_dim=specification.projection_hidden_dim,
            out_dim=specification.projection_out_dim,
            seed=specification.seed,
        )

        loss_fn = ContrastiveNTXentLoss(temperature=specification.temperature)
        loader = ContrastiveBatchLoader(
            dataset=dataset,
            batch_size=specification.batch_size,
            seed=specification.seed,
        )

        # SGD velocity states
        encoder_velocities: dict[str, Any] = {}
        head_velocities: dict[str, Any] = {}

        loss_traj: list[float] = []
        pos_sim_traj: list[float] = []
        neg_sim_traj: list[float] = []
        gap_traj: list[float] = []
        lr_traj: list[float] = []

        cur_lr = specification.learning_rate

        for ep in range(specification.epochs):
            batches = loader.get_batches(epoch=ep)
            ep_losses: list[float] = []
            ep_pos_sims: list[float] = []
            ep_neg_sims: list[float] = []

            for batch in batches:
                # 1. Zero gradients
                encoder.zero_grad()
                head.zero_grad()

                # 2. Forward representations h: [2N x D_rep]
                h = encoder.forward(batch.views)

                # 3. Forward projection z: [2N x D_proj]
                z = head.forward(h)

                # 4. L2 Normalization
                hat_z, norms = normalize_embeddings(z)

                # 5. NT-Xent Loss & analytical gradient w.r.t hat_z
                loss_val, d_hat_z, metrics = loss_fn(hat_z, batch.positive_indices)
                ep_losses.append(loss_val)
                ep_pos_sims.append(metrics["positive_similarity"])
                ep_neg_sims.append(metrics["negative_similarity"])

                # 6. Backward through normalization
                d_z = backward_normalize_embeddings(d_hat_z, z, norms)

                # 7. Backward through projection head
                d_h = head.backward(d_z)

                # 8. Backward through encoder
                encoder.backward(d_h)

                # 9. Optimizer updates
                enc_params = encoder.get_parameters()
                enc_grads = encoder.get_gradients()
                self._update_parameters(
                    params=enc_params,
                    grads=enc_grads,
                    velocities=encoder_velocities,
                    lr=cur_lr,
                    momentum=specification.momentum,
                    weight_decay=specification.weight_decay,
                )
                encoder.set_parameters(enc_params)

                head_params = head.get_parameters()
                head_grads = head.get_gradients()
                self._update_parameters(
                    params=head_params,
                    grads=head_grads,
                    velocities=head_velocities,
                    lr=cur_lr,
                    momentum=specification.momentum,
                    weight_decay=specification.weight_decay,
                )
                head.set_parameters(head_params)

            # Record epoch telemetry
            mean_ep_loss = sum(ep_losses) / len(ep_losses) if ep_losses else 0.0
            mean_ep_pos = sum(ep_pos_sims) / len(ep_pos_sims) if ep_pos_sims else 1.0
            mean_ep_neg = sum(ep_neg_sims) / len(ep_neg_sims) if ep_neg_sims else 0.0

            loss_traj.append(mean_ep_loss)
            pos_sim_traj.append(mean_ep_pos)
            neg_sim_traj.append(mean_ep_neg)
            gap_traj.append(mean_ep_pos - mean_ep_neg)
            lr_traj.append(cur_lr)

            # Cosine-like decay
            cur_lr *= 0.95

        # 10. Compute final representations and collapse diagnostics
        encoder.eval()
        eval_samples = reference_samples or list(dataset.samples)
        if eval_samples:
            eval_inputs = [s.data for s in eval_samples]
            final_h = encoder.forward(eval_inputs)
        else:
            final_h = []

        collapse = compute_collapse_diagnostics(final_h)

        # 11. Create ModelStateSnapshot for encoder (discarding projection head)
        encoder_snapshot = create_model_state_snapshot(
            model=encoder.backbone,
            source_experiment_id=specification.ssl_id,
        )

        total_enc_params = sum(
            len(v) * len(v[0])
            if isinstance(v, list) and v and isinstance(v[0], list)
            else len(v)
            for v in encoder.get_parameters().values()
            if isinstance(v, list)
        )
        total_head_params = sum(
            len(v) * len(v[0])
            if isinstance(v, list) and v and isinstance(v[0], list)
            else len(v)
            for v in head.get_parameters().values()
            if isinstance(v, list)
        )

        warnings: list[str] = list(collapse.warnings)
        if specification.batch_size < 8:
            warnings.append(
                f"Small batch size (N={specification.batch_size}) "
                f"provides limited negative contrastive samples."
            )

        report = SelfSupervisedLearningReport(
            ssl_id=specification.ssl_id,
            encoder_family=specification.encoder_family.value,
            architecture=specification.encoder_spec.architecture,
            dataset_id=specification.dataset_id,
            total_encoder_parameters=total_enc_params,
            projection_head_parameters=total_head_params,
            epochs=specification.epochs,
            temperature=specification.temperature,
            loss_trajectory=loss_traj,
            positive_similarity_trajectory=pos_sim_traj,
            negative_similarity_trajectory=neg_sim_traj,
            similarity_gap_trajectory=gap_traj,
            learning_rate_trajectory=lr_traj,
            collapse_summary=collapse,
            warnings=warnings,
        )

        return encoder, encoder_snapshot, report
