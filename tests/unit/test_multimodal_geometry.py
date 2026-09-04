"""Unit tests for shared multimodal geometry, joint PCA, and collapse diagnostics."""

from __future__ import annotations

from prism.multimodal.evaluation import (
    compute_multimodal_collapse_diagnostics,
    compute_shared_multimodal_geometry,
)
from prism.ssl.projection import normalize_embeddings


def test_shared_multimodal_geometry_computation() -> None:
    """Verify joint PCA basis fitting and cross-modal centroid alignment."""
    n = 6
    # Raw Image embeddings and Text embeddings
    raw_image_embeds = [
        [0.8, 0.2, 0.0, 0.0],
        [0.7, 0.3, 0.0, 0.0],
        [0.0, 0.0, 0.8, 0.2],
        [0.0, 0.0, 0.7, 0.3],
        [0.1, 0.9, 0.0, 0.0],
        [0.0, 0.0, 0.2, 0.8],
    ]
    raw_text_embeds = [
        [0.82, 0.18, 0.0, 0.0],
        [0.68, 0.32, 0.0, 0.0],
        [0.0, 0.0, 0.85, 0.15],
        [0.0, 0.0, 0.65, 0.35],
        [0.12, 0.88, 0.0, 0.0],
        [0.0, 0.0, 0.18, 0.82],
    ]
    # L2 normalize test embeddings
    image_embeds, _ = normalize_embeddings(raw_image_embeds)
    text_embeds, _ = normalize_embeddings(raw_text_embeds)
    sample_ids = [f"s_{i}" for i in range(n)]
    class_labels = [0, 0, 1, 1, 2, 2]
    class_names = ["class_a", "class_b", "class_c"]

    geom = compute_shared_multimodal_geometry(
        image_embeddings=image_embeds,
        text_embeddings=text_embeds,
        sample_ids=sample_ids,
        class_labels=class_labels,
        class_names=class_names,
    )

    # 1. Joint PCA 2D coordinates for all images and texts
    assert len(geom["image_pca_coordinates"]) == n
    assert len(geom["text_pca_coordinates"]) == n
    assert len(geom["image_pca_coordinates"][0]) == 2
    assert len(geom["text_pca_coordinates"][0]) == 2

    # 2. Paired Distances
    assert len(geom["paired_distances"]) == n
    assert len(geom["paired_cosines"]) == n
    assert geom["mean_paired_cosine"] > 0.9

    # 3. Centroid Alignments
    centroids = geom["centroid_alignments"]
    assert len(centroids) == 3
    for ca in centroids:
        assert ca["cosine_similarity"] > 0.95
        assert ca["euclidean_distance"] >= 0.0


def test_multimodal_collapse_diagnostics() -> None:
    """Verify collapse detection on collapsed vs diverse representation spaces."""
    # Diverse representations
    diverse_v = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    diverse_t = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]

    diag_diverse = compute_multimodal_collapse_diagnostics(diverse_v, diverse_t)
    assert diag_diverse.is_collapsed is False
    assert diag_diverse.visual_feature_std > 0.1
    assert diag_diverse.similarity_gap > 0.5

    # Collapsed representations (all vectors identical)
    collapsed_v = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    collapsed_t = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]

    diag_collapsed = compute_multimodal_collapse_diagnostics(collapsed_v, collapsed_t)
    assert diag_collapsed.is_collapsed is True
