"""Helper functions for evaluating model performance"""
from typing import Dict, Optional

import numpy as np
from scvi.model._metrics import unsupervised_clustering_accuracy
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture


def evaluate_latent_representations(
    labels: np.ndarray,
    latent_representations: np.ndarray,
    clustering_seed: Optional[int] = None,
    cluster_algorithm="kmeans",
) -> Dict[str, float]:
    """Evaluate latent representations against ground truth labels"""
    if cluster_algorithm == "kmeans":
        latent_clusters = (
            KMeans(n_clusters=len(np.unique(labels)), random_state=clustering_seed)
            .fit(latent_representations)
            .labels_
        )
    elif cluster_algorithm == "gmm":
        latent_clusters = GaussianMixture(
            n_components=len(np.unique(labels)), random_state=clustering_seed
        ).fit_predict(latent_representations)
    else:
        raise ValueError("Clustering algorithm must be one of kmeans or gmm")

    silhouette = silhouette_score(latent_representations, labels)

    adjusted_rand_index = adjusted_rand_score(labels, latent_clusters)
    normalized_mutual_info = normalized_mutual_info_score(labels, latent_clusters)
    unsupervised_cluster_accuracy = unsupervised_clustering_accuracy(
        labels, latent_clusters
    )[0]

    return {
        "silhouette": silhouette,
        "adjusted_rand_index": adjusted_rand_index,
        "normalized_mutual_info": normalized_mutual_info,
        "unsupervised_cluster_accuracy": unsupervised_cluster_accuracy,
    }


def nan_metrics() -> Dict[str, float]:
    """Return nan for all latent representation evaluation metrics."""
    return {
        "silhouette": float("nan"),
        "adjusted_rand_index": float("nan"),
        "normalized_mutual_info": float("nan"),
        "unsupervised_cluster_accuracy": float("nan"),
    }
