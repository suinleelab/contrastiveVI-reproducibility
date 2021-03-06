"""Evaluate trained models."""
import os

import constants
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from eval_utils import evaluate_latent_representations, nan_metrics
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

datasets = ["zheng_2017"]  # Modify for datasets of interest.
latent_sizes = [2, 10, 32, 64]
dataset_split_lookup = constants.DATASET_SPLIT_LOOKUP

deterministic_methods = []
non_deterministic_methods = [
    "CPLVM",
    "CGLVM",
    "scVI",
    "contrastiveVI",
]
methods = deterministic_methods + non_deterministic_methods

result_df_list = []

for dataset in datasets:
    print(f"Evaluating models with dataset {dataset}...")
    adata = sc.read_h5ad(
        os.path.join(
            constants.DEFAULT_DATA_PATH,
            f"{dataset}/preprocessed/adata_top_2000_genes_tc.h5ad",
        )
    )
    split_key = dataset_split_lookup[dataset]["split_key"]
    background_value = dataset_split_lookup[dataset]["background_value"]
    label_key = dataset_split_lookup[dataset]["label_key"]
    target_labels = adata[adata.obs[split_key] != background_value].obs[label_key]
    target_labels = LabelEncoder().fit_transform(target_labels)

    for method in tqdm(methods):
        if method in deterministic_methods:
            method_seeds = [""]
        else:
            method_seeds = constants.DEFAULT_SEEDS
        if method in constants.METHODS_WITHOUT_LIB_NORMALIZATION:
            normalization_suffixes = [
                f"_{normalization}" for normalization in constants.NORMALIZATION_LIST
            ]
        else:
            normalization_suffixes = [""]

        for latent_size in latent_sizes:
            for method_seed in method_seeds:
                for normalization_suffix in normalization_suffixes:
                    full_method = f"{method}{normalization_suffix}"
                    results_path = constants.DEFAULT_RESULTS_PATH
                    if dataset == "norman_2019":
                        cluster_algorithm = "gmm"
                    else:
                        cluster_algorithm = "kmeans"

                    output_dir = os.path.join(
                        results_path,
                        dataset,
                        full_method,
                        f"latent_{latent_size}",
                        f"{method_seed}",
                    )

                    model_filepath = os.path.join(output_dir, "model.ckpt")
                    if os.path.exists(model_filepath):
                        model = torch.load(model_filepath, map_location="cpu")
                        num_epochs = model.history["reconstruction_loss_train"].shape[0]
                    else:
                        model = None
                        num_epochs = float("nan")

                    representation_filepath = os.path.join(
                        output_dir, "latent_representations.npy"
                    )
                    if os.path.exists(representation_filepath):
                        latent_representations = np.load(representation_filepath)
                        nan_exists = np.isnan(latent_representations).sum() > 0
                        if nan_exists:
                            message = f"{representation_filepath} contains nan!"
                            print(message)
                            metrics = nan_metrics()
                    else:
                        message = f"{representation_filepath} does not exist!"
                        print(message)
                        latent_representations = None
                        nan_exists = False
                        metrics = nan_metrics()
                        message = "representation file does not exist"

                    if latent_representations is not None and not nan_exists:
                        metrics = evaluate_latent_representations(
                            target_labels,
                            latent_representations,
                            clustering_seed=123,
                            cluster_algorithm=cluster_algorithm,
                        )
                        message = "successful evaluation"
                    metrics = pd.DataFrame({key: [val] for key, val in metrics.items()})
                    metrics["dataset"] = dataset
                    metrics["method"] = full_method
                    metrics["latent_size"] = latent_size
                    metrics["seed"] = (
                        "Deterministic"
                        if method in deterministic_methods
                        else method_seed
                    )
                    metrics["num_epochs"] = num_epochs
                    metrics["message"] = message
                    result_df_list.append(metrics)

result_df = pd.concat(result_df_list).reset_index(drop=True)
result_df.to_csv(
    os.path.join(constants.DEFAULT_RESULTS_PATH, "performance_summary.csv"),
    index=False,
)
