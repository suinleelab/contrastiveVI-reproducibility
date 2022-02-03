"""Evaluate reconstruction performance for auto-encoders."""
import argparse
import os
import sys
from functools import partial

import constants
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from eval_utils import evaluate_reconstruction, nan_reconstruction_metrics
from scvi._settings import settings
from tqdm import tqdm

from contrastive_vi.model.contrastive_vi import ContrastiveVIModel
from contrastive_vi.model.cvae import CVAEModel

settings.seed = 0

parser = argparse.ArgumentParser()
parser.add_argument(
    "--use-gpu",
    action="store_true",
    help="Flag for enabling GPU usage.",
    dest="use_gpu",
)
parser.add_argument(
    "--gpu-num",
    type=int,
    help="If --use-gpu is enabled, controls which specific GPU to use for training.",
    dest="gpu_num",
)
args = parser.parse_args()
print(f"Running {sys.argv[0]} with arguments")
for arg in vars(args):
    print(f"\t{arg}={getattr(args, arg)}")

if args.use_gpu:
    if args.gpu_num is not None:
        device = f"cuda:{args.gpu_num}"
    else:
        device = "cuda:0"
else:
    device = "cpu"

datasets = ["mcfarland_2020", "zheng_2017", "haber_2017"]
methods = ["contrastiveVI", "cVAE"]
latent_sizes = [2, 10, 32, 64]
dataset_split_lookup = constants.DATASET_SPLIT_LOOKUP

result_df_list = []
for dataset in datasets:
    print(f"Evaluating model reconstruction performance with dataset {dataset}...")
    split_key = dataset_split_lookup[dataset]["split_key"]
    background_value = dataset_split_lookup[dataset]["background_value"]

    for method in tqdm(methods):
        if method == "cVAE":
            normalization_suffixes = [
                f"_{normalization}" for normalization in constants.NORMALIZATION_LIST
            ]
            adata_setup_function = partial(CVAEModel.setup_anndata, layer=None)
            true_layer = "X"
        else:  # If method is contrastiveVI.
            normalization_suffixes = [""]
            adata_setup_function = partial(
                ContrastiveVIModel.setup_anndata, layer="count"
            )
            true_layer = "count"

        for normalization_suffix in normalization_suffixes:
            if normalization_suffix == "":  # When method is contrastiveVI.
                data_suffix = "_tc"
                if dataset == "zheng_2017":  # Use previous preprocessed data.
                    data_suffix = ""
            else:
                data_suffix = normalization_suffix
            adata = sc.read_h5ad(
                os.path.join(
                    constants.DEFAULT_DATA_PATH,
                    f"{dataset}/preprocessed/adata_top_2000_genes{data_suffix}.h5ad",
                )
            )
            adata.layers["X"] = adata.X
            adata_setup_function(adata)
            setup_attr_key = adata.uns["_scvi"]["data_registry"]["X"]["attr_key"]
            if method == "cVAE":
                assert (
                    setup_attr_key == "None"
                ), "the data registered for cVAE should be adata.X!"
            else:
                assert (
                    setup_attr_key == "count"
                ), "the data registered for contrastiveVI should be the count layer!"

            target_adata = adata[adata.obs[split_key] != background_value].copy()
            background_adata = adata[adata.obs[split_key] == background_value].copy()

            true_target_value = target_adata.layers[true_layer]
            true_background_value = background_adata.layers[true_layer]
            true_value = np.concatenate((true_target_value, true_background_value))

            for latent_size in latent_sizes:
                for seed in constants.DEFAULT_SEEDS:
                    full_method = f"{method}{normalization_suffix}"
                    if dataset == "zheng_2017" and full_method == "contrastiveVI":
                        # Use models trained on previous preprocessed data.
                        results_path = os.path.join(
                            "/projects/leelab/contrastiveVI",
                            "results-fixed-background-size",
                        )
                    else:
                        results_path = constants.DEFAULT_RESULTS_PATH
                    output_dir = os.path.join(
                        results_path,
                        dataset,
                        full_method,
                        f"latent_{latent_size}",
                        f"{seed}",
                    )
                    model_filepath = os.path.join(output_dir, "model.ckpt")
                    if os.path.exists(model_filepath):
                        model = torch.load(model_filepath, map_location=device)
                        num_epochs = model.history["reconstruction_loss_train"].shape[0]

                        target_sample_mean = model.get_sample_mean(
                            data_source="target", adata=target_adata, n_samples=100
                        )
                        background_sample_mean = model.get_sample_mean(
                            data_source="background",
                            adata=background_adata,
                            n_samples=100,
                        )
                        metrics = evaluate_reconstruction(
                            true_value,
                            np.concatenate(
                                (target_sample_mean, background_sample_mean)
                            ),
                        )
                        message = "successful evaluation."
                    else:
                        num_epochs = float("nan")
                        metrics = nan_reconstruction_metrics()
                        message = "model file does not exist!"

                    metrics = pd.DataFrame({key: [val] for key, val in metrics.items()})
                    metrics["dataset"] = dataset
                    metrics["method"] = full_method
                    metrics["latent_size"] = latent_size
                    metrics["seed"] = seed
                    metrics["num_epochs"] = num_epochs
                    metrics["message"] = message
                    result_df_list.append(metrics)

result_df = pd.concat(result_df_list).reset_index(drop=True)
result_df.to_csv(
    os.path.join(
        constants.DEFAULT_RESULTS_PATH, "reconstruction_performance_summary.csv"
    ),
    index=False,
)
