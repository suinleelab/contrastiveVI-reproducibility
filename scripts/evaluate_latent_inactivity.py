"""Evaluate latent dimension inactivity for auto-encoders."""
import argparse
import os
import sys
from functools import partial

import constants
import numpy as np
import pandas as pd
import scanpy as sc
import torch
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
            if normalization_suffix == "":  # For contrastiveVI.
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

            for latent_size in latent_sizes:
                for seed in constants.DEFAULT_SEEDS:
                    full_method = f"{method}{normalization_suffix}"
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

                        background_representations = model.get_latent_representation(
                            adata=adata, representation_kind="background"
                        )
                        salient_representations = model.get_latent_representation(
                            adata=target_adata, representation_kind="salient"
                        )

                        background_metrics = pd.DataFrame(
                            {
                                "variance": np.var(background_representations, axis=0),
                                "mean": np.mean(background_representations, axis=0),
                                "median": np.median(background_representations, axis=0),
                                "representation_type": "background",
                                "latent_dim": [
                                    i for i in range(model.module.n_background_latent)
                                ],
                            }
                        )
                        salient_metrics = pd.DataFrame(
                            {
                                "variance": np.var(salient_representations, axis=0),
                                "mean": np.mean(salient_representations, axis=0),
                                "median": np.median(salient_representations, axis=0),
                                "representation_type": "salient",
                                "latent_dim": [
                                    i for i in range(model.module.n_salient_latent)
                                ],
                            }
                        )
                        metrics = pd.concat([background_metrics, salient_metrics])
                        message = "successful evaluation."
                    else:
                        num_epochs = float("nan")
                        metrics = pd.DataFrame(
                            {
                                "variance": [float("nan")],
                                "representation_type": [float("nan")],
                                "latent_dim": [float("nan")],
                            }
                        )
                        message = "model file does not exist!"

                    metrics["dataset"] = dataset
                    metrics["method"] = full_method
                    metrics["latent_size"] = latent_size
                    metrics["seed"] = seed
                    metrics["num_epochs"] = num_epochs
                    metrics["message"] = message
                    result_df_list.append(metrics)

result_df = pd.concat(result_df_list).reset_index(drop=True)
result_df.to_csv(
    os.path.join(constants.DEFAULT_RESULTS_PATH, "latent_inactivity_summary.csv"),
    index=False,
)
