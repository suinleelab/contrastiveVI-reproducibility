"""Run a model training experiment."""
import argparse
import os
import pickle
import sys

import constants
import numpy as np
import scanpy as sc
import tensorflow as tf
import torch
from contrastive import CPCA
from cplvm import CGLVM, CPLVM, CGLVMMFGaussianApprox, CPLVMLogNormalApprox
from pcpca import PCPCA
from scvi._settings import settings
from scvi.model import SCVI, TOTALVI
from sklearn.preprocessing import StandardScaler

from contrastive_vi.baselines.mixscape import run_mixscape
from contrastive_vi.model.contrastive_vi import ContrastiveVIModel
from contrastive_vi.model.cvae import CVAEModel
from contrastive_vi.model.total_contrastive_vi import TotalContrastiveVIModel

settings.num_threads = 1
settings.dl_num_workers = 1

parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset",
    type=str,
    choices=constants.DATASET_LIST,
    help="Which dataset to use for the experiment.",
)
parser.add_argument(
    "method",
    type=str,
    choices=[
        "contrastiveVI",
        "TC_contrastiveVI",
        "mmd_contrastiveVI",
        "scVI",
        "scVI_with_background",
        "PCPCA",
        "cPCA",
        "CPLVM",
        "CGLVM",
        "cVAE",
        "totalVI",
        "total_contrastiveVI",
        "mixscape",
    ],
    help="Which model to train",
)
parser.add_argument(
    "-use_gpu", action="store_true", help="Flag for enabling GPU usage."
)
parser.add_argument(
    "--latent_size",
    type=int,
    default=10,
    help=(
        "Size of the model's latent space. For contrastive models, this is the size "
        "of the salient latent space. For non-contrastive models, this is the size "
        "of the single latent space."
    ),
)
parser.add_argument(
    "--n_genes",
    type=int,
    default=2000,
    help="Number of highly variable genes in dataset.",
)
parser.add_argument(
    "--normalization_method",
    type=str,
    choices=constants.NORMALIZATION_LIST,
    default="tc",
    dest="normalization_method",
    help=(
        "Normalization method used for scaling cell-specific library sizes. "
        f"Only applicable to {constants.METHODS_WITHOUT_LIB_NORMALIZATION}"
    ),
)
parser.add_argument(
    "--gpu_num",
    type=int,
    help="If -use_gpu is enabled, controls which specific GPU to use for training.",
)
parser.add_argument(
    "--random_seeds",
    nargs="+",
    type=int,
    default=constants.DEFAULT_SEEDS,
    help="List of random seeds to use for experiments, with one model trained per "
    "seed.",
)

args = parser.parse_args()
print(f"Running {sys.argv[0]} with arguments")
for arg in vars(args):
    print(f"\t{arg}={getattr(args, arg)}")

# totalVI and variants can only be used with the papalexi_2021 joint RNA + protein
# dataset. We use mixscape as a baseline for papalexi_2021 (as it was
# developed by the Papalexi authors), but we disallow other RNA-only methods from
# being applied to this dataset.
rna_protein_methods = ["totalVI", "total_contrastiveVI"]
if args.method in rna_protein_methods:
    assert (
        args.dataset == "papalexi_2021"
    ), f"{args.method} can only be applied to papalexi_2021!"
elif args.method != "mixscape":
    assert (
        args.dataset != "papalexi_2021"
    ), "RNA-only models cannot be applied to papalexi_2021!"

if args.method in constants.METHODS_WITHOUT_LIB_NORMALIZATION:
    preprocessed_file_suffix = f"_{args.normalization_method}"
    output_suffix = preprocessed_file_suffix
else:
    preprocessed_file_suffix = "_tc"
    # The actual file choice doesn't matter since adata count layers are all the same.

    output_suffix = ""

adata_file = os.path.join(
    constants.DEFAULT_DATA_PATH,
    args.dataset,
    "preprocessed",
    f"adata_top_{args.n_genes}_genes{preprocessed_file_suffix}.h5ad",
)
adata = sc.read_h5ad(adata_file)
print(f"Data read from {adata_file}")

dataset_split_lookup = constants.DATASET_SPLIT_LOOKUP
if args.dataset in dataset_split_lookup.keys():
    split_key = dataset_split_lookup[args.dataset]["split_key"]
    background_value = dataset_split_lookup[args.dataset]["background_value"]
else:
    raise NotImplementedError("Dataset not yet implemented.")

torch_models = [
    "scVI",
    "scVI_with_background",
    "cVAE",
    "contrastiveVI",
    "TC_contrastiveVI",
    "mmd_contrastiveVI",
    "totalVI",
    "total_contrastiveVI",
]
tf_models = ["CPLVM", "CGLVM"]
normalized_expressions = None

# For deep learning methods, we experiment with multiple random initializations
# to get error bars
if args.method in torch_models:
    if args.use_gpu:
        if args.gpu_num is not None:
            use_gpu = args.gpu_num
        else:
            use_gpu = True
    else:
        use_gpu = False

    for seed in args.random_seeds:
        settings.seed = seed

        if args.method == "contrastiveVI":
            ContrastiveVIModel.setup_anndata(adata, layer="count")
            model = ContrastiveVIModel(
                adata,
                disentangle=False,
                use_mmd=False,
                n_salient_latent=args.latent_size,
                n_background_latent=10,
                use_observed_lib_size=False,
            )

            # np.where returns a list of indices, one for each dimension of the input
            # array. Since we have 1d arrays, we simply grab the first (and only)
            # returned list.
            background_indices = np.where(adata.obs[split_key] == background_value)[0]
            target_indices = np.where(adata.obs[split_key] != background_value)[0]
            model.train(
                check_val_every_n_epoch=1,
                train_size=0.8,
                background_indices=background_indices,
                target_indices=target_indices,
                use_gpu=use_gpu,
                early_stopping=True,
                max_epochs=500,
            )
            target_adata = adata[adata.obs[split_key] != background_value].copy()
            latent_representations = model.get_latent_representation(
                adata=target_adata, representation_kind="salient"
            )
            shared_latent_representations = model.get_latent_representation(
                adata=target_adata,
                representation_kind="background",
            )
            normalized_expressions = model.get_normalized_expression(
                adata=adata, n_samples=100
            )

        elif args.method == "TC_contrastiveVI":
            ContrastiveVIModel.setup_anndata(adata, layer="count")
            model = ContrastiveVIModel(
                adata,
                disentangle=True,
                use_mmd=False,
                n_salient_latent=args.latent_size,
                n_background_latent=10,
                use_observed_lib_size=False,
            )

            # np.where returns a list of indices, one for each dimension of the input
            # array. Since we have 1d arrays, we simply grab the first (and only)
            # returned list.
            background_indices = np.where(adata.obs[split_key] == background_value)[0]
            target_indices = np.where(adata.obs[split_key] != background_value)[0]
            model.train(
                check_val_every_n_epoch=1,
                train_size=0.8,
                background_indices=background_indices,
                target_indices=target_indices,
                use_gpu=use_gpu,
                early_stopping=True,
                max_epochs=500,
            )
            target_adata = adata[adata.obs[split_key] != background_value].copy()
            latent_representations = model.get_latent_representation(
                adata=target_adata, representation_kind="salient"
            )
            shared_latent_representations = model.get_latent_representation(
                adata=target_adata,
                representation_kind="background",
            )
            normalized_expressions = model.get_normalized_expression(
                adata=adata, n_samples=100
            )

        elif args.method == "mmd_contrastiveVI":
            ContrastiveVIModel.setup_anndata(adata, layer="count")
            gammas = np.array([10 ** x for x in range(-6, 7, 1)])
            model = ContrastiveVIModel(
                adata,
                disentangle=False,
                use_mmd=True,
                gammas=gammas,
                mmd_weight=10000,
                n_salient_latent=args.latent_size,
                n_background_latent=10,
                use_observed_lib_size=False,
            )

            # np.where returns a list of indices, one for each dimension of the input
            # array. Since we have 1d arrays, we simply grab the first (and only)
            # returned list.
            background_indices = np.where(adata.obs[split_key] == background_value)[0]
            target_indices = np.where(adata.obs[split_key] != background_value)[0]
            model.train(
                check_val_every_n_epoch=1,
                train_size=0.8,
                background_indices=background_indices,
                target_indices=target_indices,
                use_gpu=use_gpu,
                early_stopping=True,
                max_epochs=500,
            )
            target_adata = adata[adata.obs[split_key] != background_value].copy()
            latent_representations = model.get_latent_representation(
                adata=target_adata, representation_kind="salient"
            )
            shared_latent_representations = model.get_latent_representation(
                adata=target_adata,
                representation_kind="background",
            )
            normalized_expressions = model.get_normalized_expression(
                adata=adata, n_samples=100
            )

        elif args.method == "scVI":
            # We only train scVI with target samples
            target_adata = adata[adata.obs[split_key] != background_value].copy()
            SCVI.setup_anndata(target_adata, layer="count")
            model = SCVI(
                target_adata, n_latent=args.latent_size, use_observed_lib_size=False
            )
            model.train(
                check_val_every_n_epoch=1,
                train_size=0.8,
                use_gpu=use_gpu,
                early_stopping=True,
                max_epochs=500,
            )
            latent_representations = model.get_latent_representation(adata=target_adata)
            shared_latent_representations = None

        elif args.method == "scVI_with_background":
            # Train scVI with both target and background samples to test target vs.
            # background differential expression tests
            SCVI.setup_anndata(adata, layer="count")
            model = SCVI(adata, n_latent=args.latent_size, use_observed_lib_size=False)
            model.train(
                check_val_every_n_epoch=1,
                train_size=0.8,
                use_gpu=use_gpu,
                early_stopping=True,
                max_epochs=500,
            )
            target_adata = adata[adata.obs[split_key] != background_value].copy()
            latent_representations = model.get_latent_representation(adata=target_adata)
            shared_latent_representations = None

        elif args.method == "total_contrastiveVI":
            TotalContrastiveVIModel.setup_anndata(
                adata,
                protein_expression_obsm_key=constants.PROTEIN_EXPRESSION_KEY,
                layer="count",
            )

            model = TotalContrastiveVIModel(
                adata,
                n_hidden=128,
                n_background_latent=10,
                n_salient_latent=args.latent_size,
                n_layers=1,
                dropout_rate=0.1,
                protein_batch_mask=None,
                use_observed_lib_size=False,
            )
            background_indices = np.where(adata.obs[split_key] == background_value)[0]
            target_indices = np.where(adata.obs[split_key] != background_value)[0]
            model.train(
                check_val_every_n_epoch=1,
                train_size=0.8,
                background_indices=background_indices,
                target_indices=target_indices,
                use_gpu=use_gpu,
                early_stopping=True,
                max_epochs=500,
            )
            target_adata = adata[adata.obs[split_key] != background_value].copy()
            latent_representations = model.get_latent_representation(
                adata=target_adata,
                representation_kind="salient",
            )
            shared_latent_representations = model.get_latent_representation(
                adata=target_adata,
                representation_kind="background",
            )

        elif args.method == "totalVI":
            # We only train totalVI with target samples
            target_adata = adata[adata.obs[split_key] != background_value].copy()
            TOTALVI.setup_anndata(
                target_adata,
                protein_expression_obsm_key=constants.PROTEIN_EXPRESSION_KEY,
                layer="count",
            )
            model = TOTALVI(
                target_adata,
                n_hidden=128,
                n_layers_encoder=1,
                n_layers_decoder=1,
                dropout_rate_decoder=0.1,
                dropout_rate_encoder=0.1,
                n_latent=args.latent_size,
                use_observed_lib_size=False,
            )
            model.train(
                check_val_every_n_epoch=1,
                train_size=0.8,
                use_gpu=use_gpu,
                early_stopping=True,
                max_epochs=500,
            )
            latent_representations = model.get_latent_representation(adata=target_adata)
            shared_latent_representations = None

        elif args.method == "cVAE":
            CVAEModel.setup_anndata(adata)
            model = CVAEModel(
                adata,
                n_salient_latent=args.latent_size,
                n_background_latent=10,
            )

            # np.where returns a list of indices, one for each dimension of the input
            # array. Since we have 1d arrays, we simply grab the first (and only)
            # returned list.
            background_indices = np.where(adata.obs[split_key] == background_value)[0]
            target_indices = np.where(adata.obs[split_key] != background_value)[0]
            model.train(
                check_val_every_n_epoch=1,
                train_size=0.8,
                background_indices=background_indices,
                target_indices=target_indices,
                use_gpu=use_gpu,
                early_stopping=True,
                max_epochs=500,
            )
            target_adata = adata[adata.obs[split_key] != background_value].copy()
            latent_representations = model.get_latent_representation(
                adata=target_adata, representation_kind="salient"
            )
            shared_latent_representations = model.get_latent_representation(
                adata=target_adata,
                representation_kind="background",
            )

        results_dir = os.path.join(
            constants.DEFAULT_RESULTS_PATH,
            args.dataset,
            f"{args.method}{output_suffix}",
            f"latent_{args.latent_size}",
            str(seed),
        )
        os.makedirs(results_dir, exist_ok=True)
        torch.save(
            model, os.path.join(results_dir, "model.ckpt"), pickle_protocol=4
        )  # Protocol version >= 4 is required to save large model files.
        np.save(
            arr=latent_representations,
            file=os.path.join(results_dir, "latent_representations.npy"),
        )
        if shared_latent_representations is not None:
            np.save(
                arr=shared_latent_representations,
                file=os.path.join(results_dir, "shared_latent_representations.npy"),
            )
        if normalized_expressions is not None:
            background_normalized_expression = normalized_expressions["background"]
            salient_normalized_expression = normalized_expressions["salient"]
            np.save(
                arr=background_normalized_expression,
                file=os.path.join(results_dir, "background_normalized_expression.npy"),
            )
            np.save(
                arr=salient_normalized_expression,
                file=os.path.join(results_dir, "salient_normalized_expression.npy"),
            )

elif args.method in tf_models:
    if args.use_gpu:
        if args.gpu_num is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Use CPU.

    for seed in args.random_seeds:
        tf.random.set_seed(seed)

        if args.method == "CPLVM" or args.method == "CGLVM":
            background_data = (
                adata[adata.obs[split_key] == background_value]
                .layers["count"]
                .transpose()
            )
            target_data = (
                adata[adata.obs[split_key] != background_value]
                .layers["count"]
                .transpose()
            )

            if args.method == "CPLVM":
                lvm = CPLVM(
                    k_shared=10,
                    k_foreground=args.latent_size,
                    compute_size_factors=True,
                    offset_term=False,
                )

                # Set up approximate model
                approx_model = CPLVMLogNormalApprox(
                    background_data,
                    target_data,
                    k_shared=10,
                    k_foreground=args.latent_size,
                    compute_size_factors=True,
                    offset_term=False,
                )
            elif args.method == "CGLVM":
                lvm = CGLVM(k_shared=10, k_foreground=args.latent_size)

                # Set up approximate model
                approx_model = CGLVMMFGaussianApprox(
                    background_data,
                    target_data,
                    k_shared=10,
                    k_foreground=args.latent_size,
                    compute_size_factors=True,
                )

            # Fit model
            model_output = lvm.fit_model_vi(
                background_data,
                target_data,
                approximate_model=approx_model,
            )

            # The CPLVM package author used different keys for the two models
            # (this was a mistake I assume)
            approx_model_key = (
                "approximate_model" if args.method == "CPLVM" else "approx_model"
            )
            model_output = {
                "qty_mean": model_output[approx_model_key]
                .qty_mean.numpy()
                .transpose(),  # Salient
                "qzy_mean": model_output[approx_model_key]
                .qzy_mean.numpy()
                .transpose(),  # Background
            }
            latent_representations = model_output["qty_mean"]
            shared_latent_representations = model_output["qzy_mean"]

            results_dir = os.path.join(
                constants.DEFAULT_RESULTS_PATH,
                args.dataset,
                f"{args.method}{output_suffix}",
                f"latent_{args.latent_size}",
                str(seed),
            )

            os.makedirs(results_dir, exist_ok=True)
            pickle.dump(
                model_output, open(os.path.join(results_dir, "model.pkl"), "wb")
            )
            np.save(
                arr=latent_representations,
                file=os.path.join(results_dir, "latent_representations.npy"),
            )
            np.save(
                arr=shared_latent_representations,
                file=os.path.join(results_dir, "shared_latent_representations.npy"),
            )

elif args.method == "PCPCA":
    # In the original PCPCA paper they standardize data to 0-mean and unit variance, so
    # we do the same thing here.
    background_data = StandardScaler().fit_transform(
        adata[adata.obs[split_key] == background_value].X
    )
    target_data = StandardScaler().fit_transform(
        adata[adata.obs[split_key] != background_value].X
    )

    model = PCPCA(n_components=args.latent_size, gamma=0.7)
    # The PCPCA package expects data to have rows be features and columns be samples
    # so we transpose the data here.
    model.fit(target_data.transpose(), background_data.transpose())

    # model.transform() returns a tuple of transformed target and background data (in
    # this order).
    latent_representations = model.transform(
        target_data.transpose(), background_data.transpose()
    )[0].transpose()

    results_dir = os.path.join(
        constants.DEFAULT_RESULTS_PATH,
        args.dataset,
        f"{args.method}{output_suffix}",
        f"latent_{args.latent_size}",
    )

    os.makedirs(results_dir, exist_ok=True)
    pickle.dump(model, open(os.path.join(results_dir, "model.pkl"), "wb"))
    np.save(
        arr=latent_representations,
        file=os.path.join(results_dir, "latent_representations.npy"),
    )

elif args.method == "cPCA":
    background_data = adata[adata.obs[split_key] == background_value].X
    target_data = adata[adata.obs[split_key] != background_value].X

    model = CPCA(n_components=args.latent_size, standardize=True)
    model.fit(
        foreground=target_data,
        background=background_data,
        preprocess_with_pca_dim=args.n_genes,  # Avoid preprocessing with standard PCA.
    )

    # model.transform() returns a list of transformed data for varying alpha values.
    latent_representations = model.transform(target_data, n_alphas_to_return=1)[0]

    results_dir = os.path.join(
        constants.DEFAULT_RESULTS_PATH,
        args.dataset,
        f"{args.method}{output_suffix}",
        f"latent_{args.latent_size}",
    )
    os.makedirs(results_dir, exist_ok=True)
    pickle.dump(model, open(os.path.join(results_dir, "model.pkl"), "wb"))
    np.save(
        arr=latent_representations,
        file=os.path.join(results_dir, "latent_representations.npy"),
    )

elif args.method == "mixscape":

    # Since mixscape relies on a (non-deterministic) approximate nearest neighbors
    # algorithm, we run it for multiple seeds
    for seed in args.random_seeds:
        settings.seed = seed

        # Using the `split_by` argument results in mixscape being run
        # separately for each replicate with the results aggregated at the end.
        # Here I used the argument for Papalexi to match the authors' workflow
        # in https://satijalab.org/seurat/articles/mixscape_vignette.html (see
        # the call to `CalcPerturbSig`).
        run_mixscape(
            adata=adata,
            pert_key=split_key,
            control=background_value,
            n_pcs=15,  # Default value in Seurat implementation
            split_by="replicate" if args.dataset == "papalexi_2021" else None,
        )

        adata_pert = adata.copy()
        adata_pert.X = adata_pert.layers["X_pert"]
        adata_pert = adata_pert[adata_pert.obs[split_key] != background_value]

        # Since the mixscape authors recommend running PCA on the mixscape values,
        # we do so here and treat the result as a latent embedding
        sc.pp.pca(adata_pert, n_comps=args.latent_size)
        latent_representations = adata_pert.obsm["X_pca"]

        results_dir = os.path.join(
            constants.DEFAULT_RESULTS_PATH,
            args.dataset,
            f"{args.method}{output_suffix}",
            f"latent_{args.latent_size}",
            str(seed),
        )

        os.makedirs(results_dir, exist_ok=True)
        np.save(
            arr=latent_representations,
            file=os.path.join(results_dir, "latent_representations.npy"),
        )

print("Done!")
