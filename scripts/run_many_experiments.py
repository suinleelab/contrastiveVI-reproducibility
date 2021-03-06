"""Run multiple model training experiments."""
import argparse
import os

import constants

dataset_list = ["mcfarland_2020", "zheng_2017", "haber_2017"]
latent_size_list = [2, 10, 32, 64]

parser = argparse.ArgumentParser()
parser.add_argument(
    "method",
    type=str,
    choices=[
        "contrastiveVI",
        "TC_contrastiveVI",
        "mmd_contrastiveVI",
        "scVI",
        "PCPCA",
        "cPCA",
        "CPLVM",
        "CGLVM",
        "cVAE",
    ],
    help="Which model to train",
)
parser.add_argument(
    "-use_gpu", action="store_true", help="Flag for enabling GPU usage."
)
parser.add_argument(
    "--gpu_num",
    type=int,
    help="If -use_gpu is enabled, controls which specific GPU to use for training.",
)
args = parser.parse_args()

if args.method in constants.METHODS_WITHOUT_LIB_NORMALIZATION:
    normalization_methods = constants.NORMALIZATION_LIST
else:
    normalization_methods = [None]

for dataset in dataset_list:
    for normalization_method in normalization_methods:
        for latent_size in latent_size_list:
            command = f"python scripts/run_experiment.py {dataset} {args.method}"
            command += f" --latent_size {latent_size}"
            if args.use_gpu:
                command += " -use_gpu"
            if args.gpu_num is not None:
                command += f" --gpu_num {args.gpu_num}"
            if normalization_method is not None:
                command += f" --normalization_method {normalization_method}"
            os.system(command)
