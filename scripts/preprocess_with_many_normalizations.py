"""Preprocess dataset with different normalization methods."""

import argparse
import os

import constants

parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset",
    type=str,
    choices=constants.DATASET_LIST,
    help="Which dataset to preprocess",
)
args = parser.parse_args()
for normalization_method in constants.NORMALIZATION_LIST:
    command = f"python scripts/preprocess_data.py {args.dataset}"
    command += f" --normalization-method {normalization_method}"
    os.system(command)
