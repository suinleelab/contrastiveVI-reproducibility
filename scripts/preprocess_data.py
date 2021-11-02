"""Download, preprocess, and save data as AnnData."""
import argparse
import os
import sys

import constants

from contrastive_vi.data.datasets.blish_2020 import (
    download_blish_2020,
    preprocess_blish_2020,
)
from contrastive_vi.data.datasets.fasolino_2021 import (
    download_fasolino_2021,
    preprocess_fasolino_2021,
)
from contrastive_vi.data.datasets.haber_2017 import (
    download_haber_2017,
    preprocess_haber_2017,
)
from contrastive_vi.data.datasets.xiang_2020 import (
    download_xiang_2020,
    preprocess_xiang_2020,
)
from contrastive_vi.data.datasets.zheng_2017 import (
    download_zheng_2017,
    preprocess_zheng_2017,
)
from contrastive_vi.data.utils import save_preprocessed_adata


def download_and_preprocess_zheng_2017(output_path: str, n_top_genes: int) -> None:
    """Download, preprocess, and save data from Zheng et al. 2017.

    Args:
        output_path: Path to save output files.
        n_top_genes: Number of most variable genes to retain.

    Returns:
        None. Raw Data are saved in output_path. Preprocessed data are saved
        in a sub-directory called "preprocessed" in output_path.
    """
    download_zheng_2017(output_path)
    adata = preprocess_zheng_2017(output_path, n_top_genes)
    save_preprocessed_adata(adata, output_path)


def download_and_preprocess_haber_2017(output_path: str, n_top_genes: int) -> None:
    """Download, preprocess, and save data from Haber et al. 2017.

    Args:
        output_path: Path to save output files.
        n_top_genes: Number of most variable genes to retain.

    Returns:
        None. Raw Data are saved in output_path. Preprocessed data are saved
        in a sub-directory called "preprocessed" in output_path.
    """
    download_haber_2017(output_path)
    adata = preprocess_haber_2017(output_path, n_top_genes)
    save_preprocessed_adata(adata, output_path)


def download_and_preprocess_blish_2020(output_path: str, n_top_genes: int) -> None:
    """Download, preprocess, and save data from Blish et al. 2020.

    Args:
        output_path: Path to save output files.
        n_top_genes: Number of most variable genes to retain.

    Returns:
        None. Raw Data are saved in output_path. Preprocessed data are saved
        in a sub-directory called "preprocessed" in output_path.
    """
    download_blish_2020(output_path)
    adata = preprocess_blish_2020(output_path, n_top_genes)
    save_preprocessed_adata(adata, output_path)


def download_and_preprocess_xiang_2020(output_path: str, n_top_genes: int) -> None:
    """Download, preprocess, and save data from Xiang et al. 2020.

    Args:
        output_path: Path to save output files.
        n_top_genes: Number of most variable genes to retain.

    Returns:
        None. Raw Data are saved in output_path. Preprocessed data are saved
        in a sub-directory called "preprocessed" in output_path.
    """
    download_xiang_2020(output_path)
    adata = preprocess_xiang_2020(output_path, n_top_genes)
    save_preprocessed_adata(adata, output_path)


def download_and_preprocess_fasolino_2021(output_path: str, n_top_genes: int) -> None:
    """Download, preprocess, and save data from Xiang et al. 2020.

    Args:
        output_path: Path to save output files.
        n_top_genes: Number of most variable genes to retain.

    Returns:
        None. Raw Data are saved in output_path. Preprocessed data are saved
        in a sub-directory called "preprocessed" in output_path.
    """
    download_fasolino_2021(output_path)
    adata = preprocess_fasolino_2021(output_path, n_top_genes)
    save_preprocessed_adata(adata, output_path)


def main():
    """Run main function."""
    preprocess_function_dict = {
        "zheng_2017": download_and_preprocess_zheng_2017,
        "haber_2017": download_and_preprocess_haber_2017,
        "blish_2020": download_and_preprocess_blish_2020,
        "xiang_2020": download_and_preprocess_xiang_2020,
        "fasolino_2021": download_and_preprocess_fasolino_2021,
    }
    parser = argparse.ArgumentParser(description="Preprocess data.")
    parser.add_argument(
        "dataset",
        type=str,
        choices=constants.DATASET_LIST,
        help="Preprocess single-cell expression data from Zheng et al. 2017, Haber "
        "et al. 2017, Blish et al., 2020, Xiang et al., 2020, or Fasolino et al., 2021",
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=2000,
        dest="n_top_genes",
        help="Number of top variable genes to retain during preprocessing",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/projects/leelab/data/single-cell",
        dest="output_path",
        help="Path for storing a directory for the preprocessed dataset",
    )
    args = parser.parse_args()
    print(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        print(f"\t{arg}={getattr(args, arg)}")
    preprocess_data_function = preprocess_function_dict[args.dataset]
    output_path = os.path.join(args.output_path, args.dataset)
    os.makedirs(output_path, exist_ok=True)
    preprocess_data_function(output_path=output_path, n_top_genes=args.n_top_genes)
    print("Done!")


if __name__ == "__main__":
    main()