"""
Download, read, and preprocess Cain et al. (2020) expression data.

Single-cell expression data from Cain et al. Multi-cellular communities
are perturbed in the aging human brain and with Alzheimer’s disease.
bioRxiv (2021).
"""
import gzip
import os

import numpy as np
import pandas as pd
from anndata import AnnData

from contrastive_vi.data.utils import preprocess_workflow


def download_cain_2020(output_path: str) -> None:
    """
    Because access to this dataset is restricted, we can't download the data files
    programatically. Instead, this function redirects the user to the webpage where the
    file can be downloaded.

    Args:
    ----
        output_path: Path where raw data file should live.

    Returns
    -------
        None. This function redirects the user to the Synapse AD knowledge
        portal to download the data file if it doesn't already exist.
    """
    if not os.path.exists(
        os.path.join(
            output_path, "ROSMAP_Brain.snRNAseq_counts_sparse_format_20201107.csv.gz"
        )
    ):
        raise FileNotFoundError(
            "Files cannot be downloaded automatically. Please download"
            "files from https://adknowledgeportal.synapse.org/"
            f"and place them in {output_path} to continue."
        )


def read_cain_2020(file_directory: str) -> pd.DataFrame:
    """
    Read the expression data for Cain et al. 2020 in the given directory.

    Args:
    ----
        file_directory: Directory containing Cain et al. 2020 data.

    Returns
    -------
        A data frame containing single-cell gene expression counts. The count
        matrix is stored in triplet format. I.e., each row of the data frame
        has the format (row, column, count) stored in columns (i, j, x) respectively.
    """

    with gzip.open(
        os.path.join(
            file_directory, "ROSMAP_Brain.snRNAseq_counts_sparse_format_20201107.csv.gz"
        ),
        "rb",
    ) as f:
        df = pd.read_csv(f)

    return df


def preprocess_cain_2020(
    download_path: str, n_top_genes: int, normalization_method: str = "tc"
) -> AnnData:
    """
    Preprocess expression data from Cain et al., 2020.

    Args:
    ----
        download_path: Path containing the downloaded Cain et al. 2020 data file.
        n_top_genes: Number of most variable genes to retain.
        normalization_method: Normalization method. Available options are "tc" (total
        count), "tmm" (trimmed-mean-of-M-values), "scran" (scran deconvolution), and
        "basics" (BASiCS).

    Returns
    -------
        An AnnData object containing single-cell expression data. The layer
        "count" contains the count data for the most variable genes. The .X
        variable contains the normalized and log-transformed data for the most variable
        genes. A copy of data with all genes is stored in .raw.
    """

    df = read_cain_2020(download_path)

    # The Cain count data is in a sparse triplet format represented
    # by three columns 'i', 'j', and 'x'. 'i' refers to a row number, 'j' refers to
    # a column number, and 'x' refers to a count value.
    counts = df["x"]
    rows = (
        df["i"] - 1
    )  # Indices were originally 1-base-indexed --> switch to 0-base-indexing
    cols = df["j"] - 1

    # Convert the triplets into a numpy array
    count_matrix = np.zeros([max(rows) + 1, max(cols) + 1])
    count_matrix[rows, cols] = counts

    # Switch matrix from gene rows and cell columns to cell rows and gene columns
    count_matrix = count_matrix.T

    cell_metadata = pd.read_csv(
        os.path.join(
            download_path, "ROSMAP_Brain.snRNAseq_metadata_cells_20201107.csv"
        ),
        index_col=0,
    )

    gene_metadata = pd.read_csv(
        os.path.join(
            download_path,
            "ROSMAP_Brain.snRNAseq_metadata_genes_20201107.csv",
        ),
        index_col=1,
    )

    adata = AnnData(X=count_matrix, obs=cell_metadata)
    adata.var.index = gene_metadata.index

    # Filtering as described in the methods section of Cain et al. 2020
    adata = adata[adata.obs["broad_class"] != "None"]

    adata = preprocess_workflow(
        adata=adata, n_top_genes=n_top_genes, normalization_method=normalization_method
    )
    return adata
