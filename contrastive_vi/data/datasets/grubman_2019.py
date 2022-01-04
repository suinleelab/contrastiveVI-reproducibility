"""
Download, read, and preprocess Grubman et al. (2019) expression data.

Single-cell expression data from Grubman et al. A single-cell atlas of entorhinal cortex
from individuals with Alzheimer’s disease reveals cell-type-specific gene expression
regulation. Nature Neuroscience (2019).
"""
import gzip
import os

import pandas as pd
import scanpy as sc
from anndata import AnnData

from contrastive_vi.data.utils import download_binary_file


def download_grubman_2019(output_path: str) -> None:
    """
    Download Grubman et al. 2019 data from the hosting URLs.

    Args:
    ----
        output_path: Output path to store the downloaded and unzipped
        directories.

    Returns
    -------
        None. File directories are downloaded to output_path.
    """

    data_url = (
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE138852&format=file"
        "&file=GSE138852_counts.csv.gz "
    )
    data_output_filename = os.path.join(output_path, data_url.split("=")[-1])
    download_binary_file(data_url, data_output_filename)


def read_grubman_2019(file_directory: str) -> pd.DataFrame:
    """
    Read the expression data for Grubman et al. 2019 in the given directory.

    Args:
    ----
        file_directory: Directory containing Grubman et al. 2019 data.

    Returns
    -------
        A data frame containing single-cell gene expression count, with cell
        identification barcodes as column names and gene IDs as indices.
    """

    with gzip.open(os.path.join(file_directory, "GSE138852_counts.csv.gz"), "rb") as f:
        df = pd.read_csv(f, index_col=0)

    return df


def preprocess_grubman_2019(download_path: str, n_top_genes: int) -> AnnData:
    """
    Preprocess expression data from Grubman et al. 2019.

    Args:
    ----
        download_path: Path containing the downloaded Grubman et al. 2019 data file.
        n_top_genes: Number of most variable genes to retain.

    Returns
    -------
        An AnnData object containing single-cell expression data. The layer
        "count" contains the count data for the most variable genes. The X
        variable contains the total-count-normalized and log-transformed data
        for the most variable genes (a copy with all the genes is stored in
        .raw).
    """

    df = read_grubman_2019(download_path)
    df = df.transpose()

    metadata_df = pd.read_csv(
        os.path.join(download_path, "scRNA_metadata.tsv"), sep="\t", index_col=0
    )
    adata = AnnData(df)
    adata.obs = metadata_df

    # To avoid weirdness in downstream analyses, I chose to exclude cells for which the
    # cell type could not be identified by the authors
    adata = adata[adata.obs["cellType"] != "doublet"]
    adata = adata[adata.obs["cellType"] != "unID"]

    adata.layers["count"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(
        adata, flavor="seurat_v3", n_top_genes=n_top_genes, layer="count", subset=True
    )
    adata = adata[adata.layers["count"].sum(1) != 0]  # Remove cells with all zeros.
    return adata
