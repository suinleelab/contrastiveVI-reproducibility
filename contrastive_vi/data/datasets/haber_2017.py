"""
Download, read, and preprocess Haber et al. (2017) expression data.

Single-cell expression data from Haber et al. A single-cell survey of the small
intestinal epithelium. Nature (2017).
"""
import gzip
import os

import pandas as pd
from anndata import AnnData

from contrastive_vi.data.utils import download_binary_file, preprocess_workflow


def download_haber_2017(output_path: str) -> None:
    """
    Download Haber et al. 2017 data from the hosting URLs.

    Args:
    ----
        output_path: Output path to store the downloaded and unzipped
        directories.

    Returns
    -------
        None. File directories are downloaded to output_path.
    """

    url = (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92332/suppl/GSE92332"
        "_SalmHelm_UMIcounts.txt.gz"
    )

    output_filename = os.path.join(output_path, url.split("/")[-1])

    download_binary_file(url, output_filename)


def read_haber_2017(file_directory: str) -> pd.DataFrame:
    """
    Read the expression data for Download Haber et al. 2017 the given directory.

    Args:
    ----
        file_directory: Directory containing Haber et al. 2017 data.

    Returns
    -------
        A data frame containing single-cell gene expression count, with cell
        identification barcodes as column names and gene IDs as indices.
    """

    with gzip.open(
        os.path.join(file_directory, "GSE92332_SalmHelm_UMIcounts.txt.gz"), "rb"
    ) as f:
        df = pd.read_csv(f, sep="\t")

    return df


def preprocess_haber_2017(
    download_path: str, n_top_genes: int, normalization_method: str = "tc"
) -> AnnData:
    """
    Preprocess expression data from Haber et al. 2017.

    Args:
    ----
        download_path: Path containing the downloaded Haber et al. 2017 data file.
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

    df = read_haber_2017(download_path)
    df = df.transpose()
    df.columns = [x.upper() for x in df.columns] # Converts gene feature names to uppercase

    cell_groups = []
    barcodes = []
    conditions = []
    cell_types = []

    for cell in df.index:
        cell_group, barcode, condition, cell_type = cell.split("_")
        cell_groups.append(cell_group)
        barcodes.append(barcode)
        conditions.append(condition)
        cell_types.append(cell_type)

    metadata_df = pd.DataFrame(
        {
            "cell_group": cell_groups,
            "barcode": barcodes,
            "condition": conditions,
            "cell_type": cell_types,
        }
    )

    adata = AnnData(df)
    adata.obs = metadata_df
    adata = adata[adata.obs["condition"] != "Hpoly.Day3"]
    adata = preprocess_workflow(
        adata=adata, n_top_genes=n_top_genes, normalization_method=normalization_method
    )
    return adata
