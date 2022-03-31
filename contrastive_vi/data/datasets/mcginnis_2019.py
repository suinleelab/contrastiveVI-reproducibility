"""
Download, read, and preprocess McGinnis et al. (2019) expression data.

Single-cell expression data from McGinnis et al. MULTI-seq: sample multiplexing
for single-cell RNA sequencing using lipid-tagged indices. Nature Methods (2019).
"""
import gzip
import os
import shutil

import pandas as pd
from anndata import AnnData
from scipy.io import mmread
from scipy.sparse import coo_matrix

from contrastive_vi.data.utils import download_binary_file, preprocess_workflow


def download_mcginnis_2019(output_path: str) -> None:
    """
    Download McGinnis et al. 2019 data from the hosting URLs.

    Args:
    ----
        output_path: Output path to store the downloaded and unzipped
        directories.

    Returns
    -------
        None. File directories are downloaded to output_path.
    """

    counts_matrix_url = (
        "https://www.ncbi.nlm.nih.gov/geo/download/"
        "?acc=GSE129578&format=file&file=GSE129578_PDX_mm_matrix.mtx.gz"
    )
    data_output_filename = os.path.join(output_path, counts_matrix_url.split("=")[-1])
    download_binary_file(counts_matrix_url, data_output_filename)

    metadata_url = (
        "https://static-content.springer.com/esm/"
        "art%3A10.1038%2Fs41592-019-0433-8/MediaObjects/41592_2019_433_MOESM5_ESM.xlsx"
    )
    metadata_filename = os.path.join(output_path, metadata_url.split("/")[-1])
    download_binary_file(metadata_url, metadata_filename)

    barcodes_url = (
        "https://www.ncbi.nlm.nih.gov/geo/download/"
        "?acc=GSE129578&format=file&file=GSE129578_processed_data_files.tsv.tar.gz"
    )
    barcodes_filename = os.path.join(output_path, barcodes_url.split("=")[-1])
    download_binary_file(barcodes_url, barcodes_filename)
    shutil.unpack_archive(barcodes_filename, output_path)


def read_mcginnis_2019(file_directory: str) -> coo_matrix:
    """
    Read the expression data for McGinnis et al. 2019 in the given directory.

    Args:
    ----
        file_directory: Directory containing McGinnis et al. 2019 data.

    Returns
    -------
        A scipy-coo format sparse matrix, with each column representing a cell
        and each row representing a gene feature.
    """

    with gzip.open(
        os.path.join(file_directory, "GSE129578_PDX_mm_matrix.mtx.gz"), "rb"
    ) as f:
        matrix = mmread(f)

    return matrix


def preprocess_mcginnis_2019(
    download_path: str, n_top_genes: int, normalization_method: str = "tc"
) -> AnnData:
    """
    Preprocess expression data from McGinnis et al. 2019.

    Args:
    ----
        download_path: Path containing the downloaded McGinnis et al. 2019 data file.
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

    count_matrix = read_mcginnis_2019(download_path)

    gene_metadata = pd.read_csv(
        os.path.join(download_path, "PDX_mm_genes.tsv"),
        sep="\t",
        header=None,
        index_col=0,
        names=["gene_id"],
    )
    cell_barcodes = pd.read_csv(
        os.path.join(download_path, "PDX_mm_barcodes.tsv"),
        sep="\t",
        header=None,
        index_col=0,
    )
    adata = AnnData(X=count_matrix.T.tocsr(), obs=cell_barcodes, var=gene_metadata)

    # Contains metadata for the cells used to create Figure 3d in McGillin
    # et al 2019. This is a subset of the cells present in `count_matrix`.
    cell_metadata = pd.read_excel(
        os.path.join(download_path, "41592_2019_433_MOESM5_ESM.xlsx"),
        sheet_name=1,  # Relevant information is stored on sheet 1
        skiprows=1,  # The first row contains notes by the authors that we can skip
        nrows=10739,  # Rows past this contain more notes that we can skip
        index_col=0,
    )

    # In the barcodes file, each cell has a "-1" appended to the end of its barcode.
    # Here we edit the index of "cell_metadata" to match this format, so that
    # we can easily index into our Anndata object.
    cell_metadata.index = [barcode + "-1" for barcode in cell_metadata.index]

    # Subset our AnnData object to just contain the cells that have metadata.
    adata = adata[cell_metadata.index].copy()
    adata.X = adata.X.todense()
    adata.obs = cell_metadata

    # Only consider lung cells
    adata = adata[adata.obs["Location"] == "Lung"]

    # To avoid weirdness in downstream analyses, I chose to exclude cells for which
    # the tumor stage and/or cell type could not be identified by the authors.
    adata = adata[adata.obs["TumorStage"] != "Doublet"]
    adata = adata[adata.obs["TumorStage"] != "Negative"]
    adata = adata[adata.obs["CellType_LungImmune"] != "Undetermined"]
    adata = adata[adata.obs["CellType_LungImmune"] != "Lung"]

    adata = preprocess_workflow(
        adata=adata, n_top_genes=n_top_genes, normalization_method=normalization_method
    )
    return adata
