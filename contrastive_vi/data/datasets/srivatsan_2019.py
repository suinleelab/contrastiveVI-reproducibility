"""
Download, read, and preprocess Srivatsan et al. (2019) expression data.

Single-cell expression data from Srivatsan et al. Massively multiplex chemical
transcriptomics at single-cell resolution. Science (2019).
"""
import gzip
import os

import numpy as np
import pandas as pd
from anndata import AnnData

from contrastive_vi.data.utils import download_binary_file, preprocess_workflow


def download_srivatsan_2019(output_path: str) -> None:
    """
    Download Srivatsan et al. 2019 data from the hosting URLs.

    Args:
    ----
        output_path: Output path to store the downloaded and unzipped
        directories.

    Returns
    -------
        None. File directories are downloaded to output_path.
    """

    count_matrix_url = (
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM4150377&format=file&file="
        "GSM4150377_sciPlex2_A549_Transcription_Modulators_UMI.count.matrix.gz"
    )
    count_matrix_filename = os.path.join(output_path, count_matrix_url.split("=")[-1])
    download_binary_file(count_matrix_url, count_matrix_filename)

    cell_metadata_url = (
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM4150377&format=file"
        "&file=GSM4150377_sciPlex2_pData.txt.gz"
    )
    cell_metadata_filename = os.path.join(output_path, cell_metadata_url.split("=")[-1])
    download_binary_file(cell_metadata_url, cell_metadata_filename)

    gene_metadata_url = (
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM4150377&format=file&file="
        "GSM4150377_sciPlex2_A549_Transcription_Modulators_gene.annotations.txt.gz"
    )
    cell_metadata_filename = os.path.join(output_path, gene_metadata_url.split("=")[-1])
    download_binary_file(gene_metadata_url, cell_metadata_filename)


def read_srivatsan_2019(file_directory: str) -> pd.DataFrame:
    """
    Read the expression data for Srivatsan et al. 2019 in the given directory.

    Args:
    ----
        file_directory: Directory containing Srivatsan et al. 2019 data.

    Returns
    -------
        A data frame containing single-cell gene expression counts. The count
        matrix is stored in triplet format. I.e., each row of the data frame
        has the format (row, column, count) stored in columns (i, j, x) respectively.
    """

    with gzip.open(
        os.path.join(
            file_directory,
            "GSM4150377_sciPlex2_A549_Transcription_Modulators_UMI.count.matrix.gz",
        ),
        "rb",
    ) as f:
        df = pd.read_csv(f, sep="\t", header=None, names=["i", "j", "x"])

    return df


def preprocess_srivatsan_2019(
    download_path: str, n_top_genes: int, normalization_method: str = "tc"
) -> AnnData:
    """
    Preprocess expression data from Srivatsan et al., 2019.

    Args:
    ----
        download_path: Path containing the downloaded Srivatsan et al. 2019 data file.
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

    df = read_srivatsan_2019(download_path)

    # The Srivatsan count data is in a sparse triplet format represented
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
            download_path,
            "GSM4150377_sciPlex2_pData.txt.gz",
        ),
        sep=" ",
    )

    gene_metadata = pd.read_csv(
        os.path.join(
            download_path,
            "GSM4150377_sciPlex2_A549_Transcription_Modulators_gene.annotations.txt.gz",
        ),
        sep="\t",
        header=None,
        index_col=0,
    )

    adata = AnnData(
        X=count_matrix, obs=cell_metadata, var=pd.DataFrame(index=gene_metadata.index)
    )

    # Index needs string names or else the write_h5ad call will throw an error
    adata.var.index.name = "gene_id"

    # Treatment information is contained in the `top_oligo` column
    # with the format <drug>_<dose>.
    adata = adata[adata.obs['top_oligo'].notna()]
    adata.obs["drug"] = [
        treatment.split("_")[0] for treatment in adata.obs["top_oligo"]
    ]
    adata = adata[adata.obs["drug"] != "nan"]
    adata.obs["dose"] = [
        treatment.split("_")[1] for treatment in adata.obs["top_oligo"]
    ]
    adata.obs["dose"] = adata.obs["dose"].apply(pd.to_numeric, args=("coerce",))

    # If a drug is listed with dosage of 0, the cell was only exposed to vehicle control
    adata.obs["drug"][adata.obs["dose"] == 0.0] = "Vehicle"

    # Filtering the data as done by the authors here
    # https://github.com/cole-trapnell-lab/sci-plex/blob/079639c50811dd43a206a779ab2f0199a147c98f/small_screen/Notebook_1_small_screen_analysis.R
    adata = adata[adata.obs["hash_umis"] > 30]
    adata = adata[adata.obs["top_to_second_best_ratio"] > 10]
    adata = adata[adata.obs["qval"] < 0.01]

    adata = preprocess_workflow(
        adata=adata, n_top_genes=n_top_genes, normalization_method=normalization_method
    )
    return adata
