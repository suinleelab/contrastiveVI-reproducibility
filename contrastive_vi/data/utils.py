"""Data preprocessing utilities."""
import os
from typing import Tuple

import numpy as np
import pandas as pd
import requests
import scanpy as sc
from anndata import AnnData
from rpy2 import robjects
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.pandas2ri import rpy2py


def download_binary_file(
    file_url: str, output_path: str, overwrite: bool = False
) -> None:
    """
    Download binary data file from a URL.

    Args:
    ----
        file_url: URL where the file is hosted.
        output_path: Output path for the downloaded file.
        overwrite: Whether to overwrite existing downloaded file.

    Returns
    -------
        None.
    """
    file_exists = os.path.exists(output_path)
    if (not file_exists) or (file_exists and overwrite):
        request = requests.get(file_url)
        with open(output_path, "wb") as f:
            f.write(request.content)
        print(f"Downloaded data from {file_url} at {output_path}")
    else:
        print(
            f"File {output_path} already exists. "
            "No files downloaded to overwrite the existing file."
        )


def read_seurat_raw_counts(file_path: str) -> pd.DataFrame:
    """
    Read raw expression count data from a Seurat R object.

    Args:
    ----
        file_path: Path to RDS file containing Seurat R object.

    Returns
    -------
        A (pandas) dataframe containing the count data stored in the Seurat object. This
        data frame has cell identification barcodes as column names and gene IDs as
        indices.
    """
    try:
        readRDS = robjects.r["readRDS"]
        base = importr("base")
        seurat_object = importr("SeuratObject")
    except RRuntimeError:
        raise ImportError(
            "Unable to load R environment with SeuratObject package. Please ensure you "
            "have a working installation of R with the SeuratObject "
            "package installed before continuing."
        )

    rds_object = readRDS(file_path)

    r_df = base.as_data_frame(
        base.as_matrix(seurat_object.GetAssayData(object=rds_object, slot="counts"))
    )

    # Converts the R dataframe object to a pandas dataframe via rpy2
    pandas_df = rpy2py(r_df)
    return pandas_df


def read_seurat_cell_metadata(file_path: str) -> pd.DataFrame:
    """
    Read cell metadata from a Seurat R object.

    Args:
    ----
        file_path: Path to RDS file containing Seurat R object.

    Returns
    -------
        A (pandas) dataframe containing metadata for each cell in the Seurat object.
        For this dataframe rows are cells while columns represent metadata features.
    """
    try:
        readRDS = robjects.r["readRDS"]
    except RRuntimeError:
        raise ImportError(
            "Unable to load R environment. Please ensure you "
            "have a working installation of R before continuing."
        )
    rds_object = readRDS(file_path)

    metadata_r_df = rds_object.slots["meta.data"]
    metadata_pandas_df = rpy2py(metadata_r_df)
    return metadata_pandas_df


def read_seurat_feature_metadata(file_path: str) -> pd.DataFrame:
    """
    Read feature metadata from a Seurat R object.

    Args:
    ----
        file_path: Path to RDS file containing Seurat R object.

    Returns
    -------
        A (pandas) dataframe containing metadata for each gene feature in the Seurat
        object. For this dataframe rows are genes while columns represent metadata
        features.
    """
    try:
        readRDS = robjects.r["readRDS"]
        dollar_sign = robjects.r["$"]
        double_bracket = robjects.r["[["]
    except RRuntimeError:
        raise ImportError(
            "Unable to load R environment. Please ensure you "
            "have a working installation of R before continuing."
        )
    rds_object = readRDS(file_path)

    # This line is equivalent to the R code `rds_object$RNA[[]]`, which is used to
    # access metadata on the features stored in the Seurat object
    feature_metadata_r_df = double_bracket(dollar_sign(rds_object, "RNA"))

    feature_metadata_pandas_df = rpy2py(feature_metadata_r_df)
    return feature_metadata_pandas_df


def normalize_edger_tmm(raw_count: np.ndarray) -> np.ndarray:
    """
    Normalize count using trimmed mean of M values implemented in the R package edgeR.

    Args:
    ----
        raw_count: Numpy array of raw count before normalization. Rows correspond to
            cells (samples), and columns correspond to genes.

    Returns
    -------
        A numpy array of the normalized count, with the same shape as `raw_count`. Each
        row of raw count is divided by the normalization factor found via edgeR.
    """
    try:
        dollar_sign = robjects.r["$"]
    except RRuntimeError:
        raise ImportError(
            "Unable to load R environment. Please ensure you have a working "
            "installation of R!"
        )
    try:
        edger = importr("edgeR")
    except RRuntimeError:
        raise ImportError("The R package edgeR is not installed!")
    numpy2ri.activate()  # Enable numpy-to-matrix conversion.
    results = edger.calcNormFactors(edger.DGEList(counts=raw_count.transpose()))

    # results$samples contains information for each sample. For each result in
    # results$samples, result[0] contains the group that the sample belongs to,
    # result[1] contains the un-normalized library size, and result[2] contains the
    # normalization factor.
    norm_factors = [result[2] for result in dollar_sign(results, "samples")]
    norm_factors = np.array(norm_factors)[:, np.newaxis]
    numpy2ri.deactivate()
    return raw_count / norm_factors


def normalize_scran_deconv(raw_count: np.ndarray) -> np.ndarray:
    """
    Normalize count using deconvolution as implemented in the R package scran.

    Args:
    ----
        raw_count: Numpy array of raw count before normalization. Rows correspond to
            cells (samples), and columns correspond to genes.

    Returns
    -------
        A numpy array of the normalized count, with the same shape as `raw_count`. Each
        row of raw count is divided by the normalization factor found via scran.
    """
    try:
        dollar_sign = robjects.r["$"]
    except RRuntimeError:
        raise ImportError(
            "Unable to load R environment. Please ensure you have a working "
            "installation of R!"
        )
    try:
        single_cell_experiment = importr("SingleCellExperiment")
    except RRuntimeError:
        raise ImportError("The R package SingleCellExperiment is not installed!")
    try:
        scran = importr("scran")
    except RRuntimeError:
        raise ImportError("The R package scran is not installed!")
    numpy2ri.activate()  # Enable numpy-to-matrix conversion.
    count = single_cell_experiment.SingleCellExperiment(
        robjects.r.list(counts=raw_count.transpose())
    )
    results = scran.computeSumFactors(count)
    results = scran.convertTo(results, type="edgeR")

    # results$samples contains information for each sample. For each result in
    # results$samples, result[0] contains the group that the sample belongs to,
    # result[1] contains the un-normalized library size, and result[2] contains the
    # normalization factor.
    norm_factors = [result[2] for result in dollar_sign(results, "samples")]
    norm_factors = np.array(norm_factors)[:, np.newaxis]
    numpy2ri.deactivate()
    return raw_count / norm_factors


def normalize_basics_denoise(raw_count: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Normalize count using the denoising algorithm implemented in the R package BASiCS.

    Args:
    ----
        raw_count: Numpy array of raw count before normalization. Rows correspond to
            cells (samples), and columns correspond to genes.
        seed: Random seed for the MCMC.

    Returns
    -------
        A numpy array of the normalized count, with the same shape as `raw_count`.
    """
    try:
        set_seed = robjects.r("set.seed")
    except RRuntimeError:
        raise ImportError(
            "Unable to load R environment. Please ensure you have a working "
            "installation of R!"
        )
    try:
        single_cell_experiment = importr("SingleCellExperiment")
    except RRuntimeError:
        raise ImportError("The R package SingleCellExperiment is not installed!")
    try:
        basics = importr("BASiCS")
    except RRuntimeError:
        raise ImportError("The R package BASiCS is not installed!")
    numpy2ri.activate()
    set_seed(seed)
    np.random.seed(seed)
    count = single_cell_experiment.SingleCellExperiment(
        robjects.r.list(counts=raw_count.transpose())
    )
    mcmc_chain = basics.BASiCS_MCMC(
        Data=count,
        N=5000,  # Reasonable settings for large datasets.
        Thin=20,
        Burn=2500,
        Regression=True,
        WithSpikes=False,
    )
    denoised_count = basics.BASiCS_DenoisedCounts(
        Data=count, Chain=mcmc_chain, WithSpikes=False
    )
    numpy2ri.deactivate()
    return denoised_count.transpose()


def preprocess_workflow(
    adata: AnnData, n_top_genes: int, normalization_method: str = "tc"
) -> AnnData:
    """
    Preprocess single-cell data in a pipeline workflow.

    NOTE: The AnnData passed into this function is modified inplace. If the original
        AnnData is intended to be unaltered, make a copy of the data first.
    Args:
    ----
        adata: An AnnData object containing single-cell expression count data.
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
    available_normalization_methods = ["tc", "tmm", "scran", "basics"]
    assert normalization_method in available_normalization_methods, (
        f"normalization_method = {normalization_method} should be one of "
        f"{available_normalization_methods}!"
    )

    adata.layers["count"] = adata.X.copy()
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
        layer="count",
        subset=False,
    )

    adata = adata[adata.layers["count"].sum(1) != 0]  # Remove cells with all zeros.

    if normalization_method == "tc":
        sc.pp.normalize_total(adata)
    elif normalization_method == "tmm":
        adata.X = normalize_edger_tmm(adata.layers["count"].copy())
    elif normalization_method == "scran":
        adata.X = normalize_scran_deconv(adata.layers["count"].copy())
    elif normalization_method == "basics":
        adata.X = normalize_basics_denoise(adata.layers["count"].copy())
    else:
        raise NotImplementedError(
            f"normalization_method = {normalization_method} is not implemented!"
        )
    sc.pp.log1p(adata)

    adata.raw = adata
    adata = adata[:, adata.var["highly_variable"]]
    return adata


def get_library_log_means_and_vars(adata: AnnData) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the mean and variance of log library size for each experimental batch.

    Args:
    ----
        adata: AnnData object that has been registered via `setup_anndata`.

    Returns
    -------
        A tuple of numpy array `library_log_means` and `library_log_vars` for the mean
        and variance, respectively. Each has shape `(1, n_batch)`.
    """
    count_data_registry = adata.uns["_scvi"]["data_registry"]["X"]
    if count_data_registry["attr_name"] == "layers":
        count_data = adata.layers[count_data_registry["attr_key"]]
    else:
        count_data = adata.X

    library_log_means = []
    library_log_vars = []
    batches = adata.obs["_scvi_batch"].unique()
    for batch in batches:
        if len(batches) > 1:
            library = count_data[adata.obs["_scvi_batch"] == batch].sum(1)
        else:
            library = count_data.sum(1)
        library_log = np.ma.log(library)
        library_log = library_log.filled(0.0)  # Fill invalid log values with zeros.
        library_log_means.append(library_log.mean())
        library_log_vars.append(library_log.var())
    library_log_means = np.array(library_log_means)[np.newaxis, :]
    library_log_vars = np.array(library_log_vars)[np.newaxis, :]
    return library_log_means, library_log_vars


def save_preprocessed_adata(
    adata: AnnData, output_path: str, normalization_method: str
) -> None:
    """
    Save given AnnData object with preprocessed data to disk using our dataset file
    naming convention.

    Args:
    ----
        adata: AnnData object containing expression count data as well as metadata.
        output_path: Path to save resulting file.
        normalization_method: Name of the normalization method to tag the output
            filename.

    Returns
    -------
        None. Provided AnnData object is saved to disk in a subdirectory called
        "preprocessed" in output_path.
    """
    preprocessed_directory = os.path.join(output_path, "preprocessed")
    os.makedirs(preprocessed_directory, exist_ok=True)
    n_genes = adata.shape[1]
    filename = os.path.join(
        preprocessed_directory,
        f"adata_top_{n_genes}_genes_{normalization_method}.h5ad",
    )
    adata.write_h5ad(filename=filename)
