"""
Download, read, and preprocess Mcfarland et al. (2020) expression data.

Single-cell expression data from Mcfarland et al. Multiplexed single-cell
transcriptional response profiling to define cancer vulnerabilities and therapeutic
mechanism of action. Nature Communications (2020).
"""
import os
import shutil
from typing import List, Tuple

import anndata
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.io import mmread

from contrastive_vi.data.utils import download_binary_file, preprocess_workflow


def download_mcfarland_2020(output_path: str) -> None:
    """
    Download Mcfarland et al. 2020 data from the hosting URLs.

    Args:
    ----
        output_path: Output path to store the downloaded and unzipped
        directories.

    Returns
    -------
        None. File directories are downloaded and unzipped in output_path.
    """
    idasanutlin_url = "https://figshare.com/ndownloader/files/18716351"
    idasanutlin_output_filename = os.path.join(output_path, "idasanutlin.zip")

    download_binary_file(idasanutlin_url, idasanutlin_output_filename)
    idasanutlin_output_dir = idasanutlin_output_filename.replace(".zip", "")
    shutil.unpack_archive(idasanutlin_output_filename, idasanutlin_output_dir)

    dmso_url = "https://figshare.com/ndownloader/files/18716354"
    dmso_output_filename = os.path.join(output_path, "dmso.zip")

    download_binary_file(dmso_url, dmso_output_filename)
    dmso_output_dir = dmso_output_filename.replace(".zip", "")
    shutil.unpack_archive(dmso_output_filename, dmso_output_dir)

    # DepMap 19Q3 mutation data; this was used by McFarland et al. to determine
    # mutation statuses for different genes for each cell line
    cell_line_mutations_url = "https://ndownloader.figshare.com/files/16757702"
    cell_line_mutations_output_filename = os.path.join(output_path, "mutations.csv")
    download_binary_file(cell_line_mutations_url, cell_line_mutations_output_filename)

    # Metadata file that contains mappings from DepMap cell line IDs (used in
    # the mutation file above) to the human-readable cell line names used
    # by McFarland et al.
    cell_line_info_url = "https://ndownloader.figshare.com/files/16757723"
    cell_line_info_output_filename = os.path.join(output_path, "cell_line_info.csv")
    download_binary_file(cell_line_info_url, cell_line_info_output_filename)


def _read_mixseq_df(directory: str) -> pd.DataFrame:
    data = mmread(os.path.join(directory, "matrix.mtx"))
    barcodes = pd.read_table(os.path.join(directory, "barcodes.tsv"), header=None)
    classifications = pd.read_csv(os.path.join(directory, "classifications.csv"))
    classifications["cell_line"] = np.array(
        [x.split("_")[0] for x in classifications.singlet_ID.values]
    )
    gene_names = pd.read_table(os.path.join(directory, "genes.tsv"), header=None)

    df = pd.DataFrame(
        data.toarray(),
        columns=barcodes.iloc[:, 0].values,
        index=gene_names.iloc[:, 0].values,
    )
    return df


def _get_cell_line_labels(directory: str) -> np.array:
    classifications = pd.read_csv(os.path.join(directory, "classifications.csv"))
    return classifications.singlet_ID.values


def _get_mutation_status(
    mutations_file_directory: str, gene: str, cell_line_labels: List[str]
):
    mutations = pd.read_csv(os.path.join(mutations_file_directory, "mutations.csv"))
    cell_line_info = pd.read_csv(
        os.path.join(mutations_file_directory, "cell_line_info.csv")
    )

    # Construct a mapping between the DepMap ID's used in the mutations file
    # and the human readable cell line names in the cell line info file
    cell_line_map = {}
    for idx, row in cell_line_info.iterrows():
        cell_line_map[row["DepMap_ID"]] = row["CCLE Name"]

    mutations["cell_line"] = [cell_line_map[x] for x in mutations["DepMap_ID"]]

    # The McFarland authors considered silent mutations to
    # be equivalent to wild type, so we do the same here
    gene_mutations = mutations[
        (mutations["Hugo_Symbol"] == gene)
        & (mutations["Variant_Classification"] != "Silent")
    ]

    gene_mutations_cell_lines = gene_mutations["cell_line"].unique()
    return [
        "Mutation" if x in gene_mutations_cell_lines else "Wild Type"
        for x in cell_line_labels
    ]


def read_mcfarland_2020(file_directory: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read the expression data for Mcfarland et al. 2020 in the given directory.

    Args:
    ----
        file_directory: Directory containing Mcfarland et al. 2020 data.

    Returns
    -------
        Two data frames of raw count expression data. The first contains
        single-cell gene expression count data from cancer cell lines exposed to
        idasanutlin with cell identification barcodes as column names and gene IDs as
        indices. The second contains count data with the same format from samples
        exposed to a control solution (DMSO).
    """
    idasanutlin_dir = os.path.join(
        file_directory, "idasanutlin", "Idasanutlin_24hr_expt1"
    )
    idasanutlin_df = _read_mixseq_df(idasanutlin_dir)

    dmso_dir = os.path.join(file_directory, "dmso", "DMSO_24hr_expt1")
    dmso_df = _read_mixseq_df(dmso_dir)

    return idasanutlin_df, dmso_df


def preprocess_mcfarland_2020(
    download_path: str, n_top_genes: int, normalization_method: str = "tc"
) -> AnnData:
    """
    Preprocess expression data from Mcfarland et al., 2020.

    Args:
    ----
        download_path: Path containing the downloaded Mcfarland et al. 2020 data files.
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

    idasanutlin_df, dmso_df = read_mcfarland_2020(download_path)
    idasanutlin_df, dmso_df = idasanutlin_df.transpose(), dmso_df.transpose()

    idasanutlin_adata = AnnData(idasanutlin_df)
    idasanutlin_dir = os.path.join(
        download_path, "idasanutlin", "Idasanutlin_24hr_expt1"
    )
    idasanutlin_adata.obs["cell_line"] = _get_cell_line_labels(idasanutlin_dir)
    idasanutlin_adata.obs["TP53_mutation_status"] = _get_mutation_status(
        mutations_file_directory=download_path,
        gene="TP53",
        cell_line_labels=idasanutlin_adata.obs["cell_line"],
    )
    idasanutlin_adata.obs["condition"] = np.repeat(
        "Idasanutlin", idasanutlin_adata.shape[0]
    )

    dmso_adata = AnnData(dmso_df)
    dmso_dir = os.path.join(download_path, "dmso", "DMSO_24hr_expt1")
    dmso_adata.obs["cell_line"] = _get_cell_line_labels(dmso_dir)
    dmso_adata.obs["TP53_mutation_status"] = _get_mutation_status(
        mutations_file_directory=download_path,
        gene="TP53",
        cell_line_labels=dmso_adata.obs["cell_line"],
    )
    dmso_adata.obs["condition"] = np.repeat("DMSO", dmso_adata.shape[0])

    full_adata = anndata.concat([idasanutlin_adata, dmso_adata])
    full_adata = preprocess_workflow(
        adata=full_adata,
        n_top_genes=n_top_genes,
        normalization_method=normalization_method,
    )
    return full_adata
