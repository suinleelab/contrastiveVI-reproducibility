"""Model class for contrastive-VI for single cell expression data."""

import logging
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from anndata import AnnData
from scvi import _CONSTANTS
from scvi._docs import setup_anndata_dsp
from scvi.data._anndata import _setup_anndata
from scvi.dataloaders import AnnDataLoader
from scvi.model._totalvi import _get_totalvi_protein_priors
from scvi.model.base import BaseModelClass

from contrastive_vi.data.utils import get_library_log_means_and_vars
from contrastive_vi.model.base.training_mixin import ContrastiveTrainingMixin
from contrastive_vi.module.total_contrastive_vi import TotalContrastiveVIModule

logger = logging.getLogger(__name__)
Number = Union[int, float]


class TotalContrastiveVIModel(ContrastiveTrainingMixin, BaseModelClass):
    """
    Model class for total-contrastiveVI.
    Args:
    ----
        adata: AnnData object that has been registered via
            `TotalContrastiveVIModel.setup_anndata`.
        n_batch: Number of batches. If 0, no batch effect correction is performed.
        n_hidden: Number of nodes per hidden layer.
        n_background_latent: Dimensionality of the background latent space.
        n_salient_latent: Dimensionality of the salient latent space.
        n_layers: Number of hidden layers used for encoder and decoder NNs.
        dropout_rate: Dropout rate for neural networks.
        protein_batch_mask: Dictionary where each key is a batch code, and value is for
            each protein, whether it was observed or not.
        use_observed_lib_size: Use observed library size for RNA as scaling factor in
            mean of conditional distribution.
        empirical_protein_background_prior: Set the initialization of protein
            background prior empirically. This option fits a GMM for each of
            100 cells per batch and averages the distributions. Note that even with
            this option set to `True`, this only initializes a parameter that is
            learned during inference. If `False`, randomly initializes. The default
            (`None`), sets this to `True` if greater than 10 proteins are used.
    """

    def __init__(
        self,
        adata: AnnData,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_background_latent: int = 10,
        n_salient_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        protein_batch_mask: Dict[Union[str, int], np.ndarray] = None,
        use_observed_lib_size: bool = True,
        empirical_protein_background_prior: Optional[bool] = None,
    ) -> None:
        super(TotalContrastiveVIModel, self).__init__(adata)
        # self.summary_stats from BaseModelClass gives info about anndata dimensions
        # and other tensor info.
        if "totalvi_batch_mask" in self.scvi_setup_dict_.keys():
            batch_mask = self.scvi_setup_dict_["totalvi_batch_mask"]
        else:
            batch_mask = None

        emp_prior = (
            empirical_protein_background_prior
            if empirical_protein_background_prior is not None
            else (self.summary_stats["n_proteins"] > 10)
        )
        if emp_prior:
            prior_mean, prior_scale = _get_totalvi_protein_priors(adata)
        else:
            prior_mean, prior_scale = None, None

        if use_observed_lib_size:
            library_log_means, library_log_vars = None, None
        else:
            library_log_means, library_log_vars = get_library_log_means_and_vars(adata)

        self.module = TotalContrastiveVIModule(
            n_input_genes=self.summary_stats["n_vars"],
            n_input_proteins=self.summary_stats["n_proteins"],
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_background_latent=n_background_latent,
            n_salient_latent=n_salient_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            protein_batch_mask=batch_mask,
            use_observed_lib_size=use_observed_lib_size,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
        )
        self._model_summary_string = "total-contrastiveVI."
        # Necessary line to get params to be used for saving and loading.
        self.init_params_ = self._get_init_params(locals())
        logger.info("The model has been initialized")

    @staticmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        adata: AnnData,
        protein_expression_obsm_key: str,
        protein_names_uns_key: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        layer: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        copy: bool = False,
    ) -> Optional[AnnData]:
        """
        Set up AnnData instance for contrastive-VI model.

        Args:
        ----
            adata: AnnData object containing raw counts. Rows represent cells, columns
                represent features.
            protein_expression_obsm_key: Key in `adata.obsm` for protein expression
                data.
            protein_names_uns_key: Key in `adata.uns` for protein names. If None, will
                use the column names of `adata.obsm[protein_expression_obsm_key]`
                if it is a DataFrame, else will assign sequential names to proteins.
            batch_key: Key in `adata.obs` for batch information. Categories will
                automatically be converted into integer categories and saved to
                `adata.obs["_scvi_batch"]`. If None, assign the same batch to all the
                data.
            labels_key: Key in `adata.obs` for label information. Categories will
                automatically be converted into integer categories and saved to
                `adata.obs["_scvi_labels"]`. If None, assign the same label to all the
                data.
            layer: If not None, use this as the key in `adata.layers` for raw count
                data.
            categorical_covariate_keys: Keys in `adata.obs` corresponding to categorical
                data. Used in some models.
            continuous_covariate_keys: Keys in `adata.obs` corresponding to continuous
                data. Used in some models.
            copy: If True, a copy of `adata` is returned.

        Returns
        -------
            If `copy` is True, return the modified `adata` set up for contrastive-VI
            model, otherwise `adata` is modified in place.
        """
        return _setup_anndata(
            adata,
            batch_key=batch_key,
            labels_key=labels_key,
            layer=layer,
            protein_expression_obsm_key=protein_expression_obsm_key,
            protein_names_uns_key=protein_names_uns_key,
            categorical_covariate_keys=categorical_covariate_keys,
            continuous_covariate_keys=continuous_covariate_keys,
            copy=copy,
        )

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        batch_size: Optional[int] = None,
        representation_kind: str = "salient",
    ) -> np.ndarray:
        """
        Return the background or salient latent representation for each cell.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        give_mean: Give mean of distribution or sample from it.
        batch_size: Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        representation_kind: Either "background" or "salient" for the corresponding
            representation kind.

        Returns
        -------
            A numpy array with shape `(n_cells, n_latent)`.
        """
        available_representation_kinds = ["background", "salient"]
        assert representation_kind in available_representation_kinds, (
            f"representation_kind = {representation_kind} is not one of"
            f" {available_representation_kinds}"
        )

        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )
        latent = []
        for tensors in data_loader:
            x = tensors[_CONSTANTS.X_KEY]
            y = tensors[_CONSTANTS.PROTEIN_EXP_KEY]
            batch_index = tensors[_CONSTANTS.BATCH_KEY]
            outputs = self.module._generic_inference(
                x=x, y=y, batch_index=batch_index, n_samples=1
            )

            if representation_kind == "background":
                latent_m = outputs["qz_m"]
                latent_sample = outputs["z"]
            else:
                latent_m = outputs["qs_m"]
                latent_sample = outputs["s"]

            if give_mean:
                latent_sample = latent_m

            latent += [latent_sample.detach().cpu()]
        return torch.cat(latent).numpy()
