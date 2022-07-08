# contrastiveVI reproducibility repository

<center>
    <img src="./sketch.png?raw=true" width="750">
</center>

This repository contains code for reproducing results in the contrastiveVI [paper](#references).

contrastiveVI is a generative model designed to isolate factors of variation specific to
a group of "target" cells (e.g. from specimens with a given disease) from those shared
with a group of "background" cells (e.g. from healthy specimens). contrastiveVI is
implemented in [scvi-tools](https://scvi-tools.org/).

## What you can do with contrastiveVI

* If you have a dataset with cells in a background condition (e.g. from healthy
controls) and a target condition (e.g. from diseased patients), you can train
contrastiveVI to isolate latent factors of variation specific to the target cells
from those shared with a background into separate latent spaces.
* Run clustering algorithms on the target-specific latent space to discover sub-groups
of target cells
* Perform differential expression testing for discovered sub-groups of target cells
using a procedure similar to that of [scVI
](https://www.nature.com/articles/s41592-018-0229-2).

## System requirements
This software was designed and tested on a machine running CentOS 7.8.2003, with Python 3.9, PyTorch 1.11, and CUDA 11.2. For a full list of all external Python package dependences used in this project, see the Conda environment file `environment.yml` and the package setup file `setup.cfg`.

When available, this software leverages graphical processing units (GPUs) to accelerate neural network training and evaluation. In our experiments we found that models trained with the aid of a GPU converged in less than 30 minutes (and usually much sooner), with the exact time depending on the size of a given dataset. Systems lacking suitable GPUs may take an extremely long time to train/evaluate models. Our experiments were conducted using an NVIDIA RTX 2080 TI GPU; other GPUs should also work as long as they have sufficient memory (~2GB).

## Reproducibility guide

### Installing the software
1. Git clone this repository.
2. `cd contrastive-vi`.
3. Create and activate the specified conda environment by running
    ```
    conda env create -f environment.yml
    conda activate contrastive-vi-env
    ```
4. Install the local `constrative_vi` package and necessary dependencies by running `pip install -e ".[dev]"`.

This installation procedure should take no more than 30 minutes. Beyond downloading this repository, no addition time is required for its installation.

### Modify data and result path
Modify the paths for storing data and model results in `scripts/constants.py`.
* Modify `DEFAULT_DATA_PATH` for the data storage path.
* Modify `DEFAULT_RESULTS_PATH` for the model result storage path.

### Download datasets

Run `scripts/preprocess_data.py` to download and preprocess the single-cell datasets.
For example, to download the data from
[Zheng _et al._ 2017](https://www.nature.com/articles/ncomms14049)
```
python scripts/preprocess_data.py zheng_2017
```

### Run experiments
Run `scripts/run_experiment.py` to train models and store model outputs. For example,
to train contrastiveVI models with the data from Zheng _et al._ 2017,
```
python scripts/run_experiment.py zheng_2017 contrastiveVI
```

### Evaluate model performance
Once the experiments for the baselines (CPLVM, CGLVM, and scVI) and contrastiveVI have
all been completed. Run
```
python scripts/evaluate_performance.py
```
to generate `performance_summary.csv` in the result directory.

Note: In `scripts/evaluate_performance.py`, the variable `datasets` can be modified for
evaluating model performance with particular datasets of interest.

### Plot results
Figures can be plotted using the corresponding notebooks in `notebooks/figures`.

## References

If you find contrastiveVI useful for your work, please consider citing our preprent:

```
@article{contrastiveVI,
  title={Isolating salient variations of interest in single-cell transcriptomic data with contrastiveVI},
  author={Weinberger, Ethan and Lin, Chris and Lee, Su-In},
  journal={bioRxiv},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
```
