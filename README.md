# Kaggle Ribonanza Models

Codebase of deep learning models for inferring chemical reactivity of RNA molecules, corresponding to the
[Kaggle Ribonanza Challenge](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding) and
accompanying manuscript (full citation when available).

## Setup

1) System Prequisites:
    * Unix environment (tested on Linux, other operating systems may require alternate processes for setting up external dependencies)
    * An available CUDA device
    * 40GB of free disk space (not including space needed for storing persistent output)

2) Ensure the following prerequisites are installed:
    * Python 3.9-3.11 and pip >= 23.1
    * `git` and `git-lfs` (make sure you've run `git lfs install` before cloning this repository)
    * `perl`
    * To compile external dependencies: `patch`, `git`, `gcc`, and `g++`
    * CUDA 12.1
> [!TIP]
> At risk of inducing errors, you can try using another 12.x version and commenting out the CUDA version check in the setup.py
> of the apex library (downlaoded to `external/apex` by `setup_external.sh`). The apex installation will fail if the version
> of CUDA used to build itself is different than the CUDA version used to build PyTorch

3) Install python dependencies via `pip install -r requirements.txt`
4) Prep external libraries with `setup_external.sh`
> [!TIP]
> If you get a module not found error from apex when running models even though apex appeared to have been
  installed correctly, you may want to try running `python setup.py install --cuda_ext --cpp_ext` in `external/apex`

## Organization
`external/`: External dependencies

`src/models/`: Python modules used to perform inference

`src/util/`: Common utilities, particularly for things like feature generation, caching, and output formatting 

`notebooks/simple.ipynb`: Basic example of calling a single inference function with a single sequence

`notebooks/extended.ipynb`: More complex example, demoing running multiple sequences and models, parallelizing and caching generated features, etc

`scripts/inference.py`: Command line utility for running sequence inference

## Usage
* The provided notebooks can be used to interactively explore inference results. In particular, configuration variables are
  provided in the top of the `extended` notebook which allow you to change sequences, limit models, etc.
* The `inference` script can be used to run inference non-interactively. A basic example can be run with `inference.py GGGGAAAACCCC --plot`,
  which will display a matplotlib window with heatmaps for each model and print the raw data as a TSV to the console. Run `inference.py -h`
  for full usage information.
* Both of these options also provide examples for calling inference functions from your own code. The simplest
  method is to `from src.models.<model> import infer` and then call `infer(<sequence_or_sequences>)`, however with more significant usage
  you may also want to set up feature precomputation and caching, adjust batch sizes, etc.

> [!TIP]
> By default, batch sizes used for models are optimized for an 8GB GPU and sequences up to length 457. If you have more GPU memory or shorter maximum sequence lengths, you may want to increase the batch size to run inference for more sequences at once. If you have longer sequences or a smaller GPU, you conevrsely may need to reduce the batch size.

## Differences From Original Models
There are some situations where the models are not identical to the original code used for the competition or otherwise
produce different output than the final Kaggle leaderboard submissions. Reasons for such changes include:
* Teams creating "leaderboard-optimized" submissions by ensembling previous top submissions (without reproducing all models precisely in their released code)
* Teams no longer having access to original weights when releasing their models, leaving either a subset of models to be used or models with retrained weights
* Some minor implementation details (eg, model compilation) have been changed to improve performance where it had negligeable impact on model output
* Generation of base pairing probability features uses different precision and rounding behavior for some models
* Modifications to perform all calculations in-memory (instead of requiring separate (manual) steps
  and intermediate files for generating features, ensembling submodels, etc) and with consistent interfaces
  (eg simple infer() methods, consistent availability and formatting of progress displays, etc)
* Removal of dead code or code unnecessary for inference in order to reduce required dependencies

All models have been verified with a random sample of 2500 sequences from the Ribonanza test set to produce output
with an MAE difference under 0.01 compared to that team's top-scoring leaderboard submission, or when not possible,
with an MAE difference under 0.015 AND an inference run over the entire test set confirmed to result in identical
leaderboard placement.

## Kaggle Solution References
Please see the following links for each team's description of their model. Most posts also include
links to original source code for their models (including code necessary for retraining).

| Model Name | Team Name                       | Rank  | Solution link                                                                      |
|------------|---------------------------------|-------|------------------------------------------------------------------------------------|
|k1_vigg     |vigg                             |   1   |https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460121|
|k2_hoyso    |Hoyeol Sohn                      |   2   |https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460316|
|k3_aek      |ар ен ка                         |   3   |https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460403|
|k4_gorna    |GORNA                            |   4   |https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460203|
|k5_rlj      |R.L.J                            |   5   |https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460250|
|k6_iijj     |iijj                             |   6   |https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460392|
