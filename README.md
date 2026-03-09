# [From Data Statistics to Feature Geometry: How Correlations Shape Superposition](https://openreview.net/pdf?id=7akSRQS5Xh)

**Authors:** Lucas Prieto, Edward Stevinson, Melih Barsbey, Tolga Birdal, Pedro A. M. Mediano

![Teaser](teaser.png)

## Overview

This repository accompanies the paper and contains the code used to study how data statistics shape feature geometry in superposition. The central claim of the paper is that superposition is not only about suppressing harmful interference: when features are correlated, interference can also be constructive and can organize representations into semantic clusters, cycles, and other structured arrangements.

The repository is organized around three complementary experimental settings:
- `text_bows/`: The main bag-of-words pipeline built from internet text, including dataset construction, tied-weight autoencoder training, and the plotting code used for the paper figures.
- `synthetic_bows/`: Controlled synthetic experiments for isolating how different correlation structures change the geometry learned under compression.
- `value_coding_features/`: Additional experiments on value-coding structure and tasks where geometry reflects explicit encoded values rather than only feature co-occurrence.

## Getting Started

Each subproject has its own README and minimal requirements file. A typical workflow is:
1. generate or download the relevant data for the subproject
2. train the corresponding model or sweep
3. reproduce the paper figures from the saved checkpoints

## Citation
```shell
@inproceedings{
prieto2026correlations,
title={Correlations in the Data Lead to Semantically Rich Feature Geometry Under Superposition},
author={Lucas Prieto and Edward Stevinson and Melih Barsbey and Tolga Birdal and Pedro A. M. Mediano},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=7akSRQS5Xh}
}
```
