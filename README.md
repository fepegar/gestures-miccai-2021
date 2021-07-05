# GESTURES at MICCAI 2021

This repository contains the training scripts used in [Pérez-García et al., 2021, *Transfer Learning of Deep Spatiotemporal Networks to Model Arbitrarily Long Videos of Seizures*, Accepted at the 24th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)](https://arxiv.org/abs/2106.12014).

The dataset is publicly available at the [UCL Research Data Repository](https://doi.org/10.5522/04/14781771).

## Citation

If you use this code or the dataset for your research, please cite the [paper](https://arxiv.org/abs/2106.12014) and the [dataset](https://doi.org/10.5522/04/14781771) appropriately.

## Installation

Using `conda` is recommended:

```shell
conda create -n miccai-gestures python=3.7 ipython -y && conda activate miccai-gestures
```

Using `light-the-torch` is recommended to install the best version of PyTorch automatically:

```shell
pip install light-the-torch
ltt install torch==1.7.0 torchvision==0.4.2
```

Then, clone this repository and install the rest of the requirements:

```shell
git clone https://github.com/fepegar/gestures-miccai-2021.git
cd gestures-miccai-2021
pip install -r requirements.txt
```

Finally, download the dataset:

```shell
curl -L -o dataset.zip https://ndownloader.figshare.com/files/28668096
unzip dataset.zip -d dataset
```

## Training

```shell
GAMMA=4  # gamma parameter for Beta distribution
AGG=blstm  # aggregation mode. Can be "mean", "lstm" or "blstm"
N=16  # number of segments
K=0  # fold for k-fold cross-validation
python train_features_lstm.py \
  --print-config \
  with \
  experiment_name=lstm_feats_jitter_${GAMMA}_agg_${AGG}_segs_${N} \
  jitter_mode=${GAMMA} \
  aggregation=${AGG} \
  num_segments=${N} \
  fold=${K}
```
