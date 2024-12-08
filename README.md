# EigenGS: From Eigenspace to Gaussian Image Space

EigenGS is a novel method that bridges Principal Component Analysis (PCA) and Gaussian Splatting for efficient image representation. Our approach enables instant initialization of Gaussian parameters for new images without requiring per-image training from scratch.

## Quick Start

### Installation

Clone the repository with `--recursive` option:
```bash
git clone https://github.com/lw-ethan/EigenGS.git --recursive
``` 

Install CUDA backend `gsplat`:
```bash
cd gsplat
pip install .[dev]
```
Install other python modules:
```bash
pip install -r requirements.txt
```

## Demo with ImageNet Basis

We provide a demo using EigenGS trained with PCA components from ImageNet, and evaluate on the FFHQ dataset. You can download trained EigenGS model for `step 3` evaluation, or train from the ImageNet images by following steps. 

### 1. Dataset Preparation

Parse your image dataset to the required format:

```bash
python parse.py --source <path to images> \
    --n_comps <number of pca components> \
    --n_samples <number of training sample> \
    --img_size <width, height>
```

**Note**: The `img_size` parameter should match the dimensions of test image.

### 2. Train EigenGS Model

#### W/O Frequency-Aware
```bash
python run_single_freq.py -d <path to parsed dataset> \
    --num_points <number of gaussian points> \
    --iterations <number of training iterations>
```

#### With Frequency-Aware
```bash
python run.py -d <path to parsed dataset> \
    --num_points <number of gaussian points> \
    --iterations <number of training iterations>
```

**Note**: Configure the number of low-frequency Gaussians in `run.py` before training.

### 3. Evaluation

After training, evaluate the performance on test images set using either:

#### Frequency-Aware Evaluation
```bash
python run_sets_single_freq.py -d <path to parsed dataset> \
    --model_path <path to eigengs model> \
    --num_points <number of gaussian points> \
    --iterations <number of training iterations> \
    --skip_train
```

#### Single Frequency Evaluation
```bash
python run_sets.py -d <path to parsed dataset> \
    --model_path <path to eigengs model> \
    --num_points <number of gaussian points> \
    --iterations <number of training iterations> \
    --skip_train
```

## Parameters Guide

- `<number of pca components>`: Number of PCA components, we use 300 in most experiments.
- `<number of training sample>`: Number of training samples, more samples will need more memory resouce when decomposing.
- `<width, height>`: Dimension of the PCA components, this should align the size of target test image.
- `<number of gaussian points>`: Number of Gaussian points to use, this should align in `step 2` and `step 3`.
- `<number of training iterations>`: Number of training iterations, it requires more iterations if Frequency-Aware is enabled.
- `<path to eigengs model>`: Path to pre-trained `.pkl` file.

## Acknowledgement

This work is built upon GaussianImage, we thank the authors for making their code publicly available.

```bibtex
@inproceedings{zhang2024gaussianimage,
  title={GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting},
  author={Zhang, Xinjie and Ge, Xingtong and Xu, Tongda and He, Dailan and Wang, Yan and Qin, Hongwei and Lu, Guo and Geng, Jing and Zhang, Jun},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```