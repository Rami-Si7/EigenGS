# EigenGS: From Eigenspace to Gaussian Image Space

EigenGS is a novel method that bridges Principal Component Analysis (PCA) and Gaussian Splatting for efficient image representation. Our approach enables instant initialization of Gaussian parameters for new images without requiring per-image training from scratch.

## Quick Start

### Installation

Clone the repository with `--recursive` option:
```bash
git clone https://github.com/vllab/EigenGS.git --recursive
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

We provide a demo using EigenGS trained with PCA components from [ImageNet](https://www.icloud.com/iclouddrive/04fbVOqtOeQCRA52Ef05QNsLA#ImageNet), and evaluate on the [FFHQ](https://www.icloud.com/iclouddrive/0bfGI1wc4x-2Y2w4ASynV6SfA#FFHQ) dataset. 
The [parsed data](https://www.icloud.com/iclouddrive/0787LHytmNDuWM4zSBdOMRlMg#imagenet-ffhq-300-ycbcr) and trained [EigenGS](https://www.icloud.com/iclouddrive/0d30IWn45tl4phrGCbFfk2BqQ#imagenet-ffhq-300-ycbcr-20000-15000-d37b07) can be downloaded for `step 3` evaluation. If you are interested in using the custom images, please follow the steps below. 

### 1. Dataset Preparation

Parse your image dataset to the required format, the `parse.py` will generated:

- `arrs.npy`: Numpy array with PCA components information.
- `norm_infos.pkl`: Normalization information of the components.
- `pca_object.pkl`: Sklearn PCA object.

Also, the processed images will be organized into `test_imgs` and `train_imgs` folders. You should replace their contents with target test and training images.

```bash
python parse.py --source <path to images> \
    --n_comps <number of pca components> \
    --n_samples <number of training sample> \
    --img_size <width, height>
```

**Note**: The `img_size` parameter should match the dimensions of test image.

### 2. Train EigenGS Model

We recommand to train with Frequency-Aware for larger test image.

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

After training, evaluate the performance on test images set. Please note that when running the demo we prepared, the `--num_points` should be set as 20000, and it should be run with frequency-aware script.

#### Frequency-Aware Evaluation
```bash
python run_sets.py -d <path to parsed dataset> \
    --model_path <path to eigengs model> \
    --num_points <number of gaussian points> \
    --iterations <number of training iterations> \
    --skip_train
```

#### Single Frequency Evaluation
```bash
python run_sets_single_freq.py -d <path to parsed dataset> \
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