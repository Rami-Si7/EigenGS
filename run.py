import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import *
from tracker import LogWriter, PSNRTracker
from tqdm import tqdm
import random
import torchvision.transforms as transforms
from sklearn.decomposition import PCA, SparseCoder
from skimage import color
from gaussianbasis import GaussianBasis
import pickle
import uuid
import cProfile
import pstats

# inline loss function

def loss_fn(image, gt_image, loss_type="L2", lambda_value=0.7):
    if loss_type == "L2":
        return F.mse_loss(image, gt_image)
    elif loss_type == "MS-SSIM":
        return 1.0 - ms_ssim(image, gt_image, data_range=1.0)
    elif loss_type == "Combined":
        l2 = F.mse_loss(image, gt_image)
        ssim_loss = 1.0 - ms_ssim(image, gt_image, data_range=1.0)
        return lambda_value * l2 + (1 - lambda_value) * ssim_loss
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

class GaussianTrainer:
    def __init__(self, args, cluster_means, cluster_bases):
        self.dataset_path = Path(args.dataset)
        self.device = torch.device("cuda:0")
        self.num_points = args.num_points
        self.num_comps = 300
        self.iterations = args.iterations

        self.train_imgs = sorted((self.dataset_path / "train_imgs").glob("*.png"))
        self.images = []
        for img_path in self.train_imgs:
            img = Image.open(img_path).resize((512, 512))
            ycbcr = color.rgb2ycbcr(np.array(img)) / 255.0
            tensor = torch.tensor(ycbcr.transpose(2, 0, 1), dtype=torch.float32)
            self.images.append((img_path.stem, tensor.to(self.device)))

        self.H, self.W = 512, 512
        BLOCK_H, BLOCK_W = 16, 16

        dummy_arrs = torch.zeros((self.num_comps, 3, self.H, self.W), device=self.device)
        self.freq_config = {
            "low": [0, 50, 2000],
            "high": [50, self.num_comps, self.num_points - 2000],
            "all": [0, self.num_comps, self.num_points]
        }

        self.C = cluster_bases.shape[0]
        self.j = cluster_bases.shape[1]

        self.model_dir = Path(f"./models/recons/{self.dataset_path.name}-shared-model")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.gaussian_model = GaussianBasis(
            loss_type="L2", opt_type="adan", num_points=self.num_points,
            H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
            device=self.device, lr=args.lr, num_comps=self.num_comps,
            gt_arrs=dummy_arrs, freq_config=self.freq_config,
            num_clusters=self.C, num_proj_dim=self.j
        ).to(self.device)

        self.gaussian_model.cluster_means.copy_(cluster_means.to(self.device))
        self.gaussian_model.cluster_bases.copy_(cluster_bases.to(self.device))

        self.logwriter = LogWriter(self.model_dir)
        self.psnr_tracker = PSNRTracker(self.logwriter)
        self.logwriter.write("Shared model training on all training images")

    def train_kpca_on_all(self):
        self.gaussian_model.cur_freq = "kpca"
        self.gaussian_model.scheduler_init()
        self.gaussian_model.train()
        progress_bar = tqdm(range(1, self.iterations+1), desc="Training shared model")
        start_time = time.time()

        for iter in range(1, self.iterations+1):
            total_loss = 0.0
            total_psnr = 0.0
            for name, image in self.images:
                loss, psnr = self.train_iter_on_image(image)
                total_loss += loss.item()
                total_psnr += psnr
            avg_loss = total_loss / len(self.images)
            avg_psnr = total_psnr / len(self.images)
            progress_bar.set_postfix({"Loss": f"{avg_loss:.7f}", "PSNR": f"{avg_psnr:.4f}"})
            progress_bar.update(1)

        end_time = time.time() - start_time
        progress_bar.close()
        self.logwriter.write("Training Complete in {:.4f}s".format(end_time))
        torch.save({"model_state_dict": self.gaussian_model.state_dict()}, self.model_dir / "gaussian_model_shared.pth.tar")

    def train_iter_on_image(self, gt_image):
        image = self.gaussian_model()
        loss = loss_fn(image, gt_image, "L2", lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.gaussian_model.optimizer.step()
        self.gaussian_model.optimizer.zero_grad(set_to_none=True)
        self.gaussian_model.scheduler.step()
        return loss, psnr

# --- Main entrypoint ---
def parse_args(argv):
    parser = argparse.ArgumentParser(description="k-PCA EigenGS Shared Training")
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--num_points", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args(argv)

def main(argv):
    args = parse_args(argv)
    dataset_path = Path(args.dataset)

    cluster_pcas = np.load(dataset_path / "cluster_pcas.npy")  # (C, 3, j, D)
    C, _, j, D = cluster_pcas.shape
    cluster_means = torch.zeros(C, 3)  # dummy means

    cluster_bases = []
    for c in range(C):
        V_y = cluster_pcas[c, 0]
        V_cb = cluster_pcas[c, 1]
        V_cr = cluster_pcas[c, 2]
        V_combined = np.stack([V_y, V_cb, V_cr], axis=-1)
        V_avg = V_combined.mean(axis=1)
        cluster_bases.append(V_avg)
    cluster_bases = torch.tensor(np.stack(cluster_bases), dtype=torch.float32)

    trainer = GaussianTrainer(args, cluster_means, cluster_bases)
    trainer.train_kpca_on_all()

if __name__ == "__main__":
    main(sys.argv[1:])
