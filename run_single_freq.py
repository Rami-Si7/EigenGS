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
from sklearn.decomposition import PCA
from skimage import color
from gaussianbasis_single_freq import GaussianBasis
import pickle
import uuid
import json

class GaussianTrainer:
    def __init__(
        self, args,
        image_path = None,
        num_points: int = 2000,
        iterations: int = 30000,
        model_path = None,
    ):
        self.dataset_path = Path(args.dataset)
        self.device = torch.device("cuda:0")

        # ---- NEW: read clustered metadata/artifacts ----
        meta_path = self.dataset_path / "metadata.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.W, self.H = int(meta["img_size"][0]), int(meta["img_size"][1])  # img_size saved as [width, height]
        self.num_comps = int(meta["n_comps"])
        self.num_clusters = int(meta["k_clusters"])
        self.comp_batch = int(getattr(args, "comp_batch", 0) or 0)

        # raw PCA components per cluster (K, 3, C, D), means per cluster (K, 3, D)
        self.cluster_pcas_raw = np.load(self.dataset_path / "cluster_pcas_raw.npy")   # (K, 3, C, D)
        self.cluster_means = np.load(self.dataset_path / "cluster_pca_means.npy")     # (K, 3, D)
        # kmeans model for assigning clusters to new images (Phase-B)
        with open(self.dataset_path / "kmeans.pkl", "rb") as f:
            self.kmeans = pickle.load(f)

        random_string = str(uuid.uuid4())[:6]
        if image_path is not None:
            self.image_path = Path(image_path)
            self.image_name = self.image_path.stem

            img = Image.open(self.image_path).resize((self.W, self.H))
            img_arr = (color.rgb2ycbcr(img) / 255.0).transpose(2, 0, 1)
            self.gt_image = torch.tensor(img_arr, dtype=torch.float32, device=self.device)

            model_dir = Path(f"./models/recons/{self.image_name}-{self.dataset_path.name}-{num_points}-{args.iterations}-{random_string}")
        else:
            model_dir = Path(f"./models/single-freq/{self.dataset_path.name}-{num_points}-{args.iterations}-{random_string}")

        self.num_points = num_points
        BLOCK_H, BLOCK_W = 16, 16
        self.iterations = iterations

        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # ---- Model: now includes num_clusters and per-cluster means ----
        self.gaussian_model = GaussianBasis(
            loss_type="L2", opt_type="adan", num_points=self.num_points,
            H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            device=self.device, lr=args.lr, num_comps=self.num_comps,
            num_clusters=self.num_clusters
        ).to(self.device)

        # Load cluster means into the model (expects (K, 3, H*W))
        self.gaussian_model.load_cluster_means(self.cluster_means, from_flat=True)

        self.gaussian_model.scheduler_init()
        self.logwriter = LogWriter(self.model_dir)
        self.psnr_tracker = PSNRTracker(self.logwriter)
        self.logwriter.write(f"Model Dir ID: {random_string}")

        if model_path is not None:
            self.model_path = Path(model_path)
            self.logwriter.write(f"Model loaded from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.gaussian_model.load_state_dict(checkpoint['model_state_dict'])

    # ---- NEW: build GT component stack for a cluster (C,3,H,W) ----
    def _get_cluster_target_stack(self, k: int) -> torch.Tensor:
        """
        cluster_pcas_raw[k]: (3, C, D)
        -> (C, 3, H, W) torch on device
        """
        comps_flat = self.cluster_pcas_raw[k]  # (3, C, D)
        comps_flat = np.transpose(comps_flat, (1, 0, 2))  # (C, 3, D)
        stack = torch.from_numpy(comps_flat).float().to(self.device).view(self.num_comps, 3, self.H, self.W)
        return stack

    def train(self):
        """
        Phase-A: Train per-cluster bases. We iterate over clusters in a round-robin manner.
        """
        progress_bar = tqdm(range(1, self.iterations+1), desc="Training (Phase-A) progress")
        self.gaussian_model.train()
        start_time = time.time()

        for it in range(1, self.iterations+1):
            # simple schedule: round-robin over clusters
            k = (it - 1) % self.num_clusters
            gt_stack = self._get_cluster_target_stack(k)                  # (C,3,H,W)
            loss, psnr = self.gaussian_model.train_iter(gt_stack, cluster_id=k)

            with torch.no_grad():
                progress_bar.set_postfix({
                    "Cluster": f"{k}",
                    "Loss": f"{loss.item():.7f}",
                    "PSNR": f"{psnr:.4f},"
                })
                progress_bar.update(1)

        end_time = time.time() - start_time
        progress_bar.close()
        self.logwriter.write("Phase-A Training Complete in {:.4f}s".format(end_time))
        torch.save({'model_state_dict': self.gaussian_model.state_dict()}, self.model_dir / "gaussian_model_phaseA.pth.tar")

        # optional quick viz of cluster 0
        self.vis(cluster_id=0)
        return
    # def train(self):
    #     """
    #     Phase-A: Train per-cluster bases by accumulating gradients over *all* clusters
    #     (and optionally over component chunks) in each iteration, then doing one step.
    #     """
    #     self.gaussian_model.train()
    #     progress_bar = tqdm(range(1, self.iterations + 1), desc="Training (Phase-A, all clusters)")

    #     # How many component chunks per cluster?
    #     if self.comp_batch and self.comp_batch > 0 and self.comp_batch < self.num_comps:
    #         n_chunks = (self.num_comps + self.comp_batch - 1) // self.comp_batch
    #         chunk_ranges = [range(s, min(s + self.comp_batch, self.num_comps))
    #                         for s in range(0, self.num_comps, self.comp_batch)]
    #     else:
    #         n_chunks = 1
    #         chunk_ranges = [range(self.num_comps)]  # one chunk: all components

    #     K = self.num_clusters
    #     N_accum = K * n_chunks  # scale each partial loss to keep LR behavior stable

    #     start_time = time.time()
    #     for it in range(1, self.iterations + 1):
    #         self.gaussian_model.optimizer.zero_grad(set_to_none=True)

    #         total_loss_val = 0.0
    #         sse = 0.0  # sum of squared errors for PSNR over all clusters/chunks rendered
    #         n_pix = 0  # total number of pixels*channels contributing to sse

    #         for k in range(K):
    #             # Ground-truth component stack for this cluster (C,3,H,W) as torch on device
    #             gt_stack_full = self._get_cluster_target_stack(k)

    #             # Optionally iterate in component chunks to save VRAM
    #             for comp_inds in chunk_ranges:
    #                 # Forward only these components
    #                 pred_stack = self.gaussian_model.forward(
    #                     cluster_id=k, render_colors=False, comp_indices=comp_inds
    #                 )  # (len(comp_inds), 3, H, W)

    #                 gt_chunk = gt_stack_full[comp_inds]  # match shape

    #                 # Loss for this (cluster, chunk)
    #                 loss_chunk = loss_fn(pred_stack, gt_chunk, self.gaussian_model.loss_type, lambda_value=0.7)

    #                 # Scale so that the sum over all chunks & clusters ~ same magnitude each iter
    #                 loss_scaled = loss_chunk / float(N_accum)
    #                 loss_scaled.backward()

    #                 total_loss_val += float(loss_chunk.detach().item())

    #                 # accumulate SSE for PSNR (use reduction='sum' to get SSE)
    #                 with torch.no_grad():
    #                     sse += float(F.mse_loss(pred_stack, gt_chunk, reduction='sum').item())
    #                     n_pix += pred_stack.numel()

    #         # One optimizer step per iteration, after accumulating across all clusters/chunks
    #         self.gaussian_model.optimizer.step()
    #         self.gaussian_model.scheduler.step()

    #         # PSNR over everything rendered this iter
    #         psnr = 10.0 * math.log10(1.0 / (sse / (n_pix + 1e-12) + 1e-12))

    #         # UI
    #         progress_bar.set_postfix({
    #             "Loss(sum)": f"{total_loss_val:.7f}",
    #             "PSNR(avg)": f"{psnr:.4f},",
    #             "K": f"{K}",
    #             "chunks/cluster": f"{n_chunks}",
    #         })
    #         progress_bar.update(1)

    #     end_time = time.time() - start_time
    #     progress_bar.close()
    #     self.logwriter.write("Phase-A Training Complete in {:.4f}s".format(end_time))

    #     torch.save({'model_state_dict': self.gaussian_model.state_dict()},
    #               self.model_dir / "gaussian_model_phaseA.pth.tar")

    #     # Optional quick viz of a cluster
    #     self.vis(cluster_id=0)
    #     return

    def _predict_cluster_id_for_image(self, img_tensor_3xHxW: torch.Tensor) -> int:
        """
        Use saved kmeans on concatenated Y+Cb+Cr (flattened) to choose cluster for Phase-B.
        img_tensor: (3, H, W), values in [0,1] (YCbCr)
        """
        arr = img_tensor_3xHxW.detach().cpu().numpy().reshape(3, -1)  # (3, D), usually float32
        Y, Cb, Cr = arr[0], arr[1], arr[2]
        feat = np.concatenate([Y, Cb, Cr], axis=0).reshape(1, -1)     # (1, 3D)

        model_dtype = getattr(self.kmeans.cluster_centers_, "dtype", np.float64)
        feat = np.ascontiguousarray(feat, dtype=model_dtype)

        k = int(self.kmeans.predict(feat)[0])
        print(f"cluster id: {k}")
        return k


    def _compute_pca_weights_for_image(self, k: int, img_tensor_3xHxW: torch.Tensor):
        """
        Compute per-channel PCA weights using raw components and means for cluster k:
        w_ch = (x_ch - mean_ch) @ comps_ch^T
        Returns three numpy arrays: wY, wCb, wCr of shape (C,)
        """
        arr = img_tensor_3xHxW.detach().cpu().numpy().reshape(3, -1)      # (3, D)
        comps_k = self.cluster_pcas_raw[k]                                 # (3, C, D)
        means_k = self.cluster_means[k]                                    # (3, D)

        w_list = []
        for ch in range(3):
            x = arr[ch]              # (D,)
            mu = means_k[ch]         # (D,)
            comps = comps_k[ch]      # (C, D)
            w = (x - mu) @ comps.T   # (C,)
            w_list.append(w.astype(np.float32))
        return w_list[0], w_list[1], w_list[2]  # wY, wCb, wCr

    def optimize(self):
        """
        Phase-B: Fit colors for a single image (self.gt_image) with the right cluster.
        """
        assert hasattr(self, "gt_image"), "optimize() requires --image_path"

        progress_bar = tqdm(range(1, self.iterations+1), desc="Optimizing (Phase-B) progress")

        # pick cluster for this image
        cluster_id = self._predict_cluster_id_for_image(self.gt_image)

        # initialize colors from PCA weights (good warm start)
        wY, wCb, wCr = self._compute_pca_weights_for_image(cluster_id, self.gt_image)
        self.gaussian_model.init_colors_from_weights(cluster_id, wY, wCb, wCr)

        self.gaussian_model.scheduler_init(optimize_phase=True)
        self.gaussian_model.train()
        start_time = time.perf_counter()

        # quick test before starting
        self.test(cluster_id=cluster_id, iter=0)
        self.gaussian_model.train()

        for it in range(1, 20000+1):
            loss, psnr = self.gaussian_model.optimize_iter(self.gt_image, cluster_id=cluster_id)

            with torch.no_grad():
                if it in [10,100,500,1000,2000,3000,4000,10000, 15000,20000]:
                    self.test(cluster_id=cluster_id, iter=it)
                    self.gaussian_model.train()
                if it % 10 == 0:
                    progress_bar.set_postfix({
                        "Cluster": f"{cluster_id}",
                        "Loss": f"{loss.item():.7f}",
                        "PSNR": f"{psnr:.4f},"
                    })
                    progress_bar.update(10)

        end_time = time.perf_counter() - start_time
        progress_bar.close()
        self.logwriter.write("Phase-B Optimizing Complete in {:.4f}s".format(end_time))
        torch.save({'model_state_dict': self.gaussian_model.state_dict()}, self.model_dir / "gaussian_model_phaseB.pth.tar")

        # final test
        self.test(cluster_id=cluster_id)
        return

    def vis(self, cluster_id: int = 0):
        """
        Visualize learned components for a chosen cluster.
        """
        self.gaussian_model.eval()
        with torch.no_grad():
            pred_stack = self.gaussian_model.forward(cluster_id=cluster_id, render_colors=False)  # (C,3,H,W)

        gt_stack = self._get_cluster_target_stack(cluster_id)
        mse_loss = F.mse_loss(pred_stack.float(), gt_stack.float())
        psnr = 10 * math.log10(1.0 / (mse_loss.item() + 1e-12))
        # ms-ssim over stacks (treat C as batch by flattening):
        # here we compute mean of per-component ms-ssim for a quick score
        ms_list = []
        for c in range(pred_stack.shape[0]):
            ms = ms_ssim(pred_stack[c].unsqueeze(0).float(),
                         gt_stack[c].unsqueeze(0).float(),
                         data_range=1, size_average=True).item()
            ms_list.append(ms)
        ms_ssim_value = float(np.mean(ms_list))
        self.logwriter.write(f"[Cluster {cluster_id}] Components Fitting: PSNR:{psnr:.4f}, MS_SSIM:{ms_ssim_value:.6f}")

        vis_dir = self.model_dir / f"vis_comps_cluster_{cluster_id}"
        vis_dir.mkdir(parents=True, exist_ok=True)
        transform = transforms.ToPILImage()
        array = pred_stack.float()
        for i in range(array.shape[0]):       # components
            for j in range(array.shape[1]):   # channels (3)
                img = transform(array[i, j])
                img.save(vis_dir / f"{i}-{j}.png")
        return psnr, ms_ssim_value

    def test(self, cluster_id: int, iter=None):
        """
        Render current color field + cluster mean and save RGB.
        """
        self.gaussian_model.eval()
        with torch.no_grad():
            color_field = self.gaussian_model.forward(cluster_id=cluster_id, render_colors=True)  # (3,H,W)
            image = color_field + self.gaussian_model.image_mean_k[cluster_id]                    # add mu_k
            mse_loss = F.mse_loss(image.float(), self.gt_image.float())
            psnr = 10 * math.log10(1.0 / (mse_loss.item() + 1e-12))
            ms_ssim_value = ms_ssim(image.unsqueeze(0).float(),
                                    self.gt_image.unsqueeze(0).float(),
                                    data_range=1, size_average=True).item()
            self.logwriter.write(f"Test [cluster {cluster_id}] PSNR:{psnr:.4f}, MS_SSIM:{ms_ssim_value:.6f}")

            # save RGB preview (inputs/outputs are in YCbCr [0,1])
            ycbcr_img = (image.detach().cpu().numpy() * 255.0).transpose(1, 2, 0)  # (H,W,3)
            rgb_img = color.ycbcr2rgb(ycbcr_img) * 255.0
            rgb_img = rgb_img.clip(0, 255).astype(np.uint8)
            img = Image.fromarray(rgb_img)
            name = self.image_name + (f"_{iter}_fitting.png" if iter is not None else "_fitting.png")
            img.save(str(self.model_dir / name))
        return psnr, ms_ssim_value


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Clustered EigenGS training (single-freq).")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True,
        help="Path to parse.py output dir (contains cluster_* artifacts)"
    )
    parser.add_argument(
        "--iterations", type=int, default=50000, help="training iterations (per phase)"
    )
    parser.add_argument(
        "--num_points", type=int, default=50000, help="2D GS points"
    )
    parser.add_argument(
    "--comp_batch", type=int, default=0,
    help="Number of components per forward/backward (0=all components). Use to limit VRAM."
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint")
    parser.add_argument("--image_path", type=str, default=None, help="Path to a single image for Phase-B")
    parser.add_argument("--seed", type=float, default=1, help="Random seed")
    parser.add_argument("--skip_train", action="store_true", help="Skip Phase-A and only run Phase-B optimize")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    trainer = GaussianTrainer(
        args, image_path=args.image_path, num_points=args.num_points,
        iterations=args.iterations, model_path=args.model_path
    )

    if not args.skip_train:
        trainer.train()      # Phase-A
    if args.image_path is not None:
        trainer.optimize()   # Phase-B

if __name__ == "__main__":
    main(sys.argv[1:])
