# import math
# import time
# from pathlib import Path
# import argparse
# import yaml
# import numpy as np
# import torch
# import sys
# from PIL import Image
# import torch.nn.functional as F
# from pytorch_msssim import ms_ssim
# from utils import *
# from tracker import LogWriter, PSNRTracker
# from tqdm import tqdm
# import random
# import torchvision.transforms as transforms
# from sklearn.decomposition import PCA
# from skimage import color
# from gaussianbasis_single_freq import GaussianBasis
# import pickle
# import uuid
# import json

# class GaussianTrainer:
#     def __init__(
#         self, args,
#         image_path = None,
#         num_points: int = 2000,
#         iterations: int = 30000,
#         model_path = None,
#     ):
#         self.dataset_path = Path(args.dataset)
#         self.device = torch.device("cuda:0")

#         # ---- NEW: read clustered metadata/artifacts ----
#         meta_path = self.dataset_path / "metadata.json"
#         with open(meta_path, "r") as f:
#             meta = json.load(f)
#         self.W, self.H = int(meta["img_size"][0]), int(meta["img_size"][1])  # img_size saved as [width, height]
#         self.num_comps = int(meta["n_comps"])
#         self.num_clusters = int(meta["k_clusters"])
#         self.comp_batch = int(getattr(args, "comp_batch", 0) or 0)

#         # raw PCA components per cluster (K, 3, C, D), means per cluster (K, 3, D)
#         self.cluster_pcas_raw = np.load(self.dataset_path / "cluster_pcas_raw.npy")   # (K, 3, C, D)
#         self.cluster_means = np.load(self.dataset_path / "cluster_pca_means.npy")     # (K, 3, D)
#         # kmeans model for assigning clusters to new images (Phase-B)
#         with open(self.dataset_path / "kmeans.pkl", "rb") as f:
#             self.kmeans = pickle.load(f)

#         random_string = str(uuid.uuid4())[:6]
#         if image_path is not None:
#             self.image_path = Path(image_path)
#             self.image_name = self.image_path.stem

#             img = Image.open(self.image_path).resize((self.W, self.H))
#             img_arr = (color.rgb2ycbcr(img) / 255.0).transpose(2, 0, 1)
#             self.gt_image = torch.tensor(img_arr, dtype=torch.float32, device=self.device)

#             model_dir = Path(f"./models/recons/{self.image_name}-{self.dataset_path.name}-{num_points}-{args.iterations}-{random_string}")
#         else:
#             model_dir = Path(f"./models/single-freq/{self.dataset_path.name}-{num_points}-{args.iterations}-{random_string}")

#         self.num_points = num_points
#         BLOCK_H, BLOCK_W = 16, 16
#         self.iterations = iterations

#         self.model_dir = model_dir
#         self.model_dir.mkdir(parents=True, exist_ok=True)

#         # ---- Model: now includes num_clusters and per-cluster means ----
#         self.gaussian_model = GaussianBasis(
#             loss_type="L2", opt_type="adan", num_points=self.num_points,
#             H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
#             device=self.device, lr=args.lr, num_comps=self.num_comps,
#             num_clusters=self.num_clusters
#         ).to(self.device)

#         # Load cluster means into the model (expects (K, 3, H*W))
#         self.gaussian_model.load_cluster_means(self.cluster_means, from_flat=True)

#         self.gaussian_model.scheduler_init()
#         self.logwriter = LogWriter(self.model_dir)
#         self.psnr_tracker = PSNRTracker(self.logwriter)
#         self.logwriter.write(f"Model Dir ID: {random_string}")

#         if model_path is not None:
#             self.model_path = Path(model_path)
#             self.logwriter.write(f"Model loaded from: {model_path}")
#             checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
#             self.gaussian_model.load_state_dict(checkpoint['model_state_dict'])

#     # ---- NEW: build GT component stack for a cluster (C,3,H,W) ----
#     def _get_cluster_target_stack(self, k: int) -> torch.Tensor:
#         """
#         cluster_pcas_raw[k]: (3, C, D)
#         -> (C, 3, H, W) torch on device
#         """
#         comps_flat = self.cluster_pcas_raw[k]  # (3, C, D)
#         comps_flat = np.transpose(comps_flat, (1, 0, 2))  # (C, 3, D)
#         stack = torch.from_numpy(comps_flat).float().to(self.device).view(self.num_comps, 3, self.H, self.W)
#         return stack

#     def train(self):
#         """
#         Phase-A: Train per-cluster bases by accumulating gradients over *all* clusters
#         (and optionally over component chunks) in each iteration, then doing one step.
#         """
#         self.gaussian_model.train()
#         progress_bar = tqdm(range(1, self.iterations + 1), desc="Training (Phase-A, all clusters)")

#         # How many component chunks per cluster?
#         if self.comp_batch and self.comp_batch > 0 and self.comp_batch < self.num_comps:
#             n_chunks = (self.num_comps + self.comp_batch - 1) // self.comp_batch
#             chunk_ranges = [range(s, min(s + self.comp_batch, self.num_comps))
#                             for s in range(0, self.num_comps, self.comp_batch)]
#         else:
#             n_chunks = 1
#             chunk_ranges = [range(self.num_comps)]  # one chunk: all components

#         K = self.num_clusters
#         N_accum = K * n_chunks  # scale each partial loss to keep LR behavior stable

#         start_time = time.time()
#         for it in range(1, self.iterations + 1):
#             self.gaussian_model.optimizer.zero_grad(set_to_none=True)

#             total_loss_val = 0.0
#             sse = 0.0  # sum of squared errors for PSNR over all clusters/chunks rendered
#             n_pix = 0  # total number of pixels*channels contributing to sse

#             for k in range(K):
#                 # Ground-truth component stack for this cluster (C,3,H,W) as torch on device
#                 gt_stack_full = self._get_cluster_target_stack(k)

#                 # Optionally iterate in component chunks to save VRAM
#                 for comp_inds in chunk_ranges:
#                     # Forward only these components
#                     pred_stack = self.gaussian_model.forward(
#                         cluster_id=k, render_colors=False, comp_indices=comp_inds
#                     )  # (len(comp_inds), 3, H, W)

#                     gt_chunk = gt_stack_full[comp_inds]  # match shape

#                     # Loss for this (cluster, chunk)
#                     loss_chunk = loss_fn(pred_stack, gt_chunk, self.gaussian_model.loss_type, lambda_value=0.7)

#                     # Scale so that the sum over all chunks & clusters ~ same magnitude each iter
#                     loss_scaled = loss_chunk / float(N_accum)
#                     loss_scaled.backward()

#                     total_loss_val += float(loss_chunk.detach().item())

#                     # accumulate SSE for PSNR (use reduction='sum' to get SSE)
#                     with torch.no_grad():
#                         sse += float(F.mse_loss(pred_stack, gt_chunk, reduction='sum').item())
#                         n_pix += pred_stack.numel()

#             # One optimizer step per iteration, after accumulating across all clusters/chunks
#             self.gaussian_model.optimizer.step()
#             self.gaussian_model.scheduler.step()

#             # PSNR over everything rendered this iter
#             psnr = 10.0 * math.log10(1.0 / (sse / (n_pix + 1e-12) + 1e-12))

#             # UI
#             progress_bar.set_postfix({
#                 "Loss(sum)": f"{total_loss_val:.7f}",
#                 "PSNR(avg)": f"{psnr:.4f},",
#                 "K": f"{K}",
#                 "chunks/cluster": f"{n_chunks}",
#             })
#             progress_bar.update(1)

#         end_time = time.time() - start_time
#         progress_bar.close()
#         self.logwriter.write("Phase-A Training Complete in {:.4f}s".format(end_time))

#         torch.save({'model_state_dict': self.gaussian_model.state_dict()},
#                   self.model_dir / "gaussian_model_phaseA.pth.tar")

#         # Optional quick viz of a cluster
#         self.vis(cluster_id=0)
#         return

#     def _predict_cluster_id_for_image(self, img_tensor_3xHxW: torch.Tensor) -> int:
#         """
#         Use saved kmeans on concatenated Y+Cb+Cr (flattened) to choose cluster for Phase-B.
#         img_tensor: (3, H, W), values in [0,1] (YCbCr)
#         """
#         arr = img_tensor_3xHxW.detach().cpu().numpy().reshape(3, -1)  # (3, D), usually float32
#         Y, Cb, Cr = arr[0], arr[1], arr[2]
#         feat = np.concatenate([Y, Cb, Cr], axis=0).reshape(1, -1)     # (1, 3D)

#         model_dtype = getattr(self.kmeans.cluster_centers_, "dtype", np.float64)
#         feat = np.ascontiguousarray(feat, dtype=model_dtype)

#         k = int(self.kmeans.predict(feat)[0])
#         print(f"cluster id: {k}")
#         return k


#     def _compute_pca_weights_for_image(self, k: int, img_tensor_3xHxW: torch.Tensor):
#         """
#         Compute per-channel PCA weights using raw components and means for cluster k:
#         w_ch = (x_ch - mean_ch) @ comps_ch^T
#         Returns three numpy arrays: wY, wCb, wCr of shape (C,)
#         """
#         arr = img_tensor_3xHxW.detach().cpu().numpy().reshape(3, -1)      # (3, D)
#         comps_k = self.cluster_pcas_raw[k]                                 # (3, C, D)
#         means_k = self.cluster_means[k]                                    # (3, D)

#         w_list = []
#         for ch in range(3):
#             x = arr[ch]              # (D,)
#             mu = means_k[ch]         # (D,)
#             comps = comps_k[ch]      # (C, D)
#             w = (x - mu) @ comps.T   # (C,)
#             w_list.append(w.astype(np.float32))
#         return w_list[0], w_list[1], w_list[2]  # wY, wCb, wCr

#     def optimize(self):
#         """
#         Phase-B: Fit colors for a single image (self.gt_image) with the right cluster.
#         """
#         assert hasattr(self, "gt_image"), "optimize() requires --image_path"

#         progress_bar = tqdm(range(1, self.iterations+1), desc="Optimizing (Phase-B) progress")

#         # pick cluster for this image
#         cluster_id = self._predict_cluster_id_for_image(self.gt_image)

#         # initialize colors from PCA weights (good warm start)
#         wY, wCb, wCr = self._compute_pca_weights_for_image(cluster_id, self.gt_image)
#         self.gaussian_model.init_colors_from_weights(cluster_id, wY, wCb, wCr)

#         self.gaussian_model.scheduler_init(optimize_phase=True)
#         self.gaussian_model.train()
#         start_time = time.perf_counter()

#         # quick test before starting
#         self.test(cluster_id=cluster_id, iter=0)
#         self.gaussian_model.train()

#         for it in range(1, 20000+1):
#             loss, psnr = self.gaussian_model.optimize_iter(self.gt_image, cluster_id=cluster_id)

#             with torch.no_grad():
#                 if it in [10,100,500,1000,2000,3000,4000,10000, 15000,20000]:
#                     self.test(cluster_id=cluster_id, iter=it)
#                     self.gaussian_model.train()
#                 if it % 10 == 0:
#                     progress_bar.set_postfix({
#                         "Cluster": f"{cluster_id}",
#                         "Loss": f"{loss.item():.7f}",
#                         "PSNR": f"{psnr:.4f},"
#                     })
#                     progress_bar.update(10)

#         end_time = time.perf_counter() - start_time
#         progress_bar.close()
#         self.logwriter.write("Phase-B Optimizing Complete in {:.4f}s".format(end_time))
#         torch.save({'model_state_dict': self.gaussian_model.state_dict()}, self.model_dir / "gaussian_model_phaseB.pth.tar")

#         # final test
#         self.test(cluster_id=cluster_id)
#         return

#     def vis(self, cluster_id: int = 0):
#         """
#         Visualize learned components for a chosen cluster.
#         """
#         self.gaussian_model.eval()
#         with torch.no_grad():
#             pred_stack = self.gaussian_model.forward(cluster_id=cluster_id, render_colors=False)  # (C,3,H,W)

#         gt_stack = self._get_cluster_target_stack(cluster_id)
#         mse_loss = F.mse_loss(pred_stack.float(), gt_stack.float())
#         psnr = 10 * math.log10(1.0 / (mse_loss.item() + 1e-12))
#         # ms-ssim over stacks (treat C as batch by flattening):
#         # here we compute mean of per-component ms-ssim for a quick score
#         ms_list = []
#         for c in range(pred_stack.shape[0]):
#             ms = ms_ssim(pred_stack[c].unsqueeze(0).float(),
#                          gt_stack[c].unsqueeze(0).float(),
#                          data_range=1, size_average=True).item()
#             ms_list.append(ms)
#         ms_ssim_value = float(np.mean(ms_list))
#         self.logwriter.write(f"[Cluster {cluster_id}] Components Fitting: PSNR:{psnr:.4f}, MS_SSIM:{ms_ssim_value:.6f}")

#         vis_dir = self.model_dir / f"vis_comps_cluster_{cluster_id}"
#         vis_dir.mkdir(parents=True, exist_ok=True)
#         transform = transforms.ToPILImage()
#         array = pred_stack.float()
#         for i in range(array.shape[0]):       # components
#             for j in range(array.shape[1]):   # channels (3)
#                 img = transform(array[i, j])
#                 img.save(vis_dir / f"{i}-{j}.png")
#         return psnr, ms_ssim_value

#     def test(self, cluster_id: int, iter=None):
#         """
#         Render current color field + cluster mean and save RGB.
#         """
#         self.gaussian_model.eval()
#         with torch.no_grad():
#             color_field = self.gaussian_model.forward(cluster_id=cluster_id, render_colors=True)  # (3,H,W)
#             image = color_field + self.gaussian_model.image_mean_k[cluster_id]                    # add mu_k
#             mse_loss = F.mse_loss(image.float(), self.gt_image.float())
#             psnr = 10 * math.log10(1.0 / (mse_loss.item() + 1e-12))
#             ms_ssim_value = ms_ssim(image.unsqueeze(0).float(),
#                                     self.gt_image.unsqueeze(0).float(),
#                                     data_range=1, size_average=True).item()
#             self.logwriter.write(f"Test [cluster {cluster_id}] PSNR:{psnr:.4f}, MS_SSIM:{ms_ssim_value:.6f}")

#             # save RGB preview (inputs/outputs are in YCbCr [0,1])
#             ycbcr_img = (image.detach().cpu().numpy() * 255.0).transpose(1, 2, 0)  # (H,W,3)
#             rgb_img = color.ycbcr2rgb(ycbcr_img) * 255.0
#             rgb_img = rgb_img.clip(0, 255).astype(np.uint8)
#             img = Image.fromarray(rgb_img)
#             name = self.image_name + (f"_{iter}_fitting.png" if iter is not None else "_fitting.png")
#             img.save(str(self.model_dir / name))
#         return psnr, ms_ssim_value


# def parse_args(argv):
#     parser = argparse.ArgumentParser(description="Clustered EigenGS training (single-freq).")
#     parser.add_argument(
#         "-d", "--dataset", type=str, required=True,
#         help="Path to parse.py output dir (contains cluster_* artifacts)"
#     )
#     parser.add_argument(
#         "--iterations", type=int, default=50000, help="training iterations (per phase)"
#     )
#     parser.add_argument(
#         "--num_points", type=int, default=50000, help="2D GS points"
#     )
#     parser.add_argument(
#     "--comp_batch", type=int, default=0,
#     help="Number of components per forward/backward (0=all components). Use to limit VRAM."
#     )
#     parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint")
#     parser.add_argument("--image_path", type=str, default=None, help="Path to a single image for Phase-B")
#     parser.add_argument("--seed", type=float, default=1, help="Random seed")
#     parser.add_argument("--skip_train", action="store_true", help="Skip Phase-A and only run Phase-B optimize")
#     parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
#     args = parser.parse_args(argv)
#     return args

# def main(argv):
#     args = parse_args(argv)

#     if args.seed is not None:
#         torch.manual_seed(args.seed)
#         random.seed(args.seed)
#         torch.cuda.manual_seed(args.seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#         np.random.seed(args.seed)

#     trainer = GaussianTrainer(
#         args, image_path=args.image_path, num_points=args.num_points,
#         iterations=args.iterations, model_path=args.model_path
#     )

#     if not args.skip_train:
#         trainer.train()      # Phase-A
#     if args.image_path is not None:
#         trainer.optimize()   # Phase-B

# if __name__ == "__main__":
#     main(sys.argv[1:])
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

# === NEW ===
import os, copy, wandb
from typing import List

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class GaussianTrainer:
    def __init__(
        self, args,
        image_path = None,
        num_points: int = 2000,
        iterations: int = 30000,
        model_path = None,
    ):
        self.args = args  # NEW
        self.dataset_path = Path(args.dataset)
        self.device = torch.device("cuda:0")

        # ---- clustered metadata/artifacts ----
        meta_path = self.dataset_path / "metadata.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.W, self.H = int(meta["img_size"][0]), int(meta["img_size"][1])  # [W,H]
        self.num_comps = int(meta["n_comps"])
        self.num_clusters = int(meta["k_clusters"])
        self.comp_batch = int(getattr(args, "comp_batch", 0) or 0)

        # raw PCA components per cluster (K, 3, C, D), means per cluster (K, 3, D)
        self.cluster_pcas_raw = np.load(self.dataset_path / "cluster_pcas_raw.npy")   # (K, 3, C, D)
        self.cluster_means = np.load(self.dataset_path / "cluster_pca_means.npy")     # (K, 3, D)
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
            self.image_path = None
            self.image_name = None
            self.gt_image = None
            model_dir = Path(f"./models/single-freq/{self.dataset_path.name}-{num_points}-{args.iterations}-{random_string}")

        self.num_points = num_points
        BLOCK_H, BLOCK_W = 16, 16
        self.iterations = iterations

        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # ---- Model ----
        self.gaussian_model = GaussianBasis(
            loss_type="L2", opt_type="adan", num_points=self.num_points,
            H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            device=self.device, lr=args.lr, num_comps=self.num_comps,
            num_clusters=self.num_clusters
        ).to(self.device)

        # Load per-cluster means (expects (K,3,H*W))
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

        # === NEW === W&B
        self.wandb_run = None
        if args.wandb:
            run_name = args.run_name or f"{self.dataset_path.name}-K{self.num_clusters}-C{self.num_comps}-P{self.num_points}-I{self.iterations}-{random_string}"
            self.wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                name=run_name,
                config={
                    "dataset": str(self.dataset_path),
                    "img_size": [self.W, self.H],
                    "num_points": self.num_points,
                    "iterations": self.iterations,
                    "lr": args.lr,
                    "k_clusters": self.num_clusters,
                    "n_comps": self.num_comps,
                    "comp_batch": self.comp_batch,
                    "eval_every": args.eval_every,
                    "eval_opt_iters": args.eval_opt_iters,
                    "eval_images": args.eval_images,
                    "seed": args.seed,
                },
                dir=str(self.model_dir),
                reinit=False,
            )
            if args.wandb_watch != "off":
                wandb.watch(self.gaussian_model, log=args.wandb_watch, log_freq=args.wandb_log_freq, log_graph=args.wandb_log_graph)

        # === NEW === test image cache/RNG for sampling
        self._test_images_cache = None
        self._rng = np.random.default_rng(args.seed if args.seed is not None else 0)

    # ---- helpers (NEW) ----
    def _discover_test_images(self) -> List[Path]:
        if self._test_images_cache is not None:
            return self._test_images_cache

        if self.args.test_glob:
            candidates = sorted(Path().glob(self.args.test_glob))
        elif self.args.test_dir:
            root = Path(self.args.test_dir)
            if not root.exists():
                raise FileNotFoundError(f"--test_dir not found: {root}")
            it = root.rglob("*") if self.args.test_recursive else root.glob("*")
            candidates = sorted([p for p in it if p.suffix.lower() in IMG_EXTS and p.is_file()])
        else:
            candidates = []

        self._test_images_cache = candidates
        return candidates

    def _load_ycbcr_tensor(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((self.W, self.H))
        ycbcr = (color.rgb2ycbcr(np.array(img)) / 255.0).transpose(2, 0, 1)
        return torch.tensor(ycbcr, dtype=torch.float32, device=self.device)

    # ---- existing training (Phase-A) ----
    def _get_cluster_target_stack(self, k: int) -> torch.Tensor:
        comps_flat = self.cluster_pcas_raw[k]  # (3, C, D)
        comps_flat = np.transpose(comps_flat, (1, 0, 2))  # (C, 3, D)
        stack = torch.from_numpy(comps_flat).float().to(self.device).view(self.num_comps, 3, self.H, self.W)
        return stack

    def train(self):
        self.gaussian_model.train()
        progress_bar = tqdm(range(1, self.iterations + 1), desc="Training (Phase-A, all clusters)")

        # chunking of components
        if self.comp_batch and 0 < self.comp_batch < self.num_comps:
            chunk_ranges = [range(s, min(s + self.comp_batch, self.num_comps))
                            for s in range(0, self.num_comps, self.comp_batch)]
        else:
            chunk_ranges = [range(self.num_comps)]
        n_chunks = len(chunk_ranges)
        K = self.num_clusters
        N_accum = K * n_chunks

        start_time = time.time()
        for it in range(1, self.iterations + 1):
            self.gaussian_model.optimizer.zero_grad(set_to_none=True)

            total_loss_val = 0.0
            sse = 0.0
            n_pix = 0

            for k in range(K):
                gt_stack_full = self._get_cluster_target_stack(k)
                for comp_inds in chunk_ranges:
                    pred_stack = self.gaussian_model.forward(cluster_id=k, render_colors=False, comp_indices=comp_inds)
                    gt_chunk = gt_stack_full[comp_inds]
                    loss_chunk = loss_fn(pred_stack, gt_chunk, self.gaussian_model.loss_type, lambda_value=0.7)
                    (loss_chunk / float(N_accum)).backward()
                    total_loss_val += float(loss_chunk.detach().item())
                    with torch.no_grad():
                        sse += float(F.mse_loss(pred_stack, gt_chunk, reduction='sum').item())
                        n_pix += pred_stack.numel()

            self.gaussian_model.optimizer.step()
            self.gaussian_model.scheduler.step()

            psnr = 10.0 * math.log10(1.0 / (sse / (n_pix + 1e-12) + 1e-12))

            # W&B scalars
            if self.wandb_run:
                wandb.log({"train/loss": total_loss_val, "train/psnr": psnr, "iter": it})

            progress_bar.set_postfix({
                "Loss(sum)": f"{total_loss_val:.7f}",
                "PSNR(avg)": f"{psnr:.4f},",
                "K": f"{K}",
                "chunks/cluster": f"{n_chunks}",
            })
            progress_bar.update(1)

            # === NEW === periodic Phase-B eval on test images (log to W&B)
            if self.args.eval_every and (it % self.args.eval_every == 0):
                try:
                    self.eval_phaseB_on_test_set(global_iter=it)
                    self._save_checkpoint(step=it, tag="phaseA")
                except Exception as e:
                    self.logwriter.write(f"[eval warning] eval failed at iter {it}: {e}")

        end_time = time.time() - start_time
        progress_bar.close()
        self.logwriter.write("Phase-A Training Complete in {:.4f}s".format(end_time))

        torch.save({'model_state_dict': self.gaussian_model.state_dict()},
                   self.model_dir / "gaussian_model_phaseA.pth.tar")
        if self.wandb_run:
            wandb.save(str(self.model_dir / "gaussian_model_phaseA.pth.tar"), base_path=str(self.model_dir))

        self.vis(cluster_id=0)
        return

    # ---- Phase-B for a provided single image (your original) ----
    def _predict_cluster_id_for_image(self, img_tensor_3xHxW: torch.Tensor) -> int:
        arr = img_tensor_3xHxW.detach().cpu().numpy().reshape(3, -1)
        feat = np.concatenate([arr[0], arr[1], arr[2]], axis=0).reshape(1, -1)
        model_dtype = getattr(self.kmeans.cluster_centers_, "dtype", np.float64)
        feat = np.ascontiguousarray(feat, dtype=model_dtype)
        k = int(self.kmeans.predict(feat)[0])
        print(f"cluster id: {k}")
        return k

    def _compute_pca_weights_for_image(self, k: int, img_tensor_3xHxW: torch.Tensor):
        arr = img_tensor_3xHxW.detach().cpu().numpy().reshape(3, -1)      # (3, D)
        comps_k = self.cluster_pcas_raw[k]                                 # (3, C, D)
        means_k = self.cluster_means[k]                                    # (3, D)
        w_list = []
        for ch in range(3):
            x = arr[ch]; mu = means_k[ch]; comps = comps_k[ch]             # x(D,), comps(C,D)
            w = (x - mu) @ comps.T                                        # (C,)
            w_list.append(w.astype(np.float32))
        return w_list[0], w_list[1], w_list[2]  # wY, wCb, wCr

    def optimize(self):
        assert hasattr(self, "gt_image"), "optimize() requires --image_path"
        progress_bar = tqdm(range(1, self.iterations+1), desc="Optimizing (Phase-B) progress")

        cluster_id = self._predict_cluster_id_for_image(self.gt_image)
        wY, wCb, wCr = self._compute_pca_weights_for_image(cluster_id, self.gt_image)
        self.gaussian_model.init_colors_from_weights(cluster_id, wY, wCb, wCr)

        self.gaussian_model.scheduler_init(optimize_phase=True)
        self.gaussian_model.train()
        start_time = time.perf_counter()

        self.test(cluster_id=cluster_id, iter=0)
        self.gaussian_model.train()

        for it in range(1, 200+1):
            loss, psnr = self.gaussian_model.optimize_iter(self.gt_image, cluster_id=cluster_id)

            if self.wandb_run:
                wandb.log({"phaseB/loss": float(loss.item()), "phaseB/psnr": float(psnr), "phaseB/iter": it})

            with torch.no_grad():
                if it in [10,100,500,1000,2000,3000,4000,10000,15000,20000]:
                    self.test(cluster_id=cluster_id, iter=it)
                    self.gaussian_model.train()
                if it % 10 == 0:
                    progress_bar.set_postfix({"Cluster": f"{cluster_id}", "Loss": f"{loss.item():.7f}", "PSNR": f"{psnr:.4f},"})
                    progress_bar.update(10)

        end_time = time.perf_counter() - start_time
        progress_bar.close()
        self.logwriter.write("Phase-B Optimizing Complete in {:.4f}s".format(end_time))
        torch.save({'model_state_dict': self.gaussian_model.state_dict()}, self.model_dir / "gaussian_model_phaseB.pth.tar")
        if self.wandb_run:
            wandb.save(str(self.model_dir / "gaussian_model_phaseB.pth.tar"), base_path=str(self.model_dir))

        self.test(cluster_id=cluster_id)
        return

    # ---- vis / test (unchanged) ----
    def vis(self, cluster_id: int = 0):
        self.gaussian_model.eval()
        with torch.no_grad():
            pred_stack = self.gaussian_model.forward(cluster_id=cluster_id, render_colors=False)  # (C,3,H,W)
        gt_stack = self._get_cluster_target_stack(cluster_id)
        mse_loss = F.mse_loss(pred_stack.float(), gt_stack.float())
        psnr = 10 * math.log10(1.0 / (mse_loss.item() + 1e-12))
        ms_list = []
        for c in range(pred_stack.shape[0]):
            ms = ms_ssim(pred_stack[c].unsqueeze(0).float(), gt_stack[c].unsqueeze(0).float(), data_range=1, size_average=True).item()
            ms_list.append(ms)
        ms_ssim_value = float(np.mean(ms_list))
        self.logwriter.write(f"[Cluster {cluster_id}] Components Fitting: PSNR:{psnr:.4f}, MS_SSIM:{ms_ssim_value:.6f}")

        vis_dir = self.model_dir / f"vis_comps_cluster_{cluster_id}"
        vis_dir.mkdir(parents=True, exist_ok=True)
        transform = transforms.ToPILImage()
        array = pred_stack.float()
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                img = transform(array[i, j])
                img.save(vis_dir / f"{i}-{j}.png")
        return psnr, ms_ssim_value

    def test(self, cluster_id: int, iter=None):
        self.gaussian_model.eval()
        with torch.no_grad():
            color_field = self.gaussian_model.forward(cluster_id=cluster_id, render_colors=True)  # (3,H,W)
            image = color_field + self.gaussian_model.image_mean_k[cluster_id]
            mse_loss = F.mse_loss(image.float(), self.gt_image.float())
            psnr = 10 * math.log10(1.0 / (mse_loss.item() + 1e-12))
            ms_ssim_value = ms_ssim(image.unsqueeze(0).float(), self.gt_image.unsqueeze(0).float(), data_range=1, size_average=True).item()
            self.logwriter.write(f"Test [cluster {cluster_id}] PSNR:{psnr:.4f}, MS_SSIM:{ms_ssim_value:.6f}")

            ycbcr_img = (image.detach().cpu().numpy() * 255.0).transpose(1, 2, 0)
            rgb_img = (color.ycbcr2rgb(ycbcr_img) * 255.0).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(rgb_img)
            name = self.image_name + (f"_{iter}_fitting.png" if iter is not None else "_fitting.png")
            img.save(str(self.model_dir / name))
        return psnr, ms_ssim_value

    # === NEW === periodic Phase-B eval on a set of test images (does not touch the training model)
    def _save_checkpoint(self, step: int, tag: str):
        ckpt = {
            "step": step,
            "model_state_dict": self.gaussian_model.state_dict(),
            "optimizer_state_dict": getattr(self.gaussian_model, "optimizer", None).state_dict() if hasattr(self.gaussian_model, "optimizer") else None,
            "scheduler_state_dict": getattr(self.gaussian_model, "scheduler", None).state_dict() if hasattr(self.gaussian_model, "scheduler") else None,
            "args": vars(self.args),
        }
        path = self.model_dir / f"ckpt_{tag}_{step:06d}.pth.tar"
        torch.save(ckpt, path)
        self.logwriter.write(f"[ckpt] saved {path}")
        if self.wandb_run:
            wandb.save(str(path), base_path=str(self.model_dir))

    def _phaseB_eval_single(self, base_state_dict, img_path: Path, opt_iters: int):
        # fresh model copy
        model = GaussianBasis(
            loss_type="L2", opt_type="adan", num_points=self.num_points,
            H=self.H, W=self.W, BLOCK_H=16, BLOCK_W=16,
            device=self.device, lr=self.args.lr, num_comps=self.num_comps, num_clusters=self.num_clusters
        ).to(self.device)
        model.load_state_dict(base_state_dict)
        model.load_cluster_means(self.cluster_means, from_flat=True)
        model.eval()

        gt_img = self._load_ycbcr_tensor(img_path)  # (3,H,W)
        k = self._predict_cluster_id_for_image(gt_img)

        # PCA weight init → colors
        wY, wCb, wCr = self._compute_pca_weights_for_image(k, gt_img)
        model.init_colors_from_weights(k, wY, wCb, wCr)

        # metrics at init
        with torch.no_grad():
            out0 = model.forward(cluster_id=k, render_colors=True)  # (3,H,W)
            img0 = (out0 + model.image_mean_k[k]).clamp(0,1)
            mse0 = F.mse_loss(img0, gt_img).item()
            psnr0 = 10 * math.log10(1.0 / (mse0 + 1e-12))
            ssim0 = ms_ssim(img0.unsqueeze(0), gt_img.unsqueeze(0), data_range=1, size_average=True).item()

        # short refine
        model.scheduler_init(optimize_phase=True)
        model.train()
        for _ in range(opt_iters):
            _loss, _ = model.optimize_iter(gt_img, cluster_id=k)

        # final metrics
        model.eval()
        with torch.no_grad():
            outF = model.forward(cluster_id=k, render_colors=True)
            imgF = (outF + model.image_mean_k[k]).clamp(0,1)
            mseF = F.mse_loss(imgF, gt_img).item()
            psnrF = 10 * math.log10(1.0 / (mseF + 1e-12))
            ssimF = ms_ssim(imgF.unsqueeze(0), gt_img.unsqueeze(0), data_range=1, size_average=True).item()

        # RGB preview (final)
        ycbcr_np = (imgF.detach().cpu().numpy() * 255.0).transpose(1, 2, 0)
        rgb = (color.ycbcr2rgb(ycbcr_np) * 255.0).clip(0,255).astype(np.uint8)
        from PIL import Image as _PILImage
        rgb_final = _PILImage.fromarray(rgb)

        return {
            "path": str(img_path),
            "cluster": int(k),
            "loss_init": mse0, "psnr_init": psnr0, "ssim_init": ssim0,
            "loss_final": mseF, "psnr_final": psnrF, "ssim_final": ssimF,
            "rgb_final": rgb_final,
        }

    def eval_phaseB_on_test_set(self, global_iter: int):
        if not self.wandb_run:
            return
        all_candidates = self._discover_test_images()
        if len(all_candidates) == 0:
            self.logwriter.write("[eval warning] no test images found; skipping eval")
            return

        ksel = min(self.args.eval_images, len(all_candidates))
        idxs = self._rng.choice(len(all_candidates), size=ksel, replace=False)
        selected = [all_candidates[i] for i in idxs]

        base_state = copy.deepcopy(self.gaussian_model.state_dict())

        rows = []
        gallery = []
        for p in selected:
            res = self._phaseB_eval_single(base_state, p, opt_iters=self.args.eval_opt_iters)
            rows.append([
                res["path"], res["cluster"],
                res["psnr_init"], res["ssim_init"], res["loss_init"],
                res["psnr_final"], res["ssim_final"], res["loss_final"],
            ])
            gallery.append(wandb.Image(res["rgb_final"],
                                       caption=f"{Path(res['path']).name}  k={res['cluster']}  PSNR:{res['psnr_final']:.2f} SSIM:{res['ssim_final']:.4f}"))

        table = wandb.Table(
            data=rows,
            columns=["path","cluster","psnr_init","ssim_init","loss_init","psnr_final","ssim_final","loss_final"]
        )

        arr = np.array(rows, dtype=object)
        psnrF = np.array(arr[:,5], dtype=float)
        ssimF = np.array(arr[:,6], dtype=float)
        lossF = np.array(arr[:,7], dtype=float)

        wandb.log({
            "iter": global_iter,
            "eval_kmeans/num_images": int(ksel),
            "eval_kmeans/psnr_final_mean": float(psnrF.mean()),
            "eval_kmeans/psnr_final_std": float(psnrF.std()),
            "eval_kmeans/ssim_final_mean": float(ssimF.mean()),
            "eval_kmeans/loss_final_mean": float(lossF.mean()),
            "eval_kmeans/gallery": gallery,
            "eval_kmeans/table": table,
        }, step=global_iter)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Clustered EigenGS training (single-freq).")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="Path to parse.py output dir (contains cluster_* artifacts)")
    parser.add_argument("--iterations", type=int, default=50000, help="training iterations (per phase)")
    parser.add_argument("--num_points", type=int, default=50000, help="2D GS points")
    parser.add_argument("--comp_batch", type=int, default=0,
                        help="Number of components per forward/backward (0=all components). Use to limit VRAM.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint")
    parser.add_argument("--image_path", type=str, default=None, help="Path to a single image for Phase-B")
    parser.add_argument("--seed", type=float, default=1, help="Random seed")
    parser.add_argument("--skip_train", action="store_true", help="Skip Phase-A and only run Phase-B optimize")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # === NEW === W&B + eval controls
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default=os.getenv("WANDB_PROJECT", "eigens"), help="W&B project")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (optional)")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name (optional)")
    parser.add_argument("--wandb_watch", choices=["gradients","parameters","all","off"], default="off")
    parser.add_argument("--wandb_log_freq", type=int, default=1000)
    parser.add_argument("--wandb_log_graph", action="store_true")
    parser.add_argument("--eval_every", type=int, default=1000, help="Eval cadence in iters during Phase-A")
    parser.add_argument("--eval_images", type=int, default=10, help="# of test images to evaluate each time")
    parser.add_argument("--eval_opt_iters", type=int, default=200, help="Short Phase-B iters per eval image")
    parser.add_argument("--test_glob", type=str, default=None, help="Glob for test images, e.g. './test/*.png'")
    parser.add_argument("--test_dir", type=str, default=None, help="Directory that contains test images (non-recursive).")
    parser.add_argument("--test_recursive", action="store_true", help="Recurse into subfolders of --test_dir")

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
