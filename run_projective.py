# train_eigens_clustered.py
# Clustered EigenGS training & evaluation (YCbCr per-channel projective clustering only).
#
# Usage (clustered; auto-detected by parse_emlike.py layout):
#   python train_eigens_clustered.py -d ./out/<UUID>-K6-J50/ --iterations 50000 --num_points 50000
# Optional:
#   --test_dir ./some_images/ --eval_every 2000 --eval_opt_iters 200 --psnr_threshold 35

import math
import time
from pathlib import Path
import argparse
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import *  # assumes loss_fn, etc.
from tracker import LogWriter, PSNRTracker
from tqdm import tqdm
import random
import torchvision.transforms as transforms
from skimage import color
import uuid
import copy
import json
import os
from typing import List, Dict, Any

# ====== OPTIONAL W&B ======
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

# ====== MODEL IMPORT ======
# If your class is in a different file, update this import.
from gaussianbasis_projective import GaussianBasis

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class GaussianTrainer:
    def __init__(
        self, args,
        image_path: str | None = None,
        num_points: int = 2000,
        iterations: int = 30000,
        model_path: str | None = None,
    ):
        self.args = args
        self.dataset_path = Path(args.dataset)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # ---- Enforce clustered dataset (layout from parse_emlike.py) ----
        must_exist = [
            self.dataset_path / "meta.json",
            self.dataset_path / "Y",
            self.dataset_path / "Cb",
            self.dataset_path / "Cr",
        ]
        if not all(p.exists() for p in must_exist):
            raise FileNotFoundError(
                "Clustered dataset expected. Make sure the directory contains "
                "meta.json and subfolders 'Y', 'Cb', 'Cr' produced by parse_emlike.py."
            )

        # === CLUSTERED PATH ===
        # Top-level meta has K and J (per-channel)
        with open(self.dataset_path / "meta.json", "r") as f:
            top_meta = json.load(f)
        self.K = int(top_meta["K"])
        self.J = int(top_meta["J"])

        def _load_cluster_channel(ch_name: str) -> List[np.ndarray]:
            clusters_dir = self.dataset_path / ch_name / "clusters"
            if not clusters_dir.exists():
                raise FileNotFoundError(f"Expected {clusters_dir} for clustered dataset.")
            clusters = sorted(clusters_dir.glob("cluster_*"))
            arr_list = []
            for cdir in clusters:
                A = np.load(cdir / "arrs.npy")  # (J,H,W), normalized [0,1] for visualization
                arr_list.append(A)
            return arr_list  # K entries

        Ys  = _load_cluster_channel("Y")
        Cbs = _load_cluster_channel("Cb")
        Crs = _load_cluster_channel("Cr")
        assert len(Ys) == len(Cbs) == len(Crs) == self.K

        H, W = Ys[0].shape[1], Ys[0].shape[2]
        planes = []
        # Build training stack: (3*K*J, 3, H, W), only one active channel per plane
        for c_idx, pack in enumerate([Ys, Cbs, Crs]):
            for k in range(self.K):
                for j in range(self.J):
                    img3 = np.zeros((3, H, W), dtype=np.float32)
                    img3[c_idx] = pack[k][j]  # put normalized eigenimage in its own channel
                    planes.append(img3)
        self.gt_arrs = torch.from_numpy(np.stack(planes, axis=0)).to(self.device)

        # cache per-channel min/max over raw PCA bases (not normalized arrs)
        self._vmin = np.zeros(3, dtype=np.float32)
        self._vmax = np.zeros(3, dtype=np.float32)
        for c_idx, ch_name in enumerate(["Y", "Cb", "Cr"]):
            mins, maxs = [], []
            for cdir in sorted((self.dataset_path / ch_name / "clusters").glob("cluster_*")):
                bases = np.load(cdir / "pca_bases.npy")  # (J,d)
                mins.append(bases.min())
                maxs.append(bases.max())
            self._vmin[c_idx] = float(np.min(mins))
            self._vmax[c_idx] = float(np.max(maxs))

        random_string = str(uuid.uuid4())[:6]

        if image_path is not None:
            self.image_path = Path(image_path)
            self.image_name = self.image_path.stem
            img = Image.open(self.image_path).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
            img_arr = (color.rgb2ycbcr(np.array(img)) / 255.0).transpose(2, 0, 1)
            self.gt_image = torch.tensor(img_arr, dtype=torch.float32, device=self.device)
            model_dir = Path(f"./models/recons/{self.image_name}-{self.dataset_path.name}-{num_points}-{args.iterations}-{random_string}")
        else:
            self.image_path = None
            self.image_name = None
            self.gt_image = None
            model_dir = Path(f"./models/clustered/{self.dataset_path.name}-{num_points}-{args.iterations}-{random_string}")

        self.num_points = num_points
        self.num_comps = 3 * self.K * self.J  # number of planes in phase-A target
        self.H, self.W = int(H), int(W)
        BLOCK_H, BLOCK_W = 16, 16
        self.iterations = iterations

        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # ---- Model ----
        self.gaussian_model = GaussianBasis(
            loss_type="L2", opt_type="adan", num_points=self.num_points,
            H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            device=self.device, lr=args.lr,
            num_comps=self.num_comps,
            clustered=True, K_clusters=self.K, J_dim=self.J
        ).to(self.device)
        self.gaussian_model.scheduler_init()
        self.logwriter = LogWriter(self.model_dir)
        self.psnr_tracker = PSNRTracker(self.logwriter)
        self.logwriter.write(f"Model Dir ID: {random_string}")

        if model_path is not None:
            self.model_path = Path(model_path)
            self.logwriter.write(f"Model loaded from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.gaussian_model.load_state_dict(checkpoint['model_state_dict'])

        # ---- W&B ----
        self.wandb_run = None
        if getattr(args, "wandb", False) and WANDB_AVAILABLE:
            run_name = args.run_name or f"{self.dataset_path.name}-P{self.num_points}-I{self.iterations}-{random_string}"
            self.wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                name=run_name,
                config={
                    "dataset": str(self.dataset_path),
                    "num_points": self.num_points,
                    "iterations": self.iterations,
                    "lr": args.lr,
                    "eval_every": args.eval_every,
                    "eval_opt_iters": args.eval_opt_iters,
                    "eval_images": args.eval_images,
                    "seed": args.seed,
                    "K": self.K,
                    "J": self.J,
                    "psnr_threshold": args.psnr_threshold,
                },
                dir=str(self.model_dir),
                reinit=False,
            )
            wandb.watch(self.gaussian_model, log=None, log_freq=0, log_graph=False)

        self._test_images_cache = None
        self._rng = np.random.default_rng(args.seed if args.seed is not None else 0)

    # ====== Utility: discover test images ======
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
            # fallback to common subdirs under dataset path
            subdirs = ["test_imgs", "test", "images/test", "val_imgs", "val", "Test", "Val", "validation"]
            found = []
            for sd in subdirs:
                p = self.dataset_path / sd
                if p.exists():
                    found.extend(sorted([q for q in p.rglob("*") if q.suffix.lower() in IMG_EXTS and q.is_file()]))
            if not found:
                found = sorted([q for q in self.dataset_path.rglob("*")
                                if q.suffix.lower() in IMG_EXTS and q.is_file() and ("models" not in q.parts)])
            candidates = found

        self._test_images_cache = candidates
        return candidates

    # ====== Image helpers ======
    def _load_ycbcr_tensor(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((self.W, self.H), Image.Resampling.LANCZOS)
        ycbcr = (color.rgb2ycbcr(np.array(img)) / 255.0).transpose(2, 0, 1)
        return torch.tensor(ycbcr, dtype=torch.float32, device=self.device)

    def _ycbcr_to_rgb_image(self, ycbcr_01: torch.Tensor) -> Image.Image:
        ycbcr_img = (ycbcr_01.clamp(0, 1).detach().cpu().numpy() * 255.0).transpose(1, 2, 0)
        rgb = color.ycbcr2rgb(ycbcr_img) * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb)

    # ====== Checkpoint ======
    def _save_checkpoint(self, step: int, tag: str):
        ckpt = {
            "step": step,
            "model_state_dict": self.gaussian_model.state_dict(),
            "optimizer_state_dict": getattr(self.gaussian_model, "optimizer", None).state_dict()
                if hasattr(self.gaussian_model, "optimizer") else None,
            "scheduler_state_dict": getattr(self.gaussian_model, "scheduler", None).state_dict()
                if hasattr(self.gaussian_model, "scheduler") else None,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "args": vars(self.args),
        }
        path = self.model_dir / f"ckpt_{tag}_{step:06d}.pth.tar"
        torch.save(ckpt, path)
        self.logwriter.write(f"[ckpt] saved {path}")
        if self.wandb_run:
            wandb.save(str(path), base_path=str(self.model_dir))

    # ====== Cluster selection helper (distance to affine J-flat) ======
    def _assign_cluster_for_channel(self, ch_name: str, x_flat: np.ndarray) -> int:
        """
        Pick nearest cluster for a 1×d vector using EM flats from parse_emlike.py:
          em_Vs.npy: (K,J,d) with row-orthonormal basis rows
          em_vs.npy: (K,d)   with affine offset v
        """
        ch_root = self.dataset_path / ch_name
        Vs = np.load(ch_root / "em_Vs.npy")   # (K,J,d)
        vs = np.load(ch_root / "em_vs.npy")   # (K,d)
        x = torch.from_numpy(x_flat.astype(np.float32)).to(self.device)  # (d,)
        Vs_t = torch.from_numpy(Vs.astype(np.float32)).to(self.device)   # (K,J,d)
        vs_t = torch.from_numpy(vs.astype(np.float32)).to(self.device)   # (K,d)

        diffs = x[None, :] - vs_t                 # (K,d)
        y_v = torch.einsum('kjd,kd->kj', Vs_t, diffs)  # (K,J)
        y_rec = torch.einsum('kjd,kj->kd', Vs_t, y_v)  # (K,d)
        resid = diffs - y_rec                          # (K,d)
        d2 = (resid ** 2).sum(dim=1)                   # (K,)
        return int(torch.argmin(d2).item())

    # ====== Training loop (Phase-A) ======
    def train(self):
        progress_bar = tqdm(range(1, self.iterations + 1), desc="Training progress")
        self.gaussian_model.train()
        start_time = time.time()

        for iter_i in range(1, self.iterations + 1):
            loss, psnr = self.gaussian_model.train_iter(self.gt_arrs)

            if self.wandb_run:
                wandb.log({"train/loss": float(loss.item()), "train/psnr": float(psnr), "iter": iter_i})

            with torch.no_grad():
                progress_bar.set_postfix({f"Loss": f"{loss.item():.{7}f}", "PSNR": f"{psnr:.{4}f},"})
                progress_bar.update(1)

            if self.args.eval_every and (iter_i % self.args.eval_every == 0):
                # periodic short Phase-B eval on a fixed test set
                try:
                    self.eval_phaseB_on_test_set(global_iter=iter_i)
                    self._save_checkpoint(step=iter_i, tag="phaseA")
                except Exception as e:
                    self.logwriter.write(f"[eval warning] eval failed at iter {iter_i}: {e}")

        elapsed = time.time() - start_time
        progress_bar.close()
        self.logwriter.write(f"Training Complete in {elapsed:.4f}s")
        torch.save({'model_state_dict': self.gaussian_model.state_dict()}, self.model_dir / "gaussian_model.pth.tar")
        if self.wandb_run:
            wandb.save(str(self.model_dir / "gaussian_model.pth.tar"), base_path=str(self.model_dir))
        self.vis()
        return

    # ====== Phase-B long optimize for a single image ======
    def optimize(self):
        progress_bar = tqdm(range(1, self.iterations + 1), desc="Optimizing progress")
        self.update_gaussian()  # sets colors/means/scale/shift (clustered init)
        self.gaussian_model.scheduler_init(optimize_phase=True)
        self.gaussian_model.train()
        start_time = time.perf_counter()

        self.test(iter=0)
        self.gaussian_model.train()
        for iter_i in range(1, self.iterations + 1):
            loss, psnr = self.gaussian_model.optimize_iter(self.gt_image)
            self.psnr_tracker.check(start_time, psnr, iter_i)

            if self.wandb_run:
                wandb.log({"phaseB/loss": float(loss.item()), "phaseB/psnr": float(psnr), "phaseB/iter": iter_i})

            with torch.no_grad():
                if iter_i in [10, 100, 1000]:
                    self.test(iter=iter_i)
                    self.gaussian_model.train()
                if iter_i % 10 == 0:
                    progress_bar.set_postfix({f"Loss": f"{loss.item():.{7}f}", "PSNR": f"{psnr:.{4}f},"})
                    progress_bar.update(10)

        elapsed = time.perf_counter() - start_time
        progress_bar.close()
        self.psnr_tracker.print_summary()
        self.logwriter.write(f"Optimizing Complete in {elapsed:.4f}s")
        torch.save({'model_state_dict': self.gaussian_model.state_dict()}, self.model_dir / "gaussian_model_with_colors.pth.tar")
        if self.wandb_run:
            wandb.save(str(self.model_dir / "gaussian_model_with_colors.pth.tar"), base_path=str(self.model_dir))
        self.test()
        return

    # ====== Visualize learned eigen "components" ======
    def vis(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            image = self.gaussian_model()
        mse_loss = F.mse_loss(image.float(), self.gt_arrs.float())
        psnr = 10 * math.log10(1.0 / (mse_loss.item() + 1e-12))
        ms_ssim_value = ms_ssim(image.float(), self.gt_arrs.float(), data_range=1, size_average=True).item()
        self.logwriter.write(f"Components Fitting: PSNR:{psnr:.4f}, MS_SSIM:{ms_ssim_value:.6f}")

        vis_dir = self.model_dir / f"vis_comps"
        vis_dir.mkdir(parents=True, exist_ok=True)
        transform = transforms.ToPILImage()
        array = image.float()
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                img = transform(array[i, j])
                img.save(vis_dir / f"{i}-{j}.png")
        return psnr, ms_ssim_value

    # ====== Phase-B initialization: set colors/means/scale (clustered only) ======
    def update_gaussian(self):
        model = self.gaussian_model
        d = self.H * self.W
        img_arr = self.gt_image.detach().cpu().numpy().reshape(3, d)  # (3,d)

        colors = torch.zeros((self.num_points, 3), device=self.device, dtype=torch.float32)
        mean_tensor = torch.zeros((3, d), device=self.device, dtype=torch.float32)
        sum_codes = np.zeros(3, dtype=np.float32)

        psi = model.get_features  # (3,K,J,N)
        with torch.no_grad():
          psi.zero_()
        for c_idx, ch_name in enumerate(["Y", "Cb", "Cr"]):
            x = img_arr[c_idx]  # (d,)

            # 1) nearest cluster via helper
            k_c = self._assign_cluster_for_channel(ch_name, x)

            # 2) project onto that cluster's PCA basis
            cdir = self.dataset_path / ch_name / "clusters" / f"cluster_{k_c:02d}"
            bases = np.load(cdir / "pca_bases.npy")  # (J,d)
            mean  = np.load(cdir / "pca_mean.npy")   # (d,)
            Xc = x - mean
            w = Xc @ bases.T                         # (J,)

            # 3) compose per-Gaussian color from psi[c,k_c,:,:]
            psi_ck = psi[c_idx, k_c]                 # (J,N)
            col_c = torch.from_numpy(w.astype(np.float32)).to(self.device) @ psi_ck  # (N,)
            colors[:, c_idx] = col_c

            mean_tensor[c_idx] = torch.from_numpy(mean.astype(np.float32)).to(self.device)
            sum_codes[c_idx] = float(w.sum())

        # 4) un-normalize using per-channel vmin/vmax estimated from raw bases
        max_tensor = torch.from_numpy(self._vmax).to(self.device)
        min_tensor = torch.from_numpy(self._vmin).to(self.device)
        model.shift_factor = torch.from_numpy(sum_codes).to(self.device) * min_tensor
        model.scale_factor = (max_tensor - min_tensor)
        model.image_mean = mean_tensor

        with torch.no_grad():
            model._colors.copy_(colors)
            _ = model(render_colors=True)

    # ====== Test a single image (used in long optimize and at the end) ======
    def test(self, iter: int | None = None):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model(render_colors=True)  # (3,H,W)
        image = out.reshape(3, -1) + self.gaussian_model.image_mean
        image = image.reshape(3, self.H, self.W).clamp(0, 1)

        mse_loss = F.mse_loss(image.float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / (mse_loss.item() + 1e-12))
        ms_ssim_value = ms_ssim(image.unsqueeze(0).float(), self.gt_image.unsqueeze(0).float(),
                                data_range=1, size_average=True).item()
        self.logwriter.write(f"Test PSNR:{psnr:.4f}, MS_SSIM:{ms_ssim_value:.6f}")

        rgb_img = self._ycbcr_to_rgb_image(image)
        name = f"{self.image_name}_{iter}_fitting.png" if iter is not None else f"{self.image_name}_fitting.png"
        rgb_img.save(str(self.model_dir / name))
        return psnr, ms_ssim_value

    # ====== Fixed eval-set helpers ======
    def _build_or_load_fixed_eval_set(self) -> List[Path]:
        """
        Build (or load) a deterministic list of test images.
        Saves absolute paths to model_dir/eval_set.json so it's identical every eval.
        """
        fixed_path = self.model_dir / "eval_set.json"
        if fixed_path.exists():
            with open(fixed_path, "r") as f:
                data = json.load(f)
            paths = [Path(p) for p in data.get("paths", []) if Path(p).exists()]
            if len(paths) > 0:
                self._test_images_cache = paths
                return paths

        candidates = self._discover_test_images()
        if len(candidates) == 0:
            raise FileNotFoundError(
                "No test images found. Provide --test_glob or --test_dir, "
                "or place test images under your dataset path."
            )

        abs_paths = [str(p.resolve()) for p in candidates]
        with open(fixed_path, "w") as f:
            json.dump({"paths": abs_paths}, f, indent=2)

        self._test_images_cache = [Path(p) for p in abs_paths]
        return self._test_images_cache

    def _phaseB_eval_single(self, base_state_dict: Dict[str, Any], img_path: Path, opt_iters: int,
                            save_dir: Path | None = None) -> Dict[str, Any]:
        """
        Clone model → init Phase-B for img_path → run short optimize → collect metrics and images.
        Also records PSNR/SSIM at every optimization iteration.
        """
        # fresh model copy (clustered settings)
        model = GaussianBasis(
            loss_type="L2", opt_type="adan", num_points=self.num_points,
            H=self.H, W=self.W, BLOCK_H=16, BLOCK_W=16,
            device=self.device, lr=self.args.lr,
            num_comps=self.num_comps,
            clustered=True, K_clusters=self.K, J_dim=self.J
        ).to(self.device)
        model.load_state_dict(base_state_dict)
        model.eval()

        # load image
        gt_img = self._load_ycbcr_tensor(img_path)  # (3,H,W)
        d = self.H * self.W

        # ====== clustered init (same as update_gaussian, but for gt_img here) ======
        img_arr = gt_img.detach().cpu().numpy().reshape(3, d)
        colors = torch.zeros((self.num_points, 3), device=self.device, dtype=torch.float32)
        mean_tensor = torch.zeros((3, d), device=self.device, dtype=torch.float32)
        sum_codes = np.zeros(3, dtype=np.float32)
        psi = model.get_features  # (3,K,J,N)


        for c_idx, ch_name in enumerate(["Y", "Cb", "Cr"]):
            x = img_arr[c_idx]
            # nearest cluster via helper
            k_c = self._assign_cluster_for_channel(ch_name, x)

            # local PCA projection
            cdir = self.dataset_path / ch_name / "clusters" / f"cluster_{k_c:02d}"
            bases = np.load(cdir / "pca_bases.npy")  # (J,d)
            mean  = np.load(cdir / "pca_mean.npy")   # (d,)
            Xc = x - mean
            w = Xc @ bases.T                         # (J,)

            psi_ck = psi[c_idx, k_c]                 # (J,N)
            col_c = torch.from_numpy(w.astype(np.float32)).to(self.device) @ psi_ck  # (N,)
            colors[:, c_idx] = col_c

            mean_tensor[c_idx] = torch.from_numpy(mean.astype(np.float32)).to(self.device)
            sum_codes[c_idx] = float(w.sum())

        max_tensor = torch.from_numpy(self._vmax).to(self.device)
        min_tensor = torch.from_numpy(self._vmin).to(self.device)
        model.shift_factor = torch.from_numpy(sum_codes).to(self.device) * min_tensor
        model.scale_factor = (max_tensor - min_tensor)
        model.image_mean = mean_tensor

        with torch.no_grad():
            model._colors.copy_(colors)

        # ====== metrics at init ======
        with torch.no_grad():
            out0 = model(render_colors=True)                    # (3,H,W)
            img0 = (out0.view(3, -1) + model.image_mean)        # (3,H*W)
            img0 = img0.view(3, self.H, self.W).clamp(0, 1)     # (3,H,W)
            mse0 = F.mse_loss(img0, gt_img).item()
            psnr0 = 10 * math.log10(1.0 / (mse0 + 1e-12))
            ssim0 = ms_ssim(img0.unsqueeze(0), gt_img.unsqueeze(0), data_range=1, size_average=True).item()

        # ---- per-iter histories (include iter 0) ----
        psnr_hist = [float(psnr0)]
        ssim_hist = [float(ssim0)]

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            stem = img_path.stem
            self._ycbcr_to_rgb_image(gt_img).save(save_dir / f"{stem}_gt.png")
            self._ycbcr_to_rgb_image(img0).save(save_dir / f"{stem}_init.png")

        # ====== short Phase-B refine ======
        model.scheduler_init(optimize_phase=True)
        model.train()
        last_loss, last_psnr = None, None
        for _ in range(opt_iters):
            loss, psnr = model.optimize_iter(gt_img)
            last_loss, last_psnr = float(loss.item()), float(psnr)

            # record per-iter PSNR and SSIM (after this step)
            with torch.no_grad():
                out_t = model(render_colors=True)
                img_t = (out_t.view(3, -1) + model.image_mean).view(3, self.H, self.W).clamp(0, 1)
                ssim_t = ms_ssim(img_t.unsqueeze(0), gt_img.unsqueeze(0), data_range=1, size_average=True).item()
            psnr_hist.append(float(psnr))
            ssim_hist.append(float(ssim_t))

        # ====== final metrics ======
        model.eval()
        with torch.no_grad():
            outF = model(render_colors=True)
            imgF = (outF.view(3, -1) + model.image_mean).view(3, self.H, self.W).clamp(0, 1)
            mseF = F.mse_loss(imgF, gt_img).item()
            psnrF = 10 * math.log10(1.0 / (mseF + 1e-12))
            ssimF = ms_ssim(imgF.unsqueeze(0), gt_img.unsqueeze(0), data_range=1, size_average=True).item()

        final_rgb = self._ycbcr_to_rgb_image(imgF)
        if save_dir is not None:
            final_rgb.save(save_dir / f"{img_path.stem}_final.png")

        return {
            "path": str(img_path),
            "loss_init": mse0, "psnr_init": psnr0, "ssim_init": ssim0,
            "loss_final": mseF, "psnr_final": psnrF, "ssim_final": ssimF,
            "rgb_gt": self._ycbcr_to_rgb_image(gt_img),
            "rgb_init": self._ycbcr_to_rgb_image(img0),
            "rgb_final": final_rgb,
            # per-iter histories
            "psnr_hist": psnr_hist,
            "ssim_hist": ssim_hist,
        }

    # ====== Evaluate Phase-B on a fixed test set during Phase-A ======
    def eval_phaseB_on_test_set(self, global_iter: int):
        eval_list = self._build_or_load_fixed_eval_set()

        # snapshot model state so eval never mutates training model
        base_state = copy.deepcopy(self.gaussian_model.state_dict())

        eval_root = self.model_dir / "eval" / f"iter_{global_iter:06d}"
        eval_root.mkdir(parents=True, exist_ok=True)

        rows = []
        wandb_rows = []
        psnr_hists, ssim_hists = [], []  # collect curves
        for p in tqdm(eval_list, disable=True):
            res = self._phaseB_eval_single(
                base_state_dict=base_state,
                img_path=p,
                opt_iters=self.args.eval_opt_iters,
                save_dir=eval_root
            )
            rows.append([
                res["path"],
                res["psnr_init"], res["ssim_init"], res["loss_init"],
                res["psnr_final"], res["ssim_final"], res["loss_final"],
            ])
            psnr_hists.append(res["psnr_hist"])
            ssim_hists.append(res["ssim_hist"])

            if self.wandb_run:
                wandb_rows.append([
                    res["path"],
                    wandb.Image(res["rgb_gt"],    caption=f"{Path(res['path']).name} • GT"),
                    wandb.Image(res["rgb_init"],  caption=f"{Path(res['path']).name} • INIT  PSNR:{res['psnr_init']:.2f}  SSIM:{res['ssim_init']:.4f}"),
                    wandb.Image(res["rgb_final"], caption=f"{Path(res['path']).name} • FINAL PSNR:{res['psnr_final']:.2f} SSIM:{res['ssim_final']:.4f}"),
                    float(res["psnr_init"]), float(res["ssim_init"]),
                    float(res["psnr_final"]), float(res["ssim_final"]),
                ])

        arr = np.array(rows, dtype=object)
        psnrI = np.array(arr[:, 1], dtype=float)
        ssimI = np.array(arr[:, 2], dtype=float)
        psnrF = np.array(arr[:, 4], dtype=float)
        ssimF = np.array(arr[:, 5], dtype=float)

        # ---- Curves over Phase-B iterations (length = eval_opt_iters + 1) ----
        psnr_mat = np.array(psnr_hists, dtype=float)  # [num_images, T+1]
        ssim_mat = np.array(ssim_hists, dtype=float)
        mean_psnr_curve = psnr_mat.mean(axis=0)
        mean_ssim_curve = ssim_mat.mean(axis=0)
        count_over_thr  = (psnr_mat >= float(self.args.psnr_threshold)).sum(axis=0)

        # persist curves to disk
        np.savez(eval_root / "curves.npz",
                 iters=np.arange(psnr_mat.shape[1]),
                 mean_psnr=mean_psnr_curve,
                 mean_ssim=mean_ssim_curve,
                 count_over_thr=count_over_thr,
                 psnr_threshold=float(self.args.psnr_threshold))

        txt_path = eval_root / "metrics.txt"
        with open(txt_path, "w") as f:
            print(f"[iter {global_iter}] #images={len(rows)}", file=f)
            print(f"Init:  PSNR mean={psnrI.mean():.3f}  std={psnrI.std():.3f} | "
                  f"SSIM mean={ssimI.mean():.4f}  std={ssimI.std():.4f}", file=f)
            print(f"Final: PSNR mean={psnrF.mean():.3f}  std={psnrF.std():.3f} | "
                  f"SSIM mean={ssimF.mean():.4f}  std={ssimF.std():.4f}", file=f)
            print("", file=f)
            print(f"PSNR threshold = {self.args.psnr_threshold:g}", file=f)
            print(f"Mean PSNR curve: {mean_psnr_curve.tolist()}", file=f)
            print(f"Mean SSIM curve: {mean_ssim_curve.tolist()}", file=f)
            print(f"Count >= thr:   {count_over_thr.astype(int).tolist()}", file=f)
            print("", file=f)
            for r in rows:
                print(f"{Path(r[0]).name:30s}  "
                      f"initPSNR={r[1]:.2f} initSSIM={r[2]:.4f}  "
                      f"finalPSNR={r[4]:.2f} finalSSIM={r[5]:.4f}", file=f)

        if self.wandb_run:
            # images + metrics table + curves
            table = wandb.Table(
                data=wandb_rows,
                columns=["path", "GT", "INIT", "FINAL", "psnr_init", "ssim_init", "psnr_final", "ssim_final"]
            )
            xs = list(range(mean_psnr_curve.shape[0]))
            psnr_plot = wandb.plot.line_series(
                xs=xs, ys=[mean_psnr_curve.tolist()], keys=["mean_psnr"],
                title=f"Eval mean PSNR vs Phase-B iters (thr={self.args.psnr_threshold:g})", xname="Phase-B iter"
            )
            ssim_plot = wandb.plot.line_series(
                xs=xs, ys=[mean_ssim_curve.tolist()], keys=["mean_ssim"],
                title="Eval mean SSIM vs Phase-B iters", xname="Phase-B iter"
            )
            cnt_plot = wandb.plot.line_series(
                xs=xs, ys=[count_over_thr.astype(int).tolist()],
                keys=[f"count_psnr>= {self.args.psnr_threshold:g}"],
                title="Count of images above PSNR threshold vs Phase-B iters", xname="Phase-B iter"
            )
            wandb.log({
                "iter": global_iter,
                "eval/num_images": int(len(rows)),
                "eval/mean_psnr_init": float(psnrI.mean()),
                "eval/mean_psnr_final": float(psnrF.mean()),
                "eval/mean_ssim_init": float(ssimI.mean()),
                "eval/mean_ssim_final": float(ssimF.mean()),
                "eval/table_images": table,
                "eval/mean_psnr_curve": psnr_plot,
                "eval/mean_ssim_curve": ssim_plot,
                "eval/count_psnr_over_thr_curve": cnt_plot,
            }, step=global_iter)


# ================== CLI ==================
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Clustered EigenGS training (per-channel projective clustering)")

    parser.add_argument("-d", "--dataset", type=str, required=True, help="Clustered dataset directory")
    parser.add_argument("--iterations", type=int, default=50000, help="Training iterations (Phase-A) or optimize iterations (Phase-B)")
    parser.add_argument("--num_points", type=int, default=50000, help="2D Gaussian count")
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint to resume from")
    parser.add_argument("--image_path", type=str, default=None, help="If provided, run Phase-B optimize on this image after training")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--skip_train", action="store_true", help="Skip Phase-A training and go directly to Phase-B optimize (requires --model_path and --image_path)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # W&B + Eval
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default=os.getenv("WANDB_PROJECT", "eigens"), help="W&B project")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (optional)")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name (optional)")
    parser.add_argument("--eval_every", type=int, default=1000, help="Eval cadence in iters during Phase-A")
    parser.add_argument("--eval_images", type=int, default=10, help="How many test images to evaluate (kept for compatibility)")
    parser.add_argument("--eval_opt_iters", type=int, default=200, help="Short Phase-B iters per eval image")
    parser.add_argument("--psnr_threshold", type=float, default=35.0,
                        help="PSNR threshold for counting images above it during eval curves")

    # Test-set discovery
    parser.add_argument("--test_glob", type=str, default=None, help="Glob for test images, e.g. './datasets/kodak/test/*.png'")
    parser.add_argument("--test_dir", type=str, default=None, help="Directory containing test images")
    parser.add_argument("--test_recursive", action="store_true", help="Recurse into subfolders of --test_dir")

    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(int(args.seed))
        np.random.seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(int(args.seed))
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    trainer = GaussianTrainer(
        args, image_path=args.image_path, num_points=args.num_points,
        iterations=args.iterations, model_path=args.model_path
    )

    if not args.skip_train:
        trainer.train()

    if args.test_dir:
        trainer.eval_phaseB_on_test_set(global_iter=args.eval_opt_iters)
    elif args.image_path:
        trainer.optimize()


if __name__ == "__main__":
    main(sys.argv[1:])
