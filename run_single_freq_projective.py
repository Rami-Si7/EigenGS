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
from skimage import color
from gaussianbasis_single_freq_base import GaussianBasis
import pickle
import uuid
import os
import copy
import wandb
from datetime import datetime
from typing import List, Optional, Tuple

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# --------------------- small LA helper for Y-flat residual ---------------------
@torch.no_grad()
def residual_to_flat(y_vec: torch.Tensor, V_row: torch.Tensor, v: torch.Tensor) -> float:
    """
    y_vec: (d,)          flattened Y in [0,1]
    V_row: (J, d)        row-basis of flat (from clusters/cluster_k/Y_flat_V.npy)
    v:     (d,)          translation (from clusters/cluster_k/Y_flat_v.npy)
    returns ||(I - U U^T)(y - v)||_2 where U = V_row^T column-orthonormal
    """
    # V_row came from top-J rows of Vh, so its rows are orthonormal in d-space.
    # Equivalent column-orthonormal basis:
    U = V_row.transpose(0, 1).contiguous()  # (d, J)
    z = y_vec - v
    proj = (z @ U) @ U.transpose(0, 1)      # (d,)
    r = z - proj
    return float(torch.linalg.vector_norm(r).item())

# ===============================================================================

class GaussianTrainer:
    def __init__(
        self, args,
        image_path = None,
        num_points: int = 2000,
        iterations: int = 30000,
        model_path = None,
    ):
        self.args = args
        self.dataset_path = Path(args.dataset)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # --- projective artifacts ---
        pb_path = self.dataset_path / "projective_basis.pt"
        if not pb_path.exists():
            raise FileNotFoundError(f"Expected {pb_path} from the parse step.")
        pb = torch.load(pb_path, map_location="cpu")
        self.K = int(pb["K"])
        self.J = int(pb["J"])
        self.img_size = tuple(pb["img_size"])  # (W,H)
        self.norm_mins = torch.tensor(pb["norm_mins"], dtype=torch.float32, device=self.device)  # (3,)
        self.norm_maxs = torch.tensor(pb["norm_maxs"], dtype=torch.float32, device=self.device)  # (3,)

        # --- choose training target: all clusters or single cluster ---
        self.cluster = args.cluster  # "all" or an integer (as str)
        train_arrs_path: Path
        if str(self.cluster).lower() == "all":
            train_arrs_path = self.dataset_path / "arrs.npy"           # (K*J,3,H,W)
        else:
            k = int(self.cluster)
            train_arrs_path = self.dataset_path / f"clusters/cluster_{k:02d}/arrs.npy"  # (J,3,H,W)
            if not train_arrs_path.exists():
                raise FileNotFoundError(f"No arrs for cluster {k}: {train_arrs_path}")

        arrs = np.load(train_arrs_path)
        self.gt_arrs = torch.from_numpy(arrs).to(self.device)          # float64->float32 handled by pytorch
        self.num_comps = self.gt_arrs.shape[0]

        # infer H,W from arr
        self.H, self.W = self.gt_arrs.shape[2], self.gt_arrs.shape[3]
        BLOCK_H, BLOCK_W = 16, 16
        self.iterations = iterations
        self.num_points = num_points

        # --- optional single test image path ---
        random_string = str(uuid.uuid4())[:6]
        if image_path is not None:
            self.image_path = Path(image_path)
            self.image_name = self.image_path.stem
            img = Image.open(self.image_path).convert("RGB").resize((self.W, self.H))
            img_arr = (color.rgb2ycbcr(np.array(img)) / 255.0).transpose(2, 0, 1)
            self.gt_image = torch.tensor(img_arr, dtype=torch.float32, device=self.device)
            model_dir = Path(f"./models/recons/{self.image_name}-{self.dataset_path.name}-{num_points}-{args.iterations}-{random_string}")
        else:
            self.image_path = None
            self.image_name = None
            self.gt_image = None
            model_dir = Path(f"./models/single-freq/{self.dataset_path.name}-{num_points}-{args.iterations}-{random_string}")
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # --- model ---
        self.gaussian_model = GaussianBasis(
            loss_type="L2", opt_type="adan", num_points=self.num_points,
            H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            device=self.device, lr=args.lr, num_comps=self.num_comps
        ).to(self.device)
        self.gaussian_model.scheduler_init()

        # --- logging / trackers ---
        self.logwriter = LogWriter(self.model_dir)
        self.psnr_tracker = PSNRTracker(self.logwriter)
        self.logwriter.write(f"Model Dir ID: {random_string}")
        self.logwriter.write(f"Training on cluster='{self.cluster}'  → num_comps={self.num_comps}, arrs={train_arrs_path.name}")

        # optional checkpoint load
        if model_path is not None:
            self.model_path = Path(model_path)
            self.logwriter.write(f"Model loaded from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.gaussian_model.load_state_dict(checkpoint["model_state_dict"])

        # --- W&B ---
        self.wandb_run = None
        if args.wandb:
            run_name = args.run_name or f"{self.dataset_path.name}-C{self.cluster}-P{self.num_points}-I{self.iterations}-{random_string}"
            self.wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                name=run_name,
                config={
                    "dataset": str(self.dataset_path),
                    "cluster": str(self.cluster),
                    "num_points": self.num_points,
                    "iterations": self.iterations,
                    "lr": args.lr,
                    "eval_every": args.eval_every,
                    "eval_opt_iters": args.eval_opt_iters,
                    "eval_images": args.eval_images,
                    "seed": args.seed,
                },
                dir=str(self.model_dir),
                reinit=False,
            )
            wandb.watch(self.gaussian_model, log=None, log_freq=0, log_graph=False)

        # cache for eval image discovery
        self._test_images_cache = None
        self._rng = np.random.default_rng(args.seed if args.seed is not None else 0)

    # ------------------------ data discovery helpers ------------------------

    def _discover_test_images(self) -> List[Path]:
        if self._test_images_cache is not None:
            return self._test_images_cache
        if self.args.test_glob:
            candidates = sorted(Path().glob(self.args.test_glob))
        elif self.args.test_dir:
            root = Path(self.args.test_dir)
            it = root.rglob("*") if self.args.test_recursive else root.glob("*")
            candidates = sorted([p for p in it if p.suffix.lower() in IMG_EXTS and p.is_file()])
        else:
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

    def _load_ycbcr_tensor(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((self.W, self.H))
        ycbcr = (color.rgb2ycbcr(np.array(img)) / 255.0).transpose(2, 0, 1)
        return torch.tensor(ycbcr, dtype=torch.float32, device=self.device)

    def _ycbcr_to_rgb_image(self, ycbcr_01: torch.Tensor) -> Image.Image:
        ycbcr_img = (ycbcr_01.clamp(0,1).detach().cpu().numpy() * 255.0).transpose(1,2,0)
        rgb = color.ycbcr2rgb(ycbcr_img) * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb)

    def _save_checkpoint(self, step: int, tag: str):
        ckpt = {
            "step": step,
            "model_state_dict": self.gaussian_model.state_dict(),
            "optimizer_state_dict": getattr(self.gaussian_model, "optimizer", None).state_dict() if hasattr(self.gaussian_model, "optimizer") else None,
            "scheduler_state_dict": getattr(self.gaussian_model, "scheduler", None).state_dict() if hasattr(self.gaussian_model, "scheduler") else None,
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

    # ------------------------ cluster artifacts ------------------------

    def _cluster_dir(self, k: int) -> Path:
        return self.dataset_path / f"clusters/cluster_{k:02d}"

    def _load_cluster_basis(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """returns (means_k, bases_k) with shapes (3,d) and (3,J,d) on device"""
        cdir = self._cluster_dir(k)
        means = torch.from_numpy(np.load(cdir / "means.npy")).to(self.device).float()       # (3,d)
        bases = torch.from_numpy(np.load(cdir / "bases.npy")).to(self.device).float()       # (3,J,d)
        return means, bases

    def _load_cluster_yflat(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """returns (V_row, v) with shapes (J,d) and (d,) on device"""
        cdir = self._cluster_dir(k)
        V = torch.from_numpy(np.load(cdir / "Y_flat_V.npy")).to(self.device).float()        # (J,d)
        v = torch.from_numpy(np.load(cdir / "Y_flat_v.npy")).to(self.device).float()        # (d,)
        return V, v

    # ------------------------ codes for Phase-B ------------------------

    @torch.no_grad()
    def _pick_cluster_for_image(self, img_ycbcr: torch.Tensor) -> int:
        """Auto-select best cluster for a given image by smallest Y-flat residual."""
        y_vec = img_ycbcr[0].view(-1).to(self.device)  # Y channel
        best_k, best_r = 0, float("inf")
        for k in range(self.K):
            V, v = self._load_cluster_yflat(k)
            r = residual_to_flat(y_vec, V, v)
            if r < best_r:
                best_r, best_k = r, k
        return best_k

    @torch.no_grad()
    def _codes_from_cluster(self, img_ycbcr: torch.Tensor, k: int) -> torch.Tensor:
        """
        Compute per-channel coefficients against cluster-k bases.
        Returns codes_tensor with shape:
          - if training single cluster: (3, 1, J)
          - if training all clusters:  (3, 1, K*J) with zeros outside [k*J:(k+1)*J]
        """
        means_k, bases_k = self._load_cluster_basis(k)    # (3,d), (3,J,d)
        d = bases_k.shape[-1]
        X = img_ycbcr.view(3, -1)                         # (3,d)
        centered = X - means_k                            # (3,d)
        # rows of bases_k[ch] are row-vectors in d-space
        # coes[ch] = centered[ch] @ bases_k[ch].T -> (J,)
        coes = torch.einsum("cd,cjd->cj", centered, bases_k)  # (3,J)
        if str(self.cluster).lower() == "all":
            full = torch.zeros((3, self.K * self.J), device=self.device, dtype=torch.float32)
            full[:, k*self.J:(k+1)*self.J] = coes
            return full.unsqueeze(1)                      # (3,1,K*J)
        else:
            return coes.unsqueeze(1)                      # (3,1,J)

    # ------------------------ training / optimize ------------------------

    def train(self):
        progress_bar = tqdm(range(1, self.iterations+1), desc="Training progress")
        self.gaussian_model.train()
        start_time = time.time()
        for iter in range(1, self.iterations+1):
            loss, psnr = self.gaussian_model.train_iter(self.gt_arrs)
            if self.wandb_run:
                wandb.log({"train/loss": float(loss.item()), "train/psnr": float(psnr), "iter": iter})
            with torch.no_grad():
                progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                progress_bar.update(1)
            if self.args.eval_every and (iter % self.args.eval_every == 0):
                self.eval_phaseB_on_test_set(global_iter=iter)
                self._save_checkpoint(step=iter, tag="phaseA")
        end_time = time.time() - start_time
        progress_bar.close()
        self.logwriter.write("Training Complete in {:.4f}s".format(end_time))
        torch.save({'model_state_dict': self.gaussian_model.state_dict()}, self.model_dir / "gaussian_model.pth.tar")
        if self.wandb_run:
            wandb.save(str(self.model_dir / "gaussian_model.pth.tar"), base_path=str(self.model_dir))
        self.vis()
        return

    def optimize(self):
        if self.gt_image is None:
            raise ValueError("Phase-B (--image_path) requires a target image.")
        progress_bar = tqdm(range(1, self.iterations+1), desc="Optimizing progress")
        self.update_gaussian()  # uses projective cluster bases now
        self.gaussian_model.scheduler_init(optimize_phase=True)
        self.gaussian_model.train()
        start_time = time.perf_counter()

        self.test(iter=0)
        self.gaussian_model.train()
        for iter in range(1, self.iterations+1):
            loss, psnr = self.gaussian_model.optimize_iter(self.gt_image)
            self.psnr_tracker.check(start_time, psnr, iter)
            if self.wandb_run:
                wandb.log({"phaseB/loss": float(loss.item()), "phaseB/psnr": float(psnr), "phaseB/iter": iter})
            with torch.no_grad():
                if iter in [10, 100, 1000]:
                    self.test(iter=iter); self.gaussian_model.train()
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10)
        end_time = time.perf_counter() - start_time
        progress_bar.close()
        self.psnr_tracker.print_summary()
        self.logwriter.write("Optimizing Complete in {:.4f}s".format(end_time))
        torch.save({'model_state_dict': self.gaussian_model.state_dict()}, self.model_dir / "gaussian_model_with_colors.pth.tar")
        if self.wandb_run:
            wandb.save(str(self.model_dir / "gaussian_model_with_colors.pth.tar"), base_path=str(self.model_dir))
        self.test()
        return

    def vis(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            image = self.gaussian_model()
        mse_loss = F.mse_loss(image.float(), self.gt_arrs.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(image.float(), self.gt_arrs.float(), data_range=1, size_average=True).item()
        self.logwriter.write("Components Fitting: PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))

        vis_dir = self.model_dir / f"vis_comps"
        vis_dir.mkdir(parents=True, exist_ok=True)
        transform = transforms.ToPILImage()
        array = image.float()
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                img = transform(array[i, j])
                img.save(vis_dir / f"{i}-{j}.png")
        return psnr, ms_ssim_value

    # ------------------------ Phase-B init from projective basis ------------------------

    def update_gaussian(self):
        """
        Initialize colors for Phase-B using projective cluster bases instead of PCA pickles.
        - If training 'all', choose cluster for the target image (auto or fixed),
          then embed the coefficients into a length (K*J) vector (zeros elsewhere).
        - If training a single cluster, coefficients have length J.
        """
        if self.gt_image is None:
            return  # nothing to init if we don't have a target image

        # choose cluster to use for coding
        if str(self.cluster).lower() == "all":
            if self.args.phaseB_cluster is not None:
                k = int(self.args.phaseB_cluster)
            elif self.args.auto_select_cluster:
                k = self._pick_cluster_for_image(self.gt_image)
                self.logwriter.write(f"[phaseB] auto-selected cluster {k}")
            else:
                k = 0
                self.logwriter.write(f"[phaseB] defaulting to cluster {k}")
        else:
            k = int(self.cluster)

        # per-channel coefficients
        codes_tensor = self._codes_from_cluster(self.gt_image, k)  # (3,1,num_comps)

        # normalization factors (to match how arrs were normalized)
        max_tensor = self.norm_maxs
        min_tensor = self.norm_mins

        # set factors on the model (same interface as before)
        self.gaussian_model.shift_factor = codes_tensor.sum(dim=2).squeeze(1) * min_tensor   # (3,)
        self.gaussian_model.scale_factor = (max_tensor - min_tensor)                         # (3,)
        # For projective clustering, we don't need a per-image "mean" in the model;
        # but keep the field for downstream code-paths:
        self.gaussian_model.image_mean = torch.zeros((3, self.H*self.W), device=self.device)

        with torch.no_grad():
            features = self.gaussian_model.get_features.permute(2, 0, 1)  # (3, num_comps, P)
            colors = torch.bmm(codes_tensor, features).squeeze(1).transpose(0, 1)  # (P,3)
            self.gaussian_model._colors.copy_(colors)
            _ = self.gaussian_model(render_colors=True)

    def test(self, iter=None):
        self.gaussian_model.eval()
        _, _, height, width = self.gt_arrs.shape
        with torch.no_grad():
            out = self.gaussian_model(render_colors=True)     # (3,H,W) already in [0,1] range of components
        # In projective setup we store no per-image mean; treat out as YCbCr directly
        image = out.clamp(0, 1)
        mse_loss = F.mse_loss(image.float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(image.unsqueeze(0).float(), self.gt_image.unsqueeze(0).float(), data_range=1, size_average=True).item()
        self.logwriter.write("Test PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))

        ycbcr_img = image.detach().cpu().numpy() * 255.0
        ycbcr_img = ycbcr_img.reshape(3, height, width).transpose(1, 2, 0)
        rgb_img = color.ycbcr2rgb(ycbcr_img) * 255.0
        rgb_img = rgb_img.clip(0, 255)

        img = Image.fromarray(rgb_img.astype(np.uint8))
        name = self.image_name + (f"_{iter}_fitting.png" if iter is not None else "_fitting.png")
        img.save(str(self.model_dir / name))
        return psnr, ms_ssim_value

    # ------------------------ Eval Phase-B on a small set ------------------------

    def _phaseB_eval_single(self, base_state_dict, img_path: Path, opt_iters: int):
        model = GaussianBasis(
            loss_type="L2", opt_type="adan", num_points=self.num_points,
            H=self.H, W=self.W, BLOCK_H=16, BLOCK_W=16,
            device=self.device, lr=self.args.lr, num_comps=self.num_comps
        ).to(self.device)
        model.load_state_dict(base_state_dict)
        model.eval()

        gt_img = self._load_ycbcr_tensor(img_path)  # (3,H,W)

        # choose cluster as in update_gaussian()
        if str(self.cluster).lower() == "all":
            k = int(self.args.phaseB_cluster) if self.args.phaseB_cluster is not None else self._pick_cluster_for_image(gt_img)
        else:
            k = int(self.cluster)

        codes = self._codes_from_cluster(gt_img, k)  # (3,1,num_comps)

        max_tensor = self.norm_maxs
        min_tensor = self.norm_mins

        model.shift_factor = codes.sum(dim=2).squeeze(1) * min_tensor
        model.scale_factor = (max_tensor - min_tensor)
        model.image_mean = torch.zeros((3, self.H*self.W), device=self.device)

        with torch.no_grad():
            features = model.get_features.permute(2, 0, 1)
            colors = torch.bmm(codes, features).squeeze(1).transpose(0, 1)
            model._colors.copy_(colors)

        # metrics at init
        with torch.no_grad():
            out0 = model(render_colors=True)
            img0 = out0.clamp(0, 1)
            mse0 = F.mse_loss(img0, gt_img).item()
            psnr0 = 10 * math.log10(1.0 / (mse0 + 1e-12))
            ssim0 = ms_ssim(img0.unsqueeze(0), gt_img.unsqueeze(0), data_range=1, size_average=True).item()

        # short refine
        model.scheduler_init(optimize_phase=True)
        model.train()
        for _ in range(opt_iters):
            loss, psnr = model.optimize_iter(gt_img)

        # final metrics
        model.eval()
        with torch.no_grad():
            outF = model(render_colors=True).clamp(0, 1)
            mseF = F.mse_loss(outF, gt_img).item()
            psnrF = 10 * math.log10(1.0 / (mseF + 1e-12))
            ssimF = ms_ssim(outF.unsqueeze(0), gt_img.unsqueeze(0), data_range=1, size_average=True).item()

        rgb_init = self._ycbcr_to_rgb_image(outF)  # final preview
        return {
            "path": str(img_path),
            "loss_init": mse0, "psnr_init": psnr0, "ssim_init": ssim0,
            "loss_final": mseF, "psnr_final": psnrF, "ssim_final": ssimF,
            "rgb_final": rgb_init,
        }

    def eval_phaseB_on_test_set(self, global_iter: int):
        if not self.wandb_run:
            return
        all_candidates = self._discover_test_images()
        if len(all_candidates) == 0:
            self.logwriter.write("[eval warning] no test images found; skipping eval")
            return
        k = min(self.args.eval_images, len(all_candidates))
        idxs = self._rng.choice(len(all_candidates), size=k, replace=False)
        selected = [all_candidates[i] for i in idxs]
        base_state = copy.deepcopy(self.gaussian_model.state_dict())

        rows, img_panels = [], []
        for p in selected:
            res = self._phaseB_eval_single(base_state, p, opt_iters=self.args.eval_opt_iters)
            rows.append([
                res["path"], res["psnr_init"], res["ssim_init"], res["loss_init"],
                res["psnr_final"], res["ssim_final"], res["loss_final"],
            ])
            img_panels.append(wandb.Image(res["rgb_final"], caption=f"{Path(res['path']).name}  PSNR:{res['psnr_final']:.2f}  SSIM:{res['ssim_final']:.4f}"))

        table = wandb.Table(
            data=rows,
            columns=["path","psnr_init","ssim_init","loss_init","psnr_final","ssim_final","loss_final"]
        )
        arr = np.array(rows, dtype=object)
        psnrF = np.array(arr[:,4], dtype=float)
        ssimF = np.array(arr[:,5], dtype=float)
        lossF = np.array(arr[:,6], dtype=float)
        wandb.log({
            "iter": global_iter,
            "eval/num_images": int(k),
            "eval/psnr_final_mean": float(psnrF.mean()),
            "eval/psnr_final_std": float(psnrF.std()),
            "eval/ssim_final_mean": float(ssimF.mean()),
            "eval/loss_final_mean": float(lossF.mean()),
            "eval/gallery": img_panels,
            "eval/table": table,
        }, step=global_iter)

# ------------------------------ CLI ------------------------------

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Gaussian training with projective clusters")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to parsed dataset dir")
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--num_points", type=int, default=50000)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--seed", type=float, default=1)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)

    # cluster control
    parser.add_argument("--cluster", type=str, default="all",
                        help="'all' to train on all components, or an integer k to train only cluster k.")
    parser.add_argument("--phaseB_cluster", type=int, default=None,
                        help="If training 'all', force this cluster index for Phase-B/eval. If omitted, auto-select.")
    parser.add_argument("--auto_select_cluster", action="store_true",
                        help="Auto-select best cluster per image for Phase-B when training 'all'.")

    # W&B + Eval
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=os.getenv("WANDB_PROJECT", "eigens"))
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_images", type=int, default=10)
    parser.add_argument("--eval_opt_iters", type=int, default=200)
    parser.add_argument("--test_glob", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--test_recursive", action="store_true")
    return parser.parse_args(argv)

def main(argv):
    args = parse_args(argv)
    if args.seed is not None:
        torch.manual_seed(int(args.seed)); random.seed(int(args.seed))
        torch.cuda.manual_seed(int(args.seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(int(args.seed))

    trainer = GaussianTrainer(
        args, image_path=args.image_path, num_points=args.num_points,
        iterations=args.iterations, model_path=args.model_path
    )

    if not args.skip_train:
        trainer.train()
    else:
        trainer.optimize()

if __name__ == "__main__":
    main(sys.argv[1:])
