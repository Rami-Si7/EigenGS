# run_projective.py
"""
Projective-clustering variant of the EigenGS trainer (per-channel parse, EM-based eval init).

What this version does:
- Loads components from projective_dir/{Y,Cb,Cr}/arrs.npy and meta.json for training.
- Also loads per-channel EM flats: em_Vs.npy (K,J,d), em_vs.npy (K,d).
- Builds a single training target tensor (3*K*J, 3, H, W) whose entries are single-channel.
- During evaluation (every --eval_every) and in optimize_single:
    For each image:
      1) For each channel (Y, Cb, Cr), choose the closest cluster by EM residual.
      2) Project onto that channel's affine EM flat to get a per-channel reconstruction.
      3) Stack the three channel reconstructions to a full YCbCr image.
      4) Set model.image_mean = this reconstruction (so init equals the EM recon).
      5) Run a short Phase-B refinement and report PSNR/SSIM.

Test set:
- Uses projective_dir/all_test/ (global, identical across channels).
  Can be overridden with --test_dir / --test_glob.
"""

import os
import math
import json
import random
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from skimage import color
from PIL import Image
from tqdm import tqdm
from pytorch_msssim import ms_ssim

# ----- model -----
from gaussianbasis_single_freq_base import GaussianBasis

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# ----------------------------- Argparse -----------------------------

def parse_args(argv):
    p = argparse.ArgumentParser("Projective-clustering EigenGS trainer (per-channel)")

    # projective dataset root (output dir of the per-channel parse script)
    p.add_argument("--projective_dir", type=str, required=True,
                   help="Directory produced by the per-channel parse (contains Y/, Cb/, Cr/, all_test/)")
    p.add_argument("--iterations", type=int, default=50000)
    p.add_argument("--num_points", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--loss", type=str, default="L2")

    # Phase-B optimize a single image (optional)
    p.add_argument("--image_path", type=str, default=None)
    p.add_argument("--skip_train", action="store_true")

    p.add_argument("--model_path", type=str, default=None,
                   help="Path to a Phase-A checkpoint (.pth.tar) to load before Phase-B/eval")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training and run a single full Phase-B eval on the test set")

    # evaluation controls
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--eval_opt_iters", type=int, default=200)
    p.add_argument("--test_dir", type=str, default=None,
                   help="Optional external folder of test images; if omitted, uses projective_dir/all_test")
    p.add_argument("--test_recursive", action="store_true")
    p.add_argument("--test_glob", type=str, default=None)

    # logging
    p.add_argument("--out_dir", type=str, default="./models_projective")

    # Weights & Biases
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default=os.getenv("WANDB_PROJECT", "eigens"))
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)

    return p.parse_args(argv)

# ----------------------------- Per-channel loaders & helpers -----------------------------

def _load_root_meta(root: Path) -> Tuple[int, int, int, int]:
    """Returns (W, H, K, J) from top-level meta.json (written by the new parser)."""
    meta = json.loads((root / "meta.json").read_text())
    W, H = meta["img_size"]
    K = int(meta["K"])
    J = int(meta["J"])
    return W, H, K, J

def _load_channel_pca(root: Path, name: str):
    """
    Load one channel folder (Y or Cb or Cr) PCA artifacts used for Phase-A training visuals:
      - pca_means.npy: (K, d)
      - pca_bases.npy: (K, J, d)
      - arrs.npy: (K*J, H, W)   [normalized 0..1 eigenimages]
    """
    ch_dir = root / name
    means = np.load(ch_dir / "pca_means.npy")   # (K, d)
    bases = np.load(ch_dir / "pca_bases.npy")   # (K, J, d)
    arrs  = np.load(ch_dir / "arrs.npy")        # (K*J, H, W)
    return means, bases, arrs

def _load_channel_em(root: Path, name: str):
    """
    Load one channel EM flats used for eval cluster selection + init recon:
      - em_Vs.npy: (K, J, d)  (row-orthonormal basis per cluster)
      - em_vs.npy: (K, d)     (affine offset per cluster)
    """
    ch_dir = root / name
    Vs = np.load(ch_dir / "em_Vs.npy").astype(np.float32)  # (K,J,d)
    vs = np.load(ch_dir / "em_vs.npy").astype(np.float32)  # (K,d)
    return Vs, vs

def _build_training_targets(Y_arrs, Cb_arrs, Cr_arrs, H: int, W: int) -> torch.Tensor:
    """
    Given per-channel arrs (each (K*J, H, W) in [0,1]), build a single tensor:
        gt_arrs: (3*K*J, 3, H, W)
    Component i is single-channel (only one of Y/Cb/Cr is non-zero).
    """
    KJ = Y_arrs.shape[0]
    out = np.zeros((3*KJ, 3, H, W), dtype=np.float32)
    out[0:KJ, 0, :, :] = Y_arrs
    out[KJ:2*KJ, 1, :, :] = Cb_arrs
    out[2*KJ:3*KJ, 2, :, :] = Cr_arrs
    return torch.from_numpy(out)  # (3*KJ,3,H,W)

def _rgb_path_to_ycbcr(path: Path, size_hw: Tuple[int,int]) -> torch.Tensor:
    W, H = size_hw
    img = Image.open(path).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)
    ycbcr = color.rgb2ycbcr(arr).astype(np.float32) / 255.0  # [0,1], (H,W,3)
    t = torch.from_numpy(ycbcr.transpose(2,0,1))             # (3,H,W)
    return t

# ---------- EM projection helpers ----------
def _project_em_affine(x_d: np.ndarray, V_Jd: np.ndarray, v_d: np.ndarray, T: Optional[int]) -> Tuple[np.ndarray, float]:
    """
    Orthogonal projection onto affine EM flat: rec = v + ((x-v) @ V^T) @ V
    Returns (reconstructed_d, squared_residual).
    Assumes rows of V are orthonormal (as fit by the EM step via SVD).
    """
    if T is not None:
        Juse = int(min(T, V_Jd.shape[0]))
        V = V_Jd[:Juse, :]
    else:
        V = V_Jd
    r = x_d - v_d
    r_par = (r @ V.T) @ V
    rec = v_d + r_par
    err2 = float(np.sum((r - r_par) ** 2, dtype=np.float64))
    return rec, err2

def _choose_cluster_em_and_reconstruct(x_ch: torch.Tensor, Vs_KJd: np.ndarray, vs_Kd: np.ndarray,
                                       T: Optional[int]) -> Tuple[int, np.ndarray]:
    """
    For a single channel image x_ch (H,W):
      - Evaluate residual vs every cluster's EM flat
      - Return (best_k, reconstruction_d) where reconstruction is v_k + proj onto span(V_k)
    """
    x = x_ch.reshape(-1).detach().cpu().numpy()  # (d,)
    K = Vs_KJd.shape[0]
    best_k, best_err, best_rec = 0, float("inf"), None
    for k in range(K):
        rec, e2 = _project_em_affine(x, Vs_KJd[k], vs_Kd[k], T)
        if e2 < best_err:
            best_err, best_k, best_rec = e2, k, rec
    return best_k, best_rec  # best_rec is (d,)

def _stack_recons(recY_d: np.ndarray, recCb_d: np.ndarray, recCr_d: np.ndarray, H: int, W: int) -> torch.Tensor:
    rec = np.stack([
        np.clip(recY_d,  0.0, 1.0),
        np.clip(recCb_d, 0.0, 1.0),
        np.clip(recCr_d, 0.0, 1.0),
    ], axis=0).reshape(3, H, W)
    return torch.from_numpy(rec).float()

# ----------------------------- Trainer -----------------------------

class ProjectiveTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

        self.projective_dir = Path(args.projective_dir)
        # top-level sizes
        W, H, K, J = _load_root_meta(self.projective_dir)
        self.W, self.H, self.K, self.J = W, H, K, J
        self.d = W * H

        # ----- load per-channel PCA artifacts (for Phase-A training targets) -----
        Y_means_np,  Y_bases_np,  Y_arrs_np  = _load_channel_pca(self.projective_dir, "Y")
        Cb_means_np, Cb_bases_np, Cb_arrs_np = _load_channel_pca(self.projective_dir, "Cb")
        Cr_means_np, Cr_bases_np, Cr_arrs_np = _load_channel_pca(self.projective_dir, "Cr")

        # to torch (on device) for PCA (not strictly needed in eval now, but keep for completeness)
        self.Y_means_pca  = torch.from_numpy(Y_means_np ).float().to(self.device)  # (K,d)
        self.Cb_means_pca = torch.from_numpy(Cb_means_np).float().to(self.device)
        self.Cr_means_pca = torch.from_numpy(Cr_means_np).float().to(self.device)

        self.Y_bases_pca  = torch.from_numpy(Y_bases_np ).float().to(self.device)  # (K,J,d)
        self.Cb_bases_pca = torch.from_numpy(Cb_bases_np).float().to(self.device)
        self.Cr_bases_pca = torch.from_numpy(Cr_bases_np).float().to(self.device)

        # Build Phase-A training targets (3*K*J,3,H,W)
        self.gt_arrs = _build_training_targets(Y_arrs_np, Cb_arrs_np, Cr_arrs_np, H, W).to(self.device)

        # ----- load per-channel EM flats (for eval/init) -----
        self.VsY_em,  self.vY_em  = _load_channel_em(self.projective_dir, "Y")   # (K,J,d), (K,d)
        self.VsCb_em, self.vCb_em = _load_channel_em(self.projective_dir, "Cb")
        self.VsCr_em, self.vCr_em = _load_channel_em(self.projective_dir, "Cr")

        # ----- out dir -----
        mode_tag = "all_channels"
        self.out_dir = Path(args.out_dir) / f"proj_{self.projective_dir.name}_{mode_tag}_P{args.num_points}_I{args.iterations}_seed{args.seed}"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # ----- model -----
        self.num_comps = self.gt_arrs.shape[0]  # 3*K*J
        self.model = GaussianBasis(
            loss_type=args.loss, opt_type="adan",
            num_points=args.num_points, num_comps=self.num_comps,
            H=self.H, W=self.W, BLOCK_H=16, BLOCK_W=16,
            device=self.device, lr=args.lr
        ).to(self.device)
        self.model.scheduler_init()

        # ----- wandb -----
        self.wandb_run = None
        if args.wandb:
            import wandb
            run_name = args.run_name or f"{self.projective_dir.name}-{mode_tag}-P{args.num_points}-I{args.iterations}-seed{args.seed}"
            self.wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                name=run_name,
                config={
                    "projective_dir": str(self.projective_dir),
                    "num_points": args.num_points,
                    "iterations": args.iterations,
                    "lr": args.lr,
                    "eval_every": args.eval_every,
                    "eval_opt_iters": args.eval_opt_iters,
                    "seed": args.seed,
                    "H": self.H, "W": self.W,
                    "K": self.K, "J": self.J,
                    "num_comps": self.num_comps,
                },
                dir=str(self.out_dir),
                reinit=False,
            )
            wandb.watch(self.model, log=None, log_freq=0, log_graph=False)
            self._wandb = wandb
        else:
            self._wandb = None

        # caches
        self._test_images_cache: Optional[List[Path]] = None
        self._fixed_eval_paths: Optional[List[str]] = None
        self._fixed_eval_manifest = self.out_dir / "eval_set.json"

        # Optional: how many EM rows to use in projection; None = all
        self._em_top_t: Optional[int] = None  # you can expose a CLI if you want

    # ---------- test image discovery ----------
    def _discover_test_images(self) -> List[Path]:
        if self._test_images_cache is not None:
            return self._test_images_cache

        # priority 1: global split from new parser
        all_test = self.projective_dir / "all_test"
        if all_test.exists():
            cand = sorted([p for p in all_test.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()])
            self._test_images_cache = cand
            return cand

        # priority 2: user-specified test_glob
        if self.args.test_glob:
            cand = sorted(Path().glob(self.args.test_glob))
            self._test_images_cache = cand
            return cand

        # priority 3: user-specified directory
        if self.args.test_dir:
            root = Path(self.args.test_dir)
            it = root.rglob("*") if self.args.test_recursive else root.glob("*")
            cand = sorted([p for p in it if p.suffix.lower() in IMG_EXTS and p.is_file()])
            self._test_images_cache = cand
            return cand

        # fallback: nothing
        self._test_images_cache = []
        return []

    def _prepare_fixed_eval_set(self):
        if self._fixed_eval_paths is not None:
            return
        if self._fixed_eval_manifest.exists():
            loaded = json.loads(self._fixed_eval_manifest.read_text())
            self._fixed_eval_paths = [p for p in loaded if Path(p).exists()]
            return
        cand = self._discover_test_images()
        fixed = [str(p) for p in cand]
        self._fixed_eval_manifest.write_text(json.dumps(fixed, indent=2))
        self._fixed_eval_paths = fixed

    # ---------- Phase A ----------
    def train(self, iterations: int):
        self.model.train()
        bar = tqdm(range(1, iterations + 1), desc="Phase-A (learn single-channel eigenimages)")

        for it in bar:
            loss, psnr = self.model.train_iter(self.gt_arrs)

            if self._wandb:
                self._wandb.log({"train/loss": float(loss.item()), "train/psnr": float(psnr), "iter": it})

            if it % 10 == 0:
                bar.set_postfix({"loss": f"{loss.item():.6f}", "psnr": f"{psnr:.3f}"})

            # periodic eval over the ENTIRE test set
            if self.args.eval_every and (it % self.args.eval_every == 0):
                self.eval_phaseB_on_test_set(global_iter=it)

        # save
        torch.save({"model_state_dict": self.model.state_dict()},
                   self.out_dir / "gaussian_model_phaseA.pth.tar")

        # visualize learned components
        self._dump_components()

    @torch.no_grad()
    def _dump_components(self):
        vis_dir = self.out_dir / "vis_components"
        vis_dir.mkdir(parents=True, exist_ok=True)
        comps = self.model.forward(render_colors=False).clamp(0,1)
        for i in range(comps.shape[0]):
            for ch in range(3):
                arr = (comps[i, ch].cpu().numpy() * 255.0).astype(np.uint8)
                Image.fromarray(arr, mode="L").save(vis_dir / f"{i:04d}_ch{ch}.png")

    # ---------- Phase B (single image optimize) ----------
    def optimize_single(self, image_path: Path, iters: int, log_every: int = 10):
        gt = _rgb_path_to_ycbcr(image_path, (self.W, self.H)).to(self.device)
        # EM-based per-channel init (closest cluster per channel, projected recon)
        self._init_model_for_image_em(gt)  # image_mean becomes EM recon stack

        # Switch the model to Phase-B mode (colors trainable, features frozen)
        self.model.scheduler_init(optimize_phase=True)
        self.model.train()

        bar = tqdm(range(1, iters + 1), desc=f"Phase-B optimize {image_path.name}")
        last_psnr = 0.0

        for it in bar:
            loss, psnr = self.model.optimize_iter(gt)
            last_psnr = float(psnr)

            if (it % log_every) == 0:
                bar.set_postfix({"loss": f"{loss.item():.6f}", "psnr": f"{psnr:.3f}"})
                if self._wandb:
                    self._wandb.log({"phaseB/loss": float(loss.item()),
                                     "phaseB/psnr": float(psnr),
                                     "phaseB/iter": it}, step=it)

        # Save final RGB preview
        with torch.no_grad():
            outF = self.model.forward(render_colors=True)
            imgF = (outF.view(3, -1) + self.model.image_mean).view(3, self.H, self.W).clamp(0, 1)
        self._save_rgb(imgF, self.out_dir / f"{image_path.stem}_fitting.png")

        return last_psnr

    # ---------- eval during Phase-A (EM-init, then short refine) ----------
    def eval_phaseB_on_test_set(self, global_iter: int):
        self._prepare_fixed_eval_set()
        eval_root = self.out_dir / "eval" / f"iter_{global_iter:06d}"
        (eval_root / "images").mkdir(parents=True, exist_ok=True)

        rows = []
        gallery_images = []

        psnr_init_list, psnr_final_list = [], []
        ssim_init_list, ssim_final_list = [], []

        for pstr in self._fixed_eval_paths:
            p = Path(pstr)
            gt = _rgb_path_to_ycbcr(p, (self.W, self.H)).to(self.device)

            # clone → EM-init recon (per channel, per cluster) → quick refine
            psnr0, ssim0, psnrF, ssimF, rgb_gt, rgb_init, rgb_final = \
                self._short_phaseB_eval_with_images(gt)

            psnr_init_list.append(psnr0); psnr_final_list.append(psnrF)
            ssim_init_list.append(ssim0); ssim_final_list.append(ssimF)

            # save to disk
            stem = p.stem
            rgb_gt.save(eval_root / "images" / f"{stem}_gt.png")
            rgb_init.save(eval_root / "images" / f"{stem}_init.png")
            rgb_final.save(eval_root / "images" / f"{stem}_final.png")

            rows.append([str(p), psnr0, ssim0, psnrF, ssimF])

            if self._wandb:
                cap = f"{p.name} | init PSNR:{psnr0:.2f} SSIM:{ssim0:.4f} → final PSNR:{psnrF:.2f} SSIM:{ssimF:.4f}"
                gallery_images += [
                    self._wandb.Image(rgb_gt,   caption=f"{cap} | GT"),
                    self._wandb.Image(rgb_init, caption=f"{cap} | INIT"),
                    self._wandb.Image(rgb_final,caption=f"{cap} | FINAL"),
                ]

        # means
        def _mean(x): return float(np.mean(x)) if len(x) else float("nan")
        mean_psnr_init  = _mean(psnr_init_list)
        mean_psnr_final = _mean(psnr_final_list)
        mean_ssim_init  = _mean(ssim_init_list)
        mean_ssim_final = _mean(ssim_final_list)

        # print + file
        msg = (f"[iter {global_iter}] EVAL on {len(rows)} images | "
               f"init PSNR:{mean_psnr_init:.2f} SSIM:{mean_ssim_init:.4f} | "
               f"final PSNR:{mean_psnr_final:.2f} SSIM:{mean_ssim_final:.4f}")
        print(msg)
        with open(eval_root / "metrics.txt", "w") as f:
            print(msg, file=f)
            for r in rows:
                print(f"{Path(r[0]).name:30s}  "
                      f"initPSNR={r[1]:.2f} initSSIM={r[2]:.4f}  "
                      f"finalPSNR={r[3]:.2f} finalSSIM={r[4]:.4f}", file=f)

        # wandb logging
        if self._wandb:
            table = self._wandb.Table(
                data=rows, columns=["path","psnr_init","ssim_init","psnr_final","ssim_final"]
            )
            self._wandb.log({
                "iter": global_iter,
                "eval/num_images": int(len(rows)),
                "eval/mean_psnr_init":  mean_psnr_init,
                "eval/mean_ssim_init":  mean_ssim_init,
                "eval/mean_psnr_final": mean_psnr_final,
                "eval/mean_ssim_final": mean_ssim_final,
                "eval/table": table,
                "eval/gallery": gallery_images
            }, step=global_iter)

    # ---------- short Phase-B on a CLONED model; returns metrics + PIL images ----------
    def _short_phaseB_eval_with_images(self, gt: torch.Tensor):
        # clone state
        m = GaussianBasis(loss_type=self.args.loss, opt_type="adan",
                          num_points=self.args.num_points, num_comps=self.num_comps,
                          H=self.H, W=self.W, BLOCK_H=16, BLOCK_W=16,
                          device=self.device, lr=self.args.lr).to(self.device)
        m.load_state_dict(self.model.state_dict())

        # EM-based recon init (no grad): set image_mean to stacked per-channel EM projection
        self._init_model_for_image_em(gt, model_override=m)

        with torch.no_grad():
            out0 = m.forward(render_colors=True)
            img0 = (out0.view(3, -1) + m.image_mean).view(3, self.H, self.W).clamp(0, 1)
            mse0 = F.mse_loss(img0, gt).item()
            psnr0 = 10 * math.log10(1.0 / (mse0 + 1e-12))
            ssim0 = ms_ssim(img0.unsqueeze(0), gt.unsqueeze(0), data_range=1, size_average=True).item()

        # refine briefly
        m.scheduler_init(optimize_phase=True)
        m.train()
        for _ in range(self.args.eval_opt_iters):
            out = m.forward(render_colors=True)
            img = (out.view(3, -1) + m.image_mean).view(3, self.H, self.W)
            loss = F.mse_loss(img, gt)
            loss.backward()
            m.optimizer.step()
            m.optimizer.zero_grad(set_to_none=True)
            m.scheduler.step()

        # final
        m.eval()
        with torch.no_grad():
            outF = m.forward(render_colors=True)
            imgF = (outF.view(3, -1) + m.image_mean).view(3, self.H, self.W).clamp(0, 1)
            mseF = F.mse_loss(imgF, gt).item()
            psnrF = 10 * math.log10(1.0 / (mseF + 1e-12))
            ssimF = ms_ssim(imgF.unsqueeze(0), gt.unsqueeze(0), data_range=1, size_average=True).item()

        # PIL images for gallery
        rgb_gt    = self._ycbcr_to_rgb_pil(gt)
        rgb_init  = self._ycbcr_to_rgb_pil(img0)
        rgb_final = self._ycbcr_to_rgb_pil(imgF)

        return psnr0, ssim0, psnrF, ssimF, rgb_gt, rgb_init, rgb_final

    @torch.no_grad()
    def _init_model_for_image_em(self, gt_ycbcr: torch.Tensor, model_override=None):
        """
        EM-based per-channel cluster selection + projection init:
          For each channel (Y/Cb/Cr):
            k* = argmin ||(I - P_k)(x - v_k)||^2 using EM (V_k, v_k).
            rec_ch = v_k* + Proj_{V_k*}(x - v_k*).
          Stack rec_Y, rec_Cb, rec_Cr → set as model.image_mean (so init == EM recon).
        """
        model = self.model if model_override is None else model_override

        Y  = gt_ycbcr[0]  # (H,W)
        Cb = gt_ycbcr[1]
        Cr = gt_ycbcr[2]

        kY,  recY  = _choose_cluster_em_and_reconstruct(Y,  self.VsY_em,  self.vY_em,  self._em_top_t)
        kCb, recCb = _choose_cluster_em_and_reconstruct(Cb, self.VsCb_em, self.vCb_em, self._em_top_t)
        kCr, recCr = _choose_cluster_em_and_reconstruct(Cr, self.VsCr_em, self.vCr_em, self._em_top_t)

        rec_img = _stack_recons(recY, recCb, recCr, self.H, self.W).to(self.device)  # (3,H,W)

        # set the model's mean buffer to the EM reconstruction (init image)
        model.image_mean = rec_img.view(3, -1).detach()

        # Optional: clear/zero colors for a clean start if the model exposes a method
        if hasattr(model, "zero_colors"):
            model.zero_colors()

    def load_model(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)
        # quick sanity check for num_comps
        want = self.num_comps
        got = state.get("_features_dc", None)
        if got is not None and hasattr(got, "shape") and got.shape[0] != want:
            raise RuntimeError(
                f"Checkpoint num_comps mismatch: ckpt {got.shape[0]} vs current {want}. "
                f"Make sure you trained with the same (3*K*J) configuration."
            )
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _save_rgb(img_ycbcr_01: torch.Tensor, path: Path):
        ycbcr = (img_ycbcr_01.clamp(0,1).detach().cpu().numpy() * 255.0).transpose(1,2,0)
        rgb = color.ycbcr2rgb(ycbcr) * 255.0
        Image.fromarray(np.clip(rgb,0,255).astype(np.uint8)).save(path)

    @staticmethod
    def _ycbcr_to_rgb_pil(img_ycbcr_01: torch.Tensor) -> Image.Image:
        ycbcr = (img_ycbcr_01.clamp(0,1).detach().cpu().numpy() * 255.0).transpose(1,2,0)
        rgb = color.ycbcr2rgb(ycbcr) * 255.0
        return Image.fromarray(np.clip(rgb,0,255).astype(np.uint8))

# ----------------------------- main -----------------------------

def main(argv):
    args = parse_args(argv)
    trainer = ProjectiveTrainer(args)

    # If we're not training and user supplied a model, load it
    if (args.skip_train or args.eval_only or args.image_path is not None) and args.model_path:
        trainer.load_model(args.model_path)

    # A) regular training path
    if not args.skip_train and not args.eval_only:
        trainer.train(args.iterations)

    # B) Phase-B on the whole test set (one pass)
    if args.eval_only:
        trainer.eval_phaseB_on_test_set(global_iter=0)

    # C) Optional: Phase-B on a single image
    if args.image_path is not None:
        trainer.optimize_single(Path(args.image_path), iters=args.eval_opt_iters)

if __name__ == "__main__":
    main(sys.argv[1:])