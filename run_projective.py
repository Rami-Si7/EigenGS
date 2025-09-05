# run_projective.py
"""
Projective-clustering variant of the EigenGS trainer with full Weights & Biases logging.

- Phase-A trains on eigenimages from projective clusters
  * train_mode="all": single model on all clusters (K*J comps)
  * train_mode="clusters": single model on a subset of clusters (sum J comps)
- Eval (every --eval_every iters) reconstructs the ENTIRE test set:
  1) force the correct cluster (from clusters/cluster_xx/test_imgs) or assign once
  2) project to that cluster's subspace to get codes
  3) embed (if training on all clusters) zeros elsewhere
  4) colors <- codes * learned features (global geometry shared)
  5) image = out * (max-min) + (Σ codes)*min + mean

W&B:
  - Logs train/loss, train/psnr
  - Logs eval/mean_psnr_init, eval/mean_psnr_final, eval/mean_ssim_init, eval/mean_ssim_final
  - Logs per-image metrics table + GT/INIT/FINAL gallery (fixed eval set, same every eval)
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

# ----- projective helpers -----
from projective_utils import (
    load_top_level, load_cluster_arrs,
    index_map_all, index_map_subset,
    rgb_path_to_ycbcr, choose_cluster_ycbcr,
    compute_codes, embed_codes, codes_to_colors
)

# ----- Weights & Biases -----
import wandb

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# ----------------------------- Argparse -----------------------------

def parse_args(argv):
    p = argparse.ArgumentParser("Projective-clustering EigenGS trainer")

    # projective dataset root (output dir of projective_parse.py)
    p.add_argument("--projective_dir", type=str, required=True,
                   help="Directory produced by projective_parse.py (contains projective_basis.pt, arrs.npy, clusters/*)")
    p.add_argument("--train_mode", type=str, choices=["all", "clusters"], default="all",
                   help="Train on all clusters (K*J comps) or on a subset of clusters (sum J)")
    p.add_argument("--clusters", type=str, default=None,
                   help="Comma-separated cluster IDs when train_mode=clusters, e.g., '0,2,5'")

    # training knobs
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
                   help="Optional external folder of test images; if omitted, uses projective_dir/clusters/.../test_imgs")
    p.add_argument("--test_recursive", action="store_true")
    p.add_argument("--test_glob", type=str, default=None)

    # assignment weights for Y,Cb,Cr
    p.add_argument("--assign_alphas", type=str, default="1.0,0.25,0.25",
                   help="Weights for (Y,Cb,Cr) in cluster selection and joint costs")

    # logging
    p.add_argument("--out_dir", type=str, default="./models_projective")

    # Weights & Biases
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default=os.getenv("WANDB_PROJECT", "eigens"))
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)

    return p.parse_args(argv)

# ----------------------------- Trainer -----------------------------

class ProjectiveTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

        self.projective_dir = Path(args.projective_dir)
        (self.means, self.bases, (W,H), self.K, self.J,
         self.norm_mins, self.norm_maxs, self.arrs_top) = load_top_level(self.projective_dir)

        # move artifacts to device once
        self.means     = self.means.to(self.device, non_blocking=True)
        self.bases     = self.bases.to(self.device, non_blocking=True)
        self.norm_mins = self.norm_mins.to(self.device)
        self.norm_maxs = self.norm_maxs.to(self.device)
        self.arrs_top  = self.arrs_top.to(self.device)

        self.W, self.H = W, H
        self.d = W*H
        self.assign_alphas = tuple(float(x) for x in args.assign_alphas.split(","))

        # ----- build Phase-A training targets -----
        if args.train_mode == "all":
            self.gt_arrs = self.arrs_top.clone()             # (K*J, 3, H, W)
            self.index_map = index_map_all(self.K, self.J)
            self.num_comps = self.K * self.J
            self.cluster_ids = list(range(self.K))           # for eval grouping
        else:
            assert args.clusters is not None, "--clusters must be set when train_mode=clusters"
            self.cluster_ids = [int(x) for x in args.clusters.split(",")]
            self.gt_arrs = load_cluster_arrs(self.projective_dir, self.cluster_ids)  # (sumJ,3,H,W)
            self.index_map = index_map_subset(self.cluster_ids, self.J)
            self.num_comps = len(self.index_map)

        self.gt_arrs = self.gt_arrs.to(self.device)

        # ----- out dir -----
        mode_tag = args.train_mode if args.train_mode=="all" else f"clusters_{'-'.join(map(str, self.cluster_ids))}"
        self.out_dir = Path(args.out_dir) / f"proj_{self.projective_dir.name}_{mode_tag}_P{args.num_points}_I{args.iterations}_seed{args.seed}"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # ----- model -----
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
            run_name = args.run_name or f"{self.projective_dir.name}-{mode_tag}-P{args.num_points}-I{args.iterations}-seed{args.seed}"
            self.wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                name=run_name,
                config={
                    "projective_dir": str(self.projective_dir),
                    "train_mode": args.train_mode,
                    "clusters": self.cluster_ids if args.train_mode=="clusters" else "all",
                    "num_points": args.num_points,
                    "iterations": args.iterations,
                    "lr": args.lr,
                    "eval_every": args.eval_every,
                    "eval_opt_iters": args.eval_opt_iters,
                    "seed": args.seed,
                    "assign_alphas": self.assign_alphas,
                    "H": self.H, "W": self.W,
                    "num_comps": self.num_comps,
                },
                dir=str(self.out_dir),
                reinit=False,
            )
            wandb.watch(self.model, log=None, log_freq=0, log_graph=False)

        # caches
        self._test_images_cache = None
        self._rng = np.random.default_rng(args.seed)

        
        self._fixed_eval_by_cluster: Optional[Dict[int, List[str]]] = None
        self._fixed_eval_manifest = self.out_dir / "eval_set.json"

    # ---------- test image discovery ----------
    def _discover_test_images(self) -> List[Path]:
        if self._test_images_cache is not None:
            return self._test_images_cache

        if self.args.test_glob:
            cand = sorted(Path().glob(self.args.test_glob))
        elif self.args.test_dir:
            root = Path(self.args.test_dir)
            it = root.rglob("*") if self.args.test_recursive else root.glob("*")
            cand = sorted([p for p in it if p.suffix.lower() in IMG_EXTS and p.is_file()])
        else:
            pc_root = self.projective_dir / "clusters"
            if pc_root.exists():
                cand = sorted([q for q in pc_root.rglob("test_imgs/*") if q.suffix.lower() in IMG_EXTS])
            else:
                root = self.projective_dir.parent
                cand = sorted([q for q in root.rglob("*") if q.suffix.lower() in IMG_EXTS])

        self._test_images_cache = cand
        return cand

    def _collect_cluster_test_images(self) -> Dict[int, List[Path]]:
        out: Dict[int, List[Path]] = {k: [] for k in range(self.K)}
        base = self.projective_dir / "clusters"
        if base.exists():
            for k in range(self.K):
                cdir = base / f"cluster_{k:02d}" / "test_imgs"
                if cdir.exists():
                    imgs = sorted([p for p in cdir.glob("*") if p.suffix.lower() in IMG_EXTS])
                    out[k] = imgs
        return out

    @torch.no_grad()
    def _assign_cluster(self, img_path: Path) -> int:
        """Assign a cluster by residual (YCbCr joint); used only when test folder doesn't encode cluster."""
        y = rgb_path_to_ycbcr(img_path, (self.W, self.H)).to(self.device)
        return int(choose_cluster_ycbcr(y, self.means, self.bases, self.assign_alphas))

    def _prepare_fixed_eval_set(self):
        """
        Build a fixed, deterministic eval set (dict cluster -> list of file paths).
        Saved to eval_set.json so it's identical across every eval step and restarts.
        """
        if self._fixed_eval_by_cluster is not None:
            return

        if self._fixed_eval_manifest.exists():
            with open(self._fixed_eval_manifest, "r") as f:
                loaded = json.load(f)
            # validate paths still exist; filter missing
            fixed = {int(k): [p for p in v if Path(p).exists()] for k, v in loaded.items()}
            self._fixed_eval_by_cluster = fixed
            return

        by_cluster = self._collect_cluster_test_images()
        fixed: Dict[int, List[str]] = {k: [] for k in range(self.K)}

        if any(len(v) > 0 for v in by_cluster.values()):
            # preferred: use the parser's per-cluster test splits
            for k in (self.cluster_ids if self.args.train_mode=="clusters" else range(self.K)):
                fixed[k] = [str(p) for p in by_cluster.get(k, [])]
        else:
            # fall back: discover and assign once
            cand = self._discover_test_images()
            for p in cand:
                k = self._assign_cluster(p)
                fixed.setdefault(k, []).append(str(p))

        # deterministic order
        for k in fixed:
            fixed[k] = sorted(fixed[k])

        # persist
        with open(self._fixed_eval_manifest, "w") as f:
            json.dump({int(k): v for k, v in fixed.items()}, f, indent=2)

        self._fixed_eval_by_cluster = fixed
    # ---------- Phase A ----------
    def train(self, iterations: int):
        self.model.train()
        bar = tqdm(range(1, iterations + 1), desc="Phase-A (learn eigenimages)")

        for it in bar:
            # Use the model's canonical Phase-A step (matches original repo behavior)
            loss, psnr = self.model.train_iter(self.gt_arrs)

            if self.wandb_run:
                wandb.log({"train/loss": float(loss.item()), "train/psnr": float(psnr), "iter": it})

            if it % 10 == 0:
                bar.set_postfix({"loss": f"{loss.item():.6f}", "psnr": f"{psnr:.3f}"})

            # periodic eval (cluster-subspace recon + short Phase-B refine)
            if self.args.eval_every and (it % self.args.eval_every == 0):
                self.eval_phaseB_on_test_set(global_iter=it)

        # save
        torch.save({"model_state_dict": self.model.state_dict()},
                  self.out_dir / "gaussian_model_phaseA.pth.tar")

        # visualize learned components
        self._dump_components()


    # ---------- Phase B (single image optimize) ----------
    def optimize_single(self, image_path: Path, iters: int, log_every: int = 10):
        # Prepare target (YCbCr in [0,1]) and cluster-based init
        gt = rgb_path_to_ycbcr(image_path, (self.W, self.H)).to(self.device)
        self._init_model_for_image(gt)

        # Switch the model to Phase-B mode (colors trainable, features frozen)
        self.model.scheduler_init(optimize_phase=True)
        self.model.train()

        bar = tqdm(range(1, iters + 1), desc=f"Phase-B optimize {image_path.name}")
        last_psnr = 0.0

        # (Optional) iter-0 snapshot, if you want:
        # with torch.no_grad():
        #     out0 = self.model.forward(render_colors=True)
        #     img0 = (out0.view(3, -1) + self.model.image_mean).view(3, self.H, self.W).clamp(0, 1)
        #     self._save_rgb(img0, self.out_dir / f"{image_path.stem}_init.png")

        for it in bar:
            # Use the model's canonical Phase-B step (matches original repo behavior)
            loss, psnr = self.model.optimize_iter(gt)
            last_psnr = float(psnr)

            if (it % log_every) == 0:
                bar.set_postfix({"loss": f"{loss.item():.6f}", "psnr": f"{psnr:.3f}"})
                if self.wandb_run:
                    wandb.log({"phaseB/loss": float(loss.item()),
                              "phaseB/psnr": float(psnr),
                              "phaseB/iter": it}, step=it)

        # Save final RGB preview (compute from current model state)
        with torch.no_grad():
            outF = self.model.forward(render_colors=True)
            imgF = (outF.view(3, -1) + self.model.image_mean).view(3, self.H, self.W).clamp(0, 1)
        self._save_rgb(imgF, self.out_dir / f"{image_path.stem}_fitting.png")

        return last_psnr

    # # ---------- Phase A ----------
    # def train(self, iterations: int):
    #     self.model.train()
    #     bar = tqdm(range(1, iterations+1), desc="Phase-A (learn eigenimages)")
    #     for it in bar:
    #         img = self.model.forward(render_colors=False)    # (num_comps,3,H,W)
    #         loss = F.mse_loss(img, self.gt_arrs)
    #         loss.backward()
    #         with torch.no_grad():
    #             mse = F.mse_loss(img, self.gt_arrs)
    #             psnr = 10 * math.log10(1.0 / (mse.item() + 1e-12))

    #         if self.wandb_run:
    #             wandb.log({"train/loss": float(loss.item()), "train/psnr": float(psnr), "iter": it})

    #         self.model.optimizer.step()
    #         self.model.optimizer.zero_grad(set_to_none=True)
    #         self.model.scheduler.step()

    #         if it % 10 == 0:
    #             bar.set_postfix({"loss": f"{loss.item():.6f}", "psnr": f"{psnr:.3f}"})

    #         if self.args.eval_every and (it % self.args.eval_every == 0):
    #             self.eval_phaseB_on_test_set(global_iter=it)

    #     # save
    #     torch.save({"model_state_dict": self.model.state_dict()},
    #                self.out_dir / "gaussian_model_phaseA.pth.tar")

    #     # visualize learned components
    #     self._dump_components()

    @torch.no_grad()
    def _dump_components(self):
        vis_dir = self.out_dir / "vis_components"
        vis_dir.mkdir(parents=True, exist_ok=True)
        comps = self.model.forward(render_colors=False).clamp(0,1)
        for i in range(comps.shape[0]):
            for ch in range(3):
                arr = (comps[i, ch].cpu().numpy() * 255.0).astype(np.uint8)
                Image.fromarray(arr, mode="L").save(vis_dir / f"{i:04d}_ch{ch}.png")

    # # ---------- Phase B (single image optimize) ----------
    # def optimize_single(self, image_path: Path, iters: int):
    #     gt = rgb_path_to_ycbcr(image_path, (self.W, self.H)).to(self.device)
    #     self._init_model_for_image(gt)
    #     self.model.scheduler_init(optimize_phase=True)
    #     self.model.train()

    #     bar = tqdm(range(1, iters+1), desc=f"Phase-B optimize {image_path.name}")
    #     psnr = 0.0
    #     for it in bar:
    #         out = self.model.forward(render_colors=True)
    #         img = (out.view(3, -1) + self.model.image_mean).view(3, self.H, self.W).clamp(0,1)
    #         loss = F.mse_loss(img, gt)
    #         loss.backward()
    #         with torch.no_grad():
    #             mse = F.mse_loss(img, gt)
    #             psnr = 10 * math.log10(1.0 / (mse.item() + 1e-12))
    #         self.model.optimizer.step()
    #         self.model.optimizer.zero_grad(set_to_none=True)
    #         self.model.scheduler.step()
    #         if it % 10 == 0:
    #             bar.set_postfix({"loss": f"{loss.item():.6f}", "psnr": f"{psnr:.3f}"})

    #     self._save_rgb(img, self.out_dir / f"{image_path.stem}_fitting.png")
    #     return psnr
    
    # ---------- eval during Phase-A (full test set, fixed, logged to W&B) ----------
    def eval_phaseB_on_test_set(self, global_iter: int):
        self._prepare_fixed_eval_set()
        eval_root = self.out_dir / "eval" / f"iter_{global_iter:06d}"
        eval_root.mkdir(parents=True, exist_ok=True)

        rows = []
        gallery_images = []

        # restrict to trained clusters if in 'clusters' mode; else all
        target_clusters = self.cluster_ids if self.args.train_mode == "clusters" else list(range(self.K))

        # metrics accumulators
        psnr_init_list, psnr_final_list = [], []
        ssim_init_list, ssim_final_list = [], []

        for k in target_clusters:
            paths = self._fixed_eval_by_cluster.get(int(k), [])
            if not paths:
                continue

            save_dir = eval_root / f"cluster_{k:02d}"
            save_dir.mkdir(parents=True, exist_ok=True)

            for pstr in paths:
                p = Path(pstr)
                gt = rgb_path_to_ycbcr(p, (self.W, self.H)).to(self.device)

                # clone → init (forced cluster) → quick refine
                psnr0, ssim0, psnrF, ssimF, rgb_gt, rgb_init, rgb_final = \
                    self._short_phaseB_eval_with_images(gt, override_k=k)

                # accum
                psnr_init_list.append(psnr0); psnr_final_list.append(psnrF)
                ssim_init_list.append(ssim0); ssim_final_list.append(ssimF)

                # save to disk
                stem = p.stem
                rgb_gt.save(save_dir / f"{stem}_gt.png")
                rgb_init.save(save_dir / f"{stem}_init.png")
                rgb_final.save(save_dir / f"{stem}_final.png")

                # table row
                rows.append([str(p), int(k), psnr0, ssim0, psnrF, ssimF])

                # wandb images (caption carries metrics)
                if self.wandb_run:
                    cap = f"{p.name} | k={k} | init PSNR:{psnr0:.2f} SSIM:{ssim0:.4f} → final PSNR:{psnrF:.2f} SSIM:{ssimF:.4f}"
                    gallery_images += [
                        wandb.Image(rgb_gt,   caption=f"{cap} | GT"),
                        wandb.Image(rgb_init, caption=f"{cap} | INIT"),
                        wandb.Image(rgb_final,caption=f"{cap} | FINAL"),
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
                print(f"{Path(r[0]).name:30s}  k={r[1]:>2}  "
                      f"initPSNR={r[2]:.2f} initSSIM={r[3]:.4f}  "
                      f"finalPSNR={r[4]:.2f} finalSSIM={r[5]:.4f}", file=f)

        # wandb logging
        if self.wandb_run:
            table = wandb.Table(
                data=rows,
                columns=["path","cluster","psnr_init","ssim_init","psnr_final","ssim_final"]
            )
            wandb.log({
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
    def _short_phaseB_eval_with_images(self, gt: torch.Tensor, override_k: Optional[int] = None):
        # clone state
        m = GaussianBasis(loss_type=self.args.loss, opt_type="adan",
                          num_points=self.args.num_points, num_comps=self.num_comps,
                          H=self.H, W=self.W, BLOCK_H=16, BLOCK_W=16,
                          device=self.device, lr=self.args.lr).to(self.device)
        m.load_state_dict(self.model.state_dict())

        # init (no grad); force cluster
        self._init_model_for_image(gt, model_override=m, override_k=override_k)

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

        # PIL images for wandb/gallery
        rgb_gt    = self._ycbcr_to_rgb_pil(gt)
        rgb_init  = self._ycbcr_to_rgb_pil(img0)
        rgb_final = self._ycbcr_to_rgb_pil(imgF)

        return psnr0, ssim0, psnrF, ssimF, rgb_gt, rgb_init, rgb_final

    def _init_model_for_image(self, gt_ycbcr: torch.Tensor, model_override=None, override_k: int | None = None):
        model = self.model if model_override is None else model_override
        # choose cluster (or force)
        if override_k is not None:
            k_star = int(override_k)
        else:
            k_star = choose_cluster_ycbcr(gt_ycbcr, self.means, self.bases, self.assign_alphas)

        # compute codes in that cluster
        codes_k = compute_codes(gt_ycbcr.to(self.device),
                                self.means[k_star].to(self.device),
                                self.bases[k_star].to(self.device))                  # (3,J)

        if self.args.train_mode == "all":
            codes_full = embed_codes(codes_k, self.index_map, k_star)                # (3,K*J)
        else:
            if not hasattr(self, "_subset_map"):
                self._subset_map = { (k,j):i for i,(k,j) in enumerate(self.index_map) }
            codes_full = torch.zeros((3, self.num_comps), device=self.device)
            for j in range(self.J):
                key = (k_star, j)
                if key in self._subset_map:
                    idx = self._subset_map[key]
                    codes_full[:, idx] = codes_k[:, j]

        # set colors + buffers
        codes_to_colors(model, codes_full, self.norm_mins, self.norm_maxs,
                        self.means[k_star, :, :].view(3, self.d))
    def load_model(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)
        # quick sanity check for num_comps
        want = self.num_comps
        got = state.get("_features_dc", None)
        if got is not None and got.shape[0] != want:
            raise RuntimeError(f"Checkpoint num_comps mismatch: ckpt {_features_dc.shape[0]} vs current {want}. "
                              f"Make sure --train_mode/--clusters match the training config.")
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
        # Run a single eval pass over the fixed test set
        trainer.eval_phaseB_on_test_set(global_iter=0)

    # C) Optional: Phase-B on a single image
    if args.image_path is not None:
        trainer.optimize_single(Path(args.image_path), iters=args.eval_opt_iters)

if __name__ == "__main__":
    main(sys.argv[1:])
