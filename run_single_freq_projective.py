#!/usr/bin/env python3
import math, time, os, sys, uuid, random, copy
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import color
from tqdm import tqdm

from gaussianbasis_single_freq_base import GaussianBasis
from tracker import LogWriter, PSNRTracker
from utils import *
from pytorch_msssim import ms_ssim

# Optional W&B
import wandb

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ---------------- helpers ----------------

def _ycbcr_tensor_from_path(path: Path, W: int, H: int, device) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    ycbcr = (color.rgb2ycbcr(np.array(img)) / 255.0).transpose(2, 0, 1)
    return torch.tensor(ycbcr, dtype=torch.float32, device=device)  # (3,H,W)

def _choose_cluster_y(y_flat: torch.Tensor, means_y: torch.Tensor, Uy: torch.Tensor) -> int:
    """
    y_flat: (d,)    flattened Y in [0,1]
    means_y: (K,d)
    Uy:      (K,d,J)  column-orthonormal
    """
    K, d, J = Uy.shape
    dists = torch.empty(K, device=y_flat.device)
    for k in range(K):
        Z  = y_flat - means_y[k]        # (d,)
        proj = (Z @ Uy[k]) @ Uy[k].T    # (d,)
        R = Z - proj
        dists[k] = torch.linalg.vector_norm(R)
    return int(torch.argmin(dists).item())

def _get_norm_minmax(pb):
    # Prefer values saved by the parser; otherwise compute global channel min/max from bases
    if ("norm_mins" in pb) and ("norm_maxs" in pb):
        mins = torch.as_tensor(pb["norm_mins"]).float()  # (3,)
        maxs = torch.as_tensor(pb["norm_maxs"]).float()  # (3,)
        return mins, maxs
    bases = pb["bases"].float()      # (K,3,J,d) rows
    mins = torch.empty(3)
    maxs = torch.empty(3)
    for ch in range(3):
        x = bases[:, ch].reshape(-1)   # all K*J*d values of that channel
        mins[ch] = x.min()
        maxs[ch] = x.max()
    return mins, maxs


# ---------------- Trainer ----------------

class GaussianTrainer:
    def __init__(self, args, image_path=None, num_points=2000, iterations=30000, model_path=None):
        self.args = args
        self.dataset_path = Path(args.dataset)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Training targets: (K*J, 3, H, W) produced by the projective parser
        arrs_path = self.dataset_path / "arrs.npy"
        self.gt_arrs = torch.from_numpy(np.load(arrs_path)).to(self.device)

        random_string = str(uuid.uuid4())[:6]
        if image_path is not None:
            self.image_path = Path(image_path)
            self.image_name = self.image_path.stem
            img = Image.open(self.image_path).convert("RGB")
            img_arr = (color.rgb2ycbcr(np.array(img)) / 255.0).transpose(2, 0, 1)
            self.gt_image = torch.tensor(img_arr, dtype=torch.float32, device=self.device)
            model_dir = Path(f"./models/recons/{self.image_name}-{self.dataset_path.name}-{num_points}-{args.iterations}-{random_string}")
        else:
            self.image_path = None
            self.image_name = None
            self.gt_image = None
            model_dir = Path(f"./models/single-freq/{self.dataset_path.name}-{num_points}-{args.iterations}-{random_string}")

        self.num_points = num_points
        self.num_comps = self.gt_arrs.shape[0]    # = K*J
        self.H, self.W = self.gt_arrs.shape[2], self.gt_arrs.shape[3]
        BLOCK_H, BLOCK_W = 16, 16
        self.iterations = iterations

        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.gaussian_model = GaussianBasis(
            loss_type="L2", opt_type="adan", num_points=self.num_points,
            H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            device=self.device, lr=args.lr, num_comps=self.num_comps
        ).to(self.device)
        self.gaussian_model.scheduler_init()
        self.logwriter = LogWriter(self.model_dir)
        self.psnr_tracker = PSNRTracker(self.logwriter)
        self.logwriter.write(f"Model Dir ID: {random_string}")

        if model_path is not None:
            self.model_path = Path(model_path)
            self.logwriter.write(f"Model loaded from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.gaussian_model.load_state_dict(checkpoint['model_state_dict'])

        # W&B
        self.wandb_run = None
        if args.wandb:
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
                },
                dir=str(self.model_dir),
                reinit=False,
            )
            wandb.watch(self.gaussian_model, log=None, log_freq=0, log_graph=False)

        self._test_images_cache = None
        self._rng = np.random.default_rng(args.seed if args.seed is not None else 0)

        # ---- load projective basis once ----
        pb_path = self.dataset_path / "projective_basis.pt"
        assert pb_path.exists(), f"projective_basis.pt not found in {self.dataset_path}"
        pb = torch.load(pb_path, map_location="cpu", weights_only=False)
        self.PB = {
            "means":  pb["means"].float().to(self.device),   # (K,3,d)
            "bases":  pb["bases"].float().to(self.device),   # (K,3,J,d) rows
            "K":      int(pb["K"]),
            "J":      int(pb["J"]),
            "img_sz": tuple(pb["img_size"]),
        }
        # Y-flat orth bases (d,J) for distance
        K, J = self.PB["K"], self.PB["J"]
        d = self.W * self.H
        Uy = []
        for k in range(K):
            Uk = self.PB["bases"][k, 0].T  # (d,J)
            Q, _ = torch.linalg.qr(Uk, mode='reduced')
            Uy.append(Q)
        self.PB["Uy"] = torch.stack(Uy, dim=0)                # (K,d,J)
        self.PB["means_y"] = self.PB["means"][:, 0, :]        # (K,d)
        self.PB["mins"], self.PB["maxs"] = _get_norm_minmax(pb)  # (3,), (3,)
        self.logwriter.write(f"[PB] loaded K={K} J={J}  mins={self.PB['mins'].tolist()}  maxs={self.PB['maxs'].tolist()}")

    # ---------------- files discovery ----------------
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

    # ---------------- Phase-A ----------------
    def train(self):
        progress_bar = tqdm(range(1, self.iterations+1), desc="Training progress")
        self.gaussian_model.train()
        start_time = time.time()
        for it in range(1, self.iterations+1):
            loss, psnr = self.gaussian_model.train_iter(self.gt_arrs)
            if self.wandb_run:
                wandb.log({"train/loss": float(loss.item()), "train/psnr": float(psnr), "iter": it})
            with torch.no_grad():
                progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                progress_bar.update(1)
            if self.args.eval_every and (it % self.args.eval_every == 0):
                self.eval_phaseB_on_test_set(global_iter=it)
                self._save_checkpoint(step=it, tag="phaseA")

        end_time = time.time() - start_time
        progress_bar.close()
        self.logwriter.write("Training Complete in {:.4f}s".format(end_time))
        torch.save({'model_state_dict': self.gaussian_model.state_dict()}, self.model_dir / "gaussian_model.pth.tar")
        if self.wandb_run:
            wandb.save(str(self.model_dir / "gaussian_model.pth.tar"), base_path=str(self.model_dir))
        self.vis()

    def vis(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            image = self.gaussian_model()  # (K*J,3,H,W)
        mse_loss = F.mse_loss(image.float(), self.gt_arrs.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(image.float(), self.gt_arrs.float(), data_range=1, size_average=True).item()
        self.logwriter.write("Components Fitting: PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))

        vis_dir = self.model_dir / f"vis_comps"
        vis_dir.mkdir(parents=True, exist_ok=True)
        to_img = torchvision.transforms.ToPILImage()
        array = image.float()
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                to_img(array[i, j]).save(vis_dir / f"{i}-{j}.png")
        return psnr, ms_ssim_value

    # ---------------- Phase-B init (projective) ----------------
    def _project_to_cluster_and_init(self, gt_img_3HW: torch.Tensor):
        """
        gt_img_3HW: (3,H,W) in [0,1]
        Sets model: image_mean (3,d), scale_factor (3,1,1), shift_factor (3,1,1), and colors.
        """
        PB = self.PB
        H, W = self.H, self.W
        d = H * W

        flat = gt_img_3HW.reshape(3, -1)  # (3,d)

        # 1) choose cluster by Y-flat distance
        k = _choose_cluster_y(flat[0], PB["means_y"], PB["Uy"])

        # 2) per-channel projection coefficients α (length J)
        K, J = PB["K"], PB["J"]
        codes = torch.zeros((3, 1, K * J), device=self.device, dtype=torch.float32)
        alphas_sum = torch.zeros(3, device=self.device)

        for ch in range(3):
            m = PB["means"][k, ch]    # (d,)
            B = PB["bases"][k, ch]    # (J,d) rows orthonormal
            x = flat[ch]              # (d,)
            a = (x - m) @ B.T         # (J,)
            alphas_sum[ch] = a.sum()
            start = k * J
            codes[ch, 0, start:start+J] = a

        # 3) scale/shift per channel (broadcastable to CHW)
        mins, maxs = PB["mins"].to(self.device), PB["maxs"].to(self.device)  # (3,)
        scale = (maxs - mins).view(3, 1, 1)
        shift = (alphas_sum * mins).view(3, 1, 1)
        mean  = PB["means"][k]  # (3,d)

        self.gaussian_model.scale_factor = scale
        self.gaussian_model.shift_factor = shift
        self.gaussian_model.image_mean   = mean

        # 4) mix learned features with coefficients → point colors
        with torch.no_grad():
            features = self.gaussian_model.get_features.permute(2, 0, 1)     # (3, KJ, P)
            colors = torch.bmm(codes, features).squeeze(1).transpose(0, 1)   # (P,3)
            self.gaussian_model._colors.copy_(colors)

        return k

    def optimize(self):
        assert self.gt_image is not None, "Please pass --image_path for Phase-B."
        progress_bar = tqdm(range(1, self.iterations+1), desc="Optimizing progress")

        # init from projective basis
        self._project_to_cluster_and_init(self.gt_image)

        self.gaussian_model.scheduler_init(optimize_phase=True)
        self.gaussian_model.train()
        start_time = time.perf_counter()

        self.test(iter=0)
        self.gaussian_model.train()
        for it in range(1, self.iterations+1):
            loss, psnr = self.gaussian_model.optimize_iter(self.gt_image)
            self.psnr_tracker.check(start_time, psnr, it)
            if self.wandb_run:
                wandb.log({"phaseB/loss": float(loss.item()), "phaseB/psnr": float(psnr), "phaseB/iter": it})
            with torch.no_grad():
                if it in [10, 100, 1000]:
                    self.test(iter=it); self.gaussian_model.train()
                if it % 10 == 0:
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

    # ---------------- Eval-on-the-fly (Phase-B short) ----------------
    def _phaseB_eval_single(self, base_state_dict, img_path: Path, opt_iters: int):
        model = GaussianBasis(
            loss_type="L2", opt_type="adan", num_points=self.num_points,
            H=self.H, W=self.W, BLOCK_H=16, BLOCK_W=16,
            device=self.device, lr=self.args.lr, num_comps=self.num_comps
        ).to(self.device)
        model.load_state_dict(base_state_dict)
        model.eval()

        # load image
        gt_img = _ycbcr_tensor_from_path(img_path, self.W, self.H, self.device)

        # project to cluster + init
        PB = self.PB
        flat = gt_img.view(3, -1)
        k = _choose_cluster_y(flat[0], PB["means_y"], PB["Uy"])

        K, J = PB["K"], PB["J"]
        codes = torch.zeros((3, 1, K * J), device=self.device, dtype=torch.float32)
        alphas_sum = torch.zeros(3, device=self.device)
        for ch in range(3):
            m = PB["means"][k, ch]; B = PB["bases"][k, ch]; x = flat[ch]
            a = (x - m) @ B.T
            alphas_sum[ch] = a.sum()
            codes[ch, 0, k*J:k*J+J] = a

        mins, maxs = PB["mins"].to(self.device), PB["maxs"].to(self.device)
        model.shift_factor = (alphas_sum * mins).view(3,1,1)
        model.scale_factor = (maxs - mins).view(3,1,1)
        model.image_mean   = PB["means"][k]  # (3,d)

        with torch.no_grad():
            feats = model.get_features.permute(2,0,1)                # (3,KJ,P)
            colors = torch.bmm(codes, feats).squeeze(1).transpose(0,1)
            model._colors.copy_(colors)

        # init metrics
        with torch.no_grad():
            out0 = model(render_colors=True)
            img0 = (out0.view(3, -1) + model.image_mean).view(3, self.H, self.W).clamp(0, 1)
            mse0 = F.mse_loss(img0, gt_img).item()
            psnr0 = 10 * math.log10(1.0 / (mse0 + 1e-12))
            ssim0 = ms_ssim(img0.unsqueeze(0), gt_img.unsqueeze(0), data_range=1, size_average=True).item()

        # short refine
        model.scheduler_init(optimize_phase=True)
        model.train()
        for _ in range(opt_iters):
            _, _ = model.optimize_iter(gt_img)

        # final metrics
        model.eval()
        with torch.no_grad():
            outF = model(render_colors=True)
            imgF = (outF.view(3, -1) + model.image_mean).view(3, self.H, self.W).clamp(0, 1)
            mseF = F.mse_loss(imgF, gt_img).item()
            psnrF = 10 * math.log10(1.0 / (mseF + 1e-12))
            ssimF = ms_ssim(imgF.unsqueeze(0), gt_img.unsqueeze(0), data_range=1, size_average=True).item()

        rgb_init = self._ycbcr_to_rgb_image(img0)
        rgb_final = self._ycbcr_to_rgb_image(imgF)

        return {
            "path": str(img_path),
            "loss_init": mse0, "psnr_init": psnr0, "ssim_init": ssim0,
            "loss_final": mseF, "psnr_final": psnrF, "ssim_final": ssimF,
            "rgb_init": rgb_init, "rgb_final": rgb_final,
        }

    # ---------------- utilities ----------------
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

    def test(self, iter=None):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model(render_colors=True)
        image = (out.view(3, -1) + self.gaussian_model.image_mean).view(3, self.H, self.W)
        mse_loss = F.mse_loss(image.float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(image.unsqueeze(0).float(), self.gt_image.unsqueeze(0).float(), data_range=1, size_average=True).item()
        self.logwriter.write("Test PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))

        ycbcr_img = (image.clamp(0,1).detach().cpu().numpy() * 255.0).transpose(1, 2, 0)
        rgb_img = color.ycbcr2rgb(ycbcr_img) * 255.0
        rgb_img = rgb_img.clip(0, 255).astype(np.uint8)
        img = Image.fromarray(rgb_img)
        name = self.image_name + (f"_{iter}_fitting.png" if iter is not None else "_fitting.png")
        img.save(str(self.model_dir / name))
        return psnr, ms_ssim_value

    # ---------------- eval-on-train loop ----------------
    def eval_phaseB_on_test_set(self, global_iter: int):
        if not self.wandb_run:
            return
        candidates = self._discover_test_images()
        if len(candidates) == 0:
            self.logwriter.write("[eval warning] no test images found; skipping eval")
            return
        k = min(self.args.eval_images, len(candidates))
        idxs = self._rng.choice(len(candidates), size=k, replace=False)
        selected = [candidates[i] for i in idxs]
        base_state = copy.deepcopy(self.gaussian_model.state_dict())

        rows, panels = [], []
        for p in selected:
            res = self._phaseB_eval_single(base_state, p, opt_iters=self.args.eval_opt_iters)
            rows.append([res["path"], res["psnr_init"], res["ssim_init"], res["loss_init"],
                         res["psnr_final"], res["ssim_final"], res["loss_final"]])
            panels.append(wandb.Image(res["rgb_final"], caption=f"{Path(res['path']).name}  PSNR:{res['psnr_final']:.2f}  SSIM:{res['ssim_final']:.4f}"))

        table = wandb.Table(data=rows, columns=["path","psnr_init","ssim_init","loss_init","psnr_final","ssim_final","loss_final"])
        arr = np.array(rows, dtype=object)
        psnrF = np.array(arr[:,4], dtype=float); ssimF = np.array(arr[:,5], dtype=float); lossF = np.array(arr[:,6], dtype=float)
        wandb.log({"iter": global_iter, "eval/num_images": int(k),
                   "eval/psnr_final_mean": float(psnrF.mean()), "eval/psnr_final_std": float(psnrF.std()),
                   "eval/ssim_final_mean": float(ssimF.mean()), "eval/loss_final_mean": float(lossF.mean()),
                   "eval/gallery": panels, "eval/table": table}, step=global_iter)


# ---------------- argparse / main ----------------

def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(description="Projective-cluster training/optimization")
    parser.add_argument("-d","--dataset", type=str, required=True, help="Parsed dataset folder (with arrs.npy & projective_basis.pt)")
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--num_points", type=int, default=50000)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None, help="Enable Phase-B on this image path")
    parser.add_argument("--seed", type=float, default=1)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    # W&B + eval
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
        torch.manual_seed(args.seed)
        random.seed(int(args.seed))
        torch.cuda.manual_seed(int(args.seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(int(args.seed))

    trainer = GaussianTrainer(args, image_path=args.image_path, num_points=args.num_points,
                              iterations=args.iterations, model_path=args.model_path)

    if not args.skip_train:
        trainer.train()
    else:
        trainer.optimize()

if __name__ == "__main__":
    main(sys.argv[1:])