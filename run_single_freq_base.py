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
from gaussianbasis_single_freq_base import GaussianBasis
import pickle
import uuid
import cProfile
import pstats

# === NEW ===
import os
import copy
import wandb
from datetime import datetime
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
        self.args = args  # === NEW === keep CLI handy
        self.dataset_path = Path(args.dataset)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        arrs_path = self.dataset_path / "arrs.npy"
        self.gt_arrs = torch.from_numpy(np.load(arrs_path)).to(self.device)

        random_string = str(uuid.uuid4())[:6]
        if image_path is not None:
            self.image_path = Path(image_path)
            self.image_name = self.image_path.stem
            img = Image.open(self.image_path)
            img_arr = (color.rgb2ycbcr(img) / 255.0).transpose(2, 0, 1)
            self.gt_image = torch.tensor(img_arr, dtype=torch.float32, device=self.device)
            model_dir = Path(f"./models/recons/{self.image_name}-{self.dataset_path.name}-{num_points}-{args.iterations}-{random_string}")
        else: 
            self.image_path = None
            self.image_name = None
            self.gt_image = None
            model_dir = Path(f"./models/single-freq/{self.dataset_path.name}-{num_points}-{args.iterations}-{random_string}")
        
        self.num_points = num_points
        self.num_comps = self.gt_arrs.shape[0]
        self.H, self.W = self.gt_arrs.shape[2], self.gt_arrs.shape[3]
        BLOCK_H, BLOCK_W = 16, 16
        self.iterations = iterations

        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.gaussian_model = GaussianBasis(
            loss_type="L2", opt_type="adan", num_points=self.num_points,
            H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
            device=self.device, lr=args.lr, num_comps=self.num_comps).to(self.device)
        self.gaussian_model.scheduler_init()
        self.logwriter = LogWriter(self.model_dir)
        self.psnr_tracker = PSNRTracker(self.logwriter)
        self.logwriter.write(f"Model Dir ID: {random_string}")

        if model_path is not None:
            self.model_path = Path(model_path)
            self.logwriter.write(f"Model loaded from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.gaussian_model.load_state_dict(checkpoint['model_state_dict'])

        # === NEW === W&B init
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
            # Disable gradients/parameters logging to avoid overhead
            wandb.watch(self.gaussian_model, log=None, log_freq=0, log_graph=False)


        # === NEW === cache test image list once (reproducible)
        self._test_images_cache = None
        self._rng = np.random.default_rng(args.seed if args.seed is not None else 0)

    # === UPDATED === utility: test image collection (prefers explicit folder)
    def _discover_test_images(self) -> List[Path]:
        if self._test_images_cache is not None:
            return self._test_images_cache

        # highest priority: explicit glob
        if self.args.test_glob:
            candidates = sorted(Path().glob(self.args.test_glob))

        # next: explicit directory
        elif self.args.test_dir:
            root = Path(self.args.test_dir)
            if not root.exists():
                raise FileNotFoundError(f"--test_dir not found: {root}")
            it = root.rglob("*") if self.args.test_recursive else root.glob("*")
            candidates = sorted([p for p in it if p.suffix.lower() in IMG_EXTS and p.is_file()])

        else:
            # fallback to legacy discovery (kept for convenience)
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


    # === NEW === YCbCr tensor from path
    def _load_ycbcr_tensor(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((self.W, self.H))
        ycbcr = (color.rgb2ycbcr(np.array(img)) / 255.0).transpose(2, 0, 1)
        return torch.tensor(ycbcr, dtype=torch.float32, device=self.device)

    # === NEW === RGB image from YCbCr tensor(3,H,W) in [0,1]
    def _ycbcr_to_rgb_image(self, ycbcr_01: torch.Tensor) -> Image.Image:
        ycbcr_img = (ycbcr_01.clamp(0,1).detach().cpu().numpy() * 255.0).transpose(1,2,0)
        rgb = color.ycbcr2rgb(ycbcr_img) * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb)

    # === NEW === checkpointing (full resume)
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

    def train(self):     
        progress_bar = tqdm(range(1, self.iterations+1), desc="Training progress")
        self.gaussian_model.train()
        start_time = time.time()
        for iter in range(1, self.iterations+1):
            loss, psnr = self.gaussian_model.train_iter(self.gt_arrs)
            # === NEW === W&B scalar logs
            if self.wandb_run:
                wandb.log({"train/loss": float(loss.item()), "train/psnr": float(psnr), "iter": iter})

            with torch.no_grad():
                progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                progress_bar.update(1)

            # === NEW === periodic eval: clone → short phase-B on test images
            if self.args.eval_every and (iter % self.args.eval_every == 0):
                  self.eval_phaseB_on_test_set(global_iter=iter)
                  self._save_checkpoint(step=iter, tag="phaseA")
                # try:
                #     self.eval_phaseB_on_test_set(global_iter=iter)
                #     self._save_checkpoint(step=iter, tag="phaseA")
                # except Exception as e:
                #     self.logwriter.write(f"[eval warning] eval failed at iter {iter}: {e}")

        end_time = time.time() - start_time
        progress_bar.close()     
        self.logwriter.write("Training Complete in {:.4f}s".format(end_time))
        torch.save({'model_state_dict': self.gaussian_model.state_dict()}, self.model_dir / "gaussian_model.pth.tar")
        if self.wandb_run:
            wandb.save(str(self.model_dir / "gaussian_model.pth.tar"), base_path=str(self.model_dir))
        self.vis()
        return

    def optimize(self):
        progress_bar = tqdm(range(1, self.iterations+1), desc="Optimizing progress")
        self.update_gaussian()
        self.gaussian_model.scheduler_init(optimize_phase=True)
        self.gaussian_model.train()
        start_time = time.perf_counter()

        self.test(iter=0)
        self.gaussian_model.train()
        for iter in range(1, self.iterations+1):
            loss, psnr = self.gaussian_model.optimize_iter(self.gt_image)
            self.psnr_tracker.check(start_time, psnr, iter)

            # === NEW === log phase-B scalars as well
            if self.wandb_run:
                wandb.log({"phaseB/loss": float(loss.item()), "phaseB/psnr": float(psnr), "phaseB/iter": iter})

            with torch.no_grad():
                if iter in [10, 100, 1000]:
                    self.test(iter=iter)
                    self.gaussian_model.train()
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

    def update_gaussian(self):
        with open(self.dataset_path / "pca_object.pkl", "rb") as FIN:
            pca_object = pickle.load(FIN)
        with open(self.dataset_path / "norm_infos.pkl", "rb") as FIN:
            norm_infos = pickle.load(FIN)

        self.gaussian_model.cur_freq = "all"
        img_arr = self.gt_image.detach().cpu().numpy()
        img_arr = img_arr.reshape(3, -1)
        
        codes = []
        maxs = []
        mins = []
        means = []
        for idx, ch_arr in enumerate(img_arr):
            pca = pca_object[idx]
            coes = pca.transform(ch_arr.reshape(1, -1))
            codes.append(coes)
            means.append(pca.mean_)

            maxs.append(norm_infos[idx]["max"])
            mins.append(norm_infos[idx]["min"])

        max_tensor = torch.tensor(np.array(maxs), dtype=torch.float32, device=self.gaussian_model.device)
        min_tensor = torch.tensor(np.array(mins), dtype=torch.float32, device=self.gaussian_model.device)
        mean_tensor = torch.tensor(np.array(means), dtype=torch.float32, device=self.gaussian_model.device)
        codes_tensor = torch.tensor(np.array(codes), dtype=torch.float32, device=self.gaussian_model.device)
        
        self.gaussian_model.shift_factor = codes_tensor.sum(dim=2).squeeze(1) * min_tensor
        self.gaussian_model.scale_factor = max_tensor - min_tensor
        self.gaussian_model.image_mean = mean_tensor

        # self.logwriter.write(f"Update shift factor: {self.gaussian_model.shift_factor}")
        # self.logwriter.write(f"Update scale factor: {self.gaussian_model.scale_factor}")
        # self.logwriter.write(f"Update mean tensor")

        with torch.no_grad():
            features = self.gaussian_model.get_features.permute(2, 0, 1)
            colors = torch.bmm(codes_tensor, features).squeeze(1).transpose(0, 1)
            self.gaussian_model._colors.copy_(colors)
            out = self.gaussian_model(render_colors=True)

        _, _, height, width = self.gt_arrs.shape 
        image = out.reshape(3, -1) + mean_tensor
        
        ycbcr_img = image.detach().cpu().numpy() * 255.0
        ycbcr_img = ycbcr_img.reshape(3, height, width).transpose(1, 2, 0)
        rgb_img = color.ycbcr2rgb(ycbcr_img) * 255.0
        rgb_img = rgb_img.clip(0, 255)

        img = Image.fromarray(rgb_img.astype(np.uint8))
        vis_dir = self.model_dir / f"vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        img.save(vis_dir / f"test_colors.png")
    # (your duplicated test_pca functions kept as-is)

    def test(self, iter=None):
        self.gaussian_model.eval()
        _, _, height, width = self.gt_arrs.shape

        with torch.no_grad():
            out = self.gaussian_model(render_colors=True)
        image = out.reshape(3, -1) + self.gaussian_model.image_mean
        image = image.reshape(3, height, width)
        mse_loss = F.mse_loss(image.float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(image.unsqueeze(0).float(), self.gt_image.unsqueeze(0).float(), data_range=1, size_average=True).item()
        
        self.logwriter.write("Test PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))

        ycbcr_img = image.detach().cpu().numpy() * 255.0
        ycbcr_img = ycbcr_img.reshape(3, height, width).transpose(1, 2, 0)
        rgb_img = color.ycbcr2rgb(ycbcr_img) * 255.0
        rgb_img = rgb_img.clip(0, 255)

        img = Image.fromarray(rgb_img.astype(np.uint8))
        if iter is not None:
            name = self.image_name + f"_{iter}_fitting.png"    
        else:    
            name = self.image_name + "_fitting.png"
        img.save(str(self.model_dir / name))
        return psnr, ms_ssim_value

    # === NEW === helper: compute PCA init + quick phase-B on a single image with an isolated model copy
    def _phaseB_eval_single(self, base_state_dict, img_path: Path, opt_iters: int):
        # fresh model copy
        model = GaussianBasis(
            loss_type="L2", opt_type="adan", num_points=self.num_points,
            H=self.H, W=self.W, BLOCK_H=16, BLOCK_W=16,
            device=self.device, lr=self.args.lr, num_comps=self.num_comps
        ).to(self.device)
        model.load_state_dict(base_state_dict)
        model.eval()

        # load dataset artifacts
        with open(self.dataset_path / "pca_object.pkl", "rb") as FIN:
            pca_object = pickle.load(FIN)
        with open(self.dataset_path / "norm_infos.pkl", "rb") as FIN:
            norm_infos = pickle.load(FIN)

        # load image
        print(img_path)
        gt_img = self._load_ycbcr_tensor(img_path)  # (3,H,W)
        img_arr = gt_img.detach().cpu().numpy().reshape(3, -1)

        # PCA projection → init colors (your logic)
        codes, maxs, mins, means = [], [], [], []
        for idx, ch_arr in enumerate(img_arr):
            pca = pca_object[idx]
            coes = pca.transform(ch_arr.reshape(1, -1))
            codes.append(coes)
            means.append(pca.mean_)
            maxs.append(norm_infos[idx]["max"])
            mins.append(norm_infos[idx]["min"])

        max_tensor = torch.tensor(np.array(maxs), dtype=torch.float32, device=self.device)
        min_tensor = torch.tensor(np.array(mins), dtype=torch.float32, device=self.device)
        mean_tensor = torch.tensor(np.array(means), dtype=torch.float32, device=self.device)
        codes_tensor = torch.tensor(np.array(codes), dtype=torch.float32, device=self.device)

        model.shift_factor = codes_tensor.sum(dim=2).squeeze(1) * min_tensor
        model.scale_factor = max_tensor - min_tensor
        model.image_mean = mean_tensor

        with torch.no_grad():
            features = model.get_features.permute(2, 0, 1)
            colors = torch.bmm(codes_tensor, features).squeeze(1).transpose(0, 1)
            model._colors.copy_(colors)

        # metrics at init (iter 0)
        with torch.no_grad():
            out0 = model(render_colors=True)                    # (3,H,W)
            img0 = (out0.view(3, -1) + model.image_mean)        # (3,H*W) + (3,H*W)
            img0 = img0.view(3, self.H, self.W).clamp(0, 1)     # back to (3,H,W)
         # YCbCr in [0,1]
            mse0 = F.mse_loss(img0, gt_img).item()
            psnr0 = 10 * math.log10(1.0 / (mse0 + 1e-12))
            ssim0 = ms_ssim(img0.unsqueeze(0), gt_img.unsqueeze(0), data_range=1, size_average=True).item()

        # short phase-B refine
        model.scheduler_init(optimize_phase=True)
        model.train()
        last_loss, last_psnr = None, None
        for _ in range(opt_iters):
            loss, psnr = model.optimize_iter(gt_img)
            last_loss, last_psnr = float(loss.item()), float(psnr)

        # final metrics
        model.eval()
        with torch.no_grad():
            outF = model(render_colors=True)
            imgF = (outF.view(3, -1) + model.image_mean)
            imgF = imgF.view(3, self.H, self.W).clamp(0, 1)

            mseF = F.mse_loss(imgF, gt_img).item()
            psnrF = 10 * math.log10(1.0 / (mseF + 1e-12))
            ssimF = ms_ssim(imgF.unsqueeze(0), gt_img.unsqueeze(0), data_range=1, size_average=True).item()

        # RGB preview images for W&B
        rgb_init = self._ycbcr_to_rgb_image(img0)
        rgb_final = self._ycbcr_to_rgb_image(imgF)

        return {
            "path": str(img_path),
            "loss_init": mse0, "psnr_init": psnr0, "ssim_init": ssim0,
            "loss_final": mseF, "psnr_final": psnrF, "ssim_final": ssimF,
            "rgb_init": rgb_init, "rgb_final": rgb_final,
        }

    # === NEW === public: evaluate on K test images during phase-A
    def eval_phaseB_on_test_set(self, global_iter: int):
        if not self.wandb_run:
            return  # silently skip if wandb off

        all_candidates = self._discover_test_images()
        if len(all_candidates) == 0:
            self.logwriter.write("[eval warning] no test images found; skipping eval")
            return

        k = min(self.args.eval_images, len(all_candidates))
        # fixed but reproducible sample
        idxs = self._rng.choice(len(all_candidates), size=k, replace=False)
        selected = [all_candidates[i] for i in idxs]

        base_state = copy.deepcopy(self.gaussian_model.state_dict())

        rows = []
        img_panels = []
        for p in selected:
            res = self._phaseB_eval_single(base_state, p, opt_iters=self.args.eval_opt_iters)
            rows.append([
                res["path"],
                res["psnr_init"], res["ssim_init"], res["loss_init"],
                res["psnr_final"], res["ssim_final"], res["loss_final"],
            ])
            # panel with captions
            img_panels.append(wandb.Image(res["rgb_final"], caption=f"{Path(res['path']).name}  PSNR:{res['psnr_final']:.2f}  SSIM:{res['ssim_final']:.4f}"))

        # W&B table
        table = wandb.Table(
            data=rows,
            columns=["path","psnr_init","ssim_init","loss_init","psnr_final","ssim_final","loss_final"]
        )

        # aggregate
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

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='./datasets/kodak/', help="Training dataset"
    )
    parser.add_argument(
        "--iterations", type=int, default=50000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--sh_degree", type=int, default=3, help="SH degree (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=50000,
        help="2D GS points (default: %(default)s)",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint")
    parser.add_argument("--image_path", type=str, default=None, help="Path to images")
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: %(default)s)",
    )

    # === NEW === W&B + Eval controls
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default=os.getenv("WANDB_PROJECT", "eigens"), help="W&B project")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (optional)")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name (optional)")
    parser.add_argument("--eval_every", type=int, default=1000, help="Eval cadence in iters during phase-A")
    parser.add_argument("--eval_images", type=int, default=10, help="How many test images to evaluate")
    parser.add_argument("--eval_opt_iters", type=int, default=200, help="Short phase-B iters per eval image")
    parser.add_argument("--test_glob", type=str, default=None, help="Optional glob for test images, e.g. './datasets/kodak/test/*.png'")
        # after existing eval controls
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Directory that contains test images (non-recursive).")
    parser.add_argument("--test_recursive", action="store_true",
                        help="Recurse into subfolders of --test_dir when collecting test images.")

    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(int(args.seed))
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
