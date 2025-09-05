# run_single_freq_projective.py
import math, time, sys, os, json, copy, random, uuid
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import color
from tqdm import tqdm
import torchvision.transforms as transforms
from pytorch_msssim import ms_ssim
import wandb

from gaussianbasis_single_freq_base import GaussianBasis
from tracker import LogWriter, PSNRTracker

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# --------------------- helpers ---------------------

def _is_parse_root(p: Path) -> bool:
    return (p / "projective_basis.pt").exists()

def _load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

@torch.no_grad()
def _rgb_path_to_ycbcr01(path: Path, W: int, H: int, device) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    arr = (color.rgb2ycbcr(np.asarray(im)) / 255.0).transpose(2, 0, 1)
    return torch.tensor(arr, dtype=torch.float32, device=device)  # (3,H,W)

def _ycbcr01_to_rgb_pil(ycbcr_3HW: torch.Tensor) -> Image.Image:
    ycbcr = (ycbcr_3HW.clamp(0,1).detach().cpu().numpy() * 255.0).transpose(1,2,0)
    rgb = color.ycbcr2rgb(ycbcr) * 255.0
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb)

# == small LA helper for Y-flat residual (for auto-select at root) ==
@torch.no_grad()
def residual_to_flat(y_vec: torch.Tensor, V_row: torch.Tensor, v: torch.Tensor) -> float:
    """
    y_vec: (d,) in [0,1]
    V_row: (J,d)   (rows orthonormal)
    v:     (d,)
    returns ||(I - U U^T)(y - v)||_2 with U = V_row^T (d,J)
    """
    U = V_row.transpose(0, 1).contiguous()
    z = y_vec - v
    proj = (z @ U) @ U.transpose(0, 1)
    r = z - proj
    return float(torch.linalg.vector_norm(r).item())

# --------------------- Trainer ---------------------

class GaussianTrainer:
    """
    DERMS-style trainer that consumes your Projective-Parse artifacts.

    Works with:
      A) parse ROOT:   /.../<uuid>-K{K}-J{J}/
         - has projective_basis.pt (K,J,img_size,norm_mins/maxs)
         - arrs.npy at root (K*J,3,H,W)
         - clusters/cluster_xx/{arrs.npy,means.npy,bases.npy,Y_flat_*.npy,em_Vs.npy,em_vs.npy,meta.json}
      B) single CLUSTER folder: /.../clusters/cluster_XX/
         - has {arrs.npy,means.npy,bases.npy,Y_flat_*.npy[,em_Vs.npy,em_vs.npy],meta.json}
    """
    def __init__(self, args,
                 image_path: Optional[str] = None,
                 num_points: int = 20000,
                 iterations: int = 30000,
                 model_path: Optional[str] = None):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dataset_path = Path(args.dataset)
        self.is_root = _is_parse_root(self.dataset_path)

        # ---- load dataset shape + norms + arrs (normalized eigenimages) ----
        if self.is_root:
            # root meta
            pb = torch.load(self.dataset_path / "projective_basis.pt", map_location="cpu")
            self.K = int(pb["K"]); self.J = int(pb["J"])
            W, H = pb["img_size"]; self.W, self.H = int(W), int(H)
            self.norm_mins = torch.tensor(pb["norm_mins"], dtype=torch.float32, device=self.device)  # (3,)
            self.norm_maxs = torch.tensor(pb["norm_maxs"], dtype=torch.float32, device=self.device)  # (3,)

            # which arrs to train on
            if str(args.cluster).lower() == "all":
                arrs_path = self.dataset_path / "arrs.npy"                 # (K*J,3,H,W)
                self.train_mode = ("root_all", None)
                num_comps = self.K * self.J
            else:
                k = int(args.cluster)
                arrs_path = self.dataset_path / f"clusters/cluster_{k:02d}/arrs.npy"  # (J,3,H,W)
                self.train_mode = ("root_single", k)
                num_comps = self.J
        else:
            # cluster folder only
            meta = _load_json(self.dataset_path / "meta.json")
            self.K = 1
            self.J = int(meta["J"])
            W, H = meta["img_size"]; self.W, self.H = int(W), int(H)
            # norms saved in meta (they are the global norms computed in parse)
            self.norm_mins = torch.tensor(meta["norm_mins"], dtype=torch.float32, device=self.device)
            self.norm_maxs = torch.tensor(meta["norm_maxs"], dtype=torch.float32, device=self.device)
            arrs_path = self.dataset_path / "arrs.npy"                      # (J,3,H,W)
            self.train_mode = ("cluster_dir", 0)
            num_comps = self.J

        # normalized eigenimages to fit in Phase-A (DERMS behavior)
        arrs = np.load(arrs_path)  # float32 in [0,1]
        self.gt_arrs = torch.from_numpy(arrs).to(self.device)
        self.num_comps = int(num_comps)

        # ---- bookkeeping ----
        self.iterations = int(iterations)
        self.num_points = int(num_points)

        # optional single target for Phase-B
        rid = str(uuid.uuid4())[:6]
        if image_path is not None:
            self.image_path = Path(image_path)
            self.image_name = self.image_path.stem
            self.gt_image = _rgb_path_to_ycbcr01(self.image_path, self.W, self.H, self.device)
            model_dir = Path(f"./models/recons/{self.image_name}-{self.dataset_path.name}-{self.num_points}-{args.iterations}-{rid}")
        else:
            self.image_path = None
            self.image_name = None
            self.gt_image = None
            model_dir = Path(f"./models/single-freq/{self.dataset_path.name}-{self.num_points}-{args.iterations}-{rid}")
        self.model_dir = model_dir; self.model_dir.mkdir(parents=True, exist_ok=True)

        # ---- model ----
        self.gaussian_model = GaussianBasis(
            loss_type="L2", opt_type="adan", num_points=self.num_points,
            H=self.H, W=self.W, BLOCK_H=16, BLOCK_W=16,
            device=self.device, lr=args.lr, num_comps=self.num_comps
        ).to(self.device)
        self.gaussian_model.scheduler_init()

        # ---- logging / trackers ----
        self.logwriter = LogWriter(self.model_dir)
        self.psnr_tracker = PSNRTracker(self.logwriter)
        self.logwriter.write(f"Mode: {self.train_mode[0]}  comps={self.num_comps}  arrs={arrs_path.name}")

        # optional checkpoint
        if model_path is not None:
            ck = torch.load(model_path, map_location=self.device)
            self.gaussian_model.load_state_dict(ck["model_state_dict"])
            self.logwriter.write(f"Loaded checkpoint: {model_path}")

        # ---- W&B ----
        self.wandb_run = None
        if args.wandb:
            run_name = args.run_name or f"{self.dataset_path.name}-{self.train_mode[0]}-P{self.num_points}-I{self.iterations}-{rid}"
            self.wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                name=run_name,
                config={
                    "dataset": str(self.dataset_path),
                    "mode": self.train_mode[0],
                    "cluster": str(args.cluster),
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

    # ------------------------ artifact loaders ------------------------

    def _cluster_dir(self, k: int) -> Path:
        if self.is_root:
            return self.dataset_path / f"clusters/cluster_{k:02d}"
        # cluster dir mode ignores k
        return self.dataset_path

    def _load_cluster_basis(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cdir = self._cluster_dir(k)
        means = torch.from_numpy(np.load(cdir / "means.npy")).to(self.device).float()  # (3,d)
        bases = torch.from_numpy(np.load(cdir / "bases.npy")).to(self.device).float()  # (3,J,d)
        return means, bases

    def _load_cluster_yflat(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cdir = self._cluster_dir(k)
        V = torch.from_numpy(np.load(cdir / "Y_flat_V.npy")).to(self.device).float()   # (J,d)
        v = torch.from_numpy(np.load(cdir / "Y_flat_v.npy")).to(self.device).float()   # (d,)
        return V, v

    # ------------------------ discovery ------------------------

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
            # fallbacks under dataset path
            base = self.dataset_path if self.is_root else self.dataset_path
            subdirs = ["test_imgs", "val", "validation", "Val", "Test"]
            found = []
            for sd in subdirs:
                p = base / sd
                if p.exists():
                    found += sorted([q for q in p.rglob("*") if q.suffix.lower() in IMG_EXTS and q.is_file()])
            candidates = found
        self._test_images_cache = candidates
        return candidates

    # ------------------------ codes / cluster selection ------------------------

    @torch.no_grad()
    def _pick_cluster_for_image(self, img_ycbcr: torch.Tensor) -> int:
        if not self.is_root or self.train_mode[0] != "root_all":
            return 0
        y = img_ycbcr[0].view(-1).to(self.device)
        best_k, best_r = 0, float("inf")
        for k in range(self.K):
            V, v = self._load_cluster_yflat(k)
            r = residual_to_flat(y, V, v)
            if r < best_r:
                best_r, best_k = r, k
        return best_k

    @torch.no_grad()
    def _codes_from_cluster(self, img_ycbcr: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return (codes_tensor, per_channel_sum, mean_3HW)
          codes_tensor: (3,1,num_comps) where num_comps = J or (K*J) with zeros outside [k*J:(k+1)*J)
          per_channel_sum: (3,)  sum_j alpha_{ch,j}
          mean_3HW: (3,H,W) per-pixel mean for adding AFTER rendering
        """
        means_k, bases_k = self._load_cluster_basis(k)       # (3,d), (3,J,d)
        X = img_ycbcr.view(3, -1)                            # (3,d)
        centered = X - means_k                               # (3,d)
        coes = torch.einsum("cd,cjd->cj", centered, bases_k) # (3,J), row-basis
        sumc = coes.sum(dim=1)                               # (3,)

        if self.train_mode[0] == "root_all":
            full = torch.zeros((3, self.K * self.J), device=self.device, dtype=torch.float32)
            full[:, k*self.J:(k+1)*self.J] = coes
            codes = full.unsqueeze(1)                        # (3,1,K*J)
        else:
            codes = coes.unsqueeze(1)                        # (3,1,J)

        mean_3HW = means_k.view(3, self.H, self.W)
        return codes, sumc, mean_3HW

    # ------------------------ training / optimize ------------------------

    def train(self):
        pb = tqdm(range(1, self.iterations + 1), desc="Training progress")
        self.gaussian_model.train()
        t0 = time.time()
        for it in range(1, self.iterations + 1):
            loss, psnr = self.gaussian_model.train_iter(self.gt_arrs)
            if self.wandb_run:
                wandb.log({"train/loss": float(loss.item()), "train/psnr": float(psnr), "iter": it})
            with torch.no_grad():
                pb.set_postfix({f"Loss": f"{loss.item():.7f}", "PSNR": f"{psnr:.4f},"})
                pb.update(1)

            if self.args.eval_every and (it % self.args.eval_every == 0):
                self.eval_phaseB_on_test_set(global_iter=it)
                self._save_checkpoint(it, "phaseA")

        pb.close()
        self.logwriter.write(f"Training complete in {time.time()-t0:.2f}s")
        torch.save({'model_state_dict': self.gaussian_model.state_dict()}, self.model_dir / "gaussian_model.pth.tar")
        if self.wandb_run:
            wandb.save(str(self.model_dir / "gaussian_model.pth.tar"), base_path=str(self.model_dir))
        self.vis()

    def optimize(self):
        if self.gt_image is None:
            raise ValueError("Phase-B (--image_path) needs a target image.")
        self.update_gaussian()                            # init codes from PCA (projective)
        self.gaussian_model.scheduler_init(optimize_phase=True)
        self.gaussian_model.train()

        pb = tqdm(range(1, self.iterations + 1), desc="Optimizing progress")
        t0 = time.perf_counter()
        self.test(iter=0)
        self.gaussian_model.train()
        for it in range(1, self.iterations + 1):
            loss, psnr = self.gaussian_model.optimize_iter(self.gt_image)
            self.psnr_tracker.check(t0, psnr, it)
            if self.wandb_run:
                wandb.log({"phaseB/loss": float(loss.item()), "phaseB/psnr": float(psnr), "phaseB/iter": it})
            if it % 10 == 0:
                pb.set_postfix({f"Loss": f"{loss.item():.7f}", "PSNR": f"{psnr:.4f},"})
                pb.update(10)
            if it in [10, 100, 1000]:
                self.test(iter=it); self.gaussian_model.train()
        pb.close()
        self.psnr_tracker.print_summary()
        torch.save({'model_state_dict': self.gaussian_model.state_dict()}, self.model_dir / "gaussian_model_with_colors.pth.tar")
        if self.wandb_run:
            wandb.save(str(self.model_dir / "gaussian_model_with_colors.pth.tar"), base_path=str(self.model_dir))
        self.test()

    def vis(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            pred = self.gaussian_model()                 # fits arrs directly (normalized)
        mse = F.mse_loss(pred.float(), self.gt_arrs.float()).item()
        ps = 10 * math.log10(1.0 / (mse + 1e-12))
        ssim = ms_ssim(pred.float(), self.gt_arrs.float(), data_range=1, size_average=True).item()
        self.logwriter.write(f"Components Fitting: PSNR:{ps:.4f}, MS_SSIM:{ssim:.6f}")
        vis_dir = self.model_dir / "vis_comps"; vis_dir.mkdir(exist_ok=True, parents=True)
        to_pil = transforms.ToPILImage()
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                to_pil(pred[i, j].cpu()).save(vis_dir / f"{i}-{j}.png")
        return ps, ssim

    # ------------------------ Phase-B init from projective PCA ------------------------

    def _select_cluster_for_phaseB(self, img_ycbcr: torch.Tensor) -> int:
        if self.train_mode[0] == "root_all":
            if self.args.phaseB_cluster is not None:
                return int(self.args.phaseB_cluster)
            if self.args.auto_select_cluster:
                k = self._pick_cluster_for_image(img_ycbcr)
                self.logwriter.write(f"[phaseB] auto-selected cluster {k}")
                return k
            return 0
        elif self.train_mode[0] == "root_single":
            return int(self.train_mode[1])
        else:  # cluster_dir
            return 0

    def update_gaussian(self):
        if self.gt_image is None:
            return
        k = self._select_cluster_for_phaseB(self.gt_image)
        codes, sumc, mean_3HW = self._codes_from_cluster(self.gt_image, k)  # (3,1,num_comps), (3,), (3,H,W)

        # put initial colors into model (DERMS-style)
        with torch.no_grad():
            feats = self.gaussian_model.get_features.permute(2, 0, 1)   # (3,num_comps,P)
            colors = torch.bmm(codes, feats).squeeze(1).transpose(0, 1) # (P,3)
            self.gaussian_model._colors.copy_(colors)

        # keep scalar factors neutral; we do un-norm outside
        self.gaussian_model.scale_factor = torch.ones(3, device=self.device)
        self.gaussian_model.shift_factor = torch.zeros(3, device=self.device)
        self.gaussian_model.image_mean = torch.zeros(3, self.H*self.W, device=self.device)  # unused here

        # quick preview image
        with torch.no_grad():
            out = self.gaussian_model(render_colors=True)               # (3,H,W) in normalized basis space
        den = (self.norm_maxs - self.norm_mins).view(3,1,1)
        rec = out * den + self.norm_mins.view(3,1,1) * sumc.view(3,1,1) + mean_3HW
        img = _ycbcr01_to_rgb_pil(rec)
        vis = self.model_dir / "vis"; vis.mkdir(exist_ok=True, parents=True)
        img.save(vis / "init_colors.png")

    def test(self, iter: Optional[int] = None):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model(render_colors=True)               # (3,H,W), normalized space
        # We must map back to image space using the SAME formula as update_gaussian()
        k = self._select_cluster_for_phaseB(self.gt_image)
        _, sumc, mean_3HW = self._codes_from_cluster(self.gt_image, k)
        den = (self.norm_maxs - self.norm_mins).view(3,1,1)
        image = (out * den + self.norm_mins.view(3,1,1) * sumc.view(3,1,1) + mean_3HW).clamp(0,1)

        mse = F.mse_loss(image.float(), self.gt_image.float()).item()
        psnr = 10 * math.log10(1.0 / (mse + 1e-12))
        ssim = ms_ssim(image.unsqueeze(0).float(), self.gt_image.unsqueeze(0).float(), data_range=1, size_average=True).item()
        self.logwriter.write(f"Test PSNR:{psnr:.4f}, MS_SSIM:{ssim:.6f}")

        out_img = _ycbcr01_to_rgb_pil(image)
        name = self.image_name + (f"_{iter}_fitting.png" if iter is not None else "_fitting.png")
        out_img.save(self.model_dir / name)
        return psnr, ssim

    # ------------------------ Phase-A evaluation (short Phase-B on a few test images) ------------------------

    def _phaseB_eval_single(self, base_state_dict, img_path: Path, opt_iters: int):
        model = GaussianBasis(
            loss_type="L2", opt_type="adan", num_points=self.num_points,
            H=self.H, W=self.W, BLOCK_H=16, BLOCK_W=16,
            device=self.device, lr=self.args.lr, num_comps=self.num_comps
        ).to(self.device)
        model.load_state_dict(base_state_dict)
        model.eval()

        gt_img = _rgb_path_to_ycbcr01(img_path, self.W, self.H, self.device)

        # choose cluster same way as Phase-B
        if self.train_mode[0] == "root_all":
            k = int(self.args.phaseB_cluster) if self.args.phaseB_cluster is not None else self._pick_cluster_for_image(gt_img)
        elif self.train_mode[0] == "root_single":
            k = int(self.train_mode[1])
        else:
            k = 0

        codes, sumc, mean_3HW = self._codes_from_cluster(gt_img, k)

        with torch.no_grad():
            feats = model.get_features.permute(2, 0, 1)
            colors = torch.bmm(codes, feats).squeeze(1).transpose(0, 1)
            model._colors.copy_(colors)

        # iter-0 preview
        with torch.no_grad():
            out0 = model(render_colors=True)
        den = (self.norm_maxs - self.norm_mins).view(3,1,1)
        img0 = (out0 * den + self.norm_mins.view(3,1,1) * sumc.view(3,1,1) + mean_3HW).clamp(0,1)
        mse0 = F.mse_loss(img0, gt_img).item()
        psnr0 = 10 * math.log10(1.0 / (mse0 + 1e-12))
        ssim0 = ms_ssim(img0.unsqueeze(0), gt_img.unsqueeze(0), data_range=1, size_average=True).item()

        # short refine
        model.scheduler_init(optimize_phase=True)
        model.train()
        for _ in range(opt_iters):
            loss, _ = model.optimize_iter(gt_img)
        model.eval()
        with torch.no_grad():
            outF = model(render_colors=True)
        imgF = (outF * den + self.norm_mins.view(3,1,1) * sumc.view(3,1,1) + mean_3HW).clamp(0,1)
        mseF = F.mse_loss(imgF, gt_img).item()
        psnrF = 10 * math.log10(1.0 / (mseF + 1e-12))
        ssimF = ms_ssim(imgF.unsqueeze(0), gt_img.unsqueeze(0), data_range=1, size_average=True).item()

        return {
            "path": str(img_path),
            "loss_init": mse0, "psnr_init": psnr0, "ssim_init": ssim0,
            "loss_final": mseF, "psnr_final": psnrF, "ssim_final": ssimF,
            "rgb_final": _ycbcr01_to_rgb_pil(imgF),
        }

    def eval_phaseB_on_test_set(self, global_iter: int):
        if not self.wandb_run:
            return
        cands = self._discover_test_images()
        if len(cands) == 0:
            self.logwriter.write("[eval] no test images found; skip")
            return
        k = min(self.args.eval_images, len(cands))
        idxs = self._rng.choice(len(cands), size=k, replace=False)
        chosen = [cands[i] for i in idxs]
        base_state = copy.deepcopy(self.gaussian_model.state_dict())

        rows, panels = [], []
        for p in chosen:
            res = self._phaseB_eval_single(base_state, p, opt_iters=self.args.eval_opt_iters)
            rows.append([
                res["path"],
                res["psnr_init"], res["ssim_init"], res["loss_init"],
                res["psnr_final"], res["ssim_final"], res["loss_final"]
            ])
            panels.append(wandb.Image(res["rgb_final"],
                                      caption=f"{Path(res['path']).name}  PSNR:{res['psnr_final']:.2f}  SSIM:{res['ssim_final']:.4f}"))

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
            "eval/gallery": panels,
            "eval/table": table,
        }, step=global_iter)

    # ------------------------ checkpoints ------------------------

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

# ------------------------------ CLI ------------------------------

import argparse
def parse_args(argv):
    p = argparse.ArgumentParser(description="Gaussian training with Projective Clusters (DERMS-style).")
    p.add_argument("-d","--dataset", type=str, required=True, help="Parse ROOT (…-K*-J*) or a single cluster folder (…/clusters/cluster_xx)")
    p.add_argument("--iterations", type=int, default=50000)
    p.add_argument("--num_points", type=int, default=50000)
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--image_path", type=str, default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--lr", type=float, default=1e-3)

    # cluster control (effective only when dataset is a parse ROOT)
    p.add_argument("--cluster", type=str, default="all",
                   help="'all' to train on all clusters (root only), or an integer k to train a single cluster. Ignored for cluster directories.")
    p.add_argument("--phaseB_cluster", type=int, default=None,
                   help="(root-all) force cluster index at eval/phase-B; otherwise auto-select.")
    p.add_argument("--auto_select_cluster", action="store_true",
                   help="(root-all) auto-pick cluster by Y-flat residual for each image.")

    # W&B + Eval
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default=os.getenv("WANDB_PROJECT", "eigens"))
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--eval_images", type=int, default=10)
    p.add_argument("--eval_opt_iters", type=int, default=200)
    p.add_argument("--test_glob", type=str, default=None)
    p.add_argument("--test_dir", type=str, default=None)
    p.add_argument("--test_recursive", action="store_true")
    return p.parse_args(argv)

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