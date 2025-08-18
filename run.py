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

from pytorch_msssim import ms_ssim, ssim
def _ycbcr_to_rgb_uint8(ycbcr_3hw: torch.Tensor) -> np.ndarray:
    """ycbcr_3hw in [0,1] -> RGB uint8 (H,W,3)."""
    ycbcr = (ycbcr_3hw.detach().cpu().numpy() * 255.0).transpose(1, 2, 0)  # (H,W,3) in 0..255
    rgb = color.ycbcr2rgb(ycbcr) * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)

def _save_side_by_side(rgb_gt: np.ndarray, rgb_pred: np.ndarray, out_path: Path):
    """Stack GT | PRED horizontally and save."""
    # handle shape mismatches cautiously (shouldn't happen, but just in case)
    H = min(rgb_gt.shape[0], rgb_pred.shape[0])
    W = min(rgb_gt.shape[1], rgb_pred.shape[1])
    gt = rgb_gt[:H, :W]
    pr = rgb_pred[:H, :W]
    both = np.hstack([gt, pr])
    Image.fromarray(both).save(out_path)

def _current_pred_ycbcr(trainer) -> torch.Tensor:
    """
    Render current model output (colors path) in YCbCr, shape (3,H,W), values in [0,1].
    Uses trainer.gaussian_model.image_mean exactly like test().
    """
    with torch.no_grad():
        out = trainer.gaussian_model(render_colors=True)              # (H,W,3)
    img = out.reshape(3, -1) + trainer.gaussian_model.image_mean      # add mean
    return img.reshape(3, trainer.H, trainer.W)                       # (3,H,W)

def ms_ssim_robust(x, y, *, size_average=True):
    # Accept [C,H,W] or [N,C,H,W]
    if x.dim() == 3: x = x.unsqueeze(0)
    if y.dim() == 3: y = y.unsqueeze(0)
    _, _, H, W = x.shape
    data_range = 1
    # Try to keep 5 scales by adapting the window if needed
    # Need: min(H,W) > (win-1)*16  => choose the largest odd win < min(H,W)/16 + 1
    min_side = min(H, W)
    win_try = 11
    need = (win_try - 1) * 16  # default 5 scales
    if min_side <= need:
        # pick an odd window that makes 5 scales feasible (>=3)
        win_try = max(3, int(min_side // 16) * 2 - 1)
        if win_try >= 3:
            try:
                return ms_ssim(x, y, data_range=data_range, size_average=size_average, win_size=win_try)
            except AssertionError:
                pass  # fall through to fewer scales

    # Compute the largest valid L with the default window 11
    win = 11
    denom = max(1, (win - 1))
    # Find max L such that (win-1)*2**(L-1) < min_side
    import math
    L = max(1, min(5, int(math.floor(math.log2(max(1, min_side / denom)))) + 1))
    # Use UNIFORM weights of length L (no fixed constants)
    weights = [1.0 / L] * L
    try:
        return ms_ssim(x, y, data_range=data_range, size_average=size_average, win_size=win, weights=weights)
    except AssertionError:
        # Last resort: single-scale SSIM with a window that fits
        win_ssim = max(3, (min_side // 2) * 2 - 1)  # largest odd < min_side
        return ssim(x, y, data_range=data_range, size_average=size_average, win_size=win_ssim)



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

        # ===== clustered-PCA artifacts =====
        # normalized comps per cluster/channel: (C, 3, K, D)
        self.cluster_pcas_norm = np.load(self.dataset_path / "cluster_pcas.npy")
        # raw comps & means for projection/inference
        self.cluster_pcas_raw = np.load(self.dataset_path / "cluster_pcas_raw.npy")   # (C, 3, K, D)
        self.cluster_means = np.load(self.dataset_path / "cluster_pca_means.npy")     # (C, 3, D)
        with open(self.dataset_path / "kmeans.pkl", "rb") as f:
            self.kmeans = pickle.load(f)

        # infer dimensions
        self.num_clusters = int(self.cluster_pcas_norm.shape[0])     # C
        self.num_comps    = int(self.cluster_pcas_norm.shape[2])     # K
        D = int(self.cluster_pcas_norm.shape[3])
        # deduce H,W from any training image
        any_img = next((self.dataset_path / "train_imgs").glob("*.png"))
        if any_img is None:
            raise FileNotFoundError("No images found in train_imgs/")
        w, h = Image.open(any_img).size
        if h * w != D:
            # fallback if file sizes differ: assume square or D-compatible
            h = int(np.sqrt(D))
            w = int(D // h)
        self.H, self.W = h, w

        random_string = str(uuid.uuid4())[:6]
        if image_path is not None:
            self.image_path = Path(image_path)
            self.image_name = self.image_path.stem
            img = Image.open(self.image_path)
            img_arr = (color.rgb2ycbcr(img) / 255.0).transpose(2, 0, 1)
            self.gt_image = torch.tensor(img_arr, dtype=torch.float32, device=self.device)
            model_dir = Path(f"./models/recons/{self.image_name}-{self.dataset_path.name}-{num_points}-{args.iterations}-{random_string}")
        else:
            model_dir = Path(f"./models/multi-freq/{self.dataset_path.name}-{num_points}-{args.iterations}-{random_string}")
        
        self.num_points = num_points
        BLOCK_H, BLOCK_W = 16, 16
        self.iterations = iterations

        # --- freq split (robust to small num_comps/num_points) ---
        # how many PCA comps to devote to "low" band (cap at total comps)
        low_comp_cap = min(5, self.num_comps)
        # how many Gaussians to devote to "low" band (cap at total points)
        low_pts_cap  = min(2000, self.num_points)
        high_pts_cap = self.num_points - low_pts_cap  # can be 0 if very small model

        self.freq_config = {
            "low":  [0,              low_comp_cap,     low_pts_cap],
            "high": [low_comp_cap,   self.num_comps,   max(high_pts_cap, 0)],
            "all":  [0,              self.num_comps,   self.num_points],
        }
        # ----------------------------------------------------------


        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # IMPORTANT: enable k-PCA path and pass number of clusters
        self.gaussian_model = GaussianBasis(
            loss_type="L2", opt_type="adan", num_points=self.num_points,
            H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
            device=self.device, lr=args.lr, num_comps=self.num_comps,
            gt_arrs=None, freq_config=self.freq_config,
            use_kpca=True, num_clusters=self.num_clusters
        ).to(self.device)

        self.logwriter = LogWriter(self.model_dir)
        self.psnr_tracker = PSNRTracker(self.logwriter)
        self.logwriter.write(f"Model Dir ID: {random_string}")

        if model_path is not None:
            self.model_path = Path(model_path)
            self.logwriter.write(f"Model loaded from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.gaussian_model.load_state_dict(checkpoint['model_state_dict'])

        # working buffers
        self.gt_arrs = None              # will be filled per-cluster as (K, 3, H, W)
        self.active_cluster_id = None    # set before training each cluster

    # ---------- helpers for the clustered pipeline ----------
    def _prepare_gt_arrs_for_cluster(self, cid: int):
        """
        Build the ground-truth stack of eigenimages for cluster cid:
        shape (K, 3, H, W), values in [0,1] from cluster_pcas.npy.
        """
        comps = self.cluster_pcas_norm[cid]                # (3, K, D)
        comps = np.transpose(comps, (1, 0, 2))             # (K, 3, D)
        comps = comps.reshape(self.num_comps, 3, self.H, self.W)
        self.gt_arrs = torch.from_numpy(comps).float().to(self.device)

    def _set_active_cluster(self, cid: int):
        """
        Tell the model which cluster is being trained now.
        We pass a dummy w (zeros) because w is only needed for reconstruction,
        not for per-component training.
        """
        self.active_cluster_id = int(cid)
        dummy_w = torch.zeros(3, self.num_comps, device=self.device)
        self.gaussian_model.set_cluster_and_proj(self.active_cluster_id, dummy_w)

    # ---------- training (unchanged logic, but now per-cluster) ----------
    def train(self, freq: str):
        assert self.gt_arrs is not None, "gt_arrs not set. Call _prepare_gt_arrs_for_cluster(cid) first."
        assert self.active_cluster_id is not None, "active_cluster_id not set. Call _set_active_cluster(cid) first."

        self.gaussian_model.cur_freq = freq
        self.logwriter.write(f"[Cluster {self.active_cluster_id}] Train {freq} freq params, config: {self.freq_config[freq]}")
        self.gaussian_model.scheduler_init()
        progress_bar = tqdm(range(1, self.iterations+1), desc=f"C{self.active_cluster_id}:{freq} train")
        self.gaussian_model.train()
        start_time = time.time()
        for iter in range(1, self.iterations+1):
            loss, psnr = self.gaussian_model.train_iter(self.gt_arrs)
            with torch.no_grad():
                progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                progress_bar.update(1)
        
        end_time = time.time() - start_time
        progress_bar.close()
        self.logwriter.write(f"[Cluster {self.active_cluster_id}] {freq}-training Complete in {end_time:.4f}s")
        self.test_freq()
        return

    def train_all_clusters(self):
        for cid in range(self.num_clusters):
            self._prepare_gt_arrs_for_cluster(cid)
            self._set_active_cluster(cid)

            low_cnt  = self.freq_config["low"][1]  - self.freq_config["low"][0]
            high_cnt = self.freq_config["high"][1] - self.freq_config["high"][0]
            high_pts = self.freq_config["high"][2]

            if low_cnt > 0:
                self.train(freq="low")
            if high_cnt > 0 and high_pts > 0:
                self.train(freq="high")

        self.merge_freq()
        return

    def merge_freq(self):
        # merge geometry (unchanged)
        merged_xyz = torch.cat([
            self.gaussian_model.params["low_xyz"],
            self.gaussian_model.params["high_xyz"]
        ], dim=0)
        
        merged_cholesky = torch.cat([
            self.gaussian_model.params["low_cholesky"],
            self.gaussian_model.params["high_cholesky"]
        ], dim=0)

        L = self.freq_config["low"]
        H = self.freq_config["high"]

        # merge feature params
        merged_features = torch.zeros(self.num_comps, self.num_points, 3, device=self.device)
        merged_features[L[0] : L[1],         0 : L[2],           :] = self.gaussian_model.params["low_features_dc"].data
        merged_features[H[0] : H[1],      L[2] : L[2]+H[2],      :] = self.gaussian_model.params["high_features_dc"].data

        # if using k-PCA: also merge ψ′ across low/high into 'all'
        if "low_psi_kpca" in self.gaussian_model.params and "high_psi_kpca" in self.gaussian_model.params:
            merged_psi = torch.zeros(
                self.num_comps, self.num_points, 3, self.num_clusters, device=self.device
            )
            low_psi  = self.gaussian_model.params["low_psi_kpca"].data    # (K_low, N_low, 3, C)
            high_psi = self.gaussian_model.params["high_psi_kpca"].data   # (K_high, N_high, 3, C)

            merged_psi[L[0]:L[1],            0:L[2],              :, :] = low_psi
            merged_psi[H[0]:H[1],         L[2]:L[2]+H[2],         :, :] = high_psi

            self.gaussian_model.params["all_psi_kpca"].copy_(merged_psi)

        self.gaussian_model.params["all_xyz"].copy_(merged_xyz)
        self.gaussian_model.params["all_cholesky"].copy_(merged_cholesky)
        self.gaussian_model.params["all_features_dc"].copy_(merged_features)
        
        self.gaussian_model.cur_freq = "all"
        self.logwriter.write(f"Train all freq params, config: {self.freq_config['all']}")
        torch.save({"model_state_dict": self.gaussian_model.state_dict()}, self.model_dir / "gaussian_model.pth.tar")
        self.vis()
        return

    def optimize(self):
        # Optimize a specific image (self.image_path) after initializing from cluster PCA
        assert hasattr(self, "gt_image"), "optimize() expects --image_path to be provided."

        progress_bar = tqdm(range(1, self.iterations+1), desc="Optimizing progress")
        self.update_gaussian()  # sets colors and image_mean using k-PCA projection
        self.gaussian_model.scheduler_init(optimize_phase=True)
        self.gaussian_model.train()
        start_time = time.perf_counter()

        for iter in range(1, self.iterations+1):
            loss, psnr = self.gaussian_model.optimize_iter(self.gt_image)
            self.psnr_tracker.check(start_time, psnr, iter)
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10)
        
        end_time = time.perf_counter() - start_time
        progress_bar.close()    
        self.psnr_tracker.print_summary()

        self.logwriter.write("Optimizing Complete in {:.4f}s".format(end_time))
        torch.save({'model_state_dict': self.gaussian_model.state_dict()}, self.model_dir / "gaussian_model_with_colors.pth.tar")
        self.test()
        return

    def test_freq(self):
        config = self.freq_config[self.gaussian_model.cur_freq]
        gt_arrs = self.gt_arrs[config[0]:config[1], :, :, :]
        self.gaussian_model.eval()
        with torch.no_grad():
            image = self.gaussian_model()
        mse_loss = F.mse_loss(image.float(), gt_arrs.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim_robust(image.float(), self.gt_arrs.float()).item()        
        self.logwriter.write(f"[Cluster {self.active_cluster_id}] Components Fitting: PSNR:{psnr:.4f}, MS_SSIM:{ms_ssim_value:.6f}")
        return

    def vis(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            image = self.gaussian_model()
        mse_loss = F.mse_loss(image.float(), self.gt_arrs.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim_robust(image.float(), self.gt_arrs.float()).item()
        self.logwriter.write(f"[Cluster {self.active_cluster_id}] Components Fitting: PSNR:{psnr:.4f}, MS_SSIM:{ms_ssim_value:.6f}")
        
        vis_dir = self.model_dir / f"vis_comps"
        vis_dir.mkdir(parents=True, exist_ok=True)
        transform = transforms.ToPILImage()
        array = image.float()
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                img = transform(array[i, j])
                img.save(vis_dir / f"C{self.active_cluster_id}_{i}-{j}.png")
        
        return psnr, ms_ssim_value

    def update_gaussian(self):
        """
        Initialize colors from clustered PCA for the provided image.
        Sets:
          - model.current_cluster_id and current_w
          - model.image_mean to the cluster mean (per-pixel)
          - model._colors from ψ′ and w for the 'all' block
        Also writes a preview PNG of the initialized reconstruction.
        """
        # flatten current image channels (already YCbCr in [0,1])
        _, height, width = self.gt_image.shape
        Y = self.gt_image[0].reshape(-1).cpu().numpy()
        Cb = self.gt_image[1].reshape(-1).cpu().numpy()
        Cr = self.gt_image[2].reshape(-1).cpu().numpy()
        YCC = np.concatenate([Y, Cb, Cr], axis=0)[None, :]  # (1, 3D)

        # assign cluster
        cid = int(self.kmeans.predict(YCC)[0])
        self._set_active_cluster(cid)

        # compute PCA coefficients w for the image under cluster cid
        means = self.cluster_means[cid]         # (3, D)
        comps = self.cluster_pcas_raw[cid]      # (3, K, D)

        w_list = []
        for ch, vec in enumerate([Y, Cb, Cr]):
            x_centered = vec.astype(np.float32) - means[ch].astype(np.float32)             # (D,)
            w = x_centered @ comps[ch].astype(np.float32).T                                # (K,)
            w_list.append(w.astype(np.float32))
        w_np = np.stack(w_list, axis=0)  # (3, K)

        # feed to model and set mean
        self.gaussian_model.set_cluster_and_proj(cid, w_np)
        self.gaussian_model.image_mean = torch.from_numpy(means).to(self.device).float()   # (3, D)

        # construct per-Gaussian RGB features from ψ′ and w, then preview
        self.gaussian_model.cur_freq = "all"
        with torch.no_grad():
            start_f, end_f, _ = self.freq_config["all"]
            w_group = self.gaussian_model.current_w[:, start_f:end_f]  # (3, Kf=K)
            psi = self.gaussian_model.params["all_psi_kpca"][..., cid] # (K, N, 3)
            features = torch.einsum('knc,ck->nc', psi, w_group)        # (N, 3)
            self.gaussian_model._colors.data[:features.shape[0]].copy_(features)

            # preview
            out = self.gaussian_model(render_colors=True)

        ycbcr_img = (out.reshape(3, -1) + self.gaussian_model.image_mean).reshape(3, height, width)
        ycbcr_img = (ycbcr_img.detach().cpu().numpy() * 255.0).transpose(1, 2, 0)
        rgb_img = (color.ycbcr2rgb(ycbcr_img) * 255.0).clip(0, 255).astype(np.uint8)

        vis_dir = self.model_dir / f"vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb_img).save(vis_dir / f"kpca_init.png")
        self.logwriter.write(f"Init from cluster {cid} and saved vis/kpca_init.png")
        # --- also save side-by-side GT | kPCA-init ---
        pred_init_ycbcr = self.gaussian_model.forward_kpca_init()   # (3,H,W) in [0,1]
        rgb_pred = _ycbcr_to_rgb_uint8(pred_init_ycbcr)
        rgb_gt   = _ycbcr_to_rgb_uint8(self.gt_image)
        _save_side_by_side(rgb_gt, rgb_pred, self.model_dir / "vis" / "kpca_init_side_by_side.png")


    def test_pca(self):
        """
        Reconstruct the provided image directly with PCA (mean + w @ comps) for sanity.
        """
        assert hasattr(self, "gt_image"), "test_pca() expects --image_path to be provided."
        _, height, width = self.gt_image.shape

        Y = self.gt_image[0].reshape(-1).cpu().numpy()
        Cb = self.gt_image[1].reshape(-1).cpu().numpy()
        Cr = self.gt_image[2].reshape(-1).cpu().numpy()
        YCC = np.concatenate([Y, Cb, Cr], axis=0)[None, :]
        cid = int(self.kmeans.predict(YCC)[0])

        means = self.cluster_means[cid]     # (3, D)
        comps = self.cluster_pcas_raw[cid]  # (3, K, D)

        recons = []
        for ch, vec in enumerate([Y, Cb, Cr]):
            x_centered = vec.astype(np.float32) - means[ch].astype(np.float32)
            w = x_centered @ comps[ch].astype(np.float32).T          # (K,)
            x_hat = means[ch].astype(np.float32) + w @ comps[ch].astype(np.float32)  # (D,)
            recons.append((x_hat * 255.0).reshape(height, width))
        ycbcr_img = np.stack(recons, axis=-1)
        rgb_img = (color.ycbcr2rgb(ycbcr_img) * 255.0).clip(0, 255).astype(np.uint8)

        vis_dir = self.model_dir / f"vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb_img).save(vis_dir / f"test_pca.png")
        self.logwriter.write(f"PCA test reconstruction saved to vis/test_pca.png")

    def test(self):
        self.gaussian_model.eval()
        _, _, height, width = self.gt_arrs.shape

        with torch.no_grad():
            out = self.gaussian_model(render_colors=True)
        image = out.reshape(3, -1) + self.gaussian_model.image_mean
        image = image.reshape(3, height, width)
        mse_loss = F.mse_loss(image.float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim_robust(image.unsqueeze(0).float(), self.gt_image.unsqueeze(0).float()).item()
        
        self.logwriter.write("Test PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))

        ycbcr_img = image.detach().cpu().numpy() * 255.0
        ycbcr_img = ycbcr_img.reshape(3, height, width).transpose(1, 2, 0)
        rgb_img = color.ycbcr2rgb(ycbcr_img) * 255.0
        rgb_img = rgb_img.clip(0, 255)

        img = Image.fromarray(rgb_img.astype(np.uint8))
        name = self.image_name + "_fitting.png"
        img.save(str(self.model_dir / name))
        # --- also save side-by-side GT | final prediction ---
        rgb_pred = _ycbcr_to_rgb_uint8(image)
        rgb_gt   = _ycbcr_to_rgb_uint8(self.gt_image)
        _save_side_by_side(rgb_gt, rgb_pred, self.model_dir / f"{self.image_name}_fitting_side_by_side.png")

        return psnr, ms_ssim_value

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
        trainer.train_all_clusters()  # <-- train low/high per cluster, then merge
    else:
        # Optional sanity check:
        # trainer.test_pca()
        trainer.optimize()
 
if __name__ == "__main__":
    main(sys.argv[1:])
