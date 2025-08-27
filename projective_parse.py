#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse an image set with Projective Clustering (K J-flats) instead of global PCA.

Pipeline:
1) Read + resize images -> YCbCr -> flatten per-channel matrices (N, d)
2) Run EM-like (K,J)-flats on Y to get assignments (GPU)
3) For each cluster & channel, compute top-J components (GPU SVD)
4) Build arrs.npy using global per-channel min/max (consistent scaling)
5) Save projective_basis.pt (means, bases, norms, meta) + visualizations

Outputs in: <random>-K{K}-J{J}/
  - train_imgs/, test_imgs/
  - arrs.npy                # (K*J, 3, H, W) in [0,1]
  - projective_basis.pt     # dict: means, bases, norm_mins/maxs, meta ...
  - train_assignments.npy
  - vis/ *.png              # quick look at components
"""

from __future__ import annotations
import argparse, json, time, uuid
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from skimage import color

import torch
from torch import Tensor
from tqdm import tqdm

# EM engine (your new module)
from projective_clustering_gpu import em_projective


# --------------------------- tiny logger ---------------------------

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# --------------------------- image IO ------------------------------

def prepare_imgs(args, output_dir: Path):
    train_dir = output_dir / "train_imgs"
    test_dir  = output_dir / "test_imgs"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    img_list  = sorted([f for f in args.source.glob("*.png")])
    n_samples = min(args.n_samples, len(img_list))
    n_test    = min(args.n_samples + args.n_test, len(img_list))

    log(f"Found {len(img_list)} PNGs in {args.source}")
    log(f"Preparing images: train={n_samples}, test={n_test - n_samples}, size={tuple(args.img_size)}")

    def _grab(i):
        j = i * 2
        return img_list[j] if j < len(img_list) else img_list[-1]

    c = 0
    for i in tqdm(range(n_samples), desc="Prepare training images"):
        img = Image.open(_grab(i)).convert("RGB")
        img = img.resize((args.img_size[0], args.img_size[1]), Image.Resampling.LANCZOS)
        img.save(train_dir / f"{c:03d}.png")
        c += 1

    c = 0
    for i in tqdm(range(n_samples, n_test), desc="Prepare testing images"):
        img = Image.open(_grab(i)).convert("RGB")
        img = img.resize((args.img_size[0], args.img_size[1]), Image.Resampling.LANCZOS)
        img.save(test_dir / f"{c:03d}.png")
        c += 1

    log(f"Saved training images to {train_dir} and test images to {test_dir}")


def load_train_ycbcr(train_dir: Path, img_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Y, Cb, Cr in [0,1] as float32; each is (N, d) with d = H*W."""
    img_list = sorted([f for f in train_dir.glob("*.png")])
    N = len(img_list)
    W, H = img_size
    d = W * H

    log(f"Loading {N} training images from {train_dir} → YCbCr (flattened d={d}) ...")

    Y  = np.empty((N, d), dtype=np.float32)
    Cb = np.empty((N, d), dtype=np.float32)
    Cr = np.empty((N, d), dtype=np.float32)

    for i, fp in enumerate(tqdm(img_list, desc="YCbCr load")):
        img = Image.open(fp).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        ycbcr = color.rgb2ycbcr(arr).astype(np.float32) / 255.0  # [0,1]
        y, cb, cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
        Y[i]  = y.reshape(-1)
        Cb[i] = cb.reshape(-1)
        Cr[i] = cr.reshape(-1)

    log(f"Finished YCbCr load. Shapes: Y={Y.shape}, Cb={Cb.shape}, Cr={Cr.shape}")
    return Y, Cb, Cr


# -------------------- per-cluster components (Torch) --------------------

def _to_device(x, device: torch.device) -> Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    return x.to(device)


@torch.no_grad()
def pca_topJ_torch(X: Tensor, J: int) -> Tensor:
    """Top-J right singular vectors as row-vectors (J, d). No per-row normalization."""
    if X.shape[0] <= 1:
        return torch.zeros((J, X.shape[1]), dtype=X.dtype, device=X.device)
    Xc = X - X.mean(dim=0)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    return Vh[:J, :]  # (J, d)


def per_cluster_components(Y_cpu: np.ndarray, Cb_cpu: np.ndarray, Cr_cpu: np.ndarray,
                           assign: np.ndarray, K: int, J: int, device: str) -> Tuple[Tensor, Tensor]:
    """
    Compute per-cluster, per-channel means and top-J components.
    Returns:
      means: (K, 3, d)
      bases: (K, 3, J, d)   (row-vectors per channel)
    """
    torch_device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    Y  = _to_device(Y_cpu, torch_device).float()
    Cb = _to_device(Cb_cpu, torch_device).float()
    Cr = _to_device(Cr_cpu, torch_device).float()

    N, d = Y.shape
    means = torch.empty((K, 3, d), dtype=Y.dtype, device=torch_device)
    bases = torch.empty((K, 3, J, d), dtype=Y.dtype, device=torch_device)

    log(f"Computing per-cluster components (device={torch_device.type}) ...")
    for k in range(K):
        idxs = np.where(assign == k)[0]
        if len(idxs) == 0:
            log(f"  [cluster {k}] empty → reseed random {max(32, J+2)} points")
            idxs = np.random.choice(N, size=max(32, J+2), replace=False)
        else:
            log(f"  [cluster {k}] size={len(idxs)}")

        idx_t = torch.from_numpy(idxs).to(torch_device)

        for ch, (name, Xfull) in enumerate(zip(["Y", "Cb", "Cr"], [Y, Cb, Cr])):
            t0 = time.time()
            X = Xfull.index_select(0, idx_t)       # (n_c, d)
            m = X.mean(dim=0)                      # (d,)
            means[k, ch] = m
            comps = pca_topJ_torch(X, J)           # (J, d)
            bases[k, ch] = comps
            log(f"    [cluster {k}][{name}] PCA J={J} done in {time.time()-t0:.2f}s")

    log("Per-cluster component computation complete.")
    return means.detach().cpu(), bases.detach().cpu()


# --------------------------- saving & viz ---------------------------

def visualize(output_dir: Path, arrs: np.ndarray):
    """arrs: (K*J, 3, H, W) in [0,1]"""
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    log(f"Visualizing {arrs.shape[0]} components in {vis_dir} ...")
    for i in range(arrs.shape[0]):
        for ch, name in enumerate(["Y", "Cb", "Cr"]):
            img = (arrs[i, ch] * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(img, mode="L").save(vis_dir / f"{i:04d}-{name}.png")
    log("Visualization done.")


def save_outputs(output_dir: Path,
                 means: Tensor,   # (K,3,d)
                 bases: Tensor,   # (K,3,J,d)
                 assign: np.ndarray,
                 img_size: Tuple[int,int],
                 K: int, J: int):
    W, H = img_size
    d = W * H

    # Global per-channel min/max (for consistent scaling in Phase-A & Phase-B)
    # bases is (K,3,J,d)
    mins = bases.amin(dim=(0, 2, 3)).cpu().numpy()  # (3,)
    maxs = bases.amax(dim=(0, 2, 3)).cpu().numpy()  # (3,)
    den = (maxs - mins).copy()
    den[den == 0] = 1.0

    # Build arrs.npy in [0,1] using the same per-channel min/max for *all* components
    vis_list = []
    for k in range(K):
        for j in range(J):
            triplet = []
            for ch in range(3):
                v = bases[k, ch, j].view(H, W).cpu().numpy()
                v01 = (v - mins[ch]) / den[ch]
                v01 = np.clip(v01, 0.0, 1.0)
                triplet.append(v01[None, ...])  # (1,H,W)
            vis_list.append(np.concatenate(triplet, axis=0))  # (3,H,W)
    arrs = np.stack(vis_list, axis=0)  # (K*J, 3, H, W)

    arrs_path = output_dir / "arrs.npy"
    log(f"Saving arrs to {arrs_path} with shape {arrs.shape} ...")
    np.save(arrs_path, arrs)
    visualize(output_dir, arrs)

    pb_path = output_dir / "projective_basis.pt"
    log(f"Saving projective basis to {pb_path} ...")
    torch.save({
        "means": means,                            # (K,3,d) float32 (CPU)
        "bases": bases,                            # (K,3,J,d) float32 (CPU)
        "norm_mins": mins.astype(np.float32),      # (3,)
        "norm_maxs": maxs.astype(np.float32),      # (3,)
        "img_size": (W, H),
        "K": K, "J": J,
        "color_space": "YCbCr",
        "assignment_train": torch.as_tensor(assign).long(),  # tensor (safe load)
    }, pb_path)

    meta = {
        "img_size": [W, H],
        "K": K, "J": J,
        "color_space": "YCbCr",
        "components_total": K*J,
    }
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log(f"Saved meta to {meta_path}")

    assign_path = output_dir / "train_assignments.npy"
    np.save(assign_path, assign)
    log(f"Saved train assignments to {assign_path}")


# ------------------------------ main --------------------------------

def main():
    ap = argparse.ArgumentParser(description="Parse image set with Projective Clustering (K J-flats)")
    ap.add_argument("-s", "--source", required=True, type=Path, help="Directory with *.png images")
    ap.add_argument("-n", "--n_samples", type=int, default=10000)
    ap.add_argument("-t", "--n_test", type=int, default=100)
    ap.add_argument("--img_size", type=int, nargs=2, default=[512, 512],
                    metavar=('width','height'), help="Target size (W H)")
    ap.add_argument("-K", "--K_clusters", type=int, default=6, help="Number of clusters (K)")
    ap.add_argument("-J", "--J_dim", type=int, default=50, help="Subspace dimension per cluster (J)")
    ap.add_argument("--steps", type=int, default=20, help="EM steps")
    ap.add_argument("--inits", type=int, default=10, help="Random initializations of EM")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    args = ap.parse_args()

    random_string = str(uuid.uuid4())[:6]
    output_dir = args.source.parent / f"{random_string}-K{args.K_clusters}-J{args.J_dim}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log("========== PARSE PROJECTIVE START ==========")
    log(f"Args: src={args.source}  n_samples={args.n_samples}  n_test={args.n_test}  "
        f"img_size={tuple(args.img_size)}  K={args.K_clusters}  J={args.J_dim}  "
        f"steps={args.steps}  inits={args.inits}  device={args.device}")
    log(f"Output dir: {output_dir}")
    t_all = time.time()

    # 1) Prepare images
    t0 = time.time()
    prepare_imgs(args, output_dir)
    log(f"[1/5] Image preparation done in {time.time()-t0:.2f}s")

    # 2) Load Y,Cb,Cr matrices
    t0 = time.time()
    train_dir = output_dir / "train_imgs"
    Y, Cb, Cr = load_train_ycbcr(train_dir, tuple(args.img_size))
    log(f"[2/5] Loaded Y/Cb/Cr in {time.time()-t0:.2f}s")

    # 3) EM-like clustering on Y (GPU)
    t0 = time.time()
    log("[3/5] Running EM-like (K,J)-flats on Y ...")
    # Use balanced assignment to avoid collapse by default
    U_Y, v_Y, assign, info = em_projective(
        Y, K=args.K_clusters, J=args.J_dim,
        steps=args.steps, inits=args.inits,
        device=args.device,
        robust="l2", lam=1.0, Z=2.0,
        weights=None,
        balanced=True,     # recommended to reduce collapse
        seed=0,
    )
    sizes = np.bincount(assign, minlength=args.K_clusters).tolist()
    log(f"[3/5] EM complete in {time.time()-t0:.2f}s | cluster sizes={sizes}")

    # 4) Per-cluster, per-channel components
    t0 = time.time()
    log("[4/5] Computing per-cluster components (Y, Cb, Cr) ...")
    means, bases = per_cluster_components(Y, Cb, Cr, assign, K=args.K_clusters, J=args.J_dim, device=args.device)
    log(f"[4/5] Component computation done in {time.time()-t0:.2f}s")

    # 5) Save artifacts
    t0 = time.time()
    log("[5/5] Saving outputs (arrs.npy, projective_basis.pt, meta.json, assignments) ...")
    save_outputs(output_dir, means, bases, assign, tuple(args.img_size), args.K_clusters, args.J_dim)
    log(f"[5/5] Saving done in {time.time()-t0:.2f}s")

    log(f"========== DONE in {time.time()-t_all:.2f}s ==========")
    log(f"Artifacts saved to: {output_dir}")

if __name__ == "__main__":
    main()
