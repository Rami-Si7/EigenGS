# parse_emlike.py
import numpy as np
from skimage import color
from pathlib import Path
from PIL import Image
import pickle
import argparse
from tqdm import trange
import uuid
import json
import time

import torch
from torch import Tensor

# GPU EM-like algorithm (same math as original)
from projective_clustering_gpu import EMLikeAlg, assign_points

# --------------------------- helpers ---------------------------

def visualize(output_dir: Path):
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    array = np.load(output_dir / "arrs.npy")      # (K*J, 3, H, W) in [0,1]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            img = (array[i, j] * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(img, mode="L").save(vis_dir / f"{i:04d}-{j}.png")

def prepare_imgs(args, output_dir: Path):
    train_dir = output_dir / "train_imgs"
    test_dir  = output_dir / "test_imgs"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    img_list = sorted([f for f in args.source.glob("*.png")])
    n_samples = min(args.n_samples, len(img_list))
    n_test    = min(args.n_samples + args.n_test, len(img_list))

    c = 0
    for i in trange(n_samples, desc="Prepare training images"):
        img = Image.open(img_list[i*2]).convert("RGB")
        img = img.resize((args.img_size[0], args.img_size[1]), Image.Resampling.LANCZOS)
        img.save(train_dir / f"{c:03d}.png")
        c += 1

    c = 0
    for i in trange(n_samples, n_test, desc="Prepare testing images"):
        img = Image.open(img_list[i*2]).convert("RGB")
        img = img.resize((args.img_size[0], args.img_size[1]), Image.Resampling.LANCZOS)
        img.save(test_dir / f"{c:03d}.png")
        c += 1

def load_ycbcr_matrix(train_dir: Path, img_size: tuple[int,int]):
    img_list = sorted([f for f in train_dir.glob("*.png")])
    N = len(img_list)
    W, H = img_size
    d = W * H

    Y  = np.empty((N, d), dtype=np.float32)
    Cb = np.empty((N, d), dtype=np.float32)
    Cr = np.empty((N, d), dtype=np.float32)

    for i, fp in enumerate(img_list):
        img = Image.open(fp).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        ycbcr = color.rgb2ycbcr(arr).astype(np.float32) / 255.0  # [0,1]
        y, cb, cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
        Y[i]  = y.reshape(-1)
        Cb[i] = cb.reshape(-1)
        Cr[i] = cr.reshape(-1)
    return Y, Cb, Cr

@torch.no_grad()
def pca_topJ_torch(X: Tensor, J: int) -> Tensor:
    """
    Return top-J right singular vectors as row-vectors (J, d).
    Pads with zeros if J > rank to keep shapes consistent.
    """
    device = X.device
    dtype  = X.dtype
    if X.shape[0] == 0:
        return torch.zeros((J, X.shape[1]), device=device, dtype=dtype)
    Xc = X - X.mean(dim=0)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    Juse = min(J, Vh.shape[0])
    top = Vh[:Juse, :]
    if Juse < J:
        pad = torch.zeros((J - Juse, X.shape[1]), device=device, dtype=dtype)
        top = torch.cat([top, pad], dim=0)
    return top.contiguous()  # (J,d)

def build_arrs_and_save(output_dir: Path,
                        means: torch.Tensor,      # (K,3,d)
                        bases: torch.Tensor,      # (K,3,J,d)
                        img_size: tuple[int,int]):
    W, H = img_size
    d = W * H
    K, three, J, dd = bases.shape
    assert three == 3 and dd == d

    mins, maxs, den = compute_channel_norms(bases)

    vis_list = []
    for k in range(K):
        for j in range(J):
            trip = []
            for ch in range(3):
                v = bases[k, ch, j].view(H, W).cpu().numpy()
                v01 = (v - mins[ch]) / den[ch]
                v01 = np.clip(v01, 0.0, 1.0)
                trip.append(v01[None, ...])  # (1,H,W)
            vis_list.append(np.concatenate(trip, axis=0))  # (3,H,W)
    arrs = np.stack(vis_list, axis=0)  # (K*J, 3, H, W)
    np.save(output_dir / "arrs.npy", arrs)

    visualize(output_dir)

    torch.save({
        "means": means.cpu(),                 # (K,3,d)
        "bases": bases.cpu(),                 # (K,3,J,d)
        "img_size": (W, H),
        "K": K, "J": J,
        "color_space": "YCbCr",
        "norm_mins": mins.astype(np.float32),
        "norm_maxs": maxs.astype(np.float32),
    }, output_dir / "projective_basis.pt")

    return mins, maxs

    
def compute_channel_norms(bases: torch.Tensor):
    """Global per-channel mins/maxs used for consistent visualization scaling."""
    mins = bases.amin(dim=(0,2,3)).cpu().numpy()  # (3,)
    maxs = bases.amax(dim=(0,2,3)).cpu().numpy()  # (3,)
    den = (maxs - mins).copy()
    den[den == 0] = 1.0
    return mins, maxs, den

def save_clusters(outdir: Path,
                  assign: np.ndarray,       # (N,)
                  Vs_np: np.ndarray,         # (K, J, d)  (Y-channel flats from EM)
                  vs_np: np.ndarray,         # (K, d)
                  means: torch.Tensor,       # (K, 3, d)
                  bases: torch.Tensor,       # (K, 3, J, d)
                  img_size: tuple[int,int]):
    """
    Create clusters/cluster_XX folders with:
      - indices.npy                 (training indices in this cluster)
      - Y_flat_V.npy, Y_flat_v.npy  (affine J-flat for Y channel from EM)
      - means.npy   (3,d), bases.npy (3,J,d)  per-channel PCA within cluster
      - arrs.npy    (J,3,H,W) normalized with GLOBAL per-channel min/max
      - meta.json   (all shapes, norms, image size)
      - vis/*.png   quick grayscale previews per channel component
    """
    W, H = img_size
    K, three, J, d = bases.shape
    assert three == 3 and d == W*H

    mins, maxs, den = compute_channel_norms(bases)
    clusters_dir = outdir / "clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)

    for k in range(K):
        cdir = clusters_dir / f"cluster_{k:02d}"
        (cdir / "vis").mkdir(parents=True, exist_ok=True)

        # 1) membership
        idxs = np.where(assign == k)[0]
        np.save(cdir / "indices.npy", idxs)

        # 2) Y-channel affine flat from EM (row-basis + translation)
        np.save(cdir / "Y_flat_V.npy", Vs_np[k])  # (J,d)
        np.save(cdir / "Y_flat_v.npy", vs_np[k])  # (d,)

        # 3) per-channel means + bases (from your per-cluster PCA step)
        means_k = means[k].detach().cpu().numpy()        # (3,d)
        bases_k = bases[k].detach().cpu().numpy()        # (3,J,d)
        np.save(cdir / "means.npy", means_k)
        np.save(cdir / "bases.npy", bases_k)

        # 4) normalized “eigenimages” for this cluster only (J,3,H,W)
        vis_list = []
        for j in range(J):
            trip = []
            for ch in range(3):
                v = bases[k, ch, j].view(H, W).cpu().numpy()
                v01 = (v - mins[ch]) / den[ch]
                v01 = np.clip(v01, 0.0, 1.0)
                trip.append(v01[None, ...])   # (1,H,W)
                # quick per-channel png
                Image.fromarray((v01*255.0).astype(np.uint8), mode="L").save(
                    cdir / "vis" / f"{j:03d}-ch{ch}.png"
                )
            vis_list.append(np.concatenate(trip, axis=0))  # (3,H,W)
        arrs_k = np.stack(vis_list, axis=0)  # (J,3,H,W)
        np.save(cdir / "arrs.npy", arrs_k)

        # 5) metadata
        meta = {
            "cluster": int(k),
            "num_members": int(idxs.size),
            "img_size": [W, H],
            "J": int(J),
            "d": int(d),
            "norm_mins": mins.tolist(),
            "norm_maxs": maxs.tolist(),
            "files": ["indices.npy", "Y_flat_V.npy", "Y_flat_v.npy",
                      "means.npy", "bases.npy", "arrs.npy"],
        }
        with open(cdir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

# --------------------------- main ---------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Parse image set with EM-like (K,J)-flats on GPU")
    ap.add_argument("-s", "--source", required=True, type=Path, help="Folder with *.png")
    ap.add_argument("-K", "--K_clusters", type=int, default=6, help="number of clusters")
    ap.add_argument("-J", "--J_dim", type=int, default=50, help="flat dimension per cluster")
    ap.add_argument("-n", "--n_samples", type=int, default=10000)
    ap.add_argument("-t", "--n_test", type=int, default=100)
    ap.add_argument("--steps", type=int, default=20, help="EM steps")
    ap.add_argument("--inits", type=int, default=10, help="random initializations")
    ap.add_argument("--img_size", type=int, nargs=2, default=[512, 512],
                    metavar=('width','height'))
    args = ap.parse_args()

    outdir = args.source.parent / f"{str(uuid.uuid4())[:6]}-K{args.K_clusters}-J{args.J_dim}"
    outdir.mkdir(parents=True, exist_ok=True)

    print("== [1/5] Preparing resized train/test images ==")
    t0 = time.time()
    prepare_imgs(args, outdir)
    print(f"done in {time.time()-t0:.2f}s")

    print("== [2/5] Loading YCbCr (flattened) ==")
    t0 = time.time()
    train_dir = outdir / "train_imgs"
    Y, Cb, Cr = load_ycbcr_matrix(train_dir, tuple(args.img_size))
    print(f"done in {time.time()-t0:.2f}s | Y={Y.shape} Cb={Cb.shape} Cr={Cr.shape}")

    # --- EM-like clustering on Y (GPU) ---
    print("== [3/5] EM-like clustering on Y (GPU) ==")
    t0 = time.time()
    N = Y.shape[0]
    w = np.ones((N,), dtype=np.float32)
    Vs, vs, _ = EMLikeAlg(Y, w, j=args.J_dim, k=args.K_clusters, steps=args.steps)
    assign = assign_points(Y, w, Vs, vs)         # (N,)
    sizes = np.bincount(assign, minlength=args.K_clusters)
    print(f"done in {time.time()-t0:.2f}s | sizes={sizes.tolist()}")

    # --- Per-cluster, per-channel components on GPU ---
    print("== [4/5] Per-cluster {Y,Cb,Cr} top-J components per cluster (GPU) ==")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Yt  = torch.as_tensor(Y,  dtype=torch.float32, device=device)
    Cbt = torch.as_tensor(Cb, dtype=torch.float32, device=device)
    Crt = torch.as_tensor(Cr, dtype=torch.float32, device=device)

    K = args.K_clusters
    J = args.J_dim
    d = Y.shape[1]

    means = torch.empty((K, 3, d), device=device, dtype=torch.float32)
    bases = torch.empty((K, 3, J, d), device=device, dtype=torch.float32)

    for k in range(K):
        idxs = np.where(assign == k)[0]
        if idxs.size == 0:
            # fallback: pick a small random subset to avoid empty SVD
            idxs = np.random.choice(N, size=max(J+2, 32), replace=False)
        idx_t = torch.as_tensor(idxs, device=device)

        for ch, Xfull in enumerate([Yt, Cbt, Crt]):
            Xk = Xfull.index_select(0, idx_t)        # (n_c, d)
            means[k, ch] = Xk.mean(dim=0)            # (d,)
            comps = pca_topJ_torch(Xk, J)            # (J,d) row-vectors
            bases[k, ch] = comps

    print("== [5/5] Saving arrs + basis ==")
    build_arrs_and_save(outdir, means, bases, tuple(args.img_size))
    save_clusters(outdir, assign, Vs, vs, means, bases, tuple(args.img_size))
    # also save assignments + meta for inspection
    np.save(outdir / "train_assignments.npy", assign)
    with open(outdir / "meta.json", "w") as f:
        json.dump({
            "img_size": list(args.img_size),
            "K": args.K_clusters,
            "J": args.J_dim,
            "n_samples": int(args.n_samples),
            "n_test": int(args.n_test),
            "cluster_sizes": sizes.tolist()
        }, f, indent=2)

    print(f"== DONE. Artifacts in: {outdir} ==")
