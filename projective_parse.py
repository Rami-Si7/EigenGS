# parse_emlike.py
import numpy as np
from skimage import color
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm
import uuid
import json
import time

import torch
from torch import Tensor

# GPU EM-like algorithm (same math as original)
from projective_clustering_gpu import EMLikeAlg, assign_points

# --------------------------- helpers ---------------------------
def load_ycbcr_from_paths(img_paths, img_size: tuple[int,int]):
    """Given a list of paths, load/resize and return Y,Cb,Cr and the (possibly re-ordered) paths."""
    W, H = img_size
    N = len(img_paths)
    d = W * H
    Y  = np.empty((N, d), dtype=np.float32)
    Cb = np.empty((N, d), dtype=np.float32)
    Cr = np.empty((N, d), dtype=np.float32)

    for i, fp in enumerate(tqdm(img_paths, desc="Load + resize + RGB->YCbCr")):
        img = Image.open(fp).convert("RGB")
        img = img.resize((W, H), Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.uint8)
        ycbcr = color.rgb2ycbcr(arr).astype(np.float32) / 255.0
        y, cb, cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
        Y[i]  = y.reshape(-1)
        Cb[i] = cb.reshape(-1)
        Cr[i] = cr.reshape(-1)
    return Y, Cb, Cr, img_paths

def load_ycbcr_from_source(src_dir: Path, img_size: tuple[int,int]):
    """Load ALL *.png from source, resize to (W,H), return Y,Cb,Cr matrices and the file list."""
    img_paths = sorted([f for f in src_dir.glob("*.png")])
    N = len(img_paths)
    W, H = img_size
    d = W * H

    Y  = np.empty((N, d), dtype=np.float32)
    Cb = np.empty((N, d), dtype=np.float32)
    Cr = np.empty((N, d), dtype=np.float32)

    for i, fp in enumerate(tqdm(img_paths, desc="Load + resize + RGB->YCbCr")):
        img = Image.open(fp).convert("RGB")
        img = img.resize((W, H), Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.uint8)
        ycbcr = color.rgb2ycbcr(arr).astype(np.float32) / 255.0  # [0,1]
        y, cb, cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
        Y[i]  = y.reshape(-1)
        Cb[i] = cb.reshape(-1)
        Cr[i] = cr.reshape(-1)
    return Y, Cb, Cr, img_paths

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

def compute_channel_norms(bases: torch.Tensor):
    """Global per-channel mins/maxs used for consistent visualization scaling."""
    mins = bases.amin(dim=(0,2,3)).cpu().numpy()  # (3,)
    maxs = bases.amax(dim=(0,2,3)).cpu().numpy()  # (3,)
    den = (maxs - mins).copy()
    den[den == 0] = 1.0
    return mins, maxs, den

def visualize(output_dir: Path):
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    array = np.load(output_dir / "arrs.npy")      # (K*J, 3, H, W) in [0,1]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            img = (array[i, j] * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(img, mode="L").save(vis_dir / f"{i:04d}-{j}.png")

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

def save_clusters(outdir: Path,
                  assign: np.ndarray,         # (N,)
                  train_idxs_per_k: dict,     # k -> np.ndarray
                  test_idxs_per_k: dict,      # k -> np.ndarray
                  Vs_np: np.ndarray,          # (K,3,J,d)  or legacy (K,J,d)
                  vs_np: np.ndarray,          # (K,3,d)    or legacy (K,d)
                  means: torch.Tensor,        # (K,3,d)
                  bases: torch.Tensor,        # (K,3,J,d)
                  img_size: tuple[int,int]):
    """
    Write cluster artifacts and split indices to outdir/clusters/cluster_XX.

    Saves:
      - train_indices.npy, test_indices.npy
      - em_Vs.npy, em_vs.npy            (3,J,d) and (3,d)  [NEW if available]
      - Y_flat_V.npy, Y_flat_v.npy      (J,d) and (d,)     [kept for compatibility]
      - means.npy, bases.npy            (3,d) and (3,J,d)  (per-cluster PCA, TRAIN only)
      - arrs.npy                        (J,3,H,W)          normalized eigenimages
      - meta.json
    """
    W, H = img_size
    K, three, J, d = bases.shape
    assert three == 3 and d == W * H

    # normalize for visualization
    mins, maxs, den = compute_channel_norms(bases)

    clusters_dir = outdir / "clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)

    # detect EM shapes (legacy single-channel vs multi-channel)
    em_is_multi = (Vs_np.ndim == 4 and vs_np.ndim == 3)  # expect (K,3,J,d) and (K,3,d)
    em_J = None

    for k in range(K):
        cdir = clusters_dir / f"cluster_{k:02d}"
        (cdir / "vis").mkdir(parents=True, exist_ok=True)

        # split indices
        np.save(cdir / "train_indices.npy", train_idxs_per_k[k])
        np.save(cdir / "test_indices.npy",  test_idxs_per_k[k])

        # ----- EM subspaces -----
        if em_is_multi:
            # full 3-channel EM flats
            em_Vs_k = Vs_np[k]         # (3,J,d)
            em_vs_k = vs_np[k]         # (3,d)
            em_J = em_Vs_k.shape[1]
            np.save(cdir / "em_Vs.npy", em_Vs_k)
            np.save(cdir / "em_vs.npy", em_vs_k)

            # backward-compat convenience (Y-channel slice)
            np.save(cdir / "Y_flat_V.npy", em_Vs_k[0])   # (J,d)
            np.save(cdir / "Y_flat_v.npy", em_vs_k[0])   # (d,)
        else:
            # legacy (Y-only EM)
            np.save(cdir / "Y_flat_V.npy", Vs_np[k])     # (J,d)
            np.save(cdir / "Y_flat_v.npy", vs_np[k])     # (d,)

        # ----- per-cluster PCA (TRAIN only) -----
        means_k = means[k].detach().cpu().numpy()        # (3,d)
        bases_k = bases[k].detach().cpu().numpy()        # (3,J,d)
        np.save(cdir / "means.npy", means_k)
        np.save(cdir / "bases.npy", bases_k)

        # ----- normalized eigenimages for this cluster (J,3,H,W) -----
        vis_list = []
        for j in range(J):
            trip = []
            for ch in range(3):
                v = bases[k, ch, j].view(H, W).cpu().numpy()
                v01 = (v - mins[ch]) / den[ch]
                v01 = np.clip(v01, 0.0, 1.0)
                trip.append(v01[None, ...])   # (1,H,W)
                Image.fromarray((v01 * 255.0).astype(np.uint8), mode="L").save(
                    cdir / "vis" / f"{j:03d}-ch{ch}.png"
                )
            vis_list.append(np.concatenate(trip, axis=0))  # (3,H,W)
        arrs_k = np.stack(vis_list, axis=0)  # (J,3,H,W)
        np.save(cdir / "arrs.npy", arrs_k)

        # ----- metadata -----
        idxs_all = np.where(assign == k)[0]
        files_list = [
            "train_indices.npy", "test_indices.npy",
            "Y_flat_V.npy", "Y_flat_v.npy",
            "means.npy", "bases.npy", "arrs.npy"
        ]
        if em_is_multi:
            files_list = ["em_Vs.npy", "em_vs.npy"] + files_list

        meta = {
            "cluster": int(k),
            "num_members": int(idxs_all.size),
            "num_train": int(train_idxs_per_k[k].size),
            "num_test": int(test_idxs_per_k[k].size),
            "img_size": [W, H],
            "J": int(J),
            "d": int(d),
            "em_multi_channel": bool(em_is_multi),
            "em_J": int(em_J if em_is_multi else J),
            "norm_mins": [float(x) for x in mins.tolist()],
            "norm_maxs": [float(x) for x in maxs.tolist()],
            "files": files_list,
        }
        with open(cdir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
# def save_clusters(outdir: Path,
#                   assign: np.ndarray,         # (N,)
#                   train_idxs_per_k: dict,     # k -> np.ndarray
#                   test_idxs_per_k: dict,      # k -> np.ndarray
#                   Vs_np: np.ndarray,          # (K, J, d)
#                   vs_np: np.ndarray,          # (K, d)
#                   means: torch.Tensor,        # (K, 3, d)
#                   bases: torch.Tensor,        # (K, 3, J, d)
#                   img_size: tuple[int,int]):
#     """
#     Write cluster artifacts and split indices to outdir/clusters/cluster_XX.
#     """
#     W, H = img_size
#     K, three, J, d = bases.shape
#     assert three == 3 and d == W*H

#     mins, maxs, den = compute_channel_norms(bases)
#     clusters_dir = outdir / "clusters"
#     clusters_dir.mkdir(parents=True, exist_ok=True)

#     for k in range(K):
#         cdir = clusters_dir / f"cluster_{k:02d}"
#         (cdir / "vis").mkdir(parents=True, exist_ok=True)

#         # split indices
#         np.save(cdir / "train_indices.npy", train_idxs_per_k[k])
#         np.save(cdir / "test_indices.npy",  test_idxs_per_k[k])

#         # Y-channel affine flat from EM (row-basis + translation)
#         np.save(cdir / "Y_flat_V.npy", Vs_np[k])  # (J,d)
#         np.save(cdir / "Y_flat_v.npy", vs_np[k])  # (d,)

#         # per-channel means + bases (from per-cluster PCA on TRAIN set)
#         means_k = means[k].detach().cpu().numpy()        # (3,d)
#         bases_k = bases[k].detach().cpu().numpy()        # (3,J,d)
#         np.save(cdir / "means.npy", means_k)
#         np.save(cdir / "bases.npy", bases_k)

#         # normalized eigenimages for this cluster only (J,3,H,W)
#         vis_list = []
#         for j in range(J):
#             trip = []
#             for ch in range(3):
#                 v = bases[k, ch, j].view(H, W).cpu().numpy()
#                 v01 = (v - mins[ch]) / den[ch]
#                 v01 = np.clip(v01, 0.0, 1.0)
#                 trip.append(v01[None, ...])   # (1,H,W)
#                 Image.fromarray((v01*255.0).astype(np.uint8), mode="L").save(
#                     cdir / "vis" / f"{j:03d}-ch{ch}.png"
#                 )
#             vis_list.append(np.concatenate(trip, axis=0))  # (3,H,W)
#         arrs_k = np.stack(vis_list, axis=0)  # (J,3,H,W)
#         np.save(cdir / "arrs.npy", arrs_k)

#         # metadata
#         idxs_all = np.where(assign == k)[0]
#         meta = {
#             "cluster": int(k),
#             "num_members": int(idxs_all.size),
#             "num_train": int(train_idxs_per_k[k].size),
#             "num_test": int(test_idxs_per_k[k].size),
#             "img_size": [W, H],
#             "J": int(J),
#             "d": int(d),
#             "norm_mins": mins.tolist(),
#             "norm_maxs": maxs.tolist(),
#             "files": ["train_indices.npy", "test_indices.npy",
#                       "Y_flat_V.npy", "Y_flat_v.npy",
#                       "means.npy", "bases.npy", "arrs.npy"],
#         }
#         with open(cdir / "meta.json", "w") as f:
#             json.dump(meta, f, indent=2)

def save_split_images(img_paths, train_idxs_per_k, test_idxs_per_k, outdir: Path, img_size: tuple[int,int]):
    """
    Save resized copies per-cluster (train/test) AND aggregated all_train/all_test.
    """
    W, H = img_size
    # global folders
    all_train_dir = outdir / "all_train"
    all_test_dir  = outdir / "all_test"
    all_train_dir.mkdir(parents=True, exist_ok=True)
    all_test_dir.mkdir(parents=True, exist_ok=True)

    # per-cluster folders
    for k in train_idxs_per_k.keys():
        (outdir / "clusters" / f"cluster_{k:02d}" / "train_imgs").mkdir(parents=True, exist_ok=True)
        (outdir / "clusters" / f"cluster_{k:02d}" / "test_imgs").mkdir(parents=True, exist_ok=True)

    # save helper
    def _save(idx, dest_dir: Path, k: int):
        fp = img_paths[idx]
        img = Image.open(fp).convert("RGB")
        img = img.resize((W, H), Image.Resampling.LANCZOS)
        name = f"{k:02d}_{idx:06d}.png"  # cluster-prefixed, unique
        img.save(dest_dir / name)

    # write per-cluster + aggregated
    for k, idxs in train_idxs_per_k.items():
        ctrain = outdir / "clusters" / f"cluster_{k:02d}" / "train_imgs"
        for idx in tqdm(idxs, desc=f"Save cluster {k} train", leave=False):
            _save(int(idx), ctrain, k)
            _save(int(idx), all_train_dir, k)

    for k, idxs in test_idxs_per_k.items():
        ctest = outdir / "clusters" / f"cluster_{k:02d}" / "test_imgs"
        for idx in tqdm(idxs, desc=f"Save cluster {k} test", leave=False):
            _save(int(idx), ctest, k)
            _save(int(idx), all_test_dir, k)

# --------------------------- main ---------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Parse image set with EM-like (K,J)-flats on GPU (cluster -> split -> save)")
    ap.add_argument("-s", "--source", required=True, type=Path, help="Folder with *.png")
    ap.add_argument("-K", "--K_clusters", type=int, default=6, help="number of clusters")
    ap.add_argument("-J", "--J_dim", type=int, default=50, help="flat dimension per cluster")
    ap.add_argument("--steps", type=int, default=20, help="EM steps")
    ap.add_argument("--inits", type=int, default=10, help="random inits (set inside GPU module if needed)")
    ap.add_argument("--img_size", type=int, nargs=2, default=[512, 512], metavar=('width','height'))
    ap.add_argument("--test_frac", type=float, default=0.1, help="fraction per cluster to reserve for test")
    ap.add_argument("--max_samples", type=int, default=29496,
                help="If >0, randomly pick this many images from source before EM (default: 29496)")
    ap.add_argument("--sample_seed", type=int, default=0,
                help="Random seed for sampling (default: 0)")

    args = ap.parse_args()

    outdir = args.source.parent / f"{str(uuid.uuid4())[:6]}-K{args.K_clusters}-J{args.J_dim}"
    outdir.mkdir(parents=True, exist_ok=True)

    # == [1/5] Loading and converting images ==
    print("== [1/5] Loading and converting images ==")
    t0 = time.time()
    W, H = args.img_size

    all_paths = sorted([f for f in args.source.glob("*.png")])
    M = args.max_samples
    if M > 0 and M < len(all_paths):
        rng = np.random.default_rng(args.sample_seed)
        sel = rng.choice(len(all_paths), size=M, replace=False)
        # keep a stable order for filenames/splits
        img_paths = [all_paths[i] for i in sorted(sel.tolist())]
    else:
        img_paths = all_paths

    Y, Cb, Cr, img_paths = load_ycbcr_from_paths(img_paths, (W, H))
    N = Y.shape[0]
    print(f"done in {time.time()-t0:.2f}s | picked N={N}/{len(all_paths)} | vec dim d={Y.shape[1]}")

    # == [2/5] EM-like clustering on Y (GPU) ==
    print("== [2/5] EM-like clustering on Y (GPU) ==")
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Y_t = torch.as_tensor(Y, dtype=torch.float32, device=device)
    Cb_t = torch.as_tensor(Cb, dtype=torch.float32, device=device)
    Cr_t = torch.as_tensor(Cr, dtype=torch.float32, device=device)
    w_t = torch.ones((N,), dtype=torch.float32, device=device)
    Vs_t, vs_t, _ = EMLikeAlg(Y_t, Cb_t, Cr_t, w_t, j=args.J_dim, k=args.K_clusters, steps=args.steps)
    assign_t = assign_points(Y_t, Cb_t, Cr_t, Vs_t, vs_t, w=w_t, alphas=(1.0, 1, 1))
    assign = assign_t.detach().cpu().numpy()
    Vs = Vs_t.detach().cpu().numpy()
    vs = vs_t.detach().cpu().numpy()
    sizes = np.bincount(assign, minlength=args.K_clusters)
    # Helpful top-level artifacts
    np.save(outdir / "assign.npy", assign)           # per-image cluster label
    with open(outdir / "em_shapes.json", "w") as f:  # shape audit
        json.dump({"Vs": list(Vs.shape), "vs": list(vs.shape)}, f, indent=2)
    print(f"done in {time.time()-t0:.2f}s | cluster sizes={sizes.tolist()}")

    # == [3/5] Split train/test per cluster ==
    print("== [3/5] Split train/test per cluster ==")
    rng = np.random.default_rng(0)
    train_idxs_per_k = {}
    test_idxs_per_k = {}
    for k in range(args.K_clusters):
        idxs = np.where(assign == k)[0]
        if idxs.size == 0:
            train_idxs_per_k[k] = np.array([], dtype=np.int64)
            test_idxs_per_k[k]  = np.array([], dtype=np.int64)
            continue
        idxs = rng.permutation(idxs)
        n_test = max(1, int(round(args.test_frac * idxs.size))) if idxs.size > 1 else 1
        n_test = min(n_test, idxs.size)  # clamp
        test_idxs = idxs[:n_test]
        train_idxs = idxs[n_test:]
        # if tiny cluster, allow empty train set
        train_idxs_per_k[k] = train_idxs.astype(np.int64)
        test_idxs_per_k[k]  = test_idxs.astype(np.int64)

    # == [4/5] Save images per-cluster + aggregated ==
    print("== [4/5] Saving images per cluster and aggregated train/test ==")
    save_split_images(img_paths, train_idxs_per_k, test_idxs_per_k, outdir, (W, H))

    # == [5/5] Per-cluster PCA on TRAIN members only ==
    print("== [5/5] Per-cluster {Y,Cb,Cr} top-J components per cluster (TRAIN only, GPU) ==")
    Yt  = torch.as_tensor(Y,  dtype=torch.float32, device=device)
    Cbt = torch.as_tensor(Cb, dtype=torch.float32, device=device)
    Crt = torch.as_tensor(Cr, dtype=torch.float32, device=device)

    K = args.K_clusters
    J = args.J_dim
    d = Y.shape[1]
    means = torch.empty((K, 3, d), device=device, dtype=torch.float32)
    bases = torch.empty((K, 3, J, d), device=device, dtype=torch.float32)

    for k in range(K):
        tr = train_idxs_per_k[k]
        if tr.size == 0:
            # if no train in this cluster, borrow a small random subset from all data
            tr = np.random.default_rng(1).choice(N, size=min(max(J+2, 32), N), replace=False)
        idx_t = torch.as_tensor(tr, device=device)

        for ch, Xfull in enumerate([Yt, Cbt, Crt]):
            Xk = Xfull.index_select(0, idx_t)        # (n_c, d)
            means[k, ch] = Xk.mean(dim=0)            # (d,)
            comps = pca_topJ_torch(Xk, J)            # (J,d) row-vectors
            bases[k, ch] = comps

    # Save analysis artifacts + cluster metadata (incl. split indices)
    build_arrs_and_save(outdir, means, bases, (W, H))
    save_clusters(outdir, assign, train_idxs_per_k, test_idxs_per_k,
                  Vs, vs, means, bases, (W, H))

    # top-level meta
    with open(outdir / "meta.json", "w") as f:
        json.dump({
            "img_size": [W, H],
            "K": K,
            "J": J,
            "N": int(N),
            "cluster_sizes": sizes.tolist(),
            "test_frac": float(args.test_frac),
            "outdirs": {
                "clusters": "clusters/cluster_xx/{train_imgs,test_imgs,...}",
                "all_train": "all_train/",
                "all_test": "all_test/"
            }
        }, f, indent=2)

    print(f"== DONE. Artifacts in: {outdir} ==")
