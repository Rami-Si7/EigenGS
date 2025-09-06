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
from typing import Dict, List, Tuple

import torch
from torch import Tensor

# --------------------------- imports from your GPU EM module ---------------------------
# Expect the single-channel EM + assign. If your names differ, adjust here.
try:
    from projective_clustering_gpu import EMLikeAlg_1ch, assign_points_1ch
except ImportError:
    # Fallback names if you kept them as EMLikeAlg/assign_points for 1-channel
    from projective_clustering_gpu import EMLikeAlg as EMLikeAlg_1ch   # type: ignore
    from projective_clustering_gpu import assign_points as assign_points_1ch  # type: ignore


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


def _compute_norms_1ch(bases: torch.Tensor):
    """
    bases: (K,J,d) for a single channel.
    Returns global min/max across all K*J rows (for visualization scaling).
    """
    mins = bases.amin().item()
    maxs = bases.amax().item()
    if maxs - mins < 1e-12:
        maxs = mins + 1.0
    return float(mins), float(maxs)


def _save_channel_vis_arrays(ch_dir: Path, bases: torch.Tensor, img_size: tuple[int,int]):
    """
    Save a channel-level arrs.npy = concatenated (J,H,W) per cluster, stacked along axis=0.
    Also writes per-cluster grayscale eigenimages to clusters/cluster_xx/vis.
    """
    W, H = img_size
    K, J, d = bases.shape
    assert d == W*H
    vmin, vmax = _compute_norms_1ch(bases)
    den = (vmax - vmin)

    # Save a big (K*J, H, W) for convenience
    all_list = []
    for k in range(K):
        for j in range(J):
            v = bases[k, j].view(H, W).cpu().numpy()
            v01 = np.clip((v - vmin) / den, 0.0, 1.0)
            all_list.append(v01)
    arrs = np.stack(all_list, axis=0)  # (K*J, H, W)
    np.save(ch_dir / "arrs.npy", arrs)


def _save_one(img_path: Path, dest_dir: Path, W: int, H: int, name: str):
    img = Image.open(img_path).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    img.save(dest_dir / name)


def save_global_split_images(img_paths: List[Path], train_idx: np.ndarray, test_idx: np.ndarray,
                             outdir: Path, img_size: tuple[int,int]):
    """
    Save the identical train/test copies once at the TOP LEVEL (not per channel),
    with filenames based ONLY on the global index (consistent across channels).
    """
    W, H = img_size
    all_train_dir = outdir / "all_train"
    all_test_dir  = outdir / "all_test"
    all_train_dir.mkdir(parents=True, exist_ok=True)
    all_test_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(train_idx, desc="Save global all_train", leave=False):
        name = f"{int(idx):06d}.png"
        _save_one(img_paths[int(idx)], all_train_dir, W, H, name)

    for idx in tqdm(test_idx, desc="Save global all_test", leave=False):
        name = f"{int(idx):06d}.png"
        _save_one(img_paths[int(idx)], all_test_dir, W, H, name)


def save_per_channel_cluster_images(img_paths: List[Path],
                                    train_idxs_per_k: Dict[int, np.ndarray],
                                    test_idxs_per_k: Dict[int, np.ndarray],
                                    ch_dir: Path, img_size: tuple[int,int]):
    """
    Inside a channel directory, save per-cluster train/test images.
    Filenames include a cluster prefix to keep them unique locally.
    """
    W, H = img_size
    for k in train_idxs_per_k.keys():
        (ch_dir / "clusters" / f"cluster_{k:02d}" / "train_imgs").mkdir(parents=True, exist_ok=True)
        (ch_dir / "clusters" / f"cluster_{k:02d}" / "test_imgs").mkdir(parents=True, exist_ok=True)

    def _save(idx: int, dest_dir: Path, k: int):
        name = f"{k:02d}_{idx:06d}.png"  # cluster-prefixed
        _save_one(img_paths[idx], dest_dir, W, H, name)

    for k, idxs in train_idxs_per_k.items():
        ctrain = ch_dir / "clusters" / f"cluster_{k:02d}" / "train_imgs"
        for idx in tqdm(idxs, desc=f"{ch_dir.name}: Save cluster {k} train", leave=False):
            _save(int(idx), ctrain, k)

    for k, idxs in test_idxs_per_k.items():
        ctest = ch_dir / "clusters" / f"cluster_{k:02d}" / "test_imgs"
        for idx in tqdm(idxs, desc=f"{ch_dir.name}: Save cluster {k} test", leave=False):
            _save(int(idx), ctest, k)


def save_channel_clusters(
    ch_dir: Path,
    train_idxs_per_k: Dict[int, np.ndarray],      # k -> np.ndarray of GLOBAL indices
    test_idxs_per_k: Dict[int, np.ndarray],       # k -> np.ndarray of GLOBAL indices
    Vs_np: np.ndarray,                            # (K,J,d) trained on TRAIN ONLY
    vs_np: np.ndarray,                            # (K,d)
    means: torch.Tensor,                          # (K,d)   PCA TRAIN-only
    bases: torch.Tensor,                          # (K,J,d) PCA TRAIN-only
    img_size: tuple[int,int]
):
    """
    Write per-channel cluster artifacts: splits, EM flats, PCA, visualizations, metadata.
    """
    W, H = img_size
    K, J, d = bases.shape
    assert d == W*H

    clusters_dir = ch_dir / "clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)

    # global min/max for vis scaling (for PNGs/arrs)
    vmin, vmax = _compute_norms_1ch(bases)
    den = (vmax - vmin)

    for k in range(K):
        cdir = clusters_dir / f"cluster_{k:02d}"
        (cdir / "vis").mkdir(parents=True, exist_ok=True)

        # split indices (GLOBAL indices saved)
        np.save(cdir / "train_indices.npy", train_idxs_per_k[k])
        np.save(cdir / "test_indices.npy",  test_idxs_per_k[k])

        # EM subspace for this cluster (trained on TRAIN only)
        np.save(cdir / "em_V.npy", Vs_np[k])    # (J,d)
        np.save(cdir / "em_v.npy", vs_np[k])    # (d,)

        # PCA per cluster (TRAIN-only)
        means_k = means[k].detach().cpu().numpy()   # (d,)
        bases_k = bases[k].detach().cpu().numpy()   # (J,d)
        np.save(cdir / "pca_mean.npy",  means_k)
        np.save(cdir / "pca_bases.npy", bases_k)

        # normalized eigenimages for this cluster (J,H,W) + PNGs
        vis_list = []
        for j in range(J):
            v = bases[k, j].view(H, W).cpu().numpy()
            v01 = np.clip((v - vmin) / den, 0.0, 1.0)
            vis_list.append(v01)
            Image.fromarray((v01 * 255.0).astype(np.uint8), mode="L").save(
                cdir / "vis" / f"{j:03d}.png"
            )
        arrs_k = np.stack(vis_list, axis=0)  # (J,H,W)
        np.save(cdir / "arrs.npy", arrs_k)

        # metadata
        meta = {
            "cluster": int(k),
            "num_train": int(train_idxs_per_k[k].size),
            "num_test": int(test_idxs_per_k[k].size),
            "img_size": [W, H],
            "J": int(J),
            "d": int(d),
            "files": [
                "train_indices.npy", "test_indices.npy",
                "em_V.npy", "em_v.npy",
                "pca_mean.npy", "pca_bases.npy", "arrs.npy"
            ],
        }
        with open(cdir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    # channel-level normalized array (K*J,H,W)
    _save_channel_vis_arrays(ch_dir, bases, img_size)


def run_em_for_channel(
    name: str,
    P_np: np.ndarray,
    img_paths: List[Path],
    outdir: Path,
    K: int,
    J: int,
    steps: int,
    inits: int,
    img_size: tuple[int,int],
    train_idx_global: np.ndarray,      # global TRAIN indices
    test_idx_global: np.ndarray        # global TEST indices
):
    """
    Runs single-channel EM on TRAIN ONLY, assigns TRAIN/TEST with the learned flats,
    saves artifacts to outdir/name, and returns a summary.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P = torch.as_tensor(P_np, dtype=torch.float32, device=device)  # (N,d)
    N, d = P.shape
    ch_dir = outdir / name
    ch_dir.mkdir(parents=True, exist_ok=True)

    # ----- slices -----
    train_idx_t = torch.as_tensor(train_idx_global, device=device, dtype=torch.long)
    test_idx_t  = torch.as_tensor(test_idx_global,  device=device, dtype=torch.long)
    P_tr = P.index_select(0, train_idx_t)   # (Ntr, d)
    P_te = P.index_select(0, test_idx_t)    # (Nte, d)
    w_tr = torch.ones((P_tr.shape[0],), dtype=torch.float32, device=device)
    w_te = torch.ones((P_te.shape[0],), dtype=torch.float32, device=device)

    # ----- EM on TRAIN ONLY -----
    try:
        Vs_t, vs_t, assign_tr_t, _ = EMLikeAlg_1ch(P_tr, w_tr, j=J, k=K, steps=steps, num_inits=inits, verbose=True)
    except TypeError:
        # Older signature (no assign returned); compute assignments after
        Vs_t, vs_t, _ = EMLikeAlg_1ch(P_tr, w_tr, j=J, k=K, steps=steps)  # type: ignore
        assign_tr_t = assign_points_1ch(P_tr, w_tr, Vs_t, vs_t)

    # ----- Assign TRAIN and TEST to learned flats -----
    assign_te_t = assign_points_1ch(P_te, w_te, Vs_t, vs_t)

    assign_tr = assign_tr_t.detach().cpu().numpy()      # (Ntr,)
    assign_te = assign_te_t.detach().cpu().numpy()      # (Nte,)
    Vs_np = Vs_t.detach().cpu().numpy()                 # (K,J,d)
    vs_np = vs_t.detach().cpu().numpy()                 # (K,d)

    # per-cluster GLOBAL indices (so file saving can reuse original paths)
    train_idxs_per_k: Dict[int, np.ndarray] = {k: np.array([], dtype=np.int64) for k in range(K)}
    test_idxs_per_k:  Dict[int, np.ndarray] = {k: np.array([], dtype=np.int64) for k in range(K)}

    for k in range(K):
        tr_loc = np.where(assign_tr == k)[0]                     # local to TRAIN slice
        te_loc = np.where(assign_te == k)[0]                     # local to TEST slice
        train_idxs_per_k[k] = train_idx_global[tr_loc].astype(np.int64)  # GLOBAL indices
        test_idxs_per_k[k]  = test_idx_global[te_loc].astype(np.int64)   # GLOBAL indices

    # ----- Save EM artifacts at channel root -----
    np.save(ch_dir / "assign_train.npy", assign_tr)              # local train assignments
    np.save(ch_dir / "assign_test.npy",  assign_te)              # local test assignments
    np.save(ch_dir / "em_Vs.npy", Vs_np)                          # (K,J,d) trained on TRAIN
    np.save(ch_dir / "em_vs.npy", vs_np)                          # (K,d)
    with open(ch_dir / "em_shapes.json", "w") as f:
        json.dump({"Vs": list(Vs_np.shape), "vs": list(vs_np.shape)}, f, indent=2)

    # ----- PCA per cluster (TRAIN only) for THIS channel -----
    means = torch.empty((K, d), device=device, dtype=torch.float32)
    bases = torch.empty((K, J, d), device=device, dtype=torch.float32)
    for k in range(K):
        tr_global = train_idxs_per_k[k]
        if tr_global.size == 0:
            # if no train members in this cluster, borrow a small random subset from TRAIN
            rng = np.random.default_rng(1)
            fallback = rng.choice(P_tr.shape[0], size=min(max(J+2, 32), P_tr.shape[0]), replace=False)
            Xk = P_tr.index_select(0, torch.as_tensor(fallback, device=device))
        else:
            # slice TRAIN by this cluster's members (convert to local train indices set)
            # Efficient: build a mask over TRAIN
            # But simple and fine: gather by positions corresponding to tr_global in train_idx_global
            # Build a map: global->local for train
            if k == 0:
                global_to_local_train = {int(g): i for i, g in enumerate(train_idx_global.tolist())}
            loc = [global_to_local_train[int(g)] for g in tr_global.tolist()]
            Xk = P_tr.index_select(0, torch.as_tensor(loc, device=device, dtype=torch.long))

        means[k] = Xk.mean(dim=0)
        comps = pca_topJ_torch(Xk, J)
        bases[k] = comps

    np.save(ch_dir / "pca_means.npy",  means.detach().cpu().numpy())  # (K,d)
    np.save(ch_dir / "pca_bases.npy",  bases.detach().cpu().numpy())  # (K,J,d)

    # ----- Save per-cluster images (under channel dir) -----
    save_per_channel_cluster_images(img_paths, train_idxs_per_k, test_idxs_per_k, ch_dir, img_size)

    # ----- Save per-cluster artifacts + vis + metadata -----
    save_channel_clusters(
        ch_dir=ch_dir,
        train_idxs_per_k=train_idxs_per_k,
        test_idxs_per_k=test_idxs_per_k,
        Vs_np=Vs_np,
        vs_np=vs_np,
        means=means,
        bases=bases,
        img_size=img_size
    )

    # Channel-level meta
    sizes_tr = [int(train_idxs_per_k[k].size) for k in range(K)]
    sizes_te = [int(test_idxs_per_k[k].size) for k in range(K)]
    with open(ch_dir / "meta.json", "w") as f:
        json.dump({
            "channel": name,
            "K": int(K),
            "J": int(J),
            "N_train": int(train_idx_global.size),
            "N_test": int(test_idx_global.size),
            "img_size": list(img_size),
            "cluster_sizes_train": sizes_tr,
            "cluster_sizes_test": sizes_te,
            "files": [
                "assign_train.npy", "assign_test.npy",
                "em_Vs.npy", "em_vs.npy",
                "pca_means.npy", "pca_bases.npy",
                "arrs.npy", "em_shapes.json"
            ],
            "dirs": {
                "clusters": "clusters/cluster_xx/{train_imgs,test_imgs,vis,...}"
            }
        }, f, indent=2)

    print(f"[{name}] done | train per-cluster sizes={sizes_tr} | test per-cluster sizes={sizes_te}")
    return {
        "sizes_train": sizes_tr,
        "sizes_test": sizes_te
    }


# --------------------------- main ---------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Parse image set with per-channel EM-like (K,J)-flats on GPU using a GLOBAL train/test split")
    ap.add_argument("-s", "--source", required=True, type=Path, help="Folder with *.png")
    ap.add_argument("-K", "--K_clusters", type=int, default=6, help="number of clusters")
    ap.add_argument("-J", "--J_dim", type=int, default=50, help="flat dimension per cluster")
    ap.add_argument("--steps", type=int, default=20, help="EM steps")
    ap.add_argument("--inits", type=int, default=10, help="random inits per EM")
    ap.add_argument("--img_size", type=int, nargs=2, default=[512, 512], metavar=('width','height'))
    ap.add_argument("--test_frac", type=float, default=0.1, help="GLOBAL fraction to reserve for test")
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
        img_paths = [all_paths[i] for i in sorted(sel.tolist())]  # stable order
    else:
        img_paths = all_paths

    Y, Cb, Cr, img_paths = load_ycbcr_from_paths(img_paths, (W, H))
    N = Y.shape[0]
    print(f"done in {time.time()-t0:.2f}s | picked N={N}/{len(all_paths)} | vec dim d={Y.shape[1]}")

    # == [2/5] GLOBAL split (identical across channels) ==
    print("== [2/5] Global TRAIN/TEST split ==")
    rng = np.random.default_rng(args.sample_seed)
    idxs = np.arange(N, dtype=np.int64)
    rng.shuffle(idxs)
    n_test = max(1, int(round(args.test_frac * N))) if N > 1 else 1
    n_test = min(n_test, N)
    test_idx  = np.sort(idxs[:n_test])
    train_idx = np.sort(idxs[n_test:])
    print(f"Global split: TRAIN={train_idx.size} | TEST={test_idx.size}")

    # persist global split for reproducibility
    np.save(outdir / "train_indices.npy", train_idx)
    np.save(outdir / "test_indices.npy",  test_idx)

    # save global copies once (no cluster prefix, consistent names)
    print("== [3/5] Writing global all_train/all_test copies (one time) ==")
    save_global_split_images(img_paths, train_idx, test_idx, outdir, (W, H))

    # == [4/5] EM-like clustering per channel (TRAIN ONLY) + assignments for TEST ==
    print("== [4/5] EM-like clustering per channel (GPU, TRAIN only) ==")
    ch_summary = {}
    ch_summary["Y"]  = run_em_for_channel("Y",  Y,  img_paths, outdir,
                                          K=args.K_clusters, J=args.J_dim,
                                          steps=args.steps, inits=args.inits,
                                          img_size=(W, H),
                                          train_idx_global=train_idx, test_idx_global=test_idx)
    ch_summary["Cb"] = run_em_for_channel("Cb", Cb, img_paths, outdir,
                                          K=args.K_clusters, J=args.J_dim,
                                          steps=args.steps, inits=args.inits,
                                          img_size=(W, H),
                                          train_idx_global=train_idx, test_idx_global=test_idx)
    ch_summary["Cr"] = run_em_for_channel("Cr", Cr, img_paths, outdir,
                                          K=args.K_clusters, J=args.J_dim,
                                          steps=args.steps, inits=args.inits,
                                          img_size=(W, H),
                                          train_idx_global=train_idx, test_idx_global=test_idx)

    # == [5/5] Top-level meta ==
    with open(outdir / "meta.json", "w") as f:
        json.dump({
            "img_size": [W, H],
            "K": int(args.K_clusters),
            "J": int(args.J_dim),
            "N": int(N),
            "N_train": int(train_idx.size),
            "N_test": int(test_idx.size),
            "channels": ["Y", "Cb", "Cr"],
            "channel_dirs": {"Y": "Y/", "Cb": "Cb/", "Cr": "Cr/"},
            "test_frac_global": float(args.test_frac),
            "files": ["train_indices.npy", "test_indices.npy"],
            "notes": "EM trained per channel on TRAIN only; TEST assigned post-hoc. Global all_train/all_test saved once with index-only filenames."
        }, f, indent=2)

    print(f"== DONE. Per-channel artifacts in: {outdir} ==")