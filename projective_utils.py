# projective_utils.py
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from skimage import color
from PIL import Image

Tensor = torch.Tensor

# ----------------------------- loaders -----------------------------

def load_top_level(projective_dir: Path):
    """
    Loads top-level projective artifacts:
      - projective_basis.pt: {means:(K,3,d), bases:(K,3,J,d), img_size, K, J, color_space, norm_mins, norm_maxs}
      - arrs.npy            : (K*J, 3, H, W)   normalized eigenimages used for Phase-A targets
      - meta.json           : global info
    """
    ckpt = torch.load(projective_dir / "projective_basis.pt", map_location="cpu", weights_only=False)
    means  = ckpt["means"].float()     # (K,3,d)
    bases  = ckpt["bases"].float()     # (K,3,J,d) row-basis per channel
    W, H   = ckpt["img_size"]
    K      = int(ckpt["K"])
    J      = int(ckpt["J"])
    mins   = torch.tensor(ckpt["norm_mins"], dtype=torch.float32)  # (3,)
    maxs   = torch.tensor(ckpt["norm_maxs"], dtype=torch.float32)  # (3,)

    arrs = np.load(projective_dir / "arrs.npy")  # (K*J, 3, H, W) normalized [0,1]
    arrs = torch.from_numpy(arrs).float()

    return means, bases, (W, H), K, J, mins, maxs, arrs


def load_cluster_arrs(projective_dir: Path, cluster_ids: List[int]) -> torch.Tensor:
    """
    Stack (J,3,H,W) arrays from selected clusters -> (len(cluster_ids)*J, 3, H, W)
    """
    arrs = []
    shape = None
    for k in cluster_ids:
        cdir = projective_dir / "clusters" / f"cluster_{k:02d}"
        print(cdir)
        A = np.load(cdir / "arrs.npy")  # (J,3,H,W)
        if shape is None:
            shape = A.shape
        arrs.append(torch.from_numpy(A).float())
    return torch.cat(arrs, dim=0)


def index_map_all(K: int, J: int) -> List[Tuple[int,int]]:
    """
    Global component index -> (cluster_id, local_j)
    """
    return [(k, j) for k in range(K) for j in range(J)]


def index_map_subset(cluster_ids: List[int], J: int) -> List[Tuple[int,int]]:
    """
    For subset training (per-cluster or multiple clusters)
    """
    out = []
    for k in cluster_ids:
        for j in range(J):
            out.append((k, j))
    return out

# ---------------------- YCbCr helpers ----------------------

def rgb_path_to_ycbcr(path: Path, size_wh: Tuple[int,int]) -> torch.Tensor:
    W, H = size_wh
    img = Image.open(path).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)
    ycbcr = (color.rgb2ycbcr(arr).astype(np.float32) / 255.0).transpose(2,0,1)  # (3,H,W) in [0,1]
    return torch.from_numpy(ycbcr).float()

# ---------------------- cluster selection ----------------------

@torch.no_grad()
def _proj_residual_sq(x: Tensor, mean: Tensor, Vrows: Tensor) -> Tensor:
    """
    Residual squared after projecting (x - mean) onto row-space Vrows (J,d).
    Assumes rows ~ orthonormal (from SVD). Shapes:
      x: (d,)
      mean: (d,)
      Vrows: (J,d)
    """
    r = x - mean                       # (d,)
    r_par = (r @ Vrows.T) @ Vrows      # (d,)
    diff = r - r_par
    return (diff * diff).sum()         # scalar


@torch.no_grad()
def choose_cluster_ycbcr(img_ycbcr: Tensor,
                         means: Tensor,   # (K,3,d)
                         bases: Tensor,   # (K,3,J,d)
                         weights=(1.0, 0.25, 0.25)) -> int:
    """
    Select cluster with minimal joint residual across channels.
    img_ycbcr: (3,H,W) in [0,1]
    returns k* (int)
    """
    K, _, J, d = bases.shape
    H, W = img_ycbcr.shape[1:]
    x = img_ycbcr.view(3, -1)                  # (3,d)
    aY, aCb, aCr = [float(w) for w in weights]

    best_k, best_val = -1, float("inf")
    for k in range(K):
        val  = aY  * _proj_residual_sq(x[0], means[k,0], bases[k,0])
        val += aCb * _proj_residual_sq(x[1], means[k,1], bases[k,1])
        val += aCr * _proj_residual_sq(x[2], means[k,2], bases[k,2])
        if float(val) < best_val:
            best_val = float(val)
            best_k = k
    return best_k

# ---------------------- codes + embedding ----------------------

@torch.no_grad()
def compute_codes(img_ycbcr: Tensor,
                  mean_k: Tensor,     # (3,d)
                  bases_k: Tensor     # (3,J,d)
                  ) -> Tensor:
    """
    Return per-channel PCA coefficients for the chosen cluster:
      codes: (3,J)
    """
    x = img_ycbcr.view(3, -1)                    # (3,d)
    codes = []
    for ch in range(3):
        r = x[ch] - mean_k[ch]                  # (d,)
        # Project to row-space (J,d): coeff = r @ Vrows^T
        c = r @ bases_k[ch].T                   # (J,)
        codes.append(c)
    return torch.stack(codes, dim=0)            # (3,J)


@torch.no_grad()
def embed_codes(codes_k: Tensor,    # (3,J)
                index_map: List[Tuple[int,int]],
                chosen_k: int) -> Tensor:
    """
    Embed per-cluster codes into the full (3, total_comps) vector with zeros elsewhere.
    """
    total = len(index_map)
    out = torch.zeros((3, total), dtype=codes_k.dtype, device=codes_k.device)
    for global_idx, (k, j) in enumerate(index_map):
        if k == chosen_k:
            out[:, global_idx] = codes_k[:, j]
    return out  # (3, total)


@torch.no_grad()
def codes_to_colors(model,
                    codes_rowmajor: Tensor,  # (3, num_comps)
                    norm_mins: Tensor,       # (3,)
                    norm_maxs: Tensor,       # (3,)
                    mean_for_image: Tensor   # (3,d)
                    ):
    """
    Populate model._colors from codes and set normalization/mean buffers.

    The model learned eigenimages in normalized space v01 = (v - min)/(max - min).
    To reconstruct Σ w_j v_j, we use:
       out = Σ w_j v01_j
       image = out * (max-min) + (Σ w_j)*min + mean
    """
    device = model.device
    codes_rowmajor = codes_rowmajor.to(device)           # (3, num_comps)

    # (num_comps, num_points, 3) -> (3, num_comps, num_points)
    feats = model.get_features.permute(2, 0, 1)
    # bmm: (3,1,num_comps) x (3,num_comps,num_points) -> (3,1,num_points)
    colors = torch.bmm(codes_rowmajor.unsqueeze(1), feats).squeeze(1)  # (3, num_points)
    model._colors.data.copy_(colors.transpose(0, 1))                    # (num_points,3)

    # normalization + mean
    model.scale_factor.data = (norm_maxs - norm_mins).to(device)       # (3,)
    # sum of codes per channel
    code_sums = codes_rowmajor.sum(dim=1)                               # (3,)
    model.shift_factor.data = code_sums * norm_mins.to(device)          # (3,)
    model.image_mean.data.copy_(mean_for_image.to(device))              # (d,)
