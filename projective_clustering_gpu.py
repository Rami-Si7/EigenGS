# projective_clustering_gpu.py
# GPU port of your original EM-like (K,J)-projective clustering
# with detailed logging of cluster sizes per init & per step.

from __future__ import annotations
import time, copy
import numpy as np
import torch
from torch import Tensor

# -------- original-style constants / loss --------
LAMBDA = 1.0
Z = 2.0
NUM_INIT_FOR_EM = 5
STEPS = 20

def _loss_lp(x: Tensor, Z: float = Z) -> Tensor:
    # OBJECTIVE_LOSS(dist); original default: |x|^Z / Z  (Z=2 ⇒ 0.5*x^2)
    if Z == 2.0:
        return 0.5 * (x ** 2)
    return torch.pow(torch.clamp(x, min=0.0), Z) / Z

def _log(msg: str) -> None:
    print(msg, flush=True)

# -------- linear algebra helpers (GPU-capable) --------
@torch.no_grad()
def _null_space(X: Tensor, rcond: float | None = None) -> Tensor:
    """
    Null space of X (J x d) via SVD on GPU.
    Returns N of shape (d, d-rank) with orthonormal columns s.t. X @ N = 0.
    """
    U, S, Vh = torch.linalg.svd(X, full_matrices=True)  # Vh: (d,d)
    if rcond is None:
        rcond = torch.finfo(S.dtype).eps * max(X.shape)
    tol = S.max() * rcond if S.numel() else torch.tensor(0.0, device=X.device, dtype=X.dtype)
    rank = int((S > tol).sum().item())
    V = Vh.transpose(0, 1)  # (d,d)
    return V[:, rank:]      # (d, d-rank)

@torch.no_grad()
def _distance_to_subspace(points: Tensor, X: Tensor, v: Tensor) -> Tensor:
    """
    points: (N,d) or (d,), X: (J,d) row-basis, v: (d,)
    Returns distances (N,)
    """
    if points.ndim == 1:
        points = points.unsqueeze(0)
    Nmat = _null_space(X)                    # (d, d-J)
    diffs = points - v.unsqueeze(0)          # (N,d)
    proj  = diffs @ Nmat                     # (N, d-J)
    return torch.linalg.vector_norm(proj, dim=1)  # (N,)

@torch.no_grad()
def _compute_cost(P: Tensor, w: Tensor, X: Tensor, v: Tensor, show_indices: bool = False):
    """
    Mirrors your numpy version:
      - If X.ndim==2: one flat → returns (sum_cost, per_point_cost)
      - If X.ndim==3: K flats → min across K; if show_indices, also argmin
    Costs are loss(dist) * w.
    """
    if X.ndim == 2:
        dists = _distance_to_subspace(P, X, v)  # (N,)
        per_pt = _loss_lp(dists) * w
        return per_pt.sum().item(), per_pt

    # X: (K,J,d), v: (K,d)
    K = X.shape[0]
    N = P.shape[0]
    temp = torch.empty((N, K), device=P.device, dtype=P.dtype)
    for k in range(K):
        dists = _distance_to_subspace(P, X[k], v[k])
        temp[:, k] = _loss_lp(dists) * w
    vals, idxs = torch.min(temp, dim=1)  # (N,), (N,)
    if show_indices:
        return vals.sum().item(), vals, idxs
    return vals.sum().item(), vals

@torch.no_grad()
def _compute_suboptimal_subspace(Psub: Tensor, wsub: Tensor, J: int):
    """
    Weighted mean v and top-J right singular vectors of (Psub - v).
    Return X as (J,d) row-basis and v as (d,)
    """
    t0 = time.time()
    ws = torch.clamp(wsub.sum(), min=1e-12)
    v = (Psub * wsub.unsqueeze(1)).sum(dim=0) / ws    # (d,)
    Xc = Psub - v
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    X = Vh[:J, :]                                      # (J,d)
    return X.contiguous(), v.contiguous(), time.time() - t0

# ----------------------- main EM (with prints) -----------------------
@torch.no_grad()
def EMLikeAlg(
    P_np: np.ndarray,
    w_np: np.ndarray,
    j: int,
    k: int,
    steps: int = STEPS,
    inits: int = NUM_INIT_FOR_EM,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    GPU implementation of your original EM with detailed logging.
    Returns:
      Vs_best: (k, j, d)  row-basis per flat
      vs_best: (k, d)     translation per flat
      elapsed_sec: float
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P = torch.as_tensor(P_np, dtype=torch.float32, device=device).contiguous()  # (N,d)
    w = torch.as_tensor(w_np, dtype=torch.float32, device=device).contiguous()  # (N,)
    N, d = P.shape

    min_vs = None
    min_Vs = None
    optimal_cost = float("inf")
    t_start = time.time()

    rng = torch.Generator(device=device)
    rng.manual_seed(0)

    for init in range(inits):
        # ---- init: random split + subspace per split ----
        perm = torch.randperm(N, generator=rng, device=device).cpu().numpy()
        splits = np.array_split(perm, k)

        vs = P[torch.randint(0, N, (k,), device=device)]  # (k,d)
        Vs = torch.empty((k, j, d), device=device, dtype=P.dtype)

        sizes0 = []
        for i in range(k):
            idxs_np = splits[i]
            sizes0.append(int(len(idxs_np)))
            if len(idxs_np) == 0:
                idxs_np = np.random.default_rng(1234 + i).choice(N, size=max(j+2, 32), replace=False)
            idxs = torch.as_tensor(idxs_np, device=device)
            X, v, _ = _compute_suboptimal_subspace(P.index_select(0, idxs), w.index_select(0, idxs), j)
            Vs[i] = X
            vs[i] = v

        _log(f"[init {init+1}/{inits}] initial group sizes: {sizes0}")

        prev_cost = None
        for it in range(1, steps + 1):
            # ---- E-step: per-cluster per-point costs (after loss*weight) ----
            dists = torch.empty((N, k), device=device, dtype=P.dtype)
            for l in range(k):
                _, col = _compute_cost(P, w, Vs[l], vs[l])
                dists[:, l] = col

            cluster_indices = torch.argmin(dists, dim=1)  # (N,)
            sizes = torch.bincount(cluster_indices, minlength=k).tolist()
            step_cost = dists.gather(1, cluster_indices.view(-1, 1)).sum().item()
            delta = (step_cost - prev_cost) if prev_cost is not None else float("nan")
            prev_cost = step_cost

            _log(f"  [init {init+1}/{inits}] step {it:02d}/{steps} | sizes={sizes} | cost={step_cost:.6e} | Δ={delta:.6e}")

            # ---- M-step: refit subspaces for non-empty clusters ----
            uniq = torch.unique(cluster_indices).tolist()
            for idx in uniq:
                idx = int(idx)
                mask = (cluster_indices == idx)
                sub = P[mask]
                wsub = w[mask]
                X, v, _ = _compute_suboptimal_subspace(sub, wsub, j)
                Vs[idx] = X
                vs[idx] = v

        # ---- keep best init by total cost ----
        final_cost, _ = _compute_cost(P, w, Vs, vs)
        if final_cost < optimal_cost:
            optimal_cost = final_cost
            min_Vs = Vs.detach().cpu().numpy()
            min_vs = vs.detach().cpu().numpy()
            _log(f"[init {init+1}/{inits}] new best cost = {optimal_cost:.6e}")
        else:
            _log(f"[init {init+1}/{inits}] final cost = {final_cost:.6e}")

    return min_Vs, min_vs, time.time() - t_start

# -------- helper to get assignments (unchanged API) --------
@torch.no_grad()
def assign_points(P_np: np.ndarray, w_np: np.ndarray, Vs_np: np.ndarray, vs_np: np.ndarray):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P  = torch.as_tensor(P_np, dtype=torch.float32, device=device).contiguous()
    w  = torch.as_tensor(w_np, dtype=torch.float32, device=device).contiguous()
    Vs = torch.as_tensor(Vs_np, dtype=torch.float32, device=device).contiguous()
    vs = torch.as_tensor(vs_np, dtype=torch.float32, device=device).contiguous()
    _, _, idxs = _compute_cost(P, w, Vs, vs, show_indices=True)
    return idxs.detach().cpu().numpy()
