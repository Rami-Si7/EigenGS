# projective_clustering_gpu.py
# GPU EM-like (K,J)-projective clustering WITHOUT null-space SVD
# Uses distance via projector with Q = X^T (columns orthonormal), plus batched E-steps.

from __future__ import annotations
import time
import numpy as np
import torch
from torch import Tensor

# -------- objective (same shape as your original lp loss) --------
LAMBDA = 1.0
Z = 2.0
NUM_INIT_FOR_EM = 5
STEPS = 20

def _loss_lp(x: Tensor, Z: float = Z) -> Tensor:
    # default: |x|^Z / Z  (Z=2 ⇒ 0.5*x^2)
    if Z == 2.0:
        return 0.5 * (x ** 2)
    return torch.pow(torch.clamp(x, min=0.0), Z) / Z

def _log(msg: str) -> None:
    print(msg, flush=True)

# -------- subspace fit (weighted mean + top-J right singular vectors) --------
@torch.no_grad()
def _compute_suboptimal_subspace(Psub: Tensor, wsub: Tensor, J: int):
    """
    Psub: (n_c, d), wsub: (n_c,)
    returns:
      X: (J, d)   row-orthonormal basis from SVD (take top-J rows of Vh)
      v: (d,)     weighted mean
    """
    ws = torch.clamp(wsub.sum(), min=1e-12)
    v = (Psub * wsub.unsqueeze(1)).sum(dim=0) / ws                 # (d,)
    Xc = Psub - v
    # thin SVD on (n_c, d)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)           # Vh: (min(n_c,d), d)
    X = Vh[:J, :]                                                  # (J, d) rows orthonormal
    return X.contiguous(), v.contiguous()

# -------- distances via projector with Q = X^T (columns orthonormal) --------
@torch.no_grad()
def _dist_to_flat_with_Q(Z: Tensor, Q: Tensor) -> Tensor:
    """
    Z: (B, d)   centered points (x - v)
    Q: (d, J)   column-orthonormal (Q^T Q = I)
    returns: (B,) distances
    """
    z2 = (Z * Z).sum(dim=1)           # ||z||^2
    coeffs = Z @ Q                    # (B, J) = Q^T z for each row
    proj2 = (coeffs * coeffs).sum(dim=1)
    d2 = torch.clamp(z2 - proj2, min=0.0)
    return torch.sqrt(d2)

# -------- batched E-step: get costs & assignments without building NxK matrix --------
@torch.no_grad()
def _batched_cost_and_assign(P: Tensor, w: Tensor,
                             Vs: Tensor, vs: Tensor,
                             batch: int = 32768):
    """
    P:  (N, d)
    w:  (N,)
    Vs: (K, J, d)   row-basis
    vs: (K, d)
    returns: total_cost(float), per_point_costs(N,), assignments(N,)
    """
    device = P.device
    N, d = P.shape
    K, J, d_ = Vs.shape
    assert d_ == d

    # Precompute Q_k = X_k^T once for this E-step
    Qs = [Vs[k].transpose(0, 1).contiguous() for k in range(K)]  # each (d, J)

    total_cost = 0.0
    per_point = torch.empty(N, device=device, dtype=P.dtype)
    assign = torch.empty(N, device=device, dtype=torch.long)

    for s in range(0, N, batch):
        e = min(N, s + batch)
        Zchunk = P[s:e]  # (B, d)
        wchunk = w[s:e]  # (B,)

        # initialize with cluster 0
        Z0 = Zchunk - vs[0].unsqueeze(0)
        d0 = _dist_to_flat_with_Q(Z0, Qs[0])
        best_cost = _loss_lp(d0) * wchunk
        best_idx  = torch.zeros(e - s, dtype=torch.long, device=device)

        # compare with remaining clusters
        for k in range(1, K):
            Zk = Zchunk - vs[k].unsqueeze(0)
            dk = _dist_to_flat_with_Q(Zk, Qs[k])
            ck = _loss_lp(dk) * wchunk
            better = ck < best_cost
            best_cost = torch.where(better, ck, best_cost)
            best_idx  = torch.where(better, torch.full_like(best_idx, k), best_idx)

        per_point[s:e] = best_cost
        assign[s:e]    = best_idx
        total_cost    += float(best_cost.sum().item())

    return total_cost, per_point, assign

# ----------------------- main EM -----------------------
@torch.no_grad()
def EMLikeAlg(
    P_np: np.ndarray,
    w_np: np.ndarray,
    j: int,
    k: int,
    steps: int = STEPS,
    inits: int = NUM_INIT_FOR_EM,
    batch: int = 32768,         # E-step batch size (tune to your GPU RAM)
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns:
      Vs_best: (k, j, d)  row-basis per flat
      vs_best: (k, d)     translation per flat
      elapsed_sec: float
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P = torch.as_tensor(P_np, dtype=torch.float32, device=device).contiguous()  # (N, d)
    w = torch.as_tensor(w_np, dtype=torch.float32, device=device).contiguous()  # (N,)
    N, d = P.shape

    best_vs = None
    best_Vs = None
    best_cost = float("inf")
    t0_all = time.time()

    g = torch.Generator(device=device)
    g.manual_seed(0)

    for init in range(inits):
        # ---- init: random partition into k groups (≈ balanced) ----
        perm = torch.randperm(N, generator=g, device=device).cpu().numpy()
        splits = np.array_split(perm, k)

        vs = torch.empty((k, d), device=device, dtype=P.dtype)
        Vs = torch.empty((k, j, d), device=device, dtype=P.dtype)

        init_sizes = []
        for i in range(k):
            idxs_np = splits[i]
            init_sizes.append(int(len(idxs_np)))
            if len(idxs_np) < max(j + 2, 8):  # guard tiny group
                idxs_np = np.random.default_rng(1234 + i).choice(N, size=max(j + 2, 32), replace=False)
            idxs = torch.as_tensor(idxs_np, device=device)
            X, v = _compute_suboptimal_subspace(P.index_select(0, idxs), w.index_select(0, idxs), j)
            Vs[i] = X
            vs[i] = v

        _log(f"[init {init+1}/{inits}] initial group sizes: {init_sizes}")

        prev_cost = None
        for it in range(1, steps + 1):
            # ---- E-step (batched) ----
            step_cost, per_pt, cluster_idx = _batched_cost_and_assign(P, w, Vs, vs, batch=batch)
            sizes = torch.bincount(cluster_idx, minlength=k).tolist()
            delta = (step_cost - prev_cost) if prev_cost is not None else float('nan')
            prev_cost = step_cost
            _log(f"  [init {init+1}/{inits}] step {it:02d}/{steps} | sizes={sizes} | cost={step_cost:.6e} | Δ={delta:.6e}")

            # ---- M-step: refit each occupied cluster ----
            for c in range(k):
                mask = (cluster_idx == c)
                if mask.any():
                    sub = P[mask]
                    wsub = w[mask]
                    X, v = _compute_suboptimal_subspace(sub, wsub, j)
                    Vs[c] = X
                    vs[c] = v
                else:
                    # reseed an empty cluster with a random subset
                    idxs = torch.randint(0, N, (max(j + 2, 32),), device=device, generator=g)
                    X, v = _compute_suboptimal_subspace(P.index_select(0, idxs), w.index_select(0, idxs), j)
                    Vs[c] = X
                    vs[c] = v

        # ---- final cost for this init ----
        final_cost, _, _ = _batched_cost_and_assign(P, w, Vs, vs, batch=batch)
        if final_cost < best_cost:
            best_cost = final_cost
            best_Vs = Vs.detach().cpu().numpy()
            best_vs = vs.detach().cpu().numpy()
            _log(f"[init {init+1}/{inits}] new best cost = {best_cost:.6e}")
        else:
            _log(f"[init {init+1}/{inits}] final cost = {final_cost:.6e}")

    elapsed = time.time() - t0_all
    return best_Vs, best_vs, elapsed

# -------- helper to get assignments (unchanged API) --------
@torch.no_grad()
def assign_points(P_np: np.ndarray, w_np: np.ndarray, Vs_np: np.ndarray, vs_np: np.ndarray, batch: int = 32768):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P  = torch.as_tensor(P_np, dtype=torch.float32, device=device).contiguous()
    w  = torch.as_tensor(w_np, dtype=torch.float32, device=device).contiguous()
    Vs = torch.as_tensor(Vs_np, dtype=torch.float32, device=device).contiguous()  # (K,J,d)
    vs = torch.as_tensor(vs_np, dtype=torch.float32, device=device).contiguous()  # (K,d)
    _, _, idxs = _batched_cost_and_assign(P, w, Vs, vs, batch=batch)
    return idxs.detach().cpu().numpy()