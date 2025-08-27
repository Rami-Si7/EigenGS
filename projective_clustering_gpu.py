#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU EM-like (K,J)-projective clustering with robust losses + rich prints.

Usage (as a library):
    from projective_clustering_gpu import em_projective
    U, v, assign, info = em_projective(P, K=6, J=50, steps=20, inits=5, device='cuda')

Usage (as a script, quick demo with synthetic data):
    python projective_clustering_gpu.py --demo
"""
from __future__ import annotations
import math, time, argparse
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm


# -------------------------- utils / logging --------------------------

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _to_device(x, device: torch.device) -> Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    return x.to(device)


# -------------------------- robust losses ----------------------------

def _rho_lp(x: Tensor, Z: float) -> Tensor:
    # x >= 0, rho(x) = |x|^Z / Z
    if Z == 2:
        return 0.5 * (x ** 2)
    # numerical safety
    return torch.pow(torch.clamp(x, min=0), Z) / Z

def _rho_huber(x: Tensor, lam: float) -> Tensor:
    # 0.5*x^2    if |x| <= lam
    # lam*(|x| - 0.5*lam) otherwise
    ax = torch.abs(x)
    quad = 0.5 * (x ** 2)
    lin = lam * (ax - 0.5 * lam)
    return torch.where(ax <= lam, quad, lin)

def _rho_cauchy(x: Tensor, lam: float) -> Tensor:
    # (lam^2/2)*log(1 + (x/lam)^2)
    return 0.5 * (lam ** 2) * torch.log1p((x / lam) ** 2)

def _rho_geman_mcclure(x: Tensor) -> Tensor:
    # x^2 / (2*(1 + x^2))
    x2 = x ** 2
    return x2 / (2.0 * (1.0 + x2))

def _rho_welsch(x: Tensor, lam: float) -> Tensor:
    # (lam^2/2)*(1 - exp(-(x/lam)^2))
    return 0.5 * (lam ** 2) * (1.0 - torch.exp(-(x / lam) ** 2))

def _rho_tukey(x: Tensor, lam: float) -> Tensor:
    # (lam^2/6) * (1 - (1 - (x/lam)^2)^3)  if |x|<=lam  else lam^2/6
    ax = torch.abs(x)
    out = torch.full_like(x, (lam ** 2) / 6.0)
    mask = ax <= lam
    z = x[mask] / lam
    out[mask] = (lam ** 2 / 6.0) * (1.0 - (1.0 - z ** 2) ** 3)
    return out

def _get_rho(name: str, lam: float, Z: float):
    name = name.lower()
    if name in ("l2","lp","p"):
        return lambda x: _rho_lp(x, Z)
    if name == "huber":
        return lambda x: _rho_huber(x, lam)
    if name == "cauchy":
        return lambda x: _rho_cauchy(x, lam)
    if name in ("geman_mcclure", "geman-mcclure", "gm"):
        return _rho_geman_mcclure
    if name == "welsch":
        return lambda x: _rho_welsch(x, lam)
    if name == "tukey":
        return lambda x: _rho_tukey(x, lam)
    raise ValueError(f"Unknown robust loss: {name}")


# -------------------- subspace & distance computations --------------------

@torch.no_grad()
def compute_subspace(X: Tensor, J: int, weights: Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
    """
    Given X (n_c, d), return (U, v):
      v: (d,) mean (weighted if provided)
      U: (d, J) column-orthonormal basis, top-J right singular vectors of (X - v)
    """
    if weights is None:
        v = X.mean(dim=0)
        Xc = X - v
    else:
        w = weights.view(-1, 1)
        wsum = torch.clamp(w.sum(), min=1e-8)
        v = (X * w).sum(dim=0) / wsum
        Xc = (X - v) * torch.sqrt(w)  # standard trick for weighted SVD

    # thin SVD
    # Xc shape is (n_c, d) or (n_c, d) after sqrt(w)
    U_svd, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    V = Vh.transpose(-2, -1)  # (d, min(n_c,d))
    U = V[:, :J].contiguous() # (d, J)
    return U, v


@torch.no_grad()
def distances_cost_matrix(
    P: Tensor, U: Tensor, v: Tensor,
    w: Optional[Tensor],
    rho
) -> Tensor:
    """
    Compute robust costs to each flat for all points.
      P: (N, d)
      U: (K, d, J)  column-orthonormal
      v: (K, d)
      w: (N,) or None
      rho: function mapping distances -> cost

    Returns:
      C: (N, K) with entries w_i * rho( || (I - U U^T)(x_i - v_k) || )
    """
    N, d = P.shape
    K, d2, J = U.shape
    assert d2 == d

    # We'll do it cluster-by-cluster (K is usually small), vectorizing over N
    C = torch.empty((N, K), device=P.device, dtype=P.dtype)
    for k in range(K):
        Uk = U[k]                 # (d, J)
        vk = v[k]                 # (d,)
        Z  = P - vk.unsqueeze(0)  # (N, d)
        # projection onto the sheet: Uk Uk^T Z
        proj = (Z @ Uk) @ Uk.transpose(0,1)    # (N, d)
        R = Z - proj                            # residuals (N, d)
        dist = torch.linalg.vector_norm(R, dim=1)  # (N,)
        cost = rho(dist)                        # robust per-point
        if w is not None:
            cost = cost * w
        C[:, k] = cost
    return C


# ------------------------ balanced assignment (optional) ------------------------

@torch.no_grad()
def balanced_assign(costs: Tensor, K: int) -> Tensor:
    """
    Greedy capacity-balanced assignment:
      costs: (N, K)   lower is better
    Returns:
      assign: (N,) int64 in [0..K-1] with sizes ≈ ceil(N/K)
    """
    device = costs.device
    N = costs.size(0)
    cap = int(math.ceil(N / K))
    order = torch.argsort(costs, dim=1)         # best->worst cluster for each point
    assign = torch.full((N,), -1, dtype=torch.long, device=device)
    counts = torch.zeros(K, dtype=torch.int32, device=device)

    remaining = torch.arange(N, device=device)
    for r in range(K):
        if remaining.numel() == 0:
            break
        pref = order[remaining, r]              # (R,)
        # fill each cluster up to capacity
        for k in range(K):
            mask = (pref == k)
            idxs = remaining[mask]
            if idxs.numel() == 0: continue
            room = cap - int(counts[k].item())
            if room <= 0: continue
            take = idxs[:room]
            assign[take] = k
            counts[k] += take.numel()
        remaining = torch.nonzero(assign.eq(-1), as_tuple=False).squeeze(-1)

    # spill leftovers anywhere with room (or least-filled)
    for i in remaining.tolist():
        # try by increasing cost preference
        for k in order[i].tolist():
            if counts[k] < cap:
                assign[i] = k
                counts[k] += 1
                break
        if assign[i] == -1:
            k = torch.argmin(counts).item()
            assign[i] = k
            counts[k] += 1
    return assign


# ----------------------------- main EM routine -----------------------------

@torch.no_grad()
def em_projective(
    P_np: np.ndarray | Tensor,
    K: int,
    J: int,
    steps: int = 20,
    inits: int = 5,
    device: str = "cuda",
    robust: str = "l2",         # 'l2'/'lp'/'huber'/'welsch'/'cauchy'/'geman_mcclure'/'tukey'
    lam: float = 1.0,           # lambda for robust losses
    Z: float = 2.0,             # exponent for lp
    weights: Optional[np.ndarray | Tensor] = None,
    balanced: bool = False,     # capacity-balanced E-step
    seed: int = 0,
) -> Tuple[Tensor, Tensor, np.ndarray, Dict]:
    """
    Returns:
      U_best: (K, d, J)    column-orthonormal
      v_best: (K, d)
      assign_best: (N,) numpy int64
      info: dict with history
    """
    torch_device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    P = _to_device(P_np, torch_device).float().contiguous()    # (N, d)
    N, d = P.shape
    w = None if weights is None else _to_device(weights, torch_device).float().view(-1)

    rho = _get_rho(robust, lam, Z)
    g = torch.Generator(device=torch_device)
    g.manual_seed(seed)

    best_cost = float("inf")
    bestU = bestv = bestAssign = None
    history = []

    log(f"=== EM (K={K}, J={J}, steps={steps}, inits={inits}, robust={robust}, balanced={balanced}) on device={torch_device.type} ===")
    t_all = time.time()

    for init in range(inits):
        t0 = time.time()
        log(f"[init {init+1}/{inits}] seeding...")

        # ---- init by random partition → consistent (v_k from same group used to get U_k) ----
        perm = torch.randperm(N, generator=g, device=torch_device).cpu().numpy()
        splits = np.array_split(perm, K)

        U = torch.empty((K, d, J), device=torch_device)
        v = torch.empty((K, d), device=torch_device)
        sizes0 = []
        for k in range(K):
            idxs = splits[k]
            if len(idxs) < J + 2:
                # guard tiny group
                idxs = np.random.default_rng(seed + k).choice(N, size=max(J+2, 32), replace=False)
            Xk = P.index_select(0, torch.as_tensor(idxs, device=torch_device))
            wk = None
            if w is not None:
                wk = w.index_select(0, torch.as_tensor(idxs, device=torch_device))
            Uk, vk = compute_subspace(Xk, J, wk)
            U[k] = Uk
            v[k] = vk
            sizes0.append(len(idxs))
        log(f"[init {init+1}/{inits}] init groups: sizes={sizes0} in {time.time()-t0:.2f}s")
        prev_cost = None

        # ---- EM loop ----
        for it in range(1, steps+1):
            t_it = time.time()
            # E-step: compute robust cost matrix & assign
            C = distances_cost_matrix(P, U, v, w, rho)  # (N,K)
            if balanced:
                assign = balanced_assign(C, K)
            else:
                assign = torch.argmin(C, dim=1)  # (N,)

            cost = C.gather(1, assign.view(-1,1)).sum().item()
            sizes = torch.bincount(assign, minlength=K).tolist()

            # M-step: refit each sheet
            newU = torch.empty_like(U)
            newv = torch.empty_like(v)
            for k in range(K):
                idx = torch.nonzero(assign == k, as_tuple=False).squeeze(-1)
                if idx.numel() < J + 2:
                    # reseed this cluster with random subset
                    idx = torch.randint(0, N, (max(J+2, 32),), device=torch_device, generator=g)
                Xk = P.index_select(0, idx)
                wk = None
                if w is not None:
                    wk = w.index_select(0, idx)
                Uk, vk = compute_subspace(Xk, J, wk)
                newU[k] = Uk
                newv[k] = vk
            U, v = newU, newv

            delta = (cost - prev_cost) if prev_cost is not None else float('nan')
            prev_cost = cost
            log(f"  [init {init+1}/{inits}] EM {it:02d}/{steps} | cost={cost:.4e} delta={delta:.4e} sizes={sizes} ({time.time()-t_it:.2f}s)")
            history.append(dict(init=init, it=it, cost=cost, sizes=sizes))

        # final cost for this init
        C = distances_cost_matrix(P, U, v, w, rho)
        assign = torch.argmin(C, dim=1)
        final_cost = C.gather(1, assign.view(-1,1)).sum().item()
        if final_cost < best_cost:
            best_cost = final_cost
            bestU = U.detach().cpu()
            bestv = v.detach().cpu()
            bestAssign = assign.detach().cpu().numpy()
            log(f"[init {init+1}/{inits}] new best: cost={best_cost:.4e}")
        else:
            log(f"[init {init+1}/{inits}] final cost={final_cost:.4e} (no improvement)")

    log(f"=== done in {time.time()-t_all:.2f}s | best cost={best_cost:.4e} ===")
    info = dict(best_cost=best_cost, history=history, device=str(torch_device))
    return bestU, bestv, bestAssign, info


# ----------------------------- CLI demo --------------------------------

def _demo(args):
    # Make a synthetic 2D example: 3 lines (J=1) + noise
    torch.manual_seed(0)
    N = 3000
    noise = 0.05

    # three 1D flats in 2D at different orientations
    t1 = torch.linspace(-1, 1, N//3).unsqueeze(1)
    t2 = torch.linspace(-1, 1, N//3).unsqueeze(1)
    t3 = torch.linspace(-1, 1, N - 2*(N//3)).unsqueeze(1)

    A1 = torch.tensor([[1.0, 0.0]])   # along x
    A2 = torch.tensor([[0.7, 0.7]])   # diagonal
    A3 = torch.tensor([[0.0, 1.0]])   # along y

    v1 = torch.tensor([0.0, 0.0])
    v2 = torch.tensor([0.5, -0.2])
    v3 = torch.tensor([-0.4, 0.5])

    X1 = v1 + t1 @ A1 + noise*torch.randn(t1.size(0), 2)
    X2 = v2 + t2 @ A2 + noise*torch.randn(t2.size(0), 2)
    X3 = v3 + t3 @ A3 + noise*torch.randn(t3.size(0), 2)
    X  = torch.cat([X1, X2, X3], dim=0).numpy()

    U, v, assign, info = em_projective(
        X, K=3, J=1, steps=args.steps, inits=args.inits,
        device=args.device, robust=args.robust, lam=args.lam, Z=args.Z,
        balanced=args.balanced
    )
    log(f"Cluster sizes: {np.bincount(assign)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GPU EM-like (K,J)-projective clustering")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    ap.add_argument("-K", type=int, default=6, help="number of flats")
    ap.add_argument("-J", type=int, default=50, help="flat dimension")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--inits", type=int, default=5)
    ap.add_argument("--robust", type=str, default="l2",
                    choices=["l2","lp","huber","welsch","cauchy","geman_mcclure","tukey"])
    ap.add_argument("--lam", type=float, default=1.0, help="lambda for robust losses")
    ap.add_argument("--Z", type=float, default=2.0, help="p for lp loss")
    ap.add_argument("--balanced", action="store_true", help="use capacity-balanced assignment")
    ap.add_argument("--demo", action="store_true", help="run a small synthetic demo")
    args = ap.parse_args()

    if args.demo:
        _demo(args)
    else:
        log("This file is meant to be imported and used on your own P matrix. Use --demo for a quick test.")
