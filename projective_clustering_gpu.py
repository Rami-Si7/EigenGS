# import torch
# import copy
# import time

# LAMBDA = 1
# Z = 2
# NUM_INIT_FOR_EM = 10
# STEPS = 20

# def _lp(x):
#     return (torch.abs(x) ** Z) / Z

# def _huber(x):
#     ax = torch.abs(x)
#     return torch.where(ax <= LAMBDA, 0.5 * x * x, LAMBDA * (ax - 0.5 * LAMBDA))

# def _cauchy(x):
#     return (LAMBDA**2 / 2.0) * torch.log1p((x * x) / (LAMBDA**2))

# def _geman_mcclure(x):
#     return (x * x) / (2.0 * (1.0 + x * x))

# def _welsch(x):
#     return (LAMBDA**2 / 2.0) * (1.0 - torch.exp(-(x * x) / (LAMBDA**2)))

# def _tukey(x):
#     ax2 = (x * x) / (LAMBDA**2)
#     inside = 1.0 - ax2
#     val = (LAMBDA**2 / 6.0) * (1.0 - (inside ** 3))
#     return torch.where(torch.abs(x) <= LAMBDA, val, torch.full_like(x, LAMBDA**2 / 6.0))

# M_ESTIMATOR_FUNCS = {
#     'lp': _lp,
#     'huber': _huber,
#     'cauchy': _cauchy,
#     'geman_McClure': _geman_mcclure,
#     'welsch': _welsch,
#     'tukey': _tukey,
# }
# OBJECTIVE_LOSS = M_ESTIMATOR_FUNCS['lp']

# @torch.no_grad()
# def _null_space(X: torch.Tensor, rtol: float = 1e-9):
#     U, S, Vh = torch.linalg.svd(X, full_matrices=False)
#     if S.numel() == 0:
#         n = X.shape[1]
#         return torch.eye(n, device=X.device, dtype=X.dtype)
#     eps = torch.finfo(S.dtype).eps if X.dtype in (torch.float32, torch.float64) else 1e-7
#     tol = max(X.shape) * eps * torch.max(S)
#     rank = int((S >= tol).sum().item())
#     Ns = Vh[rank:].transpose(0, 1)
#     if Ns.numel() == 0:
#         return Ns
#     Q, _ = torch.linalg.qr(Ns, mode='reduced')
#     return Q

# @torch.no_grad()
# def computeDistanceToSubspace(point: torch.Tensor, X: torch.Tensor, v: torch.Tensor = None) -> torch.Tensor:
#     """
#     Distance to affine J-flat whose row-space is spanned by rows of X.
#     Assumes rows of X are (approximately) orthonormal (true for Vh[:J,:] from SVD).
#     """
#     if point.ndim == 1:
#         R = point if v is None else (point - v)       # (d,)
#         # project onto row-space of X: R_parallel = (R X^T) X
#         R_par = (R @ X.transpose(0, 1)) @ X           # (d,)
#         return torch.norm(R - R_par)
#     else:
#         R = point if v is None else (point - v.unsqueeze(0))  # (N, d)
#         R_par = (R @ X.transpose(0, 1)) @ X                   # (N, d)
#         return torch.norm(R - R_par, dim=1)


# @torch.no_grad()
# def computeCost(P: torch.Tensor, w: torch.Tensor, X: torch.Tensor, v: torch.Tensor = None, show_indices: bool = False):
#     global OBJECTIVE_LOSS
#     if X.ndim == 2:
#         dpp = OBJECTIVE_LOSS(computeDistanceToSubspace(P, X, v))
#         cost_per_point = w * dpp
#         if not show_indices:
#             return cost_per_point.sum(), cost_per_point
#         else:
#             return cost_per_point.sum(), cost_per_point, None
#     else:
#         k = X.shape[0]
#         N = P.shape[0]
#         temp = torch.empty((N, k), device=P.device, dtype=P.dtype)
#         for i in range(k):
#             di = OBJECTIVE_LOSS(computeDistanceToSubspace(P, X[i, :, :], v[i, :]))
#             temp[:, i] = w * di
#         cost_per_point, indices = torch.min(temp, dim=1)
#         if not show_indices:
#             return cost_per_point.sum(), cost_per_point
#         else:
#             return cost_per_point.sum(), cost_per_point, indices

# # @torch.no_grad()
# # def computeSuboptimalSubspace(P: torch.Tensor, w: torch.Tensor, J: int):
# #     start = time.time()
# #     v = (P * w.unsqueeze(1)).sum(dim=0) / w.sum()
# #     Y = P - v.unsqueeze(0)
# #     U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
# #     V_rows = Vh[:J, :]
# #     return V_rows, v, (time.time() - start)
# @torch.no_grad()
# def computeSuboptimalSubspace(P: torch.Tensor, w: torch.Tensor, J: int):
#     start = time.time()
#     v = (P * w.unsqueeze(1)).sum(dim=0) / w.sum()
#     Y = P - v.unsqueeze(0)
#     # economy SVD
#     U, S, Vh = torch.linalg.svd(Y, full_matrices=False)  # Vh: (min(n,d), d)
#     Juse = min(J, Vh.shape[0])
#     V_rows = Vh[:Juse, :]
#     if Juse < J:
#         pad = torch.zeros((J - Juse, V_rows.shape[1]), device=V_rows.device, dtype=V_rows.dtype)
#         V_rows = torch.cat([V_rows, pad], dim=0)
#     return V_rows, v, (time.time() - start)

# @torch.no_grad()
# def EMLikeAlg(P: torch.Tensor, w: torch.Tensor, j: int, k: int, steps: int):
#     """
#     Torch port of the original EM-like (K,J)-projective clustering.
#     Only deviation from the original: if a cluster has n_k < j points,
#     set J_use = min(j, n_k) for its subspace fit.
#     """
#     start_time = time.time()
#     n, d = P.shape
#     _ = torch.linalg.norm(P, dim=1).max()  # parity with original

#     min_vs = None
#     min_Vs = None
#     optimal_cost = float('inf')

#     for _ in range(NUM_INIT_FOR_EM):  # run EM for 10 random initializations
#         # == init (same logic as original) ==
#         perm = torch.randperm(n, device=P.device)
#         vs = P[perm[:k], :].clone()                      # (k, d)

#         Vs = torch.empty((k, j, d), device=P.device, dtype=P.dtype)
#         idxs = torch.randperm(n, device=P.device)
#         chunks = torch.chunk(idxs, k)                    # list of index tensors

#         # initialize k subspaces
#         for i in range(k):
#             n_i = chunks[i].numel()
#             J_use = min(j, n_i)                          # <-- requested change
#             Vi, _, _ = computeSuboptimalSubspace(P[chunks[i], :], w[chunks[i]], J_use)
#             Vs[i].zero_()
#             Vs[i, :J_use, :] = Vi

#         # == EM steps ==
#         for _ in range(steps):
#             # distances of every point to each of the k flats
#             dists = torch.empty((n, k), device=P.device, dtype=P.dtype)
#             for l in range(k):
#                 _, dists[:, l] = computeCost(P, w, Vs[l, :, :], vs[l, :])

#             # E-step: closest flat per point
#             cluster_indices = torch.argmin(dists, dim=1)  # (n,)
#             unique_idxs = torch.unique(cluster_indices)   # clusters that have members

#             # M-step: refit flats using current memberships
#             for idx in unique_idxs.tolist():
#                 mask = (cluster_indices == idx)
#                 n_k = int(mask.sum().item())
#                 J_use = min(j, n_k)                       # <-- requested change
#                 Vi, vi, _ = computeSuboptimalSubspace(P[mask, :], w[mask], J_use)
#                 Vs[idx].zero_()
#                 Vs[idx, :J_use, :] = Vi
#                 vs[idx, :] = vi

#         current_cost = computeCost(P, w, Vs, vs)[0].item()
#         if current_cost < optimal_cost:
#             min_Vs = copy.deepcopy(Vs)
#             min_vs = copy.deepcopy(vs)
#             optimal_cost = current_cost
#         print(current_cost)

#     return min_Vs, min_vs, (time.time() - start_time)

# @torch.no_grad()
# def assign_points(P: torch.Tensor, w: torch.Tensor, Vs: torch.Tensor, vs: torch.Tensor) -> torch.Tensor:
#     K = Vs.shape[0]
#     N = P.shape[0]
#     dists = torch.empty((N, K), device=P.device, dtype=P.dtype)
#     for l in range(K):
#         di = OBJECTIVE_LOSS(computeDistanceToSubspace(P, Vs[l, :, :], vs[l, :]))  # (N,)
#         dists[:, l] = w * di
#     return torch.argmin(dists, dim=1)

# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     n, d = 1000, 64
#     P = torch.randn(n, d, device=device)
#     w = torch.ones(n, device=device)
#     Vs, vs, elapsed = EMLikeAlg(P, w, j=5, k=9, steps=STEPS)
#     print("Elapsed:", elapsed)


import torch, time, copy

LAMBDA = 1
Z = 2
NUM_INIT_FOR_EM = 10
STEPS = 20

def _lp(x): return (torch.abs(x) ** Z) / Z
def _huber(x):
    ax = torch.abs(x)
    return torch.where(ax <= LAMBDA, 0.5 * x * x, LAMBDA * (ax - 0.5 * LAMBDA))
def _cauchy(x): return (LAMBDA**2 / 2.0) * torch.log1p((x * x) / (LAMBDA**2))
def _geman_mcclure(x): return (x * x) / (2.0 * (1.0 + x * x))
def _welsch(x): return (LAMBDA**2 / 2.0) * (1.0 - torch.exp(-(x * x) / (LAMBDA**2)))
def _tukey(x):
    ax2 = (x * x) / (LAMBDA**2)
    inside = 1.0 - ax2
    val = (LAMBDA**2 / 6.0) * (1.0 - (inside ** 3))
    return torch.where(torch.abs(x) <= LAMBDA, val, torch.full_like(x, LAMBDA**2 / 6.0))

M_ESTIMATOR_FUNCS = {
    'lp': _lp, 'huber': _huber, 'cauchy': _cauchy,
    'geman_McClure': _geman_mcclure, 'welsch': _welsch, 'tukey': _tukey,
}
OBJECTIVE_LOSS = M_ESTIMATOR_FUNCS['lp']  # same as your one-channel code

@torch.no_grad()
def _projective_dist(point: torch.Tensor, V_rows: torch.Tensor, v: torch.Tensor | None):
    """
    Distance from point(s) to affine J-flat with row basis V_rows (J,d) and offset v (d,).
    Uses projection onto the row-space (same assumption as your SVD-based rows).
    """
    if point.ndim == 1:
        R = point if v is None else (point - v)
        R_par = (R @ V_rows.transpose(0, 1)) @ V_rows
        return torch.norm(R - R_par)
    else:
        R = point if v is None else (point - v.unsqueeze(0))
        R_par = (R @ V_rows.transpose(0, 1)) @ V_rows
        return torch.norm(R - R_par, dim=1)

@torch.no_grad()
def _compute_cost_3ch(PY, PCb, PCr, w, VY, VCb, VCr, vY, vCb, vCr, weights=(1.0,1.0,1.0)):
    a,b,c = weights
    dY  = _projective_dist(PY,  VY,  vY)
    dCb = _projective_dist(PCb, VCb, vCb)
    dCr = _projective_dist(PCr, VCr, vCr)
    loss = a * OBJECTIVE_LOSS(dY) + b * OBJECTIVE_LOSS(dCb) + c * OBJECTIVE_LOSS(dCr)
    return w * loss  # (N,)

@torch.no_grad()
def _subspace_with_padding(P: torch.Tensor, w: torch.Tensor, J: int):
    """
    Mean + top-J rows from SVD; if rank<J, pad rows with zeros so returned basis is (J,d).
    Implements J_use = min(J, n_k) as requested (no crashes on tiny clusters).
    """
    wsum = torch.clamp(w.sum(), min=1e-12)
    v = (P * w.unsqueeze(1)).sum(dim=0) / wsum           # (d,)
    Y = P - v.unsqueeze(0)
    U,S,Vh = torch.linalg.svd(Y, full_matrices=False)    # Vh: (r,d)
    Juse = int(min(J, Vh.shape[0]))
    if Juse > 0:
        V_rows = Vh[:Juse, :]
    else:
        V_rows = torch.zeros((0, P.shape[1]), device=P.device, dtype=P.dtype)
    if Juse < J:
        pad = torch.zeros((J - Juse, V_rows.shape[1]), device=P.device, dtype=P.dtype)
        V_rows = torch.cat([V_rows, pad], dim=0)
    return V_rows.contiguous(), v.contiguous()

@torch.no_grad()
def EMLikeAlg(PY: torch.Tensor, PCb: torch.Tensor, PCr: torch.Tensor,
                  w: torch.Tensor, j: int, k: int, steps: int,
                  channel_weights=(1.0,1.0,1.0),
                  num_inits: int = NUM_INIT_FOR_EM):
    """
    3-channel EM-like (K,J)-projective clustering.
      PY, PCb, PCr: (N,d) tensors (same device/dtype), e.g. YCbCr in [0,1]
      w:            (N,) per-point weights
    Returns:
      Vs: (K, 3, J, d)  per-cluster per-channel row-bases
      vs: (K, 3, d)     per-cluster per-channel means (offsets)
    """
    start_time = time.time()
    device = PY.device
    N, d = PY.shape

    best_Vs = None
    best_vs = None
    best_obj = float('inf')

    for init_id in range(num_inits):
        # == init (match original style: chunk a shuffled permutation) ==
        perm = torch.randperm(N, device=device)
        seed_idx = perm[:k]
        vs = torch.stack([PY[seed_idx], PCb[seed_idx], PCr[seed_idx]], dim=1)  # (K,3,d)

        Vs = torch.empty((k, 3, j, d), device=device, dtype=PY.dtype)

        idxs = torch.randperm(N, device=device)
        chunks = torch.chunk(idxs, k)  # original style
        # initial per-cluster, per-channel subspaces (with padding if needed)
        for i in range(k):
            ci = chunks[i]
            ViY,  vY  = _subspace_with_padding(PY.index_select(0, ci),  w.index_select(0, ci), j)
            ViCb, vCb = _subspace_with_padding(PCb.index_select(0, ci), w.index_select(0, ci), j)
            ViCr, vCr = _subspace_with_padding(PCr.index_select(0, ci), w.index_select(0, ci), j)
            Vs[i,0], Vs[i,1], Vs[i,2] = ViY, ViCb, ViCr
            vs[i,0], vs[i,1], vs[i,2] = vY,  vCb, vCr

        # == EM ==
        for _ in range(steps):
            # E-step: total (weighted) 3-channel cost to each cluster
            dists = torch.empty((N, k), device=device, dtype=PY.dtype)
            for l in range(k):
                dists[:, l] = _compute_cost_3ch(
                    PY, PCb, PCr, w,
                    Vs[l,0], Vs[l,1], Vs[l,2],
                    vs[l,0], vs[l,1], vs[l,2],
                    weights=channel_weights
                )
            assign = torch.argmin(dists, dim=1)  # (N,)

            # M-step: refit subspaces for occupied clusters
            uniq = torch.unique(assign)
            for idx in uniq.tolist():
                mask = (assign == idx)
                # per channel, with J_use = min(j, n_k) + zero-pad inside helper
                ViY,  vY  = _subspace_with_padding(PY[mask, :],  w[mask], j)
                ViCb, vCb = _subspace_with_padding(PCb[mask, :], w[mask], j)
                ViCr, vCr = _subspace_with_padding(PCr[mask, :], w[mask], j)
                Vs[idx,0], Vs[idx,1], Vs[idx,2] = ViY, ViCb, ViCr
                vs[idx,0], vs[idx,1], vs[idx,2] = vY,  vCb, vCr

        # end: pick best init by total objective
        # (sum of per-point min dists used in last E-step)
        final_obj = float(dists.gather(1, assign.unsqueeze(1)).sum().item())
        if final_obj < best_obj:
            best_obj = final_obj
            best_Vs  = copy.deepcopy(Vs)
            best_vs  = copy.deepcopy(vs)
        print(final_obj)

    return best_Vs, best_vs, (time.time() - start_time)

@torch.no_grad()
def assign_points(PY: torch.Tensor, PCb: torch.Tensor, PCr: torch.Tensor,
                      w: torch.Tensor, Vs: torch.Tensor, vs: torch.Tensor,
                      channel_weights=(1.0,1.0,1.0)) -> torch.Tensor:
    """
    Assign each point to the nearest cluster under 3-channel cost.
      Vs: (K,3,J,d), vs: (K,3,d)
    Returns: (N,) cluster indices
    """
    N = PY.shape[0]
    K = Vs.shape[0]
    dists = torch.empty((N, K), device=PY.device, dtype=PY.dtype)
    for l in range(K):
        dists[:, l] = _compute_cost_3ch(
            PY, PCb, PCr, w,
            Vs[l,0], Vs[l,1], Vs[l,2],
            vs[l,0], vs[l,1], vs[l,2],
            weights=channel_weights
        )
    return torch.argmin(dists, dim=1)