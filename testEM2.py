# """
# name:   testEM2
# author:  Alaa Maalouf

# researchers (last-name alphabetical-order):
#     Dan Feldman
#     Harry Lang
#     Alaa Maalouf
#     Daniela Rus
# """

# import numpy as np
# from scipy.linalg import null_space
# import copy
# import time
# import random

# # import getdim # file in directory to load BERT
# LAMBDA = 1
# Z = 2
# # NUM_INIT_FOR_EM = 1
# STEPS = 20
# M_ESTIMATOR_FUNCS = {
#     "lp": (lambda x: np.abs(x) ** Z / Z),
#     "huber": (
#         lambda x: x ** 2 / 2
#         if np.abs(x) <= LAMBDA
#         else LAMBDA * (np.abs(x) - LAMBDA / 2)
#     ),
#     "cauchy": (lambda x: LAMBDA ** 2 / 2 * np.log(1 + x ** 2 / LAMBDA ** 2)),
#     "geman_McClure": (lambda x: x ** 2 / (2 * (1 + x ** 2))),
#     "welsch": (
#         lambda x: LAMBDA ** 2 / 2 * (1 - np.exp(-(x ** 2) / LAMBDA ** 2))
#     ),
#     "tukey": (
#         lambda x: LAMBDA ** 2 / 6 * (1 - (1 - x ** 2 / LAMBDA ** 2) ** 3)
#         if np.abs(x) <= LAMBDA
#         else LAMBDA ** 2 / 6
#     ),
# }
# global OBJECTIVE_LOSS
# OBJECTIVE_LOSS = M_ESTIMATOR_FUNCS["lp"]


# def computeDistanceToSubspace(point, X):
#     """
#     This function is responsible for computing the distance between a point and a J dimensional affine subspace.

#     :param point: A numpy array representing a .
#     :param X: A numpy matrix representing a basis for a J dimensional subspace.
#     :param v: A numpy array representing the translation of the subspace from the origin.
#     :return: The distance between the point and the subspace which is spanned by X and translated from the origin by v.
#     """
#     if point.ndim > 1:
#         return np.linalg.norm(np.dot(point, null_space(X)), ord=2, axis=1)
#     return np.linalg.norm(np.dot(point, null_space(X)))


# def computeDistanceToSubspaceviaNullSpace(point, null_space):
#     """
#     This function is responsible for computing the distance between a point and a J dimensional affine subspace.

#     :param point: A numpy array representing a .
#     :param X: A numpy matrix representing a basis for a J dimensional subspace.
#     :param v: A numpy array representing the translation of the subspace from the origin.
#     :return: The distance between the point and the subspace which is spanned by X and translated from the origin by v.
#     """
#     if point.ndim > 1:
#         return np.linalg.norm(np.dot(point, null_space), ord=2, axis=1)
#     return np.linalg.norm(np.dot(point, null_space))


# def computeCost(P, w, X, show_indices=False):
#     """
#     This function represents our cost function which is a generalization of k-means where the means are now J-flats.

#     :param P: A weighed set, namely, a PointSet object.
#     :param X: A numpy matrix of J x d which defines the basis of the subspace which we would like to compute the
#               distance to.
#     :param v: A numpy array of d entries which defines the translation of the J-dimensional subspace spanned by the
#               rows of X.
#     :return: The sum of weighted distances of each point to the affine J dimensional flat which is denoted by (X,v)
#     """
#     global OBJECTIVE_LOSS
#     if X.ndim == 2:
#         dist_per_point = OBJECTIVE_LOSS(
#             computeDistanceToSubspaceviaNullSpace(P, null_space(X))
#         )
#         cost_per_point = np.multiply(w, dist_per_point)
#     else:
#         temp_cost_per_point = np.empty((P.shape[0], X.shape[0]))
#         for i in range(X.shape[0]):
#             temp_cost_per_point[:, i] = np.multiply(
#                 w,
#                 OBJECTIVE_LOSS(
#                     computeDistanceToSubspaceviaNullSpace(
#                         P, null_space(X[i, :, :])
#                     )
#                 ),
#             )

#         cost_per_point = np.min(temp_cost_per_point, 1)
#         indices = np.argmin(temp_cost_per_point, 1)
#     if not show_indices:
#         return np.sum(cost_per_point), cost_per_point
#     else:
#         return np.sum(cost_per_point), cost_per_point, indices


# def computeSuboptimalSubspace(P, w, J):
#     """
#     This function computes a suboptimal subspace in case of having the generalized K-means objective function.

#     :param P: A weighted set, namely, an object of PointSet.
#     :return: A tuple of a basis of J dimensional spanning subspace, namely, X and a translation vector denoted by v.
#     """

#     start_time = time.time()

#     _, _, V = np.linalg.svd(
#         P, full_matrices=False
#     )  # computing the spanning subspace
#     return V[:J, :], time.time() - start_time


# def EMLikeAlg(P, w, j, k, steps, NUM_INIT_FOR_EM=10):
#     """
#     The function at hand, is an EM-like algorithm which is heuristic in nature. It finds a suboptimal solution for the
#     (K,J)-projective clustering problem with respect to a user chosen

#     :param P: A weighted set, namely, a PointSet object
#     :param j: An integer denoting the desired dimension of each flat (affine subspace)
#     :param k: An integer denoting the number of j-flats
#     :param steps: An integer denoting the max number of EM steps
#     :return: A list of k j-flats which locally optimize the cost function
#     """

#     start_time = time.time()
#     np.random.seed(random.seed())
#     n, d = P.shape
#     min_Vs = None
#     optimal_cost = np.inf
#     # print ("started")
#     for iter in range(NUM_INIT_FOR_EM):  # run EM for 10 random initializations
#         Vs = np.empty((k, j, d))
#         idxs = np.arange(n)
#         np.random.shuffle(idxs)
#         idxs = np.array_split(idxs, k)  # ;print(idxs)
#         for i in range(k):  # initialize k random orthogonal matrices
#             Vs[i, :, :], _ = computeSuboptimalSubspace(
#                 P[idxs[i], :], w[idxs[i]], j
#             )

#         for i in range(
#             steps
#         ):  # find best k j-flats which can attain local optimum
#             dists = np.empty(
#                 (n, k)
#             )  # distance of point to each one of the k j-flats
#             for l in range(k):
#                 _, dists[:, l] = computeCost(P, w, Vs[l, :, :])

#             cluster_indices = np.argmin(
#                 dists, 1
#             )  # determine for each point, the closest flat to it
#             unique_idxs = np.unique(
#                 cluster_indices
#             )  # attain the number of clusters
#             print("Cluster sizes:", [np.sum(cluster_indices == i) for i in range(k)])

#             for (
#                 idx
#             ) in (
#                 unique_idxs
#             ):  # recompute better flats with respect to the updated cluster matching
#                 Vs[idx, :, :], _ = computeSuboptimalSubspace(
#                     P[np.where(cluster_indices == idx)[0], :],
#                     w[np.where(cluster_indices == idx)[0]],
#                     j,
#                 )

#         current_cost = computeCost(P, w, Vs)[0]
#         if current_cost < optimal_cost:
#             min_Vs = copy.deepcopy(Vs)
#             optimal_cost = current_cost
#         print(
#             "finished iteration number {} with cost {}".format(
#                 iter, optimal_cost
#             )
#         )
#     return min_Vs, time.time() - start_time


# """

# """


# def main():
#     pass


# if __name__ == "__main__":
#     main()
import torch
import numpy as np
import time

def objective_loss_lp(x, z=2):
    return (torch.abs(x) ** z) / z

def compute_null_space(X):
    u, s, vh = torch.linalg.svd(X, full_matrices=True)
    rank = torch.sum(s > 1e-10)
    return vh[rank:].T

def compute_distance_to_subspace(P, X):
    null_X = compute_null_space(X)
    projection = P @ null_X
    return torch.norm(projection, dim=1)

def computeCost(P, w, Vs, show_indices=False):
    if Vs.ndim == 2:
        dist = compute_distance_to_subspace(P, Vs)
        cost_per_point = w * objective_loss_lp(dist)
    else:
        n_subspaces = Vs.shape[0]
        all_costs = []
        with torch.no_grad():
            for i in range(n_subspaces):
                null_X = compute_null_space(Vs[i])
                dists = torch.norm(P @ null_X, dim=1)
                cost = w * objective_loss_lp(dists)
                all_costs.append(cost.unsqueeze(1))
        costs = torch.cat(all_costs, dim=1)
        cost_per_point, indices = torch.min(costs, dim=1)
    if show_indices:
        return cost_per_point.sum(), cost_per_point, indices
    return cost_per_point.sum(), cost_per_point

def compute_suboptimal_subspace(P_subset, j):
    _, _, V = torch.linalg.svd(P_subset, full_matrices=False)
    return V[:j, :]

def EMLikeAlg(P_np, j, k, steps=20, num_init=5, max_points=500000):
    if len(P_np) > max_points:
        idxs = np.random.choice(len(P_np), size=max_points, replace=False)
        P_np = P_np[idxs]

    P = torch.tensor(P_np, dtype=torch.float32).to('cuda')
    w = torch.ones(len(P), dtype=torch.float32).to('cuda')

    n, d = P.shape
    best_Vs = None
    best_cost = float('inf')

    for init in range(num_init):
        print(f"\nEM Initialization {init+1}/{num_init}")
        Vs = torch.zeros((k, j, d), device=P.device)
        indices = torch.randperm(n)
        splits = torch.chunk(indices, k)

        for i in range(k):
            if len(splits[i]) >= j:
                Vs[i] = compute_suboptimal_subspace(P[splits[i]], j)

        for step in range(steps):
            cost_val, _, cluster_indices = computeCost(P, w, Vs, show_indices=True)
            cluster_sizes = [torch.sum(cluster_indices == c).item() for c in range(k)]
            print(f"  Step {step+1}/{steps} | Cost: {cost_val.item():.6f} | Cluster sizes: {cluster_sizes}")
            for idx in range(k):
                members = (cluster_indices == idx).nonzero(as_tuple=True)[0]
                if len(members) >= j:
                    Vs[idx] = compute_suboptimal_subspace(P[members], j)

        final_cost, _ = computeCost(P, w, Vs)
        print(f"Final cost for init {init+1}: {final_cost.item():.6f}")

        if final_cost.item() < best_cost:
            best_cost = final_cost.item()
            best_Vs = Vs.clone()

    print(f"\nOptimal cost across all initializations: {best_cost:.6f}")
    return best_Vs.cpu().numpy(), best_cost

