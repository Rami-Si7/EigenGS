from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum
from utils import *
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
from optimizer import Adan

class GaussianBasis(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.num_comps = kwargs["num_comps"]              # C: components per cluster
        self.num_clusters = kwargs["num_clusters"]        # K: clusters (hard assignment)
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )
        self.device = kwargs["device"]

        # --- Geometry parameters (shared across clusters) ---
        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2, device=self.device) - 0.5)))
        self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3, device=self.device))
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))

        # --- Basis attributes: per-cluster, per-component, per-splat, 3-channels ---
        # shape: (K, C, N, 3)
        self._features_dc = nn.Parameter(
            torch.rand(self.num_clusters, self.num_comps, self.init_num_points, 3, device=self.device)
        )

        # --- Colors for Phase-B (one set per fitted image; trainer handles per-image lifecycle) ---
        # shape: (N, 3)
        self._colors = nn.Parameter(torch.empty(self.init_num_points, 3, device=self.device))

        # --- Rendering affine ---
        self.register_buffer('shift_factor', torch.tensor(0.0))
        self.register_buffer('scale_factor', torch.tensor(1.0))

        # (kept for backward-compat; no longer used in Phase-B)
        self.register_buffer('image_mean', torch.zeros(self.H * self.W))

        # NEW: per-cluster 3-channel means (e.g., YCbCr or RGB), shape (K, 3, H, W)
        self.register_buffer('image_mean_k', torch.zeros(self.num_clusters, 3, self.H, self.W))

        self.register_buffer('background', torch.zeros(3))
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))

        self.opt_type = kwargs["opt_type"]
        self.lr = kwargs["lr"]

    # ------------------------- Accessors -------------------------

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)

    @property
    def get_colors(self):
        return self._colors

    def get_features_for_cluster(self, cluster_id: int):
        # returns tensor of shape (C, N, 3) for this cluster
        return self._features_dc[cluster_id]

    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_cholesky_elements(self):
        return self._cholesky + self.cholesky_bound

    # ------------------------- Utils -------------------------

    def _project(self):
        # Project once; reuse for all component renders in this forward
        self.xys, self.depths, self.radii, self.conics, self.num_tiles_hit = project_gaussians_2d(
            self.get_xyz, self.get_cholesky_elements,
            self.H, self.W, self.tile_bounds
        )

    # Optional: load cluster means from numpy arrays
    # means: shape (K, 3, H, W) or (K, 3, H*W) if from_flat=True
    def load_cluster_means(self, means: np.ndarray, from_flat: bool = True):
        with torch.no_grad():
            if from_flat:
                # expecting (K, 3, H*W)
                K, C, D = means.shape
                assert K == self.num_clusters and C == 3 and D == self.H * self.W, \
                    f"Bad means shape {means.shape}, expected {(self.num_clusters, 3, self.H*self.W)}"
                means_t = torch.from_numpy(means).float().view(self.num_clusters, 3, self.H, self.W)
            else:
                # expecting (K, 3, H, W)
                assert means.shape == (self.num_clusters, 3, self.H, self.W), \
                    f"Bad means shape {means.shape}, expected {(self.num_clusters, 3, self.H, self.W)}"
                means_t = torch.from_numpy(means).float()
            self.image_mean_k.copy_(means_t.to(self.image_mean_k.device))

    # Optional: initialize colors from per-image PCA weights (three channels)
    # wY, wCb, wCr: numpy or torch, shape (C,)
    def init_colors_from_weights(self, cluster_id: int, wY, wCb, wCr):
        with torch.no_grad():
            feats = self._features_dc[cluster_id]  # (C, N, 3)
            if not torch.is_tensor(wY):
                wY = torch.from_numpy(np.asarray(wY))
                wCb = torch.from_numpy(np.asarray(wCb))
                wCr = torch.from_numpy(np.asarray(wCr))
            wY = wY.to(feats).view(-1, 1)   # (C,1)
            wCb = wCb.to(feats).view(-1, 1)
            wCr = wCr.to(feats).view(-1, 1)

            # channel-wise weighted sums -> (N,)
            colY  = (wY  * feats[:, :, 0]).sum(dim=0)
            colCb = (wCb * feats[:, :, 1]).sum(dim=0)
            colCr = (wCr * feats[:, :, 2]).sum(dim=0)
            colors_init = torch.stack([colY, colCb, colCr], dim=1)  # (N, 3)
            self._colors.copy_(colors_init.clamp_(0.0, 1.0))

    # ------------------------- Forwards -------------------------

    def _forward_colors(self, cluster_id: int = None):
        # cluster_id kept for symmetric API; not used here (colors are global)
        self._project()
        out_img = rasterize_gaussians_sum(
            self.xys, self.depths, self.radii, self.conics, self.num_tiles_hit,
            self.get_colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False
        )
        out_img *= self.scale_factor
        out_img += self.shift_factor
        out_img = out_img.permute(2, 0, 1).contiguous()  # (3, H, W)
        return out_img

    def _forward_features_dc(self, cluster_id: int, comp_indices=None):
        """
        Render component stack for a given cluster.
        Returns: (num_comps_selected, 3, H, W)
        """
        assert 0 <= cluster_id < self.num_clusters, f"cluster_id {cluster_id} out of range [0,{self.num_clusters-1}]"
        self._project()

        feats_k = self.get_features_for_cluster(cluster_id)  # (C, N, 3)
        if comp_indices is None:
            comp_indices = range(self.num_comps)

        comps = []
        for i in comp_indices:
            # attrs: (N,3)
            attrs = feats_k[i]
            out_img = rasterize_gaussians_sum(
                self.xys, self.depths, self.radii, self.conics, self.num_tiles_hit,
                attrs, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W,
                background=self.background, return_alpha=False
            )
            out_img = out_img.permute(2, 0, 1).contiguous()  # (3, H, W)
            comps.append(out_img)

        out_stack = torch.stack(comps, dim=0)  # (len(comp_indices), 3, H, W)
        return out_stack

    def forward(self, cluster_id: int, render_colors=False, comp_indices=None):
        if render_colors:
            return self._forward_colors(cluster_id)
        else:
            return self._forward_features_dc(cluster_id, comp_indices=comp_indices)

    # ------------------------- Training steps -------------------------

    def train_iter(self, gt_stack, cluster_id: int, comp_indices=None):
        """
        Phase-A (basis) training step, hard assignment on 'cluster_id'.
        gt_stack: (num_selected, 3, H, W) matching the rendered component stack.
        comp_indices: optional iterable of component indices to render/supervise.
        """
        pred_stack = self.forward(cluster_id=cluster_id, render_colors=False, comp_indices=comp_indices)
        loss = loss_fn(pred_stack, gt_stack, self.loss_type, lambda_value=0.7)
        loss.backward()

        with torch.no_grad():
            mse_loss = F.mse_loss(pred_stack, gt_stack)
            psnr = 10 * math.log10(1.0 / (mse_loss.item() + 1e-12))

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss, psnr

    def optimize_iter(self, gt_image, cluster_id: int):
        # forward BEFORE step
        color_field = self.forward(cluster_id=cluster_id, render_colors=True)  # (3,H,W)
        image = color_field + self.image_mean_k[cluster_id]

        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()

        # --- debug you added earlier shows grads and step_delta ---

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        # NEW: forward AGAIN AFTER step, so metrics track improvements
        with torch.no_grad():
            color_field_after = self.forward(cluster_id=cluster_id, render_colors=True)
            image_after = color_field_after + self.image_mean_k[cluster_id]
            mse_after = F.mse_loss(image_after, gt_image)
            psnr_after = 10 * math.log10(1.0 / (mse_after.item() + 1e-12))

        return loss, psnr_after


    # ------------------------- Optim/scheduler -------------------------

    def scheduler_init(self, optimize_phase=False):
        # Explicit param groups (no name filtering surprises)
        if not optimize_phase:
            # Phase-A: train basis + geometry, freeze colors
            self._features_dc.requires_grad_(True)
            self._colors.requires_grad_(False)
            self._xyz.requires_grad_(True)
            self._cholesky.requires_grad_(True)

            params = [
                self._features_dc,  # (K, C, N, 3)
                self._xyz,          # (N, 2)
                self._cholesky,     # (N, 3)
            ]
            phase = "A"
        else:
            # Phase-B: train colors + geometry, freeze basis
            self._features_dc.requires_grad_(False)
            self._colors.requires_grad_(True)
            self._xyz.requires_grad_(True)
            self._cholesky.requires_grad_(True)

            params = [
                self._colors,       # (N, 3)
                self._xyz,          # (N, 2)
                self._cholesky,     # (N, 3)
            ]
            phase = "B"

        # Destroy old optimizer/scheduler and rebuild
        if self.opt_type == "adam":
            self.optimizer = torch.optim.Adam(params, lr=self.lr)
        else:
            self.optimizer = Adan(params, lr=self.lr)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

        # Debug print (one line) so you *know* what's trainable
        n_params = sum(p.numel() for p in params)
        print(f"[scheduler_init phase {phase}] params={len(params)} (total scalars {n_params}), "
              f"features_grad={self._features_dc.requires_grad}, "
              f"colors_grad={self._colors.requires_grad}, "
              f"xyz_grad={self._xyz.requires_grad}, chol_grad={self._cholesky.requires_grad}")

