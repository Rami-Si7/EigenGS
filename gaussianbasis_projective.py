# gaussianbasis.py
from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from optimizer import Adan
import torch.nn.functional as F

class GaussianBasis(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )
        self.device = kwargs["device"]

        # ---- NEW: clustered flags/shape ----
        self.clustered = bool(kwargs.get("clustered", False))
        self.K = int(kwargs.get("K_clusters", 1))
        self.J = int(kwargs.get("J_dim", kwargs.get("num_comps", 1)))
        # For backward compatibility with the original path
        self.num_comps = int(kwargs.get("num_comps", self.J if not self.clustered else (3 * self.K * self.J)))

        # ---- shared geometry ----
        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))

        # ---- Features / weights ----
        if not self.clustered:
            # Original: per-component 3-channel colors per Gaussian
            self._features_dc = nn.Parameter(torch.rand(self.num_comps, self.init_num_points, 3))
        else:
            # NEW: psi[c, k, j, n] — single-channel weights per (channel, cluster, component, gaussian)
            self._psi = nn.Parameter(0.01 * torch.randn(3, self.K, self.J, self.init_num_points))

        # Per-image color vector for Phase-B
        self._colors = nn.Parameter(torch.empty(self.init_num_points, 3))

        # Rendering/normalization buffers (broadcast over last dim=3)
        self.register_buffer('shift_factor', torch.tensor(0.0, device=self.device))
        self.register_buffer('scale_factor', torch.tensor(1.0, device=self.device))
        # image_mean will be set to (3, H*W) at Phase-B init; keep as buffer
        self.register_buffer('image_mean', torch.zeros(self.H * self.W, device=self.device))

        self.register_buffer('background', torch.ones(3))
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))

        self.opt_type = kwargs["opt_type"]
        self.lr = kwargs["lr"]

    # -------- properties --------
    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)

    @property
    def get_colors(self):
        return self._colors

    @property
    def get_features(self):
        if self.clustered:
            return self._psi
        return self._features_dc

    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_cholesky_elements(self):
        return self._cholesky + self.cholesky_bound

    # -------- low-level renders --------
    def _project(self):
        return project_gaussians_2d(
            self.get_xyz, self.get_cholesky_elements,
            self.H, self.W, self.tile_bounds
        )

    def _render_with_colors(self, colors_3xn):
        xys, depths, radii, conics, num_tiles_hit = self._project()
        out_img = rasterize_gaussians_sum(
            xys, depths, radii, conics, num_tiles_hit,
            colors_3xn, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False
        )
        out_img *= self.scale_factor
        out_img += self.shift_factor
        return out_img.permute(2, 0, 1).contiguous()  # (3,H,W)

    def _forward_colors(self):
        # Phase-B: render a full RGB image from per-image _colors
        return self._render_with_colors(self.get_colors)

    def _forward_features_dc_original(self):
        # Original Phase-A: render each global component (num_comps, 3, H, W)
        xys, depths, radii, conics, num_tiles_hit = self._project()
        comps = []
        for i in range(self.num_comps):
            out_img = rasterize_gaussians_sum(
                xys, depths, radii, conics, num_tiles_hit,
                self._features_dc[i], self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W,
                background=self.background, return_alpha=False
            )
            comps.append(out_img.permute(2, 0, 1).contiguous())
        return torch.stack(comps, dim=0)  # (num_comps, 3, H, W)

    def _forward_features_dc_clustered(self):
        # NEW Phase-A for clustered mode: render every (c, k, j) as a separate 3-channel image
        xys, depths, radii, conics, num_tiles_hit = self._project()
        comps = []
        for c in range(3):
            for k in range(self.K):
                for j in range(self.J):
                    # Build colors per Gaussian: only channel c is active
                    col = torch.zeros(self.init_num_points, 3, device=self.device)
                    col[:, c] = self._psi[c, k, j]
                    out_img = rasterize_gaussians_sum(
                        xys, depths, radii, conics, num_tiles_hit,
                        col, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W,
                        background=self.background, return_alpha=False
                    )
                    comps.append(out_img.permute(2, 0, 1).contiguous())
        return torch.stack(comps, dim=0)  # (3*K*J, 3, H, W)

    # -------- public interface --------
    def forward(self, render_colors=False):
        if render_colors:
            return self._forward_colors()
        if self.clustered:
            return self._forward_features_dc_clustered()
        else:
            return self._forward_features_dc_original()

    # -------- one step of optimization --------
    def train_iter(self, gt_image):
        image = self.forward()
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / (mse_loss.item() + 1e-12))
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss, psnr

    def optimize_iter(self, gt_image):
        out = self.forward(render_colors=True)
        image = out.reshape(3, -1) + self.image_mean  # (3,H*W)
        image = image.reshape(3, self.H, self.W)
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / (mse_loss.item() + 1e-12))
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss, psnr

    def scheduler_init(self, optimize_phase=False):
        if not optimize_phase:
            # Phase-A: learn eigen-image weights; freeze per-image colors
            if self.clustered:
                params = [p for n, p in self.named_parameters() if n.startswith("_psi") or n.startswith("_xyz") or n.startswith("_cholesky") or n.startswith("_opacity")]
            else:
                params = [p for n, p in self.named_parameters() if n != '_colors']
            self._colors.requires_grad_(False)
            if not self.clustered:
                self._features_dc.requires_grad_(True)
        else:
            # Phase-B: optimize image colors; freeze eigen-image weights
            params = [p for n, p in self.named_parameters() if n != ('_features_dc' if not self.clustered else '_psi')]
            self._colors.requires_grad_(True)
            if not self.clustered:
                self._features_dc.requires_grad_(False)

        if self.opt_type == "adam":
            self.optimizer = torch.optim.Adam(params, lr=self.lr)
        else:
            self.optimizer = Adan(params, lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)
