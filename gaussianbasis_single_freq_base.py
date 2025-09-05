from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from optimizer import Adan
import torch.nn.functional as F  # for PSNR in train_iter

class GaussianBasis(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.num_comps = kwargs["num_comps"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )
        self.device = kwargs["device"]
        self.opt_type = kwargs["opt_type"]
        self.lr = kwargs["lr"]

        # ---- learnable params ----
        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2, device=self.device) - 0.5)))
        self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3, device=self.device))
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1), device=self.device))
        self._features_dc = nn.Parameter(torch.rand(self.num_comps, self.init_num_points, 3, device=self.device))
        self._colors = nn.Parameter(torch.empty(self.init_num_points, 3, device=self.device))

        # ---- runtime factors / means (correct shapes) ----
        # scale/shift are per-channel (Y, Cb, Cr)
        self.register_buffer('shift_factor', torch.zeros(3, device=self.device))
        self.register_buffer('scale_factor', torch.ones(3, device=self.device))
        # image_mean is per-channel, flattened image
        self.register_buffer('image_mean', torch.zeros(3, self.H * self.W, device=self.device))

        # ---- misc buffers on the right device ----
        self.register_buffer('background', torch.ones(3, device=self.device))
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0.0, 0.5], device=self.device).view(1, 3))

        # sched/opt placeholders
        self.optimizer, self.scheduler = None, None

        # In case an old checkpoint with scalar buffers is ever loaded:
        self._ensure_buffer_shapes()

    # ---------- helpers ----------
    def _ensure_buffer_shapes(self):
        # Upgrade legacy scalar buffers to vector shape
        if self.scale_factor.dim() == 0 or self.scale_factor.numel() == 1:
            self.register_buffer('scale_factor', torch.ones(3, device=self.device))
        if self.shift_factor.dim() == 0 or self.shift_factor.numel() == 1:
            self.register_buffer('shift_factor', torch.zeros(3, device=self.device))
        if self.image_mean.dim() == 1 and self.image_mean.numel() == (self.H * self.W):
            self.register_buffer('image_mean', self.image_mean.repeat(3, 1))  # -> (3, H*W)

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)

    @property
    def get_colors(self):
        return self._colors

    @property
    def get_features(self):
        return self._features_dc
    
    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_cholesky_elements(self):
        return self._cholesky + self.cholesky_bound

    # ---------- forward paths ----------
    def _forward_colors(self):
        self._ensure_buffer_shapes()
        xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
            self.get_xyz, self.get_cholesky_elements,
            self.H, self.W, self.tile_bounds
        )
        out_img = rasterize_gaussians_sum(
            xys, depths, radii, conics, num_tiles_hit,
            self.get_colors, self._opacity,
            self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False
        )
        out_img = out_img.permute(2, 0, 1).contiguous()  # -> (3,H,W)

        # per-channel affine
        out_img = out_img * self.scale_factor.view(3, 1, 1)
        out_img = out_img + self.shift_factor.view(3, 1, 1)
        return out_img  # (3,H,W)

    def _forward_featrues_dc(self):
        # (typo kept for API compatibility)
        xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
            self.get_xyz, self.get_cholesky_elements,
            self.H, self.W, self.tile_bounds
        )
        comps = []
        for i in range(self.num_comps):
            out_img = rasterize_gaussians_sum(
                xys, depths, radii, conics, num_tiles_hit,
                self.get_features[i], self._opacity,
                self.H, self.W, self.BLOCK_H, self.BLOCK_W,
                background=self.background, return_alpha=False
            )
            comps.append(out_img.permute(2, 0, 1).contiguous())  # (3,H,W)
        return torch.stack(comps, dim=0)  # (num_comps,3,H,W)

    def forward(self, render_colors=False):
        return self._forward_colors() if render_colors else self._forward_featrues_dc()

    # ---------- training loops ----------
    def train_iter(self, gt_image):
        """Phase-A: fit features to components: image ≈ gt_arrs (shape: [num_comps,3,H,W])."""
        image = self.forward(render_colors=False)
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss, psnr

    def optimize_iter(self, gt_image):
        """Phase-B: refine colors to reconstruct a target image (YCbCr in [0,1])."""
        out = self.forward(render_colors=True)              # (3,H,W) after scale/shift
        image = out.reshape(3, -1) + self.image_mean        # add per-channel mean
        image = image.reshape(3, self.H, self.W)
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss, psnr

    # ---------- opt/sched ----------
    def scheduler_init(self, optimize_phase=False):
        if not optimize_phase:
            params = [p for n, p in self.named_parameters() if n != '_colors']
            self._colors.requires_grad_(False)
            self._features_dc.requires_grad_(True)
        else:
            params = [p for n, p in self.named_parameters() if n != '_features_dc']
            self._colors.requires_grad_(True)
            self._features_dc.requires_grad_(False)

        if self.opt_type == "adam":
            self.optimizer = torch.optim.Adam(params, lr=self.lr)
        else:
            self.optimizer = Adan(params, lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)