from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from optimizer import Adan
from pytorch_msssim import ms_ssim
import torch.nn.functional as F  # <-- add; train_iter already uses F


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

        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        self._colors = nn.Parameter(torch.empty(self.init_num_points, 3))

        self.cur_freq = None
        self.config = kwargs["freq_config"]
        self.params = nn.ParameterDict()
        for freq in (["low", "high", "all"]):
            self.params[f"{freq}_xyz"] = nn.Parameter(torch.atanh(2 * (torch.rand(self.config[freq][2], 2) - 0.5)))
            self.params[f"{freq}_cholesky"] = nn.Parameter(torch.rand(self.config[freq][2], 3))
            self.params[f"{freq}_features_dc"] = nn.Parameter(torch.rand(self.config[freq][1]-self.config[freq][0], self.config[freq][2], 3))

        self.register_buffer('shift_factor', torch.tensor(0.0, device=self.device))
        self.register_buffer('scale_factor', torch.tensor(1.0, device=self.device))
        self.register_buffer('image_mean', torch.zeros(self.H * self.W, device=self.device))

        self.register_buffer('background', torch.ones(3))
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))

        self.opt_type = kwargs["opt_type"]
        self.lr = kwargs["lr"]


                # --- k-PCA extension (non-breaking) ---
        self.use_kpca = kwargs.get("use_kpca", False)           # enable multi-subspace mapping
        self.num_clusters = int(kwargs.get("num_clusters", 1))   # k from your parser

        # will be set per-image by trainer/dataset
        self.current_cluster_id = None
        self.current_w = None   # torch.tensor of shape (3, num_comps), per-image PCA coeffs

        # Create per-cluster per-component weights (ψ′) only if enabled
        if self.use_kpca and self.num_clusters > 1:
            for freq in (["low", "high", "all"]):
                start_f, end_f, num_pts = self.config[freq]  # (feature_start, feature_end, num_points_in_group)
                num_feats = end_f - start_f                  # number of PCA components in this freq block
                # shape: (num_feats, num_pts, 3, num_clusters)
                self.params[f"{freq}_psi_kpca"] = nn.Parameter(
                    0.01 * torch.randn(num_feats, num_pts, 3, self.num_clusters)
                )

    
    @property
    def get_colors(self):
        return self._colors

    @property
    def get_xyz(self):
        return self.params[f"{self.cur_freq}_xyz"]

    # @property
    # def get_features(self):
    #     return self.params[f"{self.cur_freq}_features_dc"]
    @property
    def get_features(self):
        """
        Returns per-component per-Gaussian RGB weights for the *current freq block*.
        - Default: use your original learnable features (backwards compatible).
        - If k-PCA is enabled and cluster_id is set: slice ψ′ for that cluster.
        """
        if (
            self.use_kpca and
            self.num_clusters > 1 and
            self.current_cluster_id is not None and
            f"{self.cur_freq}_psi_kpca" in self.params
        ):
            # (num_feats, num_pts, 3, C) --> cluster slice --> (num_feats, num_pts, 3)
            return self.params[f"{self.cur_freq}_psi_kpca"][..., self.current_cluster_id]
        # fallback: original single-subspace features
        return self.params[f"{self.cur_freq}_features_dc"]

    @property
    def get_cholesky_elements(self):
        return self.params[f"{self.cur_freq}_cholesky"]+self.cholesky_bound

    @property
    def get_scale_cholesky_elements(self):
        return self.params[f"{self.cur_freq}_cholesky"]*0.92 + self.cholesky_bound
    
    @property
    def get_opacity(self):
        config = self.config[self.cur_freq]
        # return self._opacity[config[0]:config[1], :]
        return self._opacity[:config[2], :]

    def set_cluster_and_proj(self, cluster_id: int, w):
        """
        Set hard cluster assignment for the current image and its PCA coefficients.
        w: (3, num_comps) tensor or numpy (Y, Cb, Cr per-component weights).
        """
        self.current_cluster_id = int(cluster_id)
        if not torch.is_tensor(w):
            w = torch.from_numpy(w)
        self.current_w = w.to(self.device).float()  # (3, num_comps)

    def _forward_colors(self):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(
            self.get_xyz, self.get_cholesky_elements, 
            self.H, self.W, self.tile_bounds
        )
        out_img = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            self.get_colors, self.get_opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False
        )
        out_img *= self.scale_factor
        out_img += self.shift_factor
        out_img = out_img.permute(2, 0, 1).contiguous()
        return out_img

    def _forward_freq_colors(self):
        self.cur_freq = "low"
        cholesky = self.get_scale_cholesky_elements.clone()
        # cholesky = self.get_cholesky_elements.clone()
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(
            self.get_xyz, cholesky, 
            self.H, self.W, self.tile_bounds
        )
        colors = self.get_colors[:2000, :].clone()
        out_img = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            colors, self.get_opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False
        )
        self.cur_freq = "all"
        out_img *= self.scale_factor
        out_img += self.shift_factor
        out_img = out_img.permute(2, 0, 1).contiguous()
        return out_img

    def _forward_featrues_dc(self):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(
            self.get_xyz, self.get_cholesky_elements, 
            self.H, self.W, self.tile_bounds
        )
        comps = []
        config = self.config[self.cur_freq]
        for i in range(config[1] - config[0]):
            out_img = rasterize_gaussians_sum(
                self.xys, depths, self.radii, conics, num_tiles_hit,
                self.get_features[i], self.get_opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W,
                background=self.background, return_alpha=False
            )
            out_img = out_img.permute(2, 0, 1).contiguous()
            comps.append(out_img)
            
        out_img = torch.stack(comps, dim=0)
        return out_img

    def forward(self, render_colors=False):
        if render_colors:
            return self._forward_colors()
        else:
            return self._forward_featrues_dc()

    def train_iter(self, gt_image):
        config = self.config[self.cur_freq]
        gt_image = gt_image[config[0]:config[1], :, :, :]
        image = self.forward()
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
        out = self.forward(render_colors=True)
        image = out.reshape(3, -1) + self.image_mean
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
        
    @torch.no_grad()
    def forward_kpca_init(self):
        """
        Render an initial reconstruction Ĩ using the current freq block, the current cluster,
        and the provided PCA coefficients self.current_w.

        Requires:
          - self.use_kpca == True and self.num_clusters > 1
          - self.current_cluster_id is not None
          - self.current_w is set with shape (3, num_comps)
        """
        assert self.use_kpca and self.num_clusters > 1, "Enable use_kpca and set num_clusters>1"
        assert self.current_cluster_id is not None, "Call set_cluster_and_proj(...) first"
        assert self.cur_freq is not None, "Set self.cur_freq to one of {'low','high','all'}"

        # Geometry & opacity for current freq block
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(
            self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds
        )

        # Select ψ′ and the matching slice of coefficients for this freq block
        start_f, end_f, _ = self.config[self.cur_freq]
        w_group = self.current_w[:, start_f:end_f]          # (3, Kf)
        psi = self.params[f"{self.cur_freq}_psi_kpca"][..., self.current_cluster_id]  # (Kf, N, 3)

        # Per-Gaussian RGB:  features[n, ch] = sum_j psi[j, n, ch] * w[ch, j]
        # psi: (Kf, N, 3), w_group: (3, Kf)  ->  (N, 3)
        features = torch.einsum('knc,ck->nc', psi, w_group)  # (N, 3)

        out_img = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            features, self.get_opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False
        )
        out_img *= self.scale_factor
        out_img += self.shift_factor
        return out_img.permute(2, 0, 1).contiguous()  # (3, H, W)


    def scheduler_init(self, optimize_phase=False):
        if not optimize_phase:
            params = [p for n, p in self.named_parameters() if n.startswith(f'params.{self.cur_freq}')]
            for name, param in self.named_parameters():
                if name.startswith(f'params.{self.cur_freq}'):
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        else:
            params = [p for n, p in self.named_parameters() if n.startswith(f'params.all') or n.startswith(f'_colors')]
            for name, param in self.named_parameters():
                if name.startswith(f'params.all') or name.startswith(f'_colors'):
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        if self.opt_type == "adam":
            self.optimizer = torch.optim.Adam(params, lr=self.lr)
        else:
            self.optimizer = Adan(params, lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)