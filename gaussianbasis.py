from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from optimizer import Adan
from pytorch_msssim import ms_ssim

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
        for freq in ("low", "high", "all"):
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

        # For k-PCA enhancement
        C = kwargs["num_clusters"]
        j = kwargs["num_proj_dim"]
        self.register_buffer("cluster_means", torch.empty(C, 3))
        self.register_buffer("cluster_bases", torch.empty(C, j, 3))
        self.alpha_logits = nn.Parameter(torch.randn(self.init_num_points, C))
        self.w_proj = nn.Parameter(torch.randn(self.init_num_points, C, j))

    @property
    def get_colors(self):
        return self._colors

    @property
    def get_xyz(self):
        return self.params[f"{self.cur_freq}_xyz"]

    @property
    def get_features(self):
        return self.params[f"{self.cur_freq}_features_dc"]

    @property
    def get_cholesky_elements(self):
        return self.params[f"{self.cur_freq}_cholesky"] + self.cholesky_bound

    @property
    def get_scale_cholesky_elements(self):
        return self.params[f"{self.cur_freq}_cholesky"] * 0.92 + self.cholesky_bound

    @property
    def get_opacity(self):
        config = self.config[self.cur_freq]
        return self._opacity[:config[2], :]

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

    def _forward_kpca_features(self):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(
            self.get_xyz, self.get_cholesky_elements, 
            self.H, self.W, self.tile_bounds
        )
        alpha_weights = torch.softmax(self.alpha_logits, dim=1)  # (N, C)
        splat_features = torch.zeros(self.init_num_points, 3, device=self.device)
        for c in range(self.cluster_means.shape[0]):
            mu_c = self.cluster_means[c]                       # (3,)
            V_c = self.cluster_bases[c]                        # (j, 3)
            w_proj_c = self.w_proj[:, c, :]                    # (N, j)
            proj_c = mu_c + torch.matmul(w_proj_c, V_c)       # (N, 3)
            splat_features += alpha_weights[:, c:c+1] * proj_c

        out_img = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            splat_features, self.get_opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False
        )
        out_img *= self.scale_factor
        out_img += self.shift_factor
        out_img = out_img.permute(2, 0, 1).contiguous()
        return out_img

    def forward(self, render_colors=False):
        if self.cur_freq == "kpca":
            return self._forward_kpca_features()
        elif render_colors:
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
