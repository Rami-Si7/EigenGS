import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum
from optimizer import Adan  # custom optimizer class you're using
from utils import loss_fn  # assumes your project has a utils.py with loss_fn defined
class GaussianBasis(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.C = kwargs["num_clusters"]
        self.j = kwargs["subspace_dim"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )
        self.device = kwargs["device"]

        # Frozen buffers (precomputed from parse.py)
        self.register_buffer("cluster_means", kwargs["cluster_means"].to(self.device))     # (C, 3)
        self.register_buffer("cluster_bases", kwargs["cluster_bases"].to(self.device))     # (C, j, 2)

        # Learnable parameters
        self.alpha_logits = nn.Parameter(torch.randn(self.init_num_points, self.C))       # (N, C)
        self.w_proj = nn.Parameter(torch.randn(self.init_num_points, self.C, self.j))     # (N, C, j)
        self.psi_weights = nn.Parameter(torch.randn(self.init_num_points, self.C, self.j))# (M=N, C, j)

        self.xyz = nn.Parameter(torch.tanh(torch.randn(self.init_num_points, 2)))         # (N, 2)
        self.cholesky = nn.Parameter(torch.rand(self.init_num_points, 3))                 # (N, 3)
        self._opacity = nn.Parameter(torch.ones((self.init_num_points, 1)))

        self.opt_type = kwargs["opt_type"]
        self.lr = kwargs["lr"]

    def get_soft_features(self):
        alpha = torch.softmax(self.alpha_logits, dim=1)            # (N, C)
        soft_part = torch.matmul(alpha, self.cluster_means)        # (N, 3)

        detail_part = torch.zeros(self.init_num_points, 2, device=self.device)

        for c in range(self.C):
            V_c = self.cluster_bases[c]                     # (j, 2)
            w_nc = self.w_proj[:, c, :]                     # (N, j)
            projection = w_nc @ V_c                         # (N, 2)
            detail_part += alpha[:, c:c+1] * projection     # (N, 2)

        return soft_part, detail_part

    def get_gaussian_feature_basis(self):
        # Computes ∑ₙ f̃ₙ ≈ ∑ₘ dₙ,ₘ * exp(−σₘ)
        N = self.init_num_points
        d_matrix = torch.zeros(N, N, device=self.device)  # d[n, m]

        alpha = torch.softmax(self.alpha_logits, dim=1)    # (N, C)

        for c in range(self.C):
            w_nc = self.w_proj[:, c, :]        # (N, j)
            psi_mc = self.psi_weights[:, c, :] # (N, j)
            for l in range(self.j):
                d_matrix += torch.outer(alpha[:, c] * w_nc[:, l], psi_mc[:, l])  # (N, N)

        return d_matrix  # used for basis rendering

    def forward(self, render_colors=False):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(
            self.xyz, self.cholesky, self.H, self.W, self.tile_bounds
        )

        # Step 1: get f̃ₙ
        soft_color, soft_feature_2d = self.get_soft_features()      # (N, 3), (N, 2)
        # Optional: add soft_feature_2d into color channels if 2D projection is spatial

        # Step 2: get basis part
        d_matrix = self.get_gaussian_feature_basis()               # (N, N)

        # Step 3: blend it all
        features = soft_color  # You could add other channels if needed

        out_img = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            features, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=torch.ones(3), return_alpha=False
        )
        return out_img.permute(2, 0, 1).contiguous()

    def scheduler_init(self):
        params = [p for p in self.parameters() if p.requires_grad]
        if self.opt_type == "adam":
            self.optimizer = torch.optim.Adam(params, lr=self.lr)
        else:
            self.optimizer = Adan(params, lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    def train_iter(self, gt_image):
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