#!/usr/bin/env python3
import argparse, json, math, os
from pathlib import Path
import numpy as np
from PIL import Image
from skimage import color

import torch
import torch.nn.functional as F
from tqdm import tqdm

from gaussianbasis_single_freq import GaussianBasis  # your model


# ------------------------- utils -------------------------

def psnr_from_mse(mse: float) -> float:
    return 10.0 * math.log10(1.0 / (mse + 1e-12))

def load_parse_artifacts(parse_dir: Path):
    """Load metadata and cluster means produced by parse.py."""
    meta = json.load(open(parse_dir / "metadata.json", "r"))
    W, H = int(meta["img_size"][0]), int(meta["img_size"][1])
    C = int(meta["n_comps"])
    K = int(meta["k_clusters"])
    # means: (K,3,D) -> (K,3,H,W)
    cluster_means = np.load(parse_dir / "cluster_pca_means.npy")  # raw means, not normalized comps
    cluster_means_hw = cluster_means.reshape(K, 3, H, W)
    return W, H, C, K, cluster_means_hw

def load_image_ycbcr_01(path: Path, W: int, H: int) -> torch.Tensor:
    """Match parse.py exactly: LANCZOS resize + rgb2ycbcr then /255.0, CHW."""
    img = Image.open(path).resize((W, H), Image.Resampling.LANCZOS)
    ycc = color.rgb2ycbcr(np.array(img)) / 255.0
    return torch.tensor(ycc.transpose(2, 0, 1), dtype=torch.float32)

@torch.no_grad()
def choose_cluster_and_weights_rendered_ls(model: GaussianBasis, K: int, C: int, gt_img_3xHxW: torch.Tensor, ridge_lambda: float = 1e-4):
    """
    Try all clusters; for each, solve ridge LS over the *learned rendered component stack*.
    Return (best_k, best_w[C], best_init_psnr).
    """
    device = next(model.parameters()).device
    y = gt_img_3xHxW.reshape(-1).cpu().numpy()  # (P,)
    best_k, best_psnr, best_w = None, -1e9, None

    for k in range(K):
        Bk = model.forward(cluster_id=k, render_colors=False)   # (C,3,H,W)
        B  = Bk.reshape(C, -1).permute(1, 0).cpu().numpy()      # (P,C)
        mu = model.image_mean_k[k].reshape(-1).cpu().numpy()    # (P,)

        BtB = B.T @ B + (ridge_lambda * np.eye(C, dtype=B.dtype))
        w   = np.linalg.solve(BtB, B.T @ (y - mu)).astype(np.float32)  # (C,)

        recon = mu + B @ w
        mse   = float(np.mean((recon - y) ** 2))
        psnr  = psnr_from_mse(mse)

        if psnr > best_psnr:
            best_psnr, best_k, best_w = psnr, k, w
    return best_k, best_w, best_psnr


def print_phaseA_parameters(model: GaussianBasis, save_to: Path = None):
    """
    Print learned Phase-A parameters (names, shapes, min/mean/max). Optionally save to file.
    """
    lines = []
    lines.append("=== Phase-A learned parameters & buffers ===")
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        try:
            t = tensor.detach()
        except Exception:
            t = tensor
        stats = []
        if t.is_floating_point():
            stats = [f"min={float(t.min()):.5f}", f"mean={float(t.mean()):.5f}", f"max={float(t.max()):.5f}"]
        shape = tuple(t.shape)
        lines.append(f"{name:35s} shape={shape!s:18s} " + (" ".join(stats)))
    text = "\n".join(lines)
    print(text)
    if save_to:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_to.write_text(text)


# ------------------------- main Phase-B runner -------------------------

def main():
    ap = argparse.ArgumentParser("Batch Phase-B on a trained Phase-A model")
    ap.add_argument("-d", "--dataset", required=True, type=Path,
                    help="Path to parse.py output dir (has metadata.json, cluster_pca_means.npy)")
    ap.add_argument("--phaseA_ckpt", required=True, type=Path,
                    help="Path to Phase-A checkpoint (gaussian_model_phaseA.pth.tar)")
    ap.add_argument("--test_dir", required=True, type=Path,
                    help="Folder with test images to fit (e.g., <dataset>/test_imgs)")
    ap.add_argument("--iters", type=int, default=10000, help="Phase-B iterations per image")
    ap.add_argument("--save_every", type=int, default=1000, help="Save RGB every N iterations")
    ap.add_argument("--lr", type=float, default=1e-3, help="Phase-B learning rate")
    ap.add_argument("--ridge", type=float, default=1e-4, help="Ridge lambda for LS init")
    ap.add_argument("--opt", type=str, default="adam", choices=["adam", "adan"], help="Optimizer to use in Phase-B")
    ap.add_argument("--out_dir", type=Path, default=Path("./phaseB_batch_outputs"))
    ap.add_argument("--freeze_geom_warmup", type=int, default=0, help="Freeze xyz/cholesky for first N iters")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- load parse artifacts ----
    W, H, C, K, cluster_means_hw = load_parse_artifacts(args.dataset)

    # ---- load Phase-A checkpoint & infer num_points ----
    state = torch.load(args.phaseA_ckpt, map_location="cpu")
    sd = state["model_state_dict"]
    if "_xyz" not in sd:
        raise RuntimeError("Checkpoint does not contain _xyz — is this a Phase-A checkpoint?")
    num_points = sd["_xyz"].shape[0]

    # ---- build model & load Phase-A weights ----
    model = GaussianBasis(
        loss_type="L2",
        opt_type=args.opt,
        num_points=num_points,
        H=H, W=W, BLOCK_H=16, BLOCK_W=16,
        device=device, lr=args.lr, num_comps=C, num_clusters=K
    ).to(device)
    model.load_state_dict(sd)

    # ---- load cluster means into model ----
    if hasattr(model, "load_cluster_means"):
        model.load_cluster_means(cluster_means_hw, from_flat=False)
    else:
        means_t = torch.tensor(cluster_means_hw, dtype=torch.float32, device=device)
        if not hasattr(model, "image_mean_k"):
            model.register_buffer("image_mean_k", means_t)
        else:
            with torch.no_grad():
                model.image_mean_k.copy_(means_t)

    # ---- print learned Phase-A parameters ----
    print_phaseA_parameters(model, save_to=args.out_dir / "phaseA_params.txt")

    # ---- make output dir ----
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- iterate test images ----
    img_paths = sorted([p for p in Path(args.test_dir).glob("*.png")])
    if not img_paths:
        raise FileNotFoundError(f"No .png images found in {args.test_dir}")

    for ipath in img_paths:
        print(f"\n=== Phase-B on: {ipath.name} ===")
        gt = load_image_ycbcr_01(ipath, W, H).to(device)  # (3,H,W)

        # choose cluster & init colors via rendered LS
        with torch.no_grad():
            k, w, init_psnr = choose_cluster_and_weights_rendered_ls(model, K, C, gt, ridge_lambda=args.ridge)
            print(f"[Init] cluster={k}, init-PSNR≈{init_psnr:.2f} dB")

            # initialize colors from single weight vector
            if hasattr(model, "init_colors_from_single_weight"):
                model.init_colors_from_single_weight(k, w)
            else:
                feats = model._features_dc[k]  # (C,N,3)
                w_t = torch.tensor(w, dtype=feats.dtype, device=feats.device).view(-1,1,1)
                colors = (w_t * feats).sum(dim=0)  # (N,3)
                model._colors.copy_(colors.clamp_(0.0, 1.0))

        # setup optimizer for Phase-B
        model.lr = args.lr
        model.scheduler_init(optimize_phase=True)

        # optional warmup: freeze geometry first N iters
        warmup = max(0, int(args.freeze_geom_warmup))
        if warmup > 0:
            model._xyz.requires_grad_(False)
            model._cholesky.requires_grad_(False)
            model.scheduler_init(optimize_phase=True)  # rebuild over current trainables only

        # per-image out dir
        img_out = args.out_dir / ipath.stem
        img_out.mkdir(parents=True, exist_ok=True)

        # Phase-B loop
        model.train()
        pbar = tqdm(range(1, args.iters + 1), desc=f"Phase-B [{ipath.stem}]")
        for it in pbar:
            # forward
            color_field = model.forward(cluster_id=k, render_colors=True)  # (3,H,W)
            pred = color_field + model.image_mean_k[k]
            loss = F.mse_loss(pred, gt)

            # backward/step
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
            model.scheduler.step()

            # unfreeze geometry after warmup
            if it == warmup and warmup > 0:
                model._xyz.requires_grad_(True)
                model._cholesky.requires_grad_(True)
                model.scheduler_init(optimize_phase=True)

            # metrics (post-step)
            with torch.no_grad():
                color_field_after = model.forward(cluster_id=k, render_colors=True)
                pred_after = color_field_after + model.image_mean_k[k]
                mse = float(F.mse_loss(pred_after, gt).item())
                psnr = psnr_from_mse(mse)

            pbar.set_postfix({"k": k, "Loss": f"{loss.item():.3e}", "PSNR": f"{psnr:.2f} dB"})

            # save preview every N iters
            if it % args.save_every == 0 or it == args.iters or it == 10 or it == 100 or it == 500:
                # ycbcr = (pred_after.detach().cpu().numpy().transpose(1, 2, 0) * 255.0)
                # rgb = color.ycbcr2rgb(ycbcr).clip(0, 255).astype(np.uint8)
                # Image.fromarray(rgb).save(str(img_out / f"{ipath.stem}_it{it}.png"))
                ycbcr_img = (pred_after.detach().cpu().numpy() * 255.0).transpose(1, 2, 0)  # (H,W,3)
                rgb_img = color.ycbcr2rgb(ycbcr_img) * 255.0
                rgb_img = rgb_img.clip(0, 255).astype(np.uint8)
                img = Image.fromarray(rgb_img).save(str(img_out / f"{ipath.stem}_it{it}.png"))

        # save per-image Phase-B checkpoint (optional)
        torch.save({"model_state_dict": model.state_dict()}, img_out / "gaussian_model_phaseB.pth.tar")
        print(f"[Done] {ipath.name}: final PSNR={psnr:.2f} dB | saved previews to {img_out}")

if __name__ == "__main__":
    main()
