import numpy as np
from sklearn.decomposition import PCA
from skimage import color
from pathlib import Path
from PIL import Image
import pickle
import argparse
import tqdm
import uuid

def visualize(output_dir: Path):
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    array = np.load(output_dir / "arrs.npy")
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            img = (array[i, j] * 255).astype(np.uint8)
            Image.fromarray(img, mode="L").save(vis_dir / f"{i}-{j}.png")

def _gather_images(root: Path, exts=(".png",)):
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()])

def prepare_imgs(args, output_dir: Path):
    """
    Creates output_dir/{train_imgs,test_imgs} with resized copies.

    Modes:
      1) DEFAULT (original): sample from --source
         - train: first args.n_samples images (using the original i*2 stepping)
         - test : next args.n_test images (using the same stepping)

      2) EXPLICIT TRAIN/TEST:
         - If --train_from is given: use ALL (or first n_samples if provided) images from that folder for TRAIN.
         - If --test_from  is given: use ALL (or first n_test if provided) images from that folder for TEST.
         - If --train_from is given and --test_from is NOT: test set will be empty.
    """
    W, H = args.img_size
    train_dir = output_dir / "train_imgs"
    test_dir  = output_dir / "test_imgs"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # ---------- TRAIN ----------
    if args.train_from is not None:
        # Use provided folder for training
        train_src = _gather_images(args.train_from)
        if args.n_samples is not None and args.n_samples > 0:
            train_src = train_src[:min(args.n_samples, len(train_src))]
        print(f"[train_from] Using {len(train_src)} images from: {args.train_from}")
        counter = 0
        for p in tqdm.tqdm(train_src, desc="Prepare training images (train_from)"):
            img = Image.open(p).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
            img.save(train_dir / f"{counter:06d}.png")
            counter += 1
    else:
        # Original behavior: sample from --source
        img_list = sorted([f for f in args.source.glob("*.png")])
        n_samples = min(args.n_samples, len(img_list))
        counter = 0
        for i in tqdm.trange(n_samples, desc="Prepare training images (source)"):
            # keep the original stride-by-2 pattern
            src_idx = min(i * 2, len(img_list) - 1)
            img = Image.open(img_list[src_idx]).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
            img.save(train_dir / f"{counter:06d}.png")
            counter += 1

    # ---------- TEST ----------
    if args.test_from is not None:
        test_src = _gather_images(args.test_from)
        if args.n_test is not None and args.n_test > 0:
            test_src = test_src[:min(args.n_test, len(test_src))]
        print(f"[test_from] Using {len(test_src)} images from: {args.test_from}")
        counter = 0
        for p in tqdm.tqdm(test_src, desc="Prepare testing images (test_from)"):
            img = Image.open(p).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
            img.save(test_dir / f"{counter:06d}.png")
            counter += 1
    elif args.train_from is None:
        # Only do the original test split if we also used --source for train
        img_list = sorted([f for f in args.source.glob("*.png")])
        n_samples = min(args.n_samples, len(img_list))
        n_test = min(args.n_samples + args.n_test, len(img_list))
        counter = 0
        for i in tqdm.trange(n_samples, n_test, desc="Prepare testing images (source)"):
            src_idx = min(i * 2, len(img_list) - 1)
            img = Image.open(img_list[src_idx]).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
            img.save(test_dir / f"{counter:06d}.png")
            counter += 1
    else:
        # train_from provided, but no test_from -> leave test empty (allowed)
        print("[info] --train_from provided without --test_from; leaving test_imgs empty.")

def prepare_arrs(args, output_dir: Path):
    """
    Load all images from output_dir/train_imgs, convert to YCbCr in [0,1],
    fit 3 separate PCAs (Y, Cb, Cr), and save normalized components to arrs.npy.
    """
    train_dir = output_dir / "train_imgs"
    img_list = sorted([f for f in train_dir.glob("*.png")])
    n_samples = len(img_list)
    assert n_samples > 0, "No training images found in train_imgs/."

    # (3, N, d)
    train_imgs = []
    for i in tqdm.trange(n_samples, desc="Loading images"):
        im = Image.open(img_list[i]).convert("RGB")
        ycbcr = color.rgb2ycbcr(np.asarray(im)).astype(np.float32)  # (H,W,3) in 0..255
        flat = ycbcr.reshape(-1, 3)                                 # (d,3)
        train_imgs.append(flat)
    train_arrs = np.stack(train_imgs)              # (N, d, 3)
    train_arrs = train_arrs.transpose(2, 0, 1)     # (3, N, d)

    pca_object = []
    norm_infos = []
    norm_comps = []

    for ch in tqdm.trange(3, desc="Fitting PCA comps"):
        ch_arr = train_arrs[ch] / 255.0            # (N, d) in [0,1]
        pca = PCA(n_components=args.n_comps, whiten=False)
        pca.fit(ch_arr)                            # fit (no need to transform for components)

        comps = pca.components_.astype(np.float32) # (n_comps, d)
        gmax  = float(comps.max())
        gmin  = float(comps.min())
        den   = (gmax - gmin) if gmax != gmin else 1.0
        norm_comp = (comps - gmin) / den           # normalized just for visualization

        pca_object.append(pca)
        norm_infos.append({"min": gmin, "max": gmax})
        norm_comps.append(norm_comp)

    norm_comps = np.stack(norm_comps, axis=-1)     # (n_comps, d, 3)
    W, H = args.img_size
    norm_comps = norm_comps.transpose(0, 2, 1)     # (n_comps, 3, d)
    norm_comps = norm_comps.reshape(args.n_comps, 3, H, W)
    np.save(output_dir / "arrs.npy", norm_comps)
    visualize(output_dir)
    return pca_object, norm_infos

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse image set (Global PCA)")

    # Original args
    parser.add_argument("-s", "--source", required=True, type=Path,
                        help="Folder with *.png (used when --train_from/--test_from not given)")
    parser.add_argument("-c", "--n_comps", type=int, default=300)
    parser.add_argument("-n", "--n_samples", type=int, default=10000)
    parser.add_argument("-t", "--n_test", type=int, default=100)
    parser.add_argument("--img_size", type=int, nargs=2, default=[512, 512],
                        metavar=('width','height'),
                        help="Target image size (e.g., 512 512)")

    # NEW: explicit training/testing sources
    parser.add_argument("--train_from", type=Path, default=None,
                        help="If set, take TRAIN images from this folder (recursively). "
                             "Resized copies will be written to output_dir/train_imgs. "
                             "If set, overrides sampling from --source.")
    parser.add_argument("--test_from", type=Path, default=None,
                        help="Optional: take TEST images from this folder (recursively). "
                             "If not set and --train_from is given, test set will be empty.")

    args = parser.parse_args()

    # Output folder naming stays the same
    random_string = str(uuid.uuid4())[:6]
    output_dir = args.source.parent / f"{random_string}-{args.n_comps}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build train_imgs/ and test_imgs/
    prepare_imgs(args, output_dir)

    # 2) Fit global PCA from train_imgs/
    pca_object, norm_infos = prepare_arrs(args, output_dir)

    # 3) Save artifacts
    with open(output_dir / "pca_object.pkl", "wb") as f:
        pickle.dump(pca_object, f)
    with open(output_dir / "norm_infos.pkl", "wb") as f:
        pickle.dump(norm_infos, f)

    print(f"[done] Global PCA saved to: {output_dir}")