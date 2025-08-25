import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage import color
from pathlib import Path
from PIL import Image
import pickle
import argparse
import tqdm
import shutil
import uuid
import json
import random

# ------------------------- utils -------------------------

def visualize(output_dir: Path):
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    array = np.load(output_dir / "arrs.npy")
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            img = (array[i, j] * 255).astype(np.uint8)
            image = Image.fromarray(img, mode="L")
            image.save(vis_dir / f"{i}-{j}.png")


def _save_set(img_paths, out_dir: Path, img_size, desc: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(tqdm.tqdm(img_paths, desc=desc)):
        img = Image.open(p)
        resized_img = img.resize((img_size[0], img_size[1]), Image.Resampling.LANCZOS)
        resized_img.save(out_dir / f"{i:03d}.png")


def _prepare_arrs_from_train_dir(train_dir: Path, n_comps: int, img_size):
    img_list = sorted([f for f in train_dir.glob("*.png")])
    n_samples = len(img_list)

    train_imgs = []
    for i in tqdm.trange(n_samples, desc="Loading images"):
        img = Image.open(img_list[i])
        img_arr = np.array(img)  # ensure numpy array
        img_arr = color.rgb2ycbcr(img_arr)
        flattened = img_arr.reshape(-1, 3)
        train_imgs.append(flattened)

    # (n_samples, n_features, 3) -> (3, n_samples, n_features)
    train_arrs = np.stack(train_imgs)  # (n_samples, n_features, 3)
    train_arrs = train_arrs.transpose(2, 0, 1)  # (3, n_samples, n_features)

    pca_object = []
    norm_infos = []
    norm_comps = []
    for ch in tqdm.trange(3, desc="Fitting PCA comps"):
        ch_arr = train_arrs[ch]
        norm_arr = ch_arr / 255.0
        pca = PCA(n_components=n_comps, whiten=False)
        _ = pca.fit_transform(norm_arr)

        comps = pca.components_  # (n_comps, n_features)
        global_max = comps.max()
        global_min = comps.min()
        norm_comp = (comps - global_min) / (global_max - global_min + 1e-12)

        pca_object.append(pca)
        norm_infos.append({"min": float(global_min), "max": float(global_max)})
        norm_comps.append(norm_comp)

    norm_comps = np.stack(norm_comps, axis=-1)  # (n_comps, n_features, 3)
    norm_comps = norm_comps.transpose(0, 2, 1)  # (n_comps, 3, n_features)
    norm_comps = norm_comps.reshape(n_comps, 3, img_size[1], img_size[0])  # (n_comps, 3, H, W)
    return pca_object, norm_infos, norm_comps


def _cluster_features_for_image(path: Path, kmeans_size: int):
    """
    Build a compact feature vector for clustering:
    - Load image
    - Resize to (kmeans_size, kmeans_size)
    - Convert to YCbCr and take only Y channel
    - Flatten to 1D
    """
    img = Image.open(path).convert("RGB")
    img_small = img.resize((kmeans_size, kmeans_size), Image.Resampling.BILINEAR)
    arr = np.array(img_small)
    ycbcr = color.rgb2ycbcr(arr) / 255.0
    y = ycbcr[:, :, 0]
    return y.reshape(-1)  # (kmeans_size*kmeans_size,)


# ------------------------- original global mode -------------------------

def prepare_imgs_global(args, output_dir: Path):
    train_dir = output_dir / "train_imgs"
    test_dir = output_dir / "test_imgs"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    img_list = sorted([f for f in args.source.glob("*.png")])
    n_samples = min(args.n_samples, len(img_list))
    n_test = min(args.n_samples + args.n_test, len(img_list))

    counter = 0
    for i in tqdm.trange(args.n_samples, desc="Prepare training images"):
        img = Image.open(img_list[i * 2])
        resized_img = img.resize((args.img_size[0], args.img_size[1]), Image.Resampling.LANCZOS)
        resized_img.save(train_dir / f"{counter:03d}.png")
        counter += 1

    counter = 0
    for i in tqdm.trange(n_samples, n_test, desc="Prepare testing images"):
        img = Image.open(img_list[i * 2])
        resized_img = img.resize((args.img_size[0], args.img_size[1]), Image.Resampling.LANCZOS)
        resized_img.save(test_dir / f"{counter:03d}.png")
        counter += 1


def run_global_pipeline(args, output_dir: Path):
    # exact original behavior
    prepare_imgs_global(args, output_dir)
    pca_object, norm_infos, norm_comps = _prepare_arrs_from_train_dir(
        output_dir / "train_imgs", args.n_comps, args.img_size
    )
    np.save(output_dir / "arrs.npy", norm_comps)
    visualize(output_dir)
    with open(output_dir / "pca_object.pkl", "wb") as FOUT:
        pickle.dump(pca_object, FOUT)
    with open(output_dir / "norm_infos.pkl", "wb") as FOUT:
        pickle.dump(norm_infos, FOUT)


# ------------------------- clustered mode -------------------------

def run_clustered_pipeline(args, output_dir: Path):
    """
    1) Collect all PNGs under source
    2) Build features (downsampled Y channel) and run KMeans (hard assignment)
    3) For each cluster:
       - split train/test by ratio
       - write cluster subdirs with train/test images resized
       - run PCA pipeline per-cluster and save artifacts
    4) Save cluster_assignments.json at the top level
    """
    img_list = sorted([f for f in args.source.glob("*.png")])
    if len(img_list) == 0:
        raise RuntimeError(f"No PNG images found under {args.source}")

    # Build features for KMeans
    feats = []
    for p in tqdm.tqdm(img_list, desc=f"Building features (kmeans_size={args.kmeans_size})"):
        feats.append(_cluster_features_for_image(p, args.kmeans_size))
    X = np.stack(feats, axis=0)

    # KMeans clustering (hard assignment)
    kmeans = KMeans(n_clusters=args.k, n_init=10, random_state=args.seed)
    labels = kmeans.fit_predict(X)

    # Save mapping filename -> cluster id
    mapping = {str(p): int(l) for p, l in zip(img_list, labels)}
    with open(output_dir / "cluster_assignments.json", "w") as f:
        json.dump(mapping, f, indent=2)


    clusters_root = output_dir / "clusters"
    clusters_root.mkdir(parents=True, exist_ok=True)

    # Group paths by cluster
    groups = [[] for _ in range(args.k)]
    for p, l in zip(img_list, labels):
        groups[int(l)].append(p)

    # For deterministic splits
    rng = random.Random(args.seed)

    for cid, paths in enumerate(groups):
        if len(paths) == 0:
            print(f"[cluster {cid}] empty cluster, skipping")
            continue

        # deterministic shuffle
        paths = paths.copy()
        rng.shuffle(paths)

        n_total = len(paths)
        n_test = int(round(args.test_ratio * n_total))
        n_train = max(0, n_total - n_test)
        train_paths = paths[:n_train]
        test_paths = paths[n_train:]

        cdir = clusters_root / f"cluster_{cid:02d}"
        (cdir / "train_imgs").mkdir(parents=True, exist_ok=True)
        (cdir / "test_imgs").mkdir(parents=True, exist_ok=True)

        # Write train/test resized
        _save_set(train_paths, cdir / "train_imgs", args.img_size, desc=f"[C{cid}] train x{n_train}")
        _save_set(test_paths, cdir / "test_imgs", args.img_size, desc=f"[C{cid}] test  x{n_test}")

        # Run PCA per-cluster
        print(f"[C{cid}] PCA over {n_train} training images @ {args.img_size} with {args.n_comps} comps")
        pca_object, norm_infos, norm_comps = _prepare_arrs_from_train_dir(
            cdir / "train_imgs", args.n_comps, args.img_size
        )
        np.save(cdir / "arrs.npy", norm_comps)
        visualize(cdir)

        with open(cdir / "pca_object.pkl", "wb") as FOUT:
            pickle.dump(pca_object, FOUT)
        with open(cdir / "norm_infos.pkl", "wb") as FOUT:
            pickle.dump(norm_infos, FOUT)


# ------------------------- main -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse image set (global PCA or per-cluster PCA).")

    parser.add_argument("-s", "--source", required=True, type=Path,
                        help="Folder with *.png images")
    parser.add_argument("-c", "--n_comps", type=int, default=300)
    parser.add_argument("--img_size", type=int, nargs=2, default=[512, 512],
                        metavar=('width', 'height'),
                        help="Target image size as width height (e.g., 512 512)")

    # ---- GLOBAL mode args (kept as-is for backward compatibility)
    parser.add_argument("-n", "--n_samples", type=int, default=10000,
                        help="(GLOBAL mode) number of training samples (paired selection pattern retained)")
    parser.add_argument("-t", "--n_test", type=int, default=100,
                        help="(GLOBAL mode) number of testing samples after the first n_samples")

    # ---- CLUSTERED mode args
    parser.add_argument("--k", type=int, default=1,
                        help="Number of KMeans clusters. Use k>=2 for clustered PCA mode.")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="(CLUSTERED mode) fraction of images per cluster for test set")
    parser.add_argument("--kmeans_size", type=int, default=96,
                        help="(CLUSTERED mode) downsample side length for KMeans Y-channel features")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for splits/KMeans")

    args = parser.parse_args()

    random_string = str(uuid.uuid4())[:6]
    output_dir = args.source.parent / f"{random_string}-{args.n_comps}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.k is None or args.k <= 1:
        print("==> GLOBAL PCA mode")
        run_global_pipeline(args, output_dir)
    else:
        print(f"==> CLUSTERED PCA mode (k={args.k}, test_ratio={args.test_ratio})")
        run_clustered_pipeline(args, output_dir)

    print(f"\nDone. Outputs under: {output_dir}\n")
