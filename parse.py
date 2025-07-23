import numpy as np
from sklearn.decomposition import PCA
from skimage import color
from pathlib import Path
from PIL import Image
import pickle
import argparse
import tqdm
import shutil
import uuid
import torch
from testEM2 import EMLikeAlg, computeCost

def visualize(output_dir):
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    array = np.load(output_dir / "arrs.npy")
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            img = (array[i, j] * 255).astype(np.uint8)
            image = Image.fromarray(img, mode="L")
            image.save(vis_dir / f"{i}-{j}.png")

def prepare_imgs(args, output_dir):
    train_dir = output_dir / "train_imgs"
    test_dir = output_dir / "test_imgs"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    img_list = sorted([f for f in args.source.glob("*.png")])
    n_samples = min(args.n_samples, len(img_list))
    n_test = min(args.n_samples + args.n_test, len(img_list))

    counter = 0
    for i in tqdm.trange(args.n_samples, desc="Prepare training images"):
        img = Image.open(img_list[i*2])
        resized_img = img.resize((args.img_size[0], args.img_size[1]), Image.Resampling.LANCZOS)
        resized_img.save(train_dir / f"{counter:03d}.png")
        counter += 1
    
    counter = 0
    for i in tqdm.trange(n_samples, n_test, desc="Prepare testing images"):
        img = Image.open(img_list[i*2])
        resized_img = img.resize((args.img_size[0], args.img_size[1]), Image.Resampling.LANCZOS)
        resized_img.save(test_dir / f"{counter:03d}.png")
        counter += 1

def prepare_arrs(args, output_dir):
    train_dir = output_dir / "train_imgs"
    img_list = sorted([f for f in train_dir.glob("*.png")])
    n_samples = len(img_list)
    
    train_imgs = []
    for i in tqdm.trange(n_samples, desc="Loading images"):
        img = Image.open(img_list[i])
        img_arr = color.rgb2ycbcr(img)
        flattened = img_arr.reshape(-1, 3)
        train_imgs.append(flattened)
    
    train_arrs = np.stack(train_imgs)  # (n_samples, n_features, 3)
    train_arrs = train_arrs.transpose(2, 0, 1)  # (3, n_samples, n_features)

    pca_object = []
    norm_infos = []
    norm_comps = []
    for i in tqdm.trange(3, desc="Fitting PCA comps"):
        ch_arr = train_arrs[i]
        norm_arr = ch_arr / 255.0
        pca = PCA(n_components=args.n_comps, whiten=False)
        pca.fit_transform(norm_arr)
        
        comps = pca.components_
        global_max = comps.max()
        global_min = comps.min()
        norm_comp = (comps - global_min) / (global_max - global_min)

        pca_object.append(pca)
        norm_infos.append({
            "min": global_min,
            "max": global_max
        })
        norm_comps.append(norm_comp)
    norm_comps = np.stack(norm_comps, axis=-1)
    norm_comps = norm_comps.transpose(0, 2, 1)
    norm_comps = norm_comps.reshape(args.n_comps, 3, args.img_size[1], args.img_size[0])
    np.save(output_dir / "arrs.npy", norm_comps)
    
    visualize(output_dir)
    return pca_object, norm_infos


import numpy as np
import tqdm
from PIL import Image
from skimage import color
from testEM2 import EMLikeAlg, computeCost

def prepare_clustered_pca(args, output_dir):
    train_dir = output_dir / "train_imgs"
    img_list = sorted(train_dir.glob("*.png"))

    flattened_images = []
    for path in tqdm.tqdm(img_list, desc="Loading images"):
        img = Image.open(path).resize(tuple(args.img_size))
        ycbcr = color.rgb2ycbcr(np.array(img)) / 255.0
        y_channel = ycbcr[:, :, 0]  # Extract luminance (Y)
        flat = y_channel.reshape(-1)  # Flatten into (H * W,)
        flattened_images.append(flat)

    # Stack into (N_images, D)
    P = np.stack(flattened_images, axis=0).astype(np.float32)
    w = np.ones(P.shape[0], dtype=np.float32)  # uniform weights

    print(f"Running EM-like algorithm for (k={args.k_clusters}, j={args.subspace_dim})")
    Vs, _ = EMLikeAlg(
        P=P,
        w=w,
        j=args.subspace_dim,
        k=args.k_clusters,
        steps=8,
        NUM_INIT_FOR_EM=1
    )

    # Compute cluster assignments
    _, _, cluster_ids = computeCost(P, w, Vs, show_indices=True)

    # Compute cluster means
    means = []
    for c in range(args.k_clusters):
        cluster_points = P[cluster_ids == c]
        if len(cluster_points) == 0:
            print(f"⚠️ Warning: cluster {c} is empty!")
            means.append(np.zeros(P.shape[1], dtype=np.float32))
        else:
            means.append(cluster_points.mean(axis=0))
    means = np.stack(means, axis=0)  # (C, D)

    # Save everything
    np.save(output_dir / "cluster_pcas.npy", Vs)
    np.save(output_dir / "cluster_means.npy", means)
    np.save(output_dir / "cluster_ids.npy", cluster_ids)
    print("✅ Clustered subspaces, means, and assignments saved.")
def prepare_clustered_pca(args, output_dir):
    train_dir = output_dir / "train_imgs"
    img_list = sorted(train_dir.glob("*.png"))

    flattened_images = []
    for path in tqdm.tqdm(img_list, desc="Loading images"):
        img = Image.open(path).resize(tuple(args.img_size))
        ycbcr = color.rgb2ycbcr(np.array(img)) / 255.0
        y_channel = ycbcr[:, :, 0]  # take Y only
        flat = torch.tensor(y_channel, dtype=torch.float32, device='cuda').reshape(-1)
        flattened_images.append(flat)

    # Already on GPU
    P = torch.stack(flattened_images, dim=0)  # (N_images, D)
    w = torch.ones(len(P), dtype=torch.float32, device='cuda')  # put weights on GPU

    print(f"Running EM-like algorithm for (k={args.k_clusters}, j={args.subspace_dim})")
    Vs, _ = EMLikeAlg(
        P_np=P,  # already torch tensor on GPU
        j=args.subspace_dim,
        k=args.k_clusters,
        steps=20,
        num_init=5,
        max_points=500000
    )

    # Now keep everything on GPU
    Vs_tensor = torch.tensor(Vs, dtype=torch.float32, device='cuda')
    _, _, cluster_ids = computeCost(P, w, Vs_tensor, show_indices=True)

    means = []
    for c in range(args.k_clusters):
        cluster_points = P[cluster_ids == c]
        if len(cluster_points) == 0:
            print(f"Warning: cluster {c} is empty!")
            means.append(torch.zeros(P.shape[1], device='cuda'))
        else:
            means.append(cluster_points.mean(dim=0))
    means = torch.stack(means, dim=0)

    # Save to disk as numpy (must move to CPU first)
    np.save(output_dir / "cluster_pcas.npy", Vs_tensor.cpu().numpy())
    np.save(output_dir / "cluster_means.npy", means.cpu().numpy())
    np.save(output_dir / "cluster_ids.npy", cluster_ids.cpu().numpy())

    print("Saved clustered EM bases and means for flattened image patches.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse image set")
    parser.add_argument("-s", "--source", required=True, type=Path)
    parser.add_argument("-c", "--subspace_dim", type=int, default=300)
    parser.add_argument("--k_clusters", type=int, default=4)
    parser.add_argument("-n", "--n_samples", type=int, default=10000)
    parser.add_argument("-t", "--n_test", type=int, default=100)
    parser.add_argument(
        "--img_size", type=int, nargs=2, 
        default=[512, 512], metavar=('width', 'height'),
        help="Target image size as width height (e.g., 512 512)"
    )

    
    args = parser.parse_args()
    random_string = str(uuid.uuid4())[:6]
    output_dir = args.source.parent / f"{random_string}-{args.subspace_dim}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prepare_imgs(args, output_dir)
    prepare_clustered_pca(args, output_dir)

    # pca_object, norm_infos = prepare_arrs(args, output_dir)
    
    # with open(output_dir / "pca_object.pkl", "wb") as FOUT:
    #     pickle.dump(pca_object, FOUT)
    # with open(output_dir / "norm_infos.pkl", "wb") as FOUT:
    #     pickle.dump(norm_infos, FOUT)