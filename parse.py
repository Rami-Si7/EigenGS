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


# import numpy as np
# import tqdm
# from PIL import Image
# from skimage import color
# from testEM2 import EMLikeAlg, computeCost

# def prepare_clustered_pca(args, output_dir):
#     train_dir = output_dir / "train_imgs"
#     img_list = sorted(train_dir.glob("*.png"))

#     flattened_images = []
#     for path in tqdm.tqdm(img_list, desc="Loading images"):
#         img = Image.open(path).resize(tuple(args.img_size))
#         ycbcr = color.rgb2ycbcr(np.array(img)) / 255.0
#         y_channel = ycbcr[:, :, 0]  # Extract luminance (Y)
#         flat = y_channel.reshape(-1)  # Flatten into (H * W,)
#         flattened_images.append(flat)

#     # Stack into (N_images, D)
#     P = np.stack(flattened_images, axis=0).astype(np.float32)
#     w = np.ones(P.shape[0], dtype=np.float32)  # uniform weights

#     print(f"Running EM-like algorithm for (k={args.k_clusters}, j={args.subspace_dim})")
#     Vs, _ = EMLikeAlg(
#         P=P,
#         w=w,
#         j=args.subspace_dim,
#         k=args.k_clusters,
#         steps=8,
#         NUM_INIT_FOR_EM=1
#     )

#     # Compute cluster assignments
#     _, _, cluster_ids = computeCost(P, w, Vs, show_indices=True)

#     # Compute cluster means
#     means = []
#     for c in range(args.k_clusters):
#         cluster_points = P[cluster_ids == c]
#         if len(cluster_points) == 0:
#             print(f"⚠️ Warning: cluster {c} is empty!")
#             means.append(np.zeros(P.shape[1], dtype=np.float32))
#         else:
#             means.append(cluster_points.mean(axis=0))
#     means = np.stack(means, axis=0)  # (C, D)

#     # Save everything
#     np.save(output_dir / "cluster_pcas.npy", Vs)
#     np.save(output_dir / "cluster_means.npy", means)
#     np.save(output_dir / "cluster_ids.npy", cluster_ids)
#     print("✅ Clustered subspaces, means, and assignments saved.")
# def prepare_clustered_pca(args, output_dir):
#     train_dir = output_dir / "train_imgs"
#     img_list = sorted(train_dir.glob("*.png"))

#     flattened_images = []
#     for path in tqdm.tqdm(img_list, desc="Loading images"):
#         img = Image.open(path).resize(tuple(args.img_size))
#         ycbcr = color.rgb2ycbcr(np.array(img)) / 255.0
#         y_channel = ycbcr[:, :, 0]  # take Y only
#         flat = torch.tensor(y_channel, dtype=torch.float32, device='cuda').reshape(-1)
#         flattened_images.append(flat)

#     # Already on GPU
#     P = torch.stack(flattened_images, dim=0)  # (N_images, D)
#     w = torch.ones(len(P), dtype=torch.float32, device='cuda')  # put weights on GPU

#     print(f"Running EM-like algorithm for (k={args.k_clusters}, j={args.subspace_dim})")
#     Vs, _ = EMLikeAlg(
#         P_np=P,  # already torch tensor on GPU
#         j=args.subspace_dim,
#         k=args.k_clusters,
#         steps=10,
#         num_init=2,
#         max_points=500000
#     )

#     # Now keep everything on GPU
#     Vs_tensor = torch.tensor(Vs, dtype=torch.float32, device='cuda')
#     _, _, cluster_ids = computeCost(P, w, Vs_tensor, show_indices=True)

#     means = []
#     for c in range(args.k_clusters):
#         cluster_points = P[cluster_ids == c]
#         if len(cluster_points) == 0:
#             print(f"Warning: cluster {c} is empty!")
#             means.append(torch.zeros(P.shape[1], device='cuda'))
#         else:
#             means.append(cluster_points.mean(dim=0))
#     means = torch.stack(means, dim=0)

#     # Save to disk as numpy (must move to CPU first)
#     np.save(output_dir / "cluster_pcas.npy", Vs_tensor.cpu().numpy())
#     np.save(output_dir / "cluster_means.npy", means.cpu().numpy())
#     np.save(output_dir / "cluster_ids.npy", cluster_ids.cpu().numpy())

#     print("Saved clustered EM bases and means for flattened image patches.")
# import numpy as np
# from pathlib import Path
# from PIL import Image
# from skimage import color
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# import tqdm

# def prepare_clustered_arrs(args, output_dir):
#     output_dir = Path(output_dir)
#     train_dir = output_dir / "train_imgs"
#     img_list = sorted([f for f in train_dir.glob("*.png")])
#     n_samples = len(img_list)

#     # Load and flatten all images
#     flat_Y, flat_Cb, flat_Cr = [], [], []
#     for path in tqdm.tqdm(img_list, desc="Loading images"):
#         img = Image.open(path).resize(tuple(args.img_size))
#         ycbcr = color.rgb2ycbcr(np.array(img)) / 255.0
#         flat_Y.append(ycbcr[:, :, 0].reshape(-1))
#         flat_Cb.append(ycbcr[:, :, 1].reshape(-1))
#         flat_Cr.append(ycbcr[:, :, 2].reshape(-1))

#     Y = np.stack(flat_Y)   # (N, D)
#     Cb = np.stack(flat_Cb)
#     Cr = np.stack(flat_Cr)

#     print(f"Running KMeans clustering on Y channel with k={args.k_clusters}")
#     kmeans = KMeans(n_clusters=args.k_clusters, n_init=15, random_state=42)
#     cluster_ids = kmeans.fit_predict(Y)

#     # For each cluster, do PCA per channel
#     cluster_pcas = []  # shape: (k, 3, n_comps, D)
#     norm_infos = []    # for min/max normalization

#     for c in range(args.k_clusters):
#         indices = np.where(cluster_ids == c)[0]
#         if len(indices) < args.n_comps:
#             print(f"Skipping cluster {c}: too small.")
#             cluster_pcas.append(np.zeros((3, args.n_comps, Y.shape[1])))
#             norm_infos.append([{"min": 0, "max": 1}] * 3)
#             continue

#         comps_per_channel = []
#         norm_info_per_channel = []
#         for channel_data in [Y, Cb, Cr]:
#             data = channel_data[indices]
#             pca = PCA(n_components=args.n_comps, whiten=False)
#             pca.fit(data)
#             comps = pca.components_
#             gmin = comps.min()
#             gmax = comps.max()
#             comps_norm = (comps - gmin) / (gmax - gmin)

#             comps_per_channel.append(comps_norm)
#             norm_info_per_channel.append({"min": gmin, "max": gmax})

#         cluster_pcas.append(np.stack(comps_per_channel))  # (3, n_comps, D)
#         norm_infos.append(norm_info_per_channel)

#     cluster_pcas = np.stack(cluster_pcas)  # (k, 3, n_comps, D)

#     np.save(output_dir / "cluster_ids.npy", cluster_ids)
#     np.save(output_dir / "cluster_pcas.npy", cluster_pcas)
#     with open(output_dir / "norm_infos.pkl", "wb") as fout:
#         import pickle
#         pickle.dump(norm_infos, fout)

#     print("✅ Saved clustered PCA components per channel.")
import numpy as np
from pathlib import Path
from PIL import Image
from skimage import color
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import tqdm
import pickle

def prepare_clustered_arrs(args, output_dir):
    output_dir = Path(output_dir)
    train_dir = output_dir / "train_imgs"
    img_list = sorted([f for f in train_dir.glob("*.png")])
    n_samples = len(img_list)

    # Load and flatten all images per channel
    flat_Y, flat_Cb, flat_Cr = [], [], []
    for path in tqdm.tqdm(img_list, desc="Loading images"):
        img = Image.open(path).resize(tuple(args.img_size))
        ycbcr = color.rgb2ycbcr(np.array(img)) / 255.0
        flat_Y.append(ycbcr[:, :, 0].reshape(-1))   # Y
        flat_Cb.append(ycbcr[:, :, 1].reshape(-1))  # Cb
        flat_Cr.append(ycbcr[:, :, 2].reshape(-1))  # Cr

    # Convert to numpy arrays (N, D)
    Y = np.stack(flat_Y)
    Cb = np.stack(flat_Cb)
    Cr = np.stack(flat_Cr)

    # 🔄 Combine all channels to form a single vector per image
    YCC = np.concatenate([Y, Cb, Cr], axis=1)  # (N, 3D)

    # 🔵 Run KMeans on combined channel info
    print(f"Running KMeans clustering on Y+Cb+Cr with k={args.k_clusters}")
    kmeans = KMeans(n_clusters=args.k_clusters, n_init=15, random_state=42)
    cluster_ids = kmeans.fit_predict(YCC)

    # 🧠 Compute PCA per channel for each cluster
    cluster_pcas = []  # shape: (k, 3, n_comps, D)
    norm_infos = []

    for c in range(args.k_clusters):
        indices = np.where(cluster_ids == c)[0]
        if len(indices) < args.n_comps:
            print(f"Skipping cluster {c}: too small.")
            cluster_pcas.append(np.zeros((3, args.n_comps, Y.shape[1])))
            norm_infos.append([{"min": 0, "max": 1}] * 3)
            continue

        comps_per_channel = []
        norm_info_per_channel = []
        for channel_data in [Y, Cb, Cr]:
            data = channel_data[indices]
            pca = PCA(n_components=args.n_comps, whiten=False)
            pca.fit(data)
            comps = pca.components_
            gmin = comps.min()
            gmax = comps.max()
            comps_norm = (comps - gmin) / (gmax - gmin)

            comps_per_channel.append(comps_norm)
            norm_info_per_channel.append({"min": gmin, "max": gmax})

        cluster_pcas.append(np.stack(comps_per_channel))  # (3, n_comps, D)
        norm_infos.append(norm_info_per_channel)


# --- keep everything above as-is ---

    cluster_pcas = np.stack(cluster_pcas)  # (k, 3, n_comps, D)

    # ✅ Save existing outputs (unchanged)
    np.save(output_dir / "cluster_ids.npy", cluster_ids)
    np.save(output_dir / "cluster_pcas.npy", cluster_pcas)
    with open(output_dir / "norm_infos.pkl", "wb") as fout:
        pickle.dump(norm_infos, fout)

    # ===== NEW: Save raw PCA components, means, EV/EVR, sizes, kmeans, metadata =====
    # We recompute light-weight stats while we still have pca objects in-scope
    # To avoid touching your existing logic above, we compute raw artifacts again cleanly.

    # Re-run per-cluster per-channel PCA to collect RAW comps & means (no normalization)
    print("🔎 Collecting raw PCA components/means/variance per cluster (added artifacts).")
    cluster_pcas_raw = []
    cluster_means = []
    cluster_expl_var = []
    cluster_expl_var_ratio = []
    cluster_sizes = []

    for c in range(args.k_clusters):
        indices = np.where(cluster_ids == c)[0]
        cluster_sizes.append(len(indices))

        comps_raw_per_ch = []
        means_per_ch = []
        ev_per_ch = []
        evr_per_ch = []

        if len(indices) < args.n_comps:
            # keep shapes consistent
            D = Y.shape[1]
            comps_raw_per_ch = [np.zeros((args.n_comps, D), dtype=np.float32) for _ in range(3)]
            means_per_ch = [np.zeros((D,), dtype=np.float32) for _ in range(3)]
            ev_per_ch = [np.zeros((args.n_comps,), dtype=np.float32) for _ in range(3)]
            evr_per_ch = [np.zeros((args.n_comps,), dtype=np.float32) for _ in range(3)]
        else:
            for channel_data in [Y, Cb, Cr]:
                data = channel_data[indices]  # (Nc, D)
                pca = PCA(n_components=args.n_comps, whiten=False)
                pca.fit(data)
                comps_raw_per_ch.append(pca.components_.astype(np.float32))
                means_per_ch.append(pca.mean_.astype(np.float32))
                ev_per_ch.append(pca.explained_variance_.astype(np.float32))
                evr_per_ch.append(pca.explained_variance_ratio_.astype(np.float32))

        cluster_pcas_raw.append(np.stack(comps_raw_per_ch))      # (3, n_comps, D)
        cluster_means.append(np.stack(means_per_ch))             # (3, D)
        cluster_expl_var.append(np.stack(ev_per_ch))             # (3, n_comps)
        cluster_expl_var_ratio.append(np.stack(evr_per_ch))      # (3, n_comps)

    cluster_pcas_raw = np.stack(cluster_pcas_raw)                # (k, 3, n_comps, D)
    cluster_means = np.stack(cluster_means)                      # (k, 3, D)
    cluster_expl_var = np.stack(cluster_expl_var)                # (k, 3, n_comps)
    cluster_expl_var_ratio = np.stack(cluster_expl_var_ratio)    # (k, 3, n_comps)

    np.save(output_dir / "cluster_pcas_raw.npy", cluster_pcas_raw)
    np.save(output_dir / "cluster_pca_means.npy", cluster_means)
    np.save(output_dir / "cluster_explained_var.npy", cluster_expl_var)
    np.save(output_dir / "cluster_explained_var_ratio.npy", cluster_expl_var_ratio)
    np.save(output_dir / "cluster_sizes.npy", np.array(cluster_sizes, dtype=np.int32))

    # Save KMeans model & training filenames for reproducibility
    with open(output_dir / "kmeans.pkl", "wb") as f:
        pickle.dump(kmeans, f)

    with open(output_dir / "train_filenames.txt", "w") as f:
        for p in img_list:
            f.write(p.name + "\n")

    # Metadata for sanity checking downstream
    import json
    meta = {
        "img_size": list(map(int, args.img_size)),
        "n_comps": int(args.n_comps),
        "k_clusters": int(args.k_clusters),
        "n_samples": int(n_samples),
        "feature_dim_per_channel": int(Y.shape[1]),
        "color_space": "YCbCr",
        "channels_order": ["Y", "Cb", "Cr"],
        "cluster_pcas_note": "cluster_pcas.npy are min-max normalized comps for viz; use cluster_pcas_raw.npy for math.",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("✅ Saved additional raw PCA artifacts, KMeans model, sizes, and metadata.")


    print("✅ Saved clustered PCA components per channel.")
    # ===== NEW: Compute and save per-image PCA weights per channel =====
    # For each cluster c, fit PCA per channel on that cluster's data exactly as before,
    # then transform ALL images in that cluster to get their scores (weights).

    print("🧮 Computing per-image PCA scores (weights) per channel...")
    weights_Y = np.zeros((n_samples, args.n_comps), dtype=np.float32)
    weights_Cb = np.zeros((n_samples, args.n_comps), dtype=np.float32)
    weights_Cr = np.zeros((n_samples, args.n_comps), dtype=np.float32)

    for c in range(args.k_clusters):
        indices = np.where(cluster_ids == c)[0]
        if len(indices) < args.n_comps:
            continue  # leave zeros; you also stored zeros for empty/small clusters

        # For each channel, refit PCA on the cluster and transform cluster members
        for chan_name, DATA, W in [
            ("Y",  Y,  weights_Y),
            ("Cb", Cb, weights_Cb),
            ("Cr", Cr, weights_Cr),
        ]:
            data_c = DATA[indices]  # (Nc, D), already in [0,1] since we divided by 255 above
            pca = PCA(n_components=args.n_comps, whiten=False)
            Z = pca.fit_transform(data_c)  # (Nc, C) scores
            W[indices] = Z.astype(np.float32)

    # Save as a single file for convenience
    np.savez(output_dir / "cluster_pca_weights_per_image.npz",
             weights_Y=weights_Y, weights_Cb=weights_Cb, weights_Cr=weights_Cr,
             cluster_ids=cluster_ids)
    print("✅ Saved per-image PCA weights (Y/Cb/Cr) and cluster_ids in NPZ.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse image set")
    parser.add_argument("-s", "--source", required=True, type=Path)
    parser.add_argument("-c", "--n_comps", type=int, default=300)
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
    output_dir = args.source.parent / f"{random_string}-{args.n_comps}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prepare_imgs(args, output_dir)
    prepare_clustered_arrs(args, output_dir)

    # pca_object, norm_infos = prepare_arrs(args, output_dir)
    
    # with open(output_dir / "pca_object.pkl", "wb") as FOUT:
    #     pickle.dump(pca_object, FOUT)
    # with open(output_dir / "norm_infos.pkl", "wb") as FOUT:
    #     pickle.dump(norm_infos, FOUT)