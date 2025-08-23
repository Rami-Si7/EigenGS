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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse image set")
    parser.add_argument("-s", "--source", required=True, type=Path)
    parser.add_argument("-c", "--n_comps", type=int, default=300)
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
    pca_object, norm_infos = prepare_arrs(args, output_dir)
    
    with open(output_dir / "pca_object.pkl", "wb") as FOUT:
        pickle.dump(pca_object, FOUT)
    with open(output_dir / "norm_infos.pkl", "wb") as FOUT:
        pickle.dump(norm_infos, FOUT)