import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import *
from tracker import LogWriter, PSNRTracker
from tqdm import tqdm
import random
import torchvision.transforms as transforms
from sklearn.decomposition import PCA, SparseCoder
from skimage import color
from gaussianbasis import GaussianBasis
import pickle
import uuid
import cProfile
import pstats
import json

class GaussianTrainer:
    def __init__(
        self, args,
        image_path = None,
        num_points: int = 2000,
        iterations: int = 30000,
        model_path = None,
    ):
        self.dataset_path = Path(args.dataset)
        self.device = torch.device("cuda:0")
        arrs_path = self.dataset_path / "arrs.npy"
        self.gt_arrs = torch.from_numpy(np.load(arrs_path)).to(self.device)

        random_string = str(uuid.uuid4())[:6]
        if image_path is not None:
            self.image_path = Path(image_path)
            self.image_name = self.image_path.stem
            img = Image.open(self.image_path)
            img_arr = (color.rgb2ycbcr(img) / 255.0).transpose(2, 0, 1)
            self.gt_image = torch.tensor(img_arr, dtype=torch.float32, device=self.device)
            model_dir = Path(f"./models/recons-ycbcr/{self.dataset_path.name}/{self.image_name}")
        else: 
            model_dir = Path(f"./models/{self.dataset_path.name}-{num_points}-{args.iterations}-{random_string}")
        
        self.num_points = num_points
        self.num_comps = self.gt_arrs.shape[0]
        self.H, self.W = self.gt_arrs.shape[2], self.gt_arrs.shape[3]
        BLOCK_H, BLOCK_W = 16, 16
        self.iterations = iterations

        # # --- 20000 points setup ---
        # self.freq_config = {
        #     "low": [0, 50, 2000],
        #     "high": [50, self.num_comps, self.num_points - 2000],
        #     "all": [0, self.num_comps, self.num_points]
        # }
        # # --------------------------

        # --- 5000 points setup ---
        self.freq_config = {
            "low": [0, 50, 500],
            "high": [50, self.num_comps, self.num_points - 500],
            "all": [0, self.num_comps, self.num_points]
        }
        # -------------------------

        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.gaussian_model = GaussianBasis(
            loss_type="L2", opt_type="adan", num_points=self.num_points,
            H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
            device=self.device, lr=args.lr, num_comps=self.num_comps,
            gt_arrs=self.gt_arrs, freq_config=self.freq_config).to(self.device)

        self.logwriter = LogWriter(self.model_dir)
        self.psnr_tracker = PSNRTracker(self.logwriter)
        # self.logwriter.write(f"Model Dir ID: {random_string}")

        if model_path is not None:
            self.model_path = Path(model_path)
            # self.logwriter.write(f"Model loaded from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.gaussian_model.load_state_dict(checkpoint['model_state_dict'])

    def train(self, freq: str):
        self.gaussian_model.cur_freq = freq
        self.logwriter.write(f"Train {freq} freq params, config: {self.freq_config[freq]}")
        self.gaussian_model.scheduler_init()
        progress_bar = tqdm(range(1, self.iterations+1), desc="Training progress")
        self.gaussian_model.train()
        start_time = time.time()
        for iter in range(1, self.iterations+1):
            loss, psnr = self.gaussian_model.train_iter(self.gt_arrs)
            with torch.no_grad():
                progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                progress_bar.update(1)
        
        end_time = time.time() - start_time
        progress_bar.close()     
        self.logwriter.write("Training Complete in {:.4f}s".format(end_time))
        self.test_freq()
        return

    def merge_freq(self):
        merged_xyz = torch.cat([
            self.gaussian_model.params["low_xyz"],
            self.gaussian_model.params["high_xyz"]
        ], dim=0)
        
        merged_cholesky = torch.cat([
            self.gaussian_model.params["low_cholesky"],
            self.gaussian_model.params["high_cholesky"]
        ], dim=0)

        L = self.freq_config["low"]
        H = self.freq_config["high"]
        merged_features = torch.zeros(self.num_comps, self.num_points, 3, device=self.device)
        merged_features[L[0] : L[1],         0 : L[2],           :] = self.gaussian_model.params["low_features_dc"].data
        merged_features[H[0] : H[1],      L[2] : L[2]+H[2],      :] = self.gaussian_model.params["high_features_dc"].data
        
        self.gaussian_model.params["all_xyz"].copy_(merged_xyz)
        self.gaussian_model.params["all_cholesky"].copy_(merged_cholesky)
        self.gaussian_model.params["all_features_dc"].copy_(merged_features)
        
        self.gaussian_model.cur_freq = "all"
        self.logwriter.write(f"Train all freq params, config: {self.freq_config['all']}")
        torch.save({"model_state_dict": self.gaussian_model.state_dict()}, self.model_dir / "gaussian_model.pth.tar")
        self.vis()
        return

    def optimize(self):
        # profiler = cProfile.Profile()
        # profiler.enable()
        progress_bar = tqdm(range(1, self.iterations+1), desc="Optimizing progress")
        self.update_gaussian()
        init_psnr, init_ssim = self.test(init=True)
        self.gaussian_model.scheduler_init(optimize_phase=True)
        self.gaussian_model.train()
        start_time = time.perf_counter()

        for iter in range(1, self.iterations+1):
            loss, psnr = self.gaussian_model.optimize_iter(self.gt_image)
            self.psnr_tracker.check(start_time, psnr, iter)
            with torch.no_grad():
                if iter % 50 == 0:
                    self.psnr_tracker.log(start_time, psnr, iter)
                    self.compute_scores(start_time, psnr, iter)
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(50)
        
        end_time = time.perf_counter() - start_time
        progress_bar.close()    
        times, iters, psnrs, log_times, log_psnrs, iter_event = self.psnr_tracker.print_summary()

        # profiler.disable()
        # stats = pstats.Stats(profiler).sort_stats("cumulative")
        # stats.dump_stats(str(self.model_dir / "profile.stats"))
        # stats.print_stats(20) 

        self.logwriter.write("Optimizing Complete in {:.4f}s".format(end_time))
        torch.save({'model_state_dict': self.gaussian_model.state_dict()}, self.model_dir / "gaussian_model_with_colors.pth.tar")
        final_psnr, final_ssim = self.test()
        return init_psnr, init_ssim, final_psnr, final_ssim, end_time, times, iters, psnrs, log_times, log_psnrs, iter_event

    def test_freq(self):
        config = self.freq_config[self.gaussian_model.cur_freq]
        gt_arrs = self.gt_arrs[config[0]:config[1], :, :, :]
        self.gaussian_model.eval()
        with torch.no_grad():
            image = self.gaussian_model()
        mse_loss = F.mse_loss(image.float(), gt_arrs.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(image.float(), gt_arrs.float(), data_range=1, size_average=True).item()
        self.logwriter.write("Components Fitting: PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))
        return

    def vis(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            image = self.gaussian_model()
        mse_loss = F.mse_loss(image.float(), self.gt_arrs.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(image.float(), self.gt_arrs.float(), data_range=1, size_average=True).item()
        self.logwriter.write("Components Fitting: PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))
        
        vis_dir = self.model_dir / f"vis_comps"
        vis_dir.mkdir(parents=True, exist_ok=True)
        transform = transforms.ToPILImage()
        array = image.float()
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                img = transform(array[i, j])
                img.save(vis_dir / f"{i}-{j}.png")
        
        return psnr, ms_ssim_value

    def update_gaussian(self):
        with open(self.dataset_path / "pca_object.pkl", "rb") as FIN:
            pca_object = pickle.load(FIN)
        with open(self.dataset_path / "norm_infos.pkl", "rb") as FIN:
            norm_infos = pickle.load(FIN)

        self.gaussian_model.cur_freq = "all"
        img_arr = self.gt_image.detach().cpu().numpy()
        img_arr = img_arr.reshape(3, -1)
        
        codes = []
        maxs = []
        mins = []
        means = []
        for idx, ch_arr in enumerate(img_arr):
            pca = pca_object[idx]
            coes = pca.transform(ch_arr.reshape(1, -1))
            codes.append(coes)
            means.append(pca.mean_)

            maxs.append(norm_infos[idx]["max"])
            mins.append(norm_infos[idx]["min"])

        max_tensor = torch.tensor(np.array(maxs), dtype=torch.float32, device=self.gaussian_model.device)
        min_tensor = torch.tensor(np.array(mins), dtype=torch.float32, device=self.gaussian_model.device)
        mean_tensor = torch.tensor(np.array(means), dtype=torch.float32, device=self.gaussian_model.device)
        codes_tensor = torch.tensor(np.array(codes), dtype=torch.float32, device=self.gaussian_model.device)
        
        self.gaussian_model.shift_factor = codes_tensor.sum(dim=2).squeeze(1) * min_tensor
        self.gaussian_model.scale_factor = max_tensor - min_tensor
        self.gaussian_model.image_mean = mean_tensor

        # self.logwriter.write(f"Update shift factor: {self.gaussian_model.shift_factor}")
        # self.logwriter.write(f"Update scale factor: {self.gaussian_model.scale_factor}")
        # self.logwriter.write(f"Update mean tensor")

        with torch.no_grad():
            features = self.gaussian_model.get_features.permute(2, 0, 1)
            colors = torch.bmm(codes_tensor, features).squeeze(1).transpose(0, 1)
            self.gaussian_model._colors.copy_(colors)
            out = self.gaussian_model(render_colors=True)

        _, _, height, width = self.gt_arrs.shape 
        image = out.reshape(3, -1) + mean_tensor
        
        ycbcr_img = image.detach().cpu().numpy() * 255.0
        ycbcr_img = ycbcr_img.reshape(3, height, width).transpose(1, 2, 0)
        rgb_img = color.ycbcr2rgb(ycbcr_img) * 255.0
        rgb_img = rgb_img.clip(0, 255)

        img = Image.fromarray(rgb_img.astype(np.uint8))
        vis_dir = self.model_dir / f"vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        img.save(vis_dir / f"test_colors.png")

    def test_pca(self):
        with open(self.dataset_path / "pca_object.pkl", "rb") as FIN:
            pca_object = pickle.load(FIN)
        _, _, height, width = self.gt_arrs.shape        # (300, 3, 256, 256)
        img_arr = self.gt_image.detach().cpu().numpy()  # (3, 256, 256)
        img_arr = img_arr.reshape(3, -1)

        ycbcr_img = []
        for idx, ch_arr in enumerate(img_arr):
            pca = pca_object[idx]
            codes = pca.transform(ch_arr.reshape(1, -1))
            recons = pca.inverse_transform(codes).copy()
            recons *= 255.0
            recons = recons.reshape(height, width)
            ycbcr_img.append(recons)

        ycbcr_img = np.stack(ycbcr_img)
        ycbcr_img = ycbcr_img.transpose(1, 2, 0)
        rgb_img = color.ycbcr2rgb(ycbcr_img) * 255.0
        rgb_img = rgb_img.clip(0, 255)

        img = Image.fromarray(rgb_img.astype(np.uint8))
        vis_dir = self.model_dir / f"vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        img.save(vis_dir / f"test_pca.png")

    def compute_scores(self, start_time, 
                       psnr: float, iteration: int, 
                       iter_thresholds=[10, 100, 500, 1000, 5000, 10000]):
        if iteration in iter_thresholds:
            current_time = time.perf_counter() - start_time
            _, _, height, width = self.gt_arrs.shape
            with torch.no_grad():
                out = self.gaussian_model(render_colors=True)
            image = out.reshape(3, -1) + self.gaussian_model.image_mean
            image = image.reshape(3, height, width)
            mse_loss = F.mse_loss(image.float(), self.gt_image.float())
            psnr = 10 * math.log10(1.0 / mse_loss.item())
            ms_ssim_value = ms_ssim(image.unsqueeze(0).float(), self.gt_image.unsqueeze(0).float(), data_range=1, size_average=True).item()
            self.psnr_tracker.iter_event.append({
                'iteration': iteration,
                'timestamp': current_time,
                'ssim': ms_ssim_value,
                'psnr': psnr
            })
        return

    def test(self, init=False):
        self.gaussian_model.eval()
        _, _, height, width = self.gt_arrs.shape

        with torch.no_grad():
            out = self.gaussian_model(render_colors=True)
        image = out.reshape(3, -1) + self.gaussian_model.image_mean
        image = image.reshape(3, height, width)
        mse_loss = F.mse_loss(image.float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(image.unsqueeze(0).float(), self.gt_image.unsqueeze(0).float(), data_range=1, size_average=True).item()
        
        if init:
            self.logwriter.write("Initial PSNR: {:.4f}, MS_SSIM: {:.6f}".format(psnr, ms_ssim_value))
        else: 
            self.logwriter.write("Final PSNR: {:.4f}, MS_SSIM: {:.6f}".format(psnr, ms_ssim_value))

        ycbcr_img = image.detach().cpu().numpy() * 255.0
        ycbcr_img = ycbcr_img.reshape(3, height, width).transpose(1, 2, 0)
        rgb_img = color.ycbcr2rgb(ycbcr_img) * 255.0
        rgb_img = rgb_img.clip(0, 255)

        img = Image.fromarray(rgb_img.astype(np.uint8))
        if init:
            name = self.image_name + "_init_fitting.png"
        else:
            name = self.image_name + "_final_fitting.png"
        img.save(str(self.model_dir / name))
        return psnr, ms_ssim_value

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='./datasets/kodak/', help="Training dataset"
    )
    parser.add_argument(
        "--iterations", type=int, default=50000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--sh_degree", type=int, default=3, help="SH degree (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=50000,
        help="2D GS points (default: %(default)s)",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint")
    parser.add_argument("--image_path", type=str, default=None, help="Path to images")
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return args

def get_report(args, results):
    model_dir = Path(f"./models/recons-ycbcr/{Path(args.dataset).name}")
    thresholds = ["25", "30", "35", "40", "45"]
    
    times = {k: 0.0 for k in thresholds}
    iters = {k: 0 for k in thresholds}
    psnrs = {k: 0.0 for k in thresholds}
    counters = {k: 0 for k in thresholds}
    
    metrics = {
        'avg_init_psnr': 0.0,
        'avg_init_ssim': 0.0,
        'avg_final_psnr': 0.0,
        'avg_final_ssim': 0.0,
        'avg_final_time': 0.0
    }
    
    all_log_times = []
    all_log_psnrs = []
    all_iter_event = []

    for init_p, init_s, final_p, final_s, final_t, t_dict, i_dict, p_dict, log_times, log_psnrs, iter_event in results:
        metrics['avg_init_psnr'] += init_p
        metrics['avg_init_ssim'] += init_s
        metrics['avg_final_psnr'] += final_p
        metrics['avg_final_ssim'] += final_s
        metrics['avg_final_time'] += final_t
        
        for thres in thresholds:
            if thres in t_dict:
                times[thres] += t_dict[thres]
                iters[thres] += i_dict[thres]
                psnrs[thres] += p_dict[thres]
                counters[thres] += 1

        all_log_times.append(log_times)
        all_log_psnrs.append(log_psnrs)
        all_iter_event.append(iter_event)
    
    avg_log_times = np.mean(all_log_times, axis=0).tolist()
    avg_log_psnrs = np.mean(all_log_psnrs, axis=0).tolist()
    
    for metric in metrics:
        metrics[metric] /= len(results)
    
    for thres in thresholds:
        if counters[thres] > 0:
            times[thres] /= counters[thres]
            iters[thres] /= counters[thres]
            psnrs[thres] /= counters[thres]

    report = {
        'metrics': metrics,
        'thresholds': {
            'times': times,
            'psnrs': psnrs,
            'iterations': iters,
            'thres_counts': counters
        },
        'total_samples': len(results)
    }
    with open(model_dir / 'report.json', 'w') as f:
        json.dump(report, f, indent=4)

    periodic_data = {
        'times': avg_log_times,
        'psnrs': avg_log_psnrs,
        'iter_event': all_iter_event,
    }
    with open(model_dir / 'periodic_data.json', 'w') as f:
        json.dump(periodic_data, f, indent=4)

    return

def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    img_dir = Path(args.dataset) / "test_imgs"
    img_list = sorted([f for f in img_dir.glob("*.png")])
    results = []
    for idx, img_path in enumerate(img_list):
        trainer = GaussianTrainer(
            args, image_path=img_path, num_points=args.num_points, 
            iterations=args.iterations, model_path=args.model_path
        )
        res = trainer.optimize()
        results.append(res)
    get_report(args, results)

if __name__ == "__main__":
    main(sys.argv[1:])
