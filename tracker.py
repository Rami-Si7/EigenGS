from pathlib import Path
from datetime import datetime
import time
from typing import List, Set, Dict

class LogWriter:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.log_path = self.log_dir / filename
        
        with open(self.log_path, 'w') as f:
            f.write(f"Log started at {datetime.now()}\n{'-'*50}\n")

    def write(self, message: str, show=True):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"
        
        if show:
            print(formatted_msg)
        with open(self.log_path, 'a') as f:
            f.write(formatted_msg + '\n')

    def section(self, title: str):
        separator = "=" * 50
        self.write(f"\n{separator}\n{title}\n{separator}")

class PSNRTracker:
    def __init__(self, logwriter: LogWriter, thresholds=[25.0, 30.0, 35.0, 40.0, 45.0]):
        self.logwriter = logwriter
        self.thresholds = set(thresholds)
        self.reached_thresholds = set()
        self.iter_event = []
        self.events = []
        self.log_psnrs = []
        self.log_times = []


    def check(self, start_time, psnr: float, iteration: int):
        # record event when PSNR threshold is reached
        current_time = time.perf_counter() - start_time
        for threshold in self.thresholds - self.reached_thresholds:
            if psnr >= threshold:
                self.reached_thresholds.add(threshold)
                self.events.append({
                    'timestamp': current_time,
                    'iteration': iteration,
                    'psnr': psnr
                })

    def log(self, start_time, psnr: float, iteration: int):
        current_time = time.perf_counter() - start_time
        self.log_psnrs.append(psnr)
        self.log_times.append(current_time)
        return

    def print_summary(self):
        # print event summary
        # if not self.events:
        #     self.logwriter.write("No PSNR thresholds reached")
        #     return
        
        # self.logwriter.section("PSNR Event Summary")
        # for event in sorted(self.events, key=lambda x: x['timestamp']):
        #     self.logwriter.write(
        #         f"Time: {event['timestamp']:.4f}s, "
        #         f"Iter: {event['iteration']}, "
        #         f"PSNR: {event['psnr']:.4f}"
        #     )

        times = [event["timestamp"] for event in self.events]
        iters = [event["iteration"] for event in self.events]
        psnrs = [event["psnr"] for event in self.events]
        
        thresholds = ["25", "30", "35", "40", "45"]
        times_dict = {k: v for k, v in zip(thresholds, times)}
        iters_dict = {k: v for k, v in zip(thresholds, iters)}
        psnrs_dict = {k: v for k, v in zip(thresholds, psnrs)}

        return times_dict, iters_dict, psnrs_dict, self.log_times, self.log_psnrs, self.iter_event