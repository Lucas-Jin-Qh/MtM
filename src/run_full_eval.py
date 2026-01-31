"""
MtM Single-Session Complete Evaluation Script
Reference: Neural Encoding and Decoding at Scale Fig.2

Functionality:
- Run 4 evaluation tasks (co-smooth, temporal, inter-region, intra-region)
- Generate BPS metrics and single-neuron visualizations (PSTH + Single-trial Raster)
"""

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import project utilities
from utils.eval_utils import load_model_data_local, co_smoothing_eval
from utils.utils import set_seed

# ================= Configuration =================
# Session Configuration
EID = "4b00df29-3769-43be-bb40-128b1cba6d35"

# Base path
BASE_DIR = "/home/jqh/Workspace/IBL foundation model/MtM"

# Model path
MODEL_PATH = f"{BASE_DIR}/results/train/num_session_1/model_NDT1/method_ssl/mask_all/stitch_True/model_best.pt"

# Output directory
SAVE_DIR = f"{BASE_DIR}/results/eval_figures"

# Dataset path
DATASET_PATH = f"{BASE_DIR}/data/4b00df29-3769-43be-bb40-128b1cba6d35_aligned"

# Config paths
MODEL_CONFIG = f"{BASE_DIR}/src/configs/ndt1.yaml"
TRAINER_CONFIG = f"{BASE_DIR}/src/configs/ssl_session_trainer.yaml"

# Data parameters
BIN_SIZE = 0.0167  # seconds (60 Hz)
TIME_WINDOW = (-0.2, 0.8)  # seconds
ALIGNMENT_BIN = int(abs(TIME_WINDOW[0]) / BIN_SIZE)  # 0.2s / 0.0167s = 12
TRIAL_LEN = 60  # 1.0s / 0.0167s = 60 bins

# Define four evaluation tasks
# mode corresponds to co_smoothing_eval mode in eval_utils.py
TASKS = {
    "co_smooth": {
        "mode": "per_neuron",
        "held_out_list": None,
        "target_regions": None,
        "description": "Leave-one-neuron-out reconstruction"
    },
    "temporal": {
        "mode": "forward_pred",
        "held_out_list": [ALIGNMENT_BIN + 2],  # Predict 2nd time bin after stimulus onset (causal prediction)
        "target_regions": None,
        "description": "Temporal prediction (causal, forward prediction)"
    },
    "intra_region": {
        "mode": "intra_region",
        "held_out_list": None,
        "target_regions": "all",
        "description": "Leave-one-neuron-out within same region"
    },
    "inter_region": {
        "mode": "inter_region",
        "held_out_list": None,
        "target_regions": "all",
        "description": "Leave-one-region-out reconstruction"
    }
}

# Evaluation parameters
N_JOBS = 1  # Number of parallel neurons, use 1 for debugging, increase for production
SUBTRACT_PSTH = "task"  # "task" | "global" | None
# ===========================================

def plot_bps_summary_all_tasks(bps_dict, save_path):
    """
    Plot BPS summary for all tasks.
    
    Args:
        bps_dict: dict, {task_name: bps_array}
        save_path: str, save path
    """
    n_tasks = len(bps_dict)
    fig, axes = plt.subplots(2, n_tasks, figsize=(5 * n_tasks, 10))
    
    if n_tasks == 1:
        axes = axes.reshape(2, 1)
    
    for idx, (task_name, bps) in enumerate(bps_dict.items()):
        # Filter NaN
        bps_valid = bps[~np.isnan(bps)]
        
        # Top: BPS histogram
        axes[0, idx].hist(bps_valid, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        mean_bps = np.nanmean(bps)
        axes[0, idx].axvline(mean_bps, color='r', linestyle='--', linewidth=2,
                            label=f'Mean: {mean_bps:.3f}')
        axes[0, idx].set_xlabel('Bits per Spike')
        axes[0, idx].set_ylabel('Count')
        axes[0, idx].set_title(f'{task_name}\n(n={len(bps_valid)}, mean={mean_bps:.3f})')
        axes[0, idx].legend()
        
        # Bottom: BPS sorted by neuron
        sorted_idx = np.argsort(bps)[::-1]
        x = np.arange(len(bps))
        axes[1, idx].scatter(x, bps[sorted_idx], s=1, alpha=0.5, c='steelblue')
        axes[1, idx].axhline(mean_bps, color='r', linestyle='--', linewidth=1)
        axes[1, idx].set_xlabel('Neuron (sorted by BPS)')
        axes[1, idx].set_ylabel('Bits per Spike')
        axes[1, idx].set_ylim([min(0, np.nanmin(bps)), max(1.0, np.nanmax(bps))])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"BPS summary plot saved to {save_path}")


def run_evaluation():
    """Main evaluation function."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create output directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger.info(f"Output directory: {SAVE_DIR}")
    
    # Check model path
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    logger.info(f"Loading model from: {MODEL_PATH}")
    
    # 1. Load model and data
    model, accelerator, dataset, dataloader = load_model_data_local(
        model_config=MODEL_CONFIG,
        trainer_config=TRAINER_CONFIG,
        model_path=MODEL_PATH,
        dataset_path=DATASET_PATH,
        test_size=0.1,
        seed=42,
        mask_name="mask_all",
        eid=EID,
        stitching=False,  # Single-session
        num_sessions=1
    )
    
    # Get dataset information
    n_neurons = len(dataset['cluster_regions'][0])
    regions = np.array(dataset['cluster_regions'][0])
    unique_regions = np.unique(regions)
    logger.info(f"Dataset: {EID}")
    logger.info(f"  - Neurons: {n_neurons}")
    logger.info(f"  - Regions: {list(unique_regions)}")
    logger.info(f"  - Trials: {len(dataset)}")
    
    # Store BPS results for all tasks
    all_bps = {}
    summary_metrics = {}
    
    # 2. Loop through all four tasks
    for task_name, task_cfg in TASKS.items():
        print(f"\n{'='*60}")
        print(f"Running Task: {task_name} - {task_cfg['description']}")
        print(f"{'='*60}")
        
        task_save_path = os.path.join(SAVE_DIR, task_name)
        os.makedirs(task_save_path, exist_ok=True)
        
        # Execute evaluation
        metrics = co_smoothing_eval(
            model=model,
            accelerator=accelerator,
            test_dataloader=dataloader,
            test_dataset=dataset,
            n=1,
            method_name="MtM_SingleSession",
            mode=task_cfg["mode"],
            is_aligned=True,  # IBL 数据是对齐的
            target_regions=task_cfg.get("target_regions"),
            held_out_list=task_cfg.get("held_out_list"),
            n_jobs=N_JOBS,
            subtract=SUBTRACT_PSTH,
            onset_alignment=[ALIGNMENT_BIN],  # 关键：对齐到 stimulus onset
            n_time_steps=TRIAL_LEN,
            save_path=task_save_path
        )
        
        # 加载保存的 BPS 数据
        bps_path = os.path.join(task_save_path, "bps.npy")
        if os.path.exists(bps_path):
            bps = np.load(bps_path)
            all_bps[task_name] = bps
            
            # 计算统计量
            mean_bps = np.nanmean(bps)
            std_bps = np.nanstd(bps)
            n_valid = np.sum(~np.isnan(bps))
            
            print(f"[{task_name}] BPS: {mean_bps:.4f} ± {std_bps:.4f} (n={n_valid})")
            
            summary_metrics[task_name] = {
                'mode': task_cfg["mode"],
                'mean_bps': mean_bps,
                'std_bps': std_bps,
                'n_valid_neurons': n_valid,
                'n_total_neurons': len(bps)
            }
        else:
            logger.warning(f"BPS file not found: {bps_path}")
    
    # 3. 生成 BPS 汇总图
    if all_bps:
        summary_plot_path = os.path.join(SAVE_DIR, "bps_summary_all_tasks.png")
        plot_bps_summary_all_tasks(all_bps, summary_plot_path)
    
    # 4. 保存汇总表格
    df = pd.DataFrame(summary_metrics).T
    summary_csv_path = os.path.join(SAVE_DIR, "bps_summary.csv")
    df.to_csv(summary_csv_path)
    
    # 5. 生成简洁报告
    report = []
    report.append("=" * 60)
    report.append("MtM Single-Session Evaluation Summary")
    report.append("=" * 60)
    report.append(f"Session: {EID}")
    report.append(f"Model: {MODEL_PATH}")
    report.append(f"Neurons: {n_neurons}")
    report.append(f"Regions: {list(unique_regions)}")
    report.append("")
    report.append("Results:")
    report.append("-" * 60)
    for task_name, metrics in summary_metrics.items():
        report.append(f"{task_name:15s} | BPS: {metrics['mean_bps']:.4f} ± {metrics['std_bps']:.4f} | n={metrics['n_valid_neurons']}/{metrics['n_total_neurons']}")
    report.append("-" * 60)
    
    report_path = os.path.join(SAVE_DIR, "eval_report.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {SAVE_DIR}")
    print(f"\nSummary:")
    print(df.to_string())
    
    return summary_metrics


if __name__ == "__main__":
    summary = run_evaluation()

