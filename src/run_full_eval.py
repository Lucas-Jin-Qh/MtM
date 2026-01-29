"""
MtM Single-Session 完整评估脚本
参考: Neural Encoding and Decoding at Scale Fig.2

功能:
- 运行 4 种评估任务 (co-smooth, temporal, inter-region, intra-region)
- 生成 BPS 指标和单神经元可视化 (PSTH + Single-trial Raster)
"""

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 引入项目中的工具
from utils.eval_utils import load_model_data_local, co_smoothing_eval
from utils.utils import set_seed

# ================= 配置区域 =================
# Session 配置
EID = "4b00df29-3769-43be-bb40-128b1cba6d35"

# 基础路径
BASE_DIR = "/home/jqh/Workspace/IBL foundation model/MtM"

# 模型路径
MODEL_PATH = f"{BASE_DIR}/results/train/num_session_1/model_NDT1/method_ssl/mask_all/stitch_True/model_best.pt"

# 输出目录
SAVE_DIR = f"{BASE_DIR}/results/eval_figures"

# 数据集路径
DATASET_PATH = f"{BASE_DIR}/data/4b00df29-3769-43be-bb40-128b1cba6d35_aligned"

# 配置路径
MODEL_CONFIG = f"{BASE_DIR}/src/configs/ndt1.yaml"
TRAINER_CONFIG = f"{BASE_DIR}/src/configs/ssl_session_trainer.yaml"

# 数据参数
BIN_SIZE = 0.0167  # seconds (60 Hz)
TIME_WINDOW = (-0.2, 0.8)  # seconds
ALIGNMENT_BIN = int(abs(TIME_WINDOW[0]) / BIN_SIZE)  # 0.2s / 0.0167s = 12
TRIAL_LEN = 60  # 1.0s / 0.0167s = 60 bins

# 定义要评估的四个任务
# mode 对应 eval_utils.py 中的 co_smoothing_eval mode
TASKS = {
    "co_smooth": {
        "mode": "per_neuron",
        "held_out_list": None,
        "target_regions": None,
        "description": "Leave-one-neuron-out reconstruction"
    },
    "temporal": {
        "mode": "forward_pred",
        "held_out_list": [ALIGNMENT_BIN + 2],  # 预测刺激后第2个时间步 (causal prediction)
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

# 评估参数
N_JOBS = 1  # 并行神经元数，调试时用1，生产环境可调大
SUBTRACT_PSTH = "task"  # "task" | "global" | None
# ===========================================

def plot_bps_summary_all_tasks(bps_dict, save_path):
    """
    绘制所有任务的 BPS 汇总图
    
    Args:
        bps_dict: dict, {task_name: bps_array}
        save_path: str, 保存路径
    """
    n_tasks = len(bps_dict)
    fig, axes = plt.subplots(2, n_tasks, figsize=(5 * n_tasks, 10))
    
    if n_tasks == 1:
        axes = axes.reshape(2, 1)
    
    for idx, (task_name, bps) in enumerate(bps_dict.items()):
        # 过滤 NaN
        bps_valid = bps[~np.isnan(bps)]
        
        # 上方: BPS 直方图
        axes[0, idx].hist(bps_valid, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        mean_bps = np.nanmean(bps)
        axes[0, idx].axvline(mean_bps, color='r', linestyle='--', linewidth=2,
                            label=f'Mean: {mean_bps:.3f}')
        axes[0, idx].set_xlabel('Bits per Spike')
        axes[0, idx].set_ylabel('Count')
        axes[0, idx].set_title(f'{task_name}\n(n={len(bps_valid)}, mean={mean_bps:.3f})')
        axes[0, idx].legend()
        
        # 下方: 按神经元排序的 BPS
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
    """主评估函数"""
    
    # 设置随机种子
    set_seed(42)
    
    # 创建输出目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger.info(f"Output directory: {SAVE_DIR}")
    
    # 检查模型路径
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    logger.info(f"Loading model from: {MODEL_PATH}")
    
    # 1. 加载模型和数据
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
    
    # 获取数据集信息
    n_neurons = len(dataset['cluster_regions'][0])
    regions = np.array(dataset['cluster_regions'][0])
    unique_regions = np.unique(regions)
    logger.info(f"Dataset: {EID}")
    logger.info(f"  - Neurons: {n_neurons}")
    logger.info(f"  - Regions: {list(unique_regions)}")
    logger.info(f"  - Trials: {len(dataset)}")
    
    # 存储所有任务的 BPS 结果
    all_bps = {}
    summary_metrics = {}
    
    # 2. 循环执行四个任务
    for task_name, task_cfg in TASKS.items():
        print(f"\n{'='*60}")
        print(f"Running Task: {task_name} - {task_cfg['description']}")
        print(f"{'='*60}")
        
        task_save_path = os.path.join(SAVE_DIR, task_name)
        os.makedirs(task_save_path, exist_ok=True)
        
        # 执行评估
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

