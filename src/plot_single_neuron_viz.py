"""
Single Neuron Visualization Script
Reference: Neural Encoding and Decoding at Scale Fig.2

Functionality:
- Plot Fig 2C: Trial-averaged Firing Rates (PSTH)
- Plot Fig 2D: Single-trial Variability (Raster)
- Support highlighting top neurons by BPS
"""

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import project utilities
from utils.eval_utils import load_model_data_local, heldout_mask
from utils.utils import set_seed, move_batch_to_device

# ================= Configuration =================
BASE_DIR = "/home/jqh/Workspace/IBL foundation model/MtM"
EID = "4b00df29-3769-43be-bb40-128b1cba6d35"

# Model and data paths
MODEL_PATH = f"{BASE_DIR}/results/train/num_session_1/model_NDT1/method_ssl/mask_all/stitch_True/model_best.pt"
DATASET_PATH = f"{BASE_DIR}/data/4b00df29-3769-43be-bb40-128b1cba6d35_aligned"
MODEL_CONFIG = f"{BASE_DIR}/src/configs/ndt1.yaml"
TRAINER_CONFIG = f"{BASE_DIR}/src/configs/ssl_session_trainer.yaml"

# Output directory
SAVE_DIR = f"{BASE_DIR}/results/single_neuron_viz"
os.makedirs(SAVE_DIR, exist_ok=True)

# Data parameters
BIN_SIZE = 0.0167
TIME_WINDOW = (-0.2, 0.8)
ALIGNMENT_BIN = int(abs(TIME_WINDOW[0]) / BIN_SIZE)  # = 12
TRIAL_LEN = 60

# Visualization parameters
N_TOP_NEURONS = 6  # Number of top neurons to highlight
SELECT_BY_BPS = True  # True: select by BPS, False: select by R2

# Tasks to visualize
VIZ_TASKS = ["co_smooth", "temporal", "intra_region", "inter_region"]
# ===========================================


def plot_fig2c_psth(gt, pred, time_axis, save_path, neuron_name="Neuron"):
    """
    Plot Fig 2C: Trial-averaged Firing Rates (PSTH)
    
    Args:
        gt: (K, T) Ground Truth
        pred: (K, T) Prediction
        time_axis: Time axis (seconds)
        save_path: Save path
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Compute PSTH (condition-averaged)
    gt_psth = gt.mean(axis=0)  # (T,)
    pred_psth = pred.mean(axis=0)
    gt_std = gt.std(axis=0)
    
    # Dynamic y-axis adjustment - use percentiles to focus on main data distribution
    psth_min = min(gt_psth.min(), pred_psth.min())
    psth_max = max(gt_psth.max(), pred_psth.max())
    
    # Use 1-99 percentile to determine range, avoid extreme values
    gt_psth_1, gt_psth_99 = np.percentile(gt_psth, [1, 99])
    pred_psth_1, pred_psth_99 = np.percentile(pred_psth, [1, 99])
    
    y_min = max(0, min(gt_psth_1, pred_psth_1) * 0.9)
    y_max = max(gt_psth_99, pred_psth_99) * 1.2
    
    # If range is too small, expand slightly for readability
    if y_max - y_min < psth_max * 0.1:
        center = (psth_max + psth_min) / 2
        half_range = max(psth_max - psth_min, center * 0.1)
        y_min = max(0, center - half_range)
        y_max = center + half_range
    
    # Plot
    ax.plot(time_axis, gt_psth, 'b-', linewidth=2.5, label='Ground Truth', alpha=0.9)
    ax.plot(time_axis, pred_psth, 'r--', linewidth=2.5, label='MtM Prediction', alpha=0.9)
    
    # Fill error band
    ax.fill_between(time_axis, gt_psth - gt_std * 0.5, gt_psth + gt_std * 0.5, 
                    color='blue', alpha=0.15, label='GT ± 0.5σ')
    
    # Mark stimulus onset
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Stimulus Onset')
    
    # Mark window start position
    ax.axvline(x=TIME_WINDOW[0], color='orange', linestyle='-.', linewidth=1.5, 
               alpha=0.6, label=f'Window start ({TIME_WINDOW[0]}s)')
    
    ax.set_xlabel('Time from stimulus onset (s)', fontsize=12)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
    ax.set_title(f'Trial-averaged Firing Rates (PSTH)\n{neuron_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"PSTH plot saved to {save_path}")


def plot_fig2c_psth_enhanced(gt, pred, time_axis, save_path, neuron_name="Neuron"):
    """
    Enhanced Fig 2C: PSTH + Difference Plot
    Top: GT vs Pred curves
    Bottom: Difference curve (Pred - GT)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), 
                                    gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.3})
    
    # Compute PSTH
    gt_psth = gt.mean(axis=0)
    pred_psth = pred.mean(axis=0)
    gt_std = gt.std(axis=0)
    diff = pred_psth - gt_psth
    
    # 动态 y 轴范围
    psth_1, psth_99 = np.percentile(np.concatenate([gt_psth, pred_psth]), [1, 99])
    y_min = max(0, psth_1 * 0.9)
    y_max = psth_99 * 1.2
    
    # 上图: GT vs Pred
    ax1.plot(time_axis, gt_psth, 'b-', linewidth=2.5, label='Ground Truth', alpha=0.9)
    ax1.plot(time_axis, pred_psth, 'r--', linewidth=2.5, label='MtM Prediction', alpha=0.9)
    ax1.fill_between(time_axis, gt_psth - gt_std * 0.5, gt_psth + gt_std * 0.5, 
                     color='blue', alpha=0.15, label='GT ± 0.5σ')
    ax1.axvline(x=0, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Stimulus Onset')
    ax1.axvline(x=TIME_WINDOW[0], color='orange', linestyle='-.', linewidth=1.5, alpha=0.6)
    
    ax1.set_ylabel('Firing Rate (Hz)', fontsize=12)
    ax1.set_ylim(y_min, y_max)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax1.spines[['right', 'top']].set_visible(False)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 下图: 差值
    diff_max = np.abs(diff).max()
    ax2.plot(time_axis, diff, 'purple', linewidth=2, alpha=0.8)
    ax2.fill_between(time_axis, 0, diff, color='purple', alpha=0.2)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax2.axvline(x=TIME_WINDOW[0], color='orange', linestyle='-.', linewidth=1.5, alpha=0.6)
    
    ax2.set_xlabel('Time from stimulus onset (s)', fontsize=12)
    ax2.set_ylabel('Prediction - GT (Hz)', fontsize=12)
    ax2.set_ylim(-diff_max * 1.3, diff_max * 1.3)
    ax2.spines[['right', 'top']].set_visible(False)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 计算 R² 并显示
    r2 = 1 - np.sum(diff**2) / np.sum((gt_psth - gt_psth.mean())**2)
    ax2.set_title(f'Prediction Error (R² = {r2:.4f})', fontsize=12, loc='right', pad=10)
    
    fig.suptitle(f'Trial-averaged Firing Rates (PSTH)\n{neuron_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Enhanced PSTH plot saved to {save_path}")


def plot_fig2d_single_trial(gt, pred, time_axis, save_path, neuron_name="Neuron", subtract_mean=True):
    """
    Plot Fig 2D: Single-trial Variability (Raster)
    
    Args:
        gt: (K, T) Ground Truth
        pred: (K, T) Prediction
        time_axis: Time axis (seconds)
        save_path: Save path
        subtract_mean: Whether to subtract mean
    """
    # Preprocessing
    if subtract_mean:
        gt_centered = gt - gt.mean(axis=0, keepdims=True)
        pred_centered = pred - pred.mean(axis=0, keepdims=True)
    else:
        gt_centered = gt
        pred_centered = pred
    
    # Compute residual
    residual = pred_centered - gt_centered
    
    # Compute correlation for sorting
    correlations = np.array([np.corrcoef(gt_centered[i], pred_centered[i])[0, 1] 
                            for i in range(len(gt_centered))])
    sort_idx = np.argsort(correlations)  # Sort from low to high correlation
    
    # Sort data
    gt_sorted = gt_centered[sort_idx]
    pred_sorted = pred_centered[sort_idx]
    residual_sorted = residual[sort_idx]
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), 
                             gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 0.3})
    
    # Common parameters
    vmax = max(np.percentile(np.abs(gt_sorted), 95),
               np.percentile(np.abs(pred_sorted), 95))
    vmin = -vmax
    
    # Top: Ground Truth
    im1 = axes[0].imshow(gt_sorted, aspect='auto', cmap='bwr', 
                          vmin=vmin, vmax=vmax, origin='upper')
    axes[0].set_ylabel('Trial (sorted)', fontsize=11)
    axes[0].set_title(f'Single-trial Variability - Ground Truth\n{neuron_name}', fontsize=13)
    axes[0].axvline(x=ALIGNMENT_BIN, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.colorbar(im1, ax=axes[0], shrink=0.6, label='Activity')
    
    # Middle: Prediction
    im2 = axes[1].imshow(pred_sorted, aspect='auto', cmap='bwr',
                          vmin=vmin, vmax=vmax, origin='upper')
    axes[1].set_ylabel('Trial (sorted)', fontsize=11)
    axes[1].set_title('MtM Prediction', fontsize=13)
    axes[1].axvline(x=ALIGNMENT_BIN, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.colorbar(im2, ax=axes[1], shrink=0.6, label='Activity')
    
    # Bottom: Residual
    resid_vmax = np.percentile(np.abs(residual_sorted), 95)
    im3 = axes[2].imshow(residual_sorted, aspect='auto', cmap='bwr',
                          vmin=-resid_vmax, vmax=resid_vmax, origin='upper')
    axes[2].set_ylabel('Trial (sorted)', fontsize=11)
    axes[2].set_xlabel('Time from stimulus onset (s)', fontsize=11)
    axes[2].set_title('Prediction Error (Residual)', fontsize=13)
    axes[2].axvline(x=ALIGNMENT_BIN, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    axes[2].set_xticks(np.linspace(0, TRIAL_LEN, 7))
    axes[2].set_xticklabels([f'{t:.1f}' for t in np.linspace(TIME_WINDOW[0], TIME_WINDOW[1], 7)])
    plt.colorbar(im3, ax=axes[2], shrink=0.6, label='Error')
    
    # Adjust layout
    for ax in axes:
        ax.spines[['right', 'top']].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Single-trial raster saved to {save_path}")


def plot_combined_fig2(gt, pred, time_axis, task_labels, save_path, neuron_name="Neuron"):
    """
    绘制组合图: Fig 2C + Fig 2D 并排显示
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 创建网格布局
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 2], hspace=0.35, wspace=0.3)
    
    # 左侧: PSTH (Fig 2C)
    ax_psth = fig.add_subplot(gs[0, 0])
    
    # 计算 PSTH
    gt_psth = gt.mean(axis=0)
    pred_psth = pred.mean(axis=0)
    gt_std = gt.std(axis=0)
    
    ax_psth.plot(time_axis, gt_psth, 'b-', linewidth=2, label='Ground Truth')
    ax_psth.plot(time_axis, pred_psth, 'r--', linewidth=2, label='MtM')
    ax_psth.fill_between(time_axis, gt_psth - gt_std, gt_psth + gt_std, 
                          color='blue', alpha=0.15)
    ax_psth.axvline(x=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax_psth.set_xlabel('Time (s)', fontsize=11)
    ax_psth.set_ylabel('Firing Rate (Hz)', fontsize=11)
    ax_psth.set_title(f'C. Trial-averaged Firing Rates (PSTH)\n{neuron_name}', fontsize=13, fontweight='bold')
    ax_psth.legend(loc='upper right', fontsize=9)
    ax_psth.spines[['right', 'top']].set_visible(False)
    ax_psth.grid(True, alpha=0.3)
    
    # 右侧: 统计信息
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_stats.axis('off')
    
    # 计算指标
    gt_centered = gt - gt.mean(axis=0, keepdims=True)
    pred_centered = pred - pred.mean(axis=0, keepdims=True)
    correlation = np.mean([np.corrcoef(gt_centered[i], pred_centered[i])[0, 1] 
                          for i in range(len(gt_centered))])
    r2_psth = 1 - np.sum((gt_psth - pred_psth)**2) / np.sum((gt_psth - gt_psth.mean())**2)
    r2_trial = 1 - np.sum((pred - gt)**2) / np.sum((gt - gt.mean())**2)
    
    stats_text = f"""
    Neuron: {neuron_name}
    Trials: {gt.shape[0]}, Time bins: {gt.shape[1]}
    
    Metrics:
    - PSTH R²: {r2_psth:.4f}
    - Single-trial R²: {r2_trial:.4f}
    - Mean correlation: {correlation:.4f}
    """
    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes, 
                  fontsize=12, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # 下侧: Single-trial Raster (Fig 2D)
    # 排序
    correlations = np.array([np.corrcoef(gt_centered[i], pred_centered[i])[0, 1] 
                            for i in range(len(gt_centered))])
    sort_idx = np.argsort(correlations)
    
    gt_sorted = gt_centered[sort_idx]
    pred_sorted = pred_centered[sort_idx]
    residual = pred_sorted - gt_sorted
    
    vmax = max(np.percentile(np.abs(gt_sorted), 95),
               np.percentile(np.abs(pred_sorted), 95))
    
    # Ground Truth
    ax_gt = fig.add_subplot(gs[1, :])
    im1 = ax_gt.imshow(gt_sorted, aspect='auto', cmap='bwr', vmin=-vmax, vmax=vmax, origin='upper')
    ax_gt.axvline(x=ALIGNMENT_BIN, color='white', linestyle='--', linewidth=2, alpha=0.8)
    ax_gt.set_ylabel('Trial (sorted)', fontsize=11)
    ax_gt.set_title(f'D. Single-trial Variability - Ground Truth', fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=ax_gt, shrink=0.5, label='Activity')
    
    # Prediction
    ax_pred = fig.add_subplot(gs[2, :])
    im2 = ax_pred.imshow(pred_sorted, aspect='auto', cmap='bwr', vmin=-vmax, vmax=vmax, origin='upper')
    ax_pred.axvline(x=ALIGNMENT_BIN, color='white', linestyle='--', linewidth=2, alpha=0.8)
    ax_pred.set_ylabel('Trial (sorted)', fontsize=11)
    ax_pred.set_xlabel('Time from stimulus onset (s)', fontsize=11)
    ax_pred.set_title('MtM Prediction', fontsize=13, fontweight='bold')
    ax_pred.set_xticks(np.linspace(0, TRIAL_LEN, 7))
    ax_pred.set_xticklabels([f'{t:.1f}' for t in np.linspace(TIME_WINDOW[0], TIME_WINDOW[1], 7)])
    plt.colorbar(im2, ax=ax_pred, shrink=0.5, label='Activity')
    
    for ax in [ax_gt, ax_pred]:
        ax.spines[['right', 'top']].set_visible(False)
    
    plt.suptitle(f'Neural Encoding Quality: {neuron_name}', fontsize=15, fontweight='bold', y=1.01)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Combined Fig2 plot saved to {save_path}")


def run_single_neuron_visualization():
    """主函数: 生成四个任务 (co-smooth, temporal, intra-region, inter-region) 的单神经元可视化"""
    
    set_seed(42)
    logger.info("Loading model and data...")
    
    # 加载模型和数据
    model, accelerator, dataset, dataloader = load_model_data_local(
        model_config=MODEL_CONFIG,
        trainer_config=TRAINER_CONFIG,
        model_path=MODEL_PATH,
        dataset_path=DATASET_PATH,
        test_size=0.1,
        seed=42,
        mask_name="mask_all",
        eid=EID,
        stitching=False,
        num_sessions=1
    )
    
    # 获取数据集信息
    n_neurons = len(dataset['cluster_regions'][0])
    regions = np.array(dataset['cluster_regions'][0])
    uuids = np.array(dataset['cluster_uuids'][0])
    
    # 获取行为数据 (choice)
    choices = np.array(dataset['choice'])
    logger.info(f"Dataset: {n_neurons} neurons, regions: {list(np.unique(regions))}")
    logger.info(f"Choice distribution: left={np.sum(choices==1)}, right={np.sum(choices==-1)}")
    
    # 时间轴
    time_axis = np.linspace(TIME_WINDOW[0], TIME_WINDOW[1], TRIAL_LEN)
    
    # 定义四个任务及其 masking mode
    TASKS = {
        "co_smooth": {
            "masking_mode": "neuron",
            "heldout_idxs_func": lambda idx: np.array([idx]),
            "eval_neurons": "all"  # 所有神经元
        },
        "temporal": {
            "masking_mode": "causal",
            "heldout_idxs_func": lambda idx: np.array([ALIGNMENT_BIN + 2]),  # 预测刺激后第2个时间步
            "eval_neurons": "all"
        },
        "intra_region": {
            "masking_mode": "intra-region",
            "heldout_idxs_func": lambda idx: np.array([idx]),
            "eval_neurons": "region_based"  # 按区域选择
        },
        "inter_region": {
            "masking_mode": "inter-region", 
            "heldout_idxs_func": lambda idx: np.array([idx]),
            "eval_neurons": "region_based"
        }
    }
    
    # 获取 BPS 结果
    bps_results = {}
    for task_name in TASKS.keys():
        bps_path = f"{BASE_DIR}/results/eval/comprehensive/{task_name}/bps.npy"
        if os.path.exists(bps_path):
            bps_results[task_name] = np.load(bps_path)
            logger.info(f"Loaded BPS for {task_name}: mean={np.nanmean(bps_results[task_name]):.4f}")
    
    # 为每个任务生成可视化
    for task_name, task_config in TASKS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing task: {task_name}")
        logger.info(f"{'='*60}")
        
        task_save_dir = os.path.join(SAVE_DIR, task_name)
        os.makedirs(task_save_dir, exist_ok=True)
        
        # 获取该任务的 BPS 或使用默认值
        if task_name in bps_results:
            task_scores = bps_results[task_name]
            select_metric = "BPS"
        else:
            task_scores = np.zeros(n_neurons)
            select_metric = "Default (0)"
        
        # 选择 top 神经元
        if task_config["eval_neurons"] == "region_based":
            # 按区域选择每个区域的 top 神经元
            unique_regions = np.unique(regions)
            top_indices = []
            for region in unique_regions:
                region_mask = regions == region
                region_scores = task_scores[region_mask]
                if len(region_scores) > 0:
                    region_top = np.where(region_mask)[0][np.argmax(region_scores)]
                    top_indices.append(region_top)
            logger.info(f"Selected top neuron per region: {list(top_indices)}")
        else:
            # 选择全局 top 神经元
            top_indices = np.argsort(task_scores)[-N_TOP_NEURONS:][::-1]
            logger.info(f"Selected top {len(top_indices)} neurons by {select_metric}: {list(top_indices)}")
        
        # 为每个 top 神经元生成可视化
        gt_data_list = []
        pred_data_list = []
        neuron_names_list = []
        bps_values_list = []
        
        for rank, neuron_idx in enumerate(top_indices):
            logger.info(f"  Neuron {neuron_idx} (rank {rank+1}/{len(top_indices)})...")
            
            region = regions[neuron_idx]
            uuid = uuids[neuron_idx][:8]
            neuron_name = f"{region}_{uuid}"
            neuron_names_list.append(neuron_name)
            
            # 获取该神经元的数据
            model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    batch = move_batch_to_device(batch, accelerator.device)
                    
                    # 创建掩码
                    heldout_idxs = task_config["heldout_idxs_func"](neuron_idx)
                    mask_result = heldout_mask(
                        batch['spikes_data'].clone(),
                        mode='manual',
                        heldout_idxs=heldout_idxs
                    )
                    
                    # 前向传播
                    outputs = model(
                        mask_result['spikes'],
                        time_attn_mask=batch['time_attn_mask'],
                        space_attn_mask=batch['space_attn_mask'],
                        spikes_timestamps=batch['spikes_timestamps'],
                        spikes_spacestamps=batch['spikes_spacestamps'],
                        targets=batch['target'],
                        neuron_regions=batch['neuron_regions'],
                        eval_mask=mask_result['eval_mask'],
                        masking_mode=task_config["masking_mode"],
                        num_neuron=batch['spikes_data'].shape[2],
                        eid=batch['eid'][0]
                    )
                    
                    # 转换预测
                    outputs.preds = torch.exp(outputs.preds)
                    
                    # 获取 GT 和 Prediction
                    gt_spikes = batch['spikes_data'].detach().cpu().numpy()
                    pred_spikes = outputs.preds.detach().cpu().numpy()
                    
                    # 提取该神经元的数据
                    gt = gt_spikes[:, :, neuron_idx]
                    pred = pred_spikes[:, :, neuron_idx]
                    
                    # 转换为 Hz
                    gt_hz = gt / BIN_SIZE
                    pred_hz = pred / BIN_SIZE
                    
                    gt_data_list.append(gt_hz)
                    pred_data_list.append(pred_hz)
                    
                    # BPS 分数
                    bps_score = task_scores[neuron_idx] if neuron_idx < len(task_scores) else np.nan
                    bps_values_list.append(bps_score)
                    
                    # 1. 绘制增强版 PSTH (Fig 2C + 差值图)
                    psth_enhanced_path = os.path.join(task_save_dir, f"{neuron_name}_psth_enhanced.png")
                    plot_fig2c_psth_enhanced(gt_hz, pred_hz, time_axis, psth_enhanced_path,
                                            f"{neuron_name} (BPS: {bps_score:.3f})")
                    
                    # 2. 绘制普通 PSTH
                    psth_path = os.path.join(task_save_dir, f"{neuron_name}_psth.png")
                    plot_fig2c_psth(gt_hz, pred_hz, time_axis, psth_path, 
                                   f"{neuron_name} (BPS: {bps_score:.3f})")
                    
                    # 3. 绘制 Single-trial Raster (Fig 2D)
                    raster_path = os.path.join(task_save_dir, f"{neuron_name}_raster.png")
                    plot_fig2d_single_trial(gt_hz, pred_hz, time_axis, raster_path,
                                           f"{neuron_name} (BPS: {bps_score:.3f})")
                    
                    # 4. 绘制组合图
                    combined_path = os.path.join(task_save_dir, f"{neuron_name}_fig2_combined.png")
                    plot_combined_fig2(gt_hz, pred_hz, time_axis, None, combined_path,
                                      f"{neuron_name} (BPS: {bps_score:.3f})")
                    
                    break  # 只处理一个 batch
        
        # 5. 生成该任务的汇总 PSTH 图
        logger.info(f"  Generating combined PSTH for {task_name}...")
        combined_psth_path = os.path.join(task_save_dir, f"combined_psth.png")
        plot_task_combined_psth(gt_data_list, pred_data_list, neuron_names_list, 
                                time_axis, combined_psth_path)
        
        # 6. 生成该任务的汇总 Raster 图
        logger.info(f"  Generating combined raster for {task_name}...")
        combined_raster_path = os.path.join(task_save_dir, f"combined_raster.png")
        plot_task_combined_raster(gt_data_list, pred_data_list, neuron_names_list,
                                  time_axis, combined_raster_path, bps_values_list)
        
        # 为该任务生成 BPS 汇总图
        plot_task_summary(task_scores, regions, uuids, top_indices, task_name, task_save_dir)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Visualization complete! Results saved to: {SAVE_DIR}")
    logger.info(f"{'='*60}")


def plot_task_summary(scores, regions, uuids, top_indices, task_name, save_dir):
    """为每个任务生成汇总图"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 过滤有效分数
    valid_mask = ~np.isnan(scores)
    valid_scores = scores[valid_mask]
    
    # 左图: BPS 分布直方图
    axes[0].hist(valid_scores, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    for idx in top_indices:
        if idx < len(scores) and not np.isnan(scores[idx]):
            axes[0].axvline(x=scores[idx], color='red', linestyle='--', alpha=0.8, linewidth=2)
    axes[0].axvline(x=np.nanmean(valid_scores), color='green', linestyle='-', linewidth=2, 
                    label=f'Mean: {np.nanmean(valid_scores):.3f}')
    axes[0].set_xlabel('Bits per Spike', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title(f'BPS Distribution - {task_name}\n(Selected neurons in red)', fontsize=13)
    axes[0].legend()
    
    # 右图: 按脑区的 BPS
    unique_regions = np.unique(regions)
    region_means = [np.nanmean(scores[regions == r]) if np.sum(regions == r) > 0 else 0 for r in unique_regions]
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regions)))
    bars = axes[1].barh(unique_regions, region_means, color=colors, edgecolor='black', alpha=0.8)
    axes[1].set_xlabel('Mean Bits per Spike', fontsize=12)
    axes[1].set_ylabel('Brain Region', fontsize=12)
    axes[1].set_title(f'BPS by Brain Region - {task_name}', fontsize=13)
    
    # 标注 top 神经元
    for i, region in enumerate(unique_regions):
        region_indices = np.where(regions == region)[0]
        top_in_region = [idx for idx in top_indices if idx in region_indices]
        if top_in_region:
            axes[1].annotate('★', xy=(region_means[i], i), fontsize=18, color='red',
                            ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'task_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Task summary saved to {os.path.join(save_dir, 'task_summary.png')}")


def plot_summary_figure(top_indices, scores, regions, uuids, save_dir):
    """生成汇总图: 展示所有 top 神经元的 BPS"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图: BPS 分布直方图
    axes[0].hist(scores, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    for idx in top_indices:
        axes[0].axvline(x=scores[idx], color='red', linestyle='--', alpha=0.7)
    axes[0].axvline(x=np.nanmean(scores), color='green', linestyle='-', linewidth=2, 
                    label=f'Mean: {np.nanmean(scores):.3f}')
    axes[0].set_xlabel('Bits per Spike', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('BPS Distribution (Top neurons marked)', fontsize=13)
    axes[0].legend()
    
    # 右图: 按脑区的 BPS
    unique_regions = np.unique(regions)
    region_means = [np.nanmean(scores[regions == r]) for r in unique_regions]
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regions)))
    bars = axes[1].barh(unique_regions, region_means, color=colors)
    axes[1].set_xlabel('Mean Bits per Spike', fontsize=12)
    axes[1].set_ylabel('Brain Region', fontsize=12)
    axes[1].set_title('BPS by Brain Region', fontsize=13)
    
    # 标注 top 神经元
    for i, region in enumerate(unique_regions):
        region_indices = np.where(regions == region)[0]
        top_in_region = [idx for idx in top_indices if idx in region_indices]
        if top_in_region:
            axes[1].annotate('★', xy=(region_means[i], i), fontsize=15, color='red',
                            ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Summary plot saved to {os.path.join(save_dir, 'summary.png')}")


def plot_task_combined_psth(gt_list, pred_list, neuron_names, time_axis, save_path):
    """
    为任务生成汇总 PSTH 图 - 显示所有 top 神经元的 PSTH
    
    Args:
        gt_list: list of (n_trials, n_time_bins) arrays
        pred_list: list of (n_trials, n_time_bins) arrays
        neuron_names: list of neuron names
        time_axis: 时间轴 (秒)
        save_path: 保存路径
    """
    n_neurons = len(neuron_names)
    n_cols = min(3, n_neurons)  # 最多3列
    n_rows = (n_neurons + n_cols - 1) // n_cols  # 自动计算行数
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)
    
    for idx in range(n_neurons):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # 计算 PSTH
        gt_psth = gt_list[idx].mean(axis=0)
        pred_psth = pred_list[idx].mean(axis=0)
        
        # 绘制
        ax.plot(time_axis, gt_psth, 'b-', linewidth=2, label='GT', alpha=0.9)
        ax.plot(time_axis, pred_psth, 'r--', linewidth=2, label='Pred', alpha=0.9)
        ax.axvline(x=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        
        ax.set_title(neuron_names[idx], fontsize=11)
        
        if col == 0:
            ax.set_ylabel('Firing Rate (Hz)', fontsize=10)
        
        if row == n_rows - 1:
            ax.set_xlabel('Time (s)', fontsize=10)
        else:
            ax.tick_params(labelbottom=False)
        
        ax.spines[['right', 'top']].set_visible(False)
        ax.grid(True, alpha=0.2)
    
    # 图例
    handles = [plt.Line2D([0], [0], color='blue', linewidth=2, label='Ground Truth'),
               plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Prediction')]
    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=12,
               bbox_to_anchor=(0.5, 1.02), frameon=True)
    
    plt.suptitle(f'Combined PSTH - All Top Neurons ({n_neurons} neurons)', 
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Combined PSTH saved to {save_path}")


def plot_task_combined_raster(gt_list, pred_list, neuron_names, time_axis, save_path, bps_list=None):
    """
    为任务生成汇总 Raster 图 - 显示所有 top 神经元的单试次变异性
    
    Args:
        gt_list: list of (n_trials, n_time_bins) arrays
        pred_list: list of (n_trials, n_time_bins) arrays
        neuron_names: list of neuron names
        time_axis: 时间轴 (秒)
        save_path: 保存路径
        bps_list: list of BPS values for each neuron
    """
    n_neurons = len(neuron_names)
    n_cols = min(3, n_neurons)  # 最多3列
    n_rows = (n_neurons + n_cols - 1) // n_cols  # 每个神经元占1行网格（每行有GT和Pred两个子行）
    
    # 每个神经元占2行 (GT行 + Pred行)
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(5 * n_cols, 2.5 * n_rows * 2))
    
    if n_cols == 1:
        axes = axes.reshape(n_rows * 2, 1)
    
    for idx in range(n_neurons):
        row = idx // n_cols * 2  # 每个神经元占2行
        col = idx % n_cols
        
        gt = gt_list[idx]
        pred = pred_list[idx]
        
        # 去均值
        gt_c = gt - gt.mean(axis=0, keepdims=True)
        pred_c = pred - pred.mean(axis=0, keepdims=True)
        
        # 按相关性排序
        correlations = np.array([np.corrcoef(gt_c[i], pred_c[i])[0, 1] 
                                for i in range(len(gt_c))])
        sort_idx = np.argsort(correlations)
        
        gt_sorted = gt_c[sort_idx]
        pred_sorted = pred_c[sort_idx]
        
        # 统一颜色范围
        vmax = max(np.percentile(np.abs(gt_sorted), 95),
                   np.percentile(np.abs(pred_sorted), 95))
        if vmax < 1e-6:
            vmax = 1.0
        vmin = -vmax
        
        # 上行: GT
        ax_top = axes[row, col]
        im1 = ax_top.imshow(gt_sorted, aspect='auto', cmap='bwr', 
                           vmin=vmin, vmax=vmax, origin='upper')
        ax_top.set_title(neuron_names[idx], fontsize=11)
        if bps_list is not None:
            ax_top.set_xlabel(f'BPS: {bps_list[idx]:.3f}', fontsize=10)
        ax_top.set_ylabel('Trials', fontsize=9)
        ax_top.axvline(x=ALIGNMENT_BIN, color='white', linestyle='--', linewidth=1, alpha=0.8)
        ax_top.spines[['right', 'top']].set_visible(False)
        
        # 下行: Prediction
        ax_bottom = axes[row + 1, col]
        im2 = ax_bottom.imshow(pred_sorted, aspect='auto', cmap='bwr',
                              vmin=vmin, vmax=vmax, origin='upper')
        ax_bottom.set_ylabel('Trials', fontsize=9)
        ax_bottom.set_xlabel('Time bins', fontsize=9)
        ax_bottom.set_xticks(np.linspace(0, TRIAL_LEN, 5))
        ax_bottom.set_xticklabels([f'{t:.1f}' for t in np.linspace(TIME_WINDOW[0], TIME_WINDOW[1], 5)])
        ax_bottom.axvline(x=ALIGNMENT_BIN, color='white', linestyle='--', linewidth=1, alpha=0.8)
        ax_bottom.spines[['right', 'top']].set_visible(False)
        
        # 只在第一列添加 y 轴标签
        if col > 0:
            ax_top.set_ylabel('')
            ax_bottom.set_ylabel('')
    
    # 添加颜色条和调整布局
    fig.subplots_adjust(right=0.92, hspace=0.3)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(im1, cax=cbar_ax, label='Activity')
    
    plt.suptitle(f'Combined Single-trial Variability - All Top Neurons ({n_neurons} neurons)', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Combined raster saved to {save_path}")


if __name__ == "__main__":
    run_single_neuron_visualization()


# ================= 旧版函数（已废弃） =================
