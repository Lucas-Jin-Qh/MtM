"""
单神经元可视化脚本
参考: Neural Encoding and Decoding at Scale Fig.2

功能:
- 绘制 Fig 2C: 试次平均放电率 (PSTH)
- 绘制 Fig 2D: 单次试次变异性 (Single-trial Raster)
- 支持选择 BPS 最高的神经元进行"高光展示"
"""

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 引入项目中的工具
from utils.eval_utils import load_model_data_local, heldout_mask
from utils.utils import set_seed, move_batch_to_device

# ================= 配置区域 =================
BASE_DIR = "/home/jqh/Workspace/IBL foundation model/MtM"
EID = "4b00df29-3769-43be-bb40-128b1cba6d35"

# 模型和数据路径
MODEL_PATH = f"{BASE_DIR}/results/train/num_session_1/model_NDT1/method_ssl/mask_all/stitch_True/model_best.pt"
DATASET_PATH = f"{BASE_DIR}/data/4b00df29-3769-43be-bb40-128b1cba6d35_aligned"
MODEL_CONFIG = f"{BASE_DIR}/src/configs/ndt1.yaml"
TRAINER_CONFIG = f"{BASE_DIR}/src/configs/ssl_session_trainer.yaml"

# 输出目录
SAVE_DIR = f"{BASE_DIR}/results/single_neuron_viz"
os.makedirs(SAVE_DIR, exist_ok=True)

# 数据参数
BIN_SIZE = 0.0167
TIME_WINDOW = (-0.2, 0.8)
ALIGNMENT_BIN = int(abs(TIME_WINDOW[0]) / BIN_SIZE)  # = 12
TRIAL_LEN = 60

# 可视化参数
N_TOP_NEURONS = 6  # 展示 BPS 最高的 N 个神经元
SELECT_BY_BPS = True  # True: 按 BPS 选择, False: 按 R2 选择

# 要生成可视化的任务
VIZ_TASKS = ["co_smooth", "temporal", "intra_region", "inter_region"]
# ===========================================


def plot_fig2c_psth(gt, pred, time_axis, save_path, neuron_name="Neuron"):
    """
    绘制 Fig 2C: 试次平均放电率 (PSTH)
    
    Args:
        gt: (K, T) Ground Truth
        pred: (K, T) Prediction
        time_axis: 时间轴 (秒)
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 计算 PSTH (按条件平均)
    gt_psth = gt.mean(axis=0)  # (T,)
    pred_psth = pred.mean(axis=0)
    gt_std = gt.std(axis=0)
    
    # 动态调整 y 轴范围 - 使用百分位数来聚焦于数据分布的主要区域
    psth_min = min(gt_psth.min(), pred_psth.min())
    psth_max = max(gt_psth.max(), pred_psth.max())
    
    # 使用 1-99 百分位数来确定范围，避免极端值影响
    gt_psth_1, gt_psth_99 = np.percentile(gt_psth, [1, 99])
    pred_psth_1, pred_psth_99 = np.percentile(pred_psth, [1, 99])
    
    y_min = max(0, min(gt_psth_1, pred_psth_1) * 0.9)
    y_max = max(gt_psth_99, pred_psth_99) * 1.2
    
    # 如果范围太小，稍微放大以保持可读性
    if y_max - y_min < psth_max * 0.1:
        center = (psth_max + psth_min) / 2
        half_range = max(psth_max - psth_min, center * 0.1)
        y_min = max(0, center - half_range)
        y_max = center + half_range
    
    # 绘制
    ax.plot(time_axis, gt_psth, 'b-', linewidth=2.5, label='Ground Truth', alpha=0.9)
    ax.plot(time_axis, pred_psth, 'r--', linewidth=2.5, label='MtM Prediction', alpha=0.9)
    
    # 填充误差带
    ax.fill_between(time_axis, gt_psth - gt_std * 0.5, gt_psth + gt_std * 0.5, 
                    color='blue', alpha=0.15, label='GT ± 0.5σ')
    
    # 标记 stimulus onset
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Stimulus Onset')
    
    # 标记刺激开始的位置 (时间窗起点)
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
    增强版 Fig 2C: PSTH + 差值图
    上图: GT vs Pred 曲线
    下图: 差值曲线 (Pred - GT)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), 
                                    gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.3})
    
    # 计算 PSTH
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
    绘制 Fig 2D: 单次试次变异性 (Single-trial Raster)
    
    Args:
        gt: (K, T) Ground Truth
        pred: (K, T) Prediction
        time_axis: 时间轴 (秒)
        save_path: 保存路径
        subtract_mean: 是否减去平均值
    """
    # 预处理
    if subtract_mean:
        gt_centered = gt - gt.mean(axis=0, keepdims=True)
        pred_centered = pred - pred.mean(axis=0, keepdims=True)
    else:
        gt_centered = gt
        pred_centered = pred
    
    # 计算残差
    residual = pred_centered - gt_centered
    
    # 计算相关性用于排序
    correlations = np.array([np.corrcoef(gt_centered[i], pred_centered[i])[0, 1] 
                            for i in range(len(gt_centered))])
    sort_idx = np.argsort(correlations)  # 按相关性从低到高排序
    
    # 排序数据
    gt_sorted = gt_centered[sort_idx]
    pred_sorted = pred_centered[sort_idx]
    residual_sorted = residual[sort_idx]
    
    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), 
                             gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 0.3})
    
    # 公共参数
    vmax = max(np.percentile(np.abs(gt_sorted), 95),
               np.percentile(np.abs(pred_sorted), 95))
    vmin = -vmax
    
    # 上图: Ground Truth
    im1 = axes[0].imshow(gt_sorted, aspect='auto', cmap='bwr', 
                          vmin=vmin, vmax=vmax, origin='upper')
    axes[0].set_ylabel('Trial (sorted)', fontsize=11)
    axes[0].set_title(f'Single-trial Variability - Ground Truth\n{neuron_name}', fontsize=13)
    axes[0].axvline(x=ALIGNMENT_BIN, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.colorbar(im1, ax=axes[0], shrink=0.6, label='Activity')
    
    # 中图: Prediction
    im2 = axes[1].imshow(pred_sorted, aspect='auto', cmap='bwr',
                          vmin=vmin, vmax=vmax, origin='upper')
    axes[1].set_ylabel('Trial (sorted)', fontsize=11)
    axes[1].set_title('MtM Prediction', fontsize=13)
    axes[1].axvline(x=ALIGNMENT_BIN, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.colorbar(im2, ax=axes[1], shrink=0.6, label='Activity')
    
    # 下图: Residual
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
    
    # 调整布局
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
        bps_path = f"{BASE_DIR}/results/eval_figures/{task_name}/bps.npy"
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
        for rank, neuron_idx in enumerate(top_indices):
            logger.info(f"  Neuron {neuron_idx} (rank {rank+1}/{len(top_indices)})...")
            
            region = regions[neuron_idx]
            uuid = uuids[neuron_idx][:8]
            neuron_name = f"{region}_{uuid}"
            
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
                    
                    # BPS 分数
                    bps_score = task_scores[neuron_idx] if neuron_idx < len(task_scores) else np.nan
                    
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
        
        # 为该任务生成汇总图
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


if __name__ == "__main__":
    run_single_neuron_visualization()
