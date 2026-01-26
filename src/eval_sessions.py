#!/usr/bin/env python3
"""
Comprehensive evaluation script for single-session model.
Generates BPS metrics and single-neuron visualizations for multiple evaluation modes:
- co-smooth (per_neuron)
- causal / temporal
- inter-region
- intra-region

Deliverables:
- bps summary (overall + per neuron)
- plots for example single neurons for each task

Usage:
    python src/eval_sessions.py \
        --model-path results/train/num_session_1/model_NDT1/method_ssl/mask_all/stitch_True/model_best.pt \
        --dataset-path data/4b00df29-3769-43be-bb40-128b1cba6d35_aligned \
        --eid 4b00df29-3769-43be-bb40-128b1cba6d35 \
        --save-path results/eval/comprehensive \
        --eval-modes co_smooth causal inter_region intra_region \
        --example-neurons 5
"""
import argparse
import os
import json
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Comprehensive evaluation for single-session model")

    # 配置路径
    p.add_argument("--model-config", default="src/configs/ndt1_stitching_prompting_eval.yaml")
    p.add_argument("--trainer-config", default="src/configs/ssl_session_trainer.yaml")

    # 模型与数据路径
    p.add_argument("--model-path", required=True,
                   help="Path to trained model checkpoint")
    p.add_argument("--dataset-path", required=True,
                   help="Local aligned dataset dir")
    p.add_argument("--eid", required=True, help="Session EID")

    # 评估参数
    p.add_argument("--mask-name", default="mask_all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-path", default="results/eval/comprehensive",
                   help="Directory to write evaluation results and plots")
    p.add_argument("--eval-modes", nargs="+", 
                   choices=["co_smooth", "causal", "temporal", "inter_region", "intra_region"],
                   default=["co_smooth", "causal", "inter_region", "intra_region"],
                   help="Evaluation modes to run")
    p.add_argument("--n-jobs", type=int, default=8,
                   help="Number of neurons processed in parallel")
    p.add_argument("--example-neurons", type=int, default=5,
                   help="Number of example neurons to visualize per mode")
    p.add_argument("--held-out-list", nargs="+", type=int, default=None,
                   help="Time indices for forward_pred mode (e.g., --held-out-list 20 25 30 35)")

    return p.parse_args()


def plot_comprehensive_summary(all_results: dict, save_path: str) -> None:
    """Plot comprehensive summary of all evaluation modes."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
    modes = list(all_results.keys())
    
    # Row 1: BPS distributions for each mode
    for i, mode in enumerate(modes[:4]):
        ax = fig.add_subplot(gs[0, i])
        bps = all_results[mode].get('bps_per_neuron', np.array([]))
        if len(bps) > 0:
            valid_bps = bps[np.isfinite(bps)]
            if len(valid_bps) > 0:
                ax.hist(valid_bps, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
                ax.axvline(np.nanmean(valid_bps), color='red', linestyle='--', 
                          label=f"Mean: {np.nanmean(valid_bps):.3f}")
                ax.set_xlabel('Bits per Spike')
                ax.set_ylabel('Count')
                ax.set_title(f'{mode.replace("_", " ").title()}\nBPS Distribution')
                ax.legend(fontsize=8)
    
    # Row 2: Mean BPS comparison
    ax = fig.add_subplot(gs[1, :2])
    mode_names = [m.replace('_', ' ').title() for m in modes]
    mean_bps = [np.nanmean(all_results[m].get('bps_per_neuron', [np.nan])) for m in modes]
    std_bps = [np.nanstd(all_results[m].get('bps_per_neuron', [np.nan])) for m in modes]
    
    colors_list = ['steelblue', 'coral', 'seagreen', 'purple', 'goldenrod'][:len(modes)]
    bars = ax.bar(range(len(modes)), mean_bps, yerr=std_bps, 
                  color=colors_list, alpha=0.7, edgecolor='black', capsize=5)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(mode_names, rotation=15, ha='right')
    ax.set_ylabel('Mean BPS')
    ax.set_title('Mean BPS Comparison Across Evaluation Modes')
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    
    for bar, val in zip(bars, mean_bps):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Row 2 continued: R2 comparison
    ax = fig.add_subplot(gs[1, 2:])
    mean_r2_psth = [np.nanmean(all_results[m].get('r2_per_neuron', np.array([[np.nan, np.nan]])[:, 0])) for m in modes]
    mean_r2_trial = [np.nanmean(all_results[m].get('r2_per_neuron', np.array([[np.nan, np.nan]])[:, 1])) for m in modes]
    
    x = np.arange(len(modes))
    width = 0.35
    bars1 = ax.bar(x - width/2, mean_r2_psth, width, label='R2 PSTH', color='steelblue', alpha=0.7)
    bars2 = ax.bar(x + width/2, mean_r2_trial, width, label='R2 Trial', color='coral', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(mode_names, rotation=15, ha='right')
    ax.set_ylabel('R2 Score')
    ax.set_title('R2 Comparison Across Evaluation Modes')
    ax.legend()
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    
    # Row 3: Summary table
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    
    table_data = []
    for mode in modes:
        bps = all_results[mode].get('bps_per_neuron', np.array([]))
        r2 = all_results[mode].get('r2_per_neuron', np.array([[np.nan, np.nan]]))
        n_valid = np.sum(np.isfinite(bps))
        
        table_data.append([
            mode.replace('_', ' ').title(),
            f"{n_valid}/{len(bps)}" if len(bps) > 0 else "N/A",
            f"{np.nanmean(bps):.4f}" if len(bps) > 0 else "N/A",
            f"{np.nanstd(bps):.4f}" if len(bps) > 0 else "N/A",
            f"{np.nanmean(r2[:, 0]):.4f}" if r2.size > 0 else "N/A",
            f"{np.nanmean(r2[:, 1]):.4f}" if r2.size > 0 else "N/A",
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Mode', 'Valid Neurons', 'Mean BPS', 'Std BPS', 'Mean R2 PSTH', 'Mean R2 Trial'],
        cellLoc='center', loc='center', colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.15]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax.set_title('Summary Statistics Table', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('Comprehensive Evaluation Results', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(save_path, 'comprehensive_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comprehensive summary to {os.path.join(save_path, 'comprehensive_summary.png')}")


def plot_example_neurons(test_dataset, bps_per_neuron, r2_per_neuron, mode, save_path, n_examples=5):
    """Plot example neurons with best, worst, and median BPS."""
    valid_mask = np.isfinite(bps_per_neuron)
    if not np.any(valid_mask):
        logger.warning(f"No valid neurons for mode {mode}")
        return
    
    valid_indices = np.where(valid_mask)[0]
    bps_valid = bps_per_neuron[valid_indices]
    sorted_idx = np.argsort(bps_valid)
    n = min(n_examples, len(sorted_idx))
    
    # Select neurons distributed across BPS range
    selected_global_idx = []
    for i in range(n):
        idx = sorted_idx[int(i * (len(sorted_idx) - 1) / (n - 1))] if n > 1 else sorted_idx[0]
        selected_global_idx.append(valid_indices[idx])
    
    for neuron_idx in selected_global_idx:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        neuron_uuid = test_dataset['cluster_uuids'][0][neuron_idx][:8] if 'cluster_uuids' in test_dataset.features else f"neu{neuron_idx}"
        neuron_region = test_dataset['cluster_regions'][0][neuron_idx] if 'cluster_regions' in test_dataset.features else "Unknown"
        bps = bps_per_neuron[neuron_idx]
        r2_psth = r2_per_neuron[neuron_idx, 0] if r2_per_neuron.ndim > 1 else np.nan
        r2_trial = r2_per_neuron[neuron_idx, 1] if r2_per_neuron.ndim > 1 else np.nan
        
        fig.suptitle(f'Neuron {neuron_idx} | Region: {neuron_region} | UUID: {neuron_uuid}\n'
                    f'BPS: {bps:.4f} | R2 PSTH: {r2_psth:.4f} | R2 Trial: {r2_trial:.4f} | Mode: {mode}',
                    fontsize=12, fontweight='bold')
        
        # Plot 1: BPS histogram
        ax1 = axes[0, 0]
        ax1.hist(bps_valid, bins=30, color='steelblue', alpha=0.5, edgecolor='black')
        ax1.axvline(bps, color='red', linewidth=2, linestyle='--', label=f'This neuron: {bps:.3f}')
        ax1.set_xlabel('Bits per Spike')
        ax1.set_ylabel('Count')
        ax1.set_title('BPS Distribution (Highlighted: This Neuron)')
        ax1.legend()
        
        # Plot 2: R2 scatter
        ax2 = axes[0, 1]
        if r2_per_neuron.ndim > 1 and r2_per_neuron.shape[0] > 1:
            r2_psth_all = r2_per_neuron[:, 0]
            r2_trial_all = r2_per_neuron[:, 1]
            valid_r2 = np.isfinite(r2_psth_all) & np.isfinite(r2_trial_all)
            ax2.scatter(r2_psth_all[valid_r2], r2_trial_all[valid_r2], alpha=0.3, s=10, c='steelblue', label='All neurons')
            ax2.scatter(r2_psth, r2_trial, c='red', s=100, marker='*', label='This neuron', zorder=5)
            ax2.set_xlabel('R2 PSTH')
            ax2.set_ylabel('R2 Trial')
            ax2.set_title('R2 Scatter Plot')
            ax2.legend()
            ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
        
        # Plot 3: BPS by region
        ax3 = axes[1, 0]
        if 'cluster_regions' in test_dataset.features:
            regions = np.array(test_dataset['cluster_regions'][0])
            unique_regions = np.unique(regions)
            region_bps = {region: bps_per_neuron[regions == region] for region in unique_regions}
            region_names = list(region_bps.keys())
            region_means = [np.nanmean(region_bps[r]) for r in region_names]
            region_stds = [np.nanstd(region_bps[r]) for r in region_names]
            
            bars = ax3.bar(range(len(region_names)), region_means, yerr=region_stds,
                          color='seagreen', alpha=0.7, edgecolor='black')
            ax3.set_xticks(range(len(region_names)))
            ax3.set_xticklabels(region_names, rotation=45, ha='right')
            ax3.set_ylabel('Mean BPS')
            ax3.set_title('BPS by Brain Region')
            if neuron_region in region_names:
                bars[region_names.index(neuron_region)].set_color('coral')
        
        # Plot 4: Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        metrics_text = f"""
        Evaluation Mode: {mode}
        
        Neuron Statistics:
        - Neuron Index: {neuron_idx}
        - Brain Region: {neuron_region}
        - UUID: {neuron_uuid}
        
        Performance Metrics:
        - Bits per Spike: {bps:.4f}
        - R2 PSTH: {r2_psth:.4f}
        - R2 Trial: {r2_trial:.4f}
        
        Population Statistics:
        - Mean BPS (all): {np.nanmean(bps_valid):.4f}
        - Std BPS (all): {np.nanstd(bps_valid):.4f}
        - Valid neurons: {len(bps_valid)}/{len(bps_per_neuron)}
        """
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        save_file = os.path.join(save_path, f'{mode}_neuron_{neuron_idx}_example.png')
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved example neuron plot: {save_file}")


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    # 延迟导入
    from utils.config_utils import config_from_kwargs, update_config
    from utils.eval_utils import load_model_data_local, co_smoothing_eval

    loader_kwargs = {
        "model_config": args.model_config,
        "trainer_config": args.trainer_config,
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "eid": args.eid,
        "mask_name": args.mask_name,
        "seed": args.seed,
    }

    logger.info("=" * 60)
    logger.info("Comprehensive Evaluation Configuration:")
    logger.info(f"  Model path: {args.model_path}")
    logger.info(f"  Dataset path: {args.dataset_path}")
    logger.info(f"  Save path: {args.save_path}")
    logger.info(f"  Evaluation modes: {args.eval_modes}")
    logger.info("=" * 60)

    logger.info("Loading model and dataset...")
    model, accelerator, test_dataset, test_dataloader = load_model_data_local(**loader_kwargs)

    for batch in test_dataloader:
        n_time_steps = batch["spikes_data"].shape[1]
        break
    logger.info(f"Number of time bins: {n_time_steps}")

    all_results = {}
    
    # 模式映射: eval_modes -> co_smoothing_eval mode
    mode_mapping = {
        "co_smooth": "per_neuron",
        "causal": "forward_pred",
        "temporal": "forward_pred",
        "inter_region": "inter_region",
        "intra_region": "intra_region",
    }

    # 为 forward_pred 模式设置默认的 held_out_list
    # 根据 NDT 论文 (Section 5.2)，forward prediction 通常在时间序列后半段选择时间点
    # 选择刺激呈现后的一些时间点进行预测
    default_held_out_list = [30, 35, 40, 45]  # 时间序列后半段，刺激响应高峰期
    
    # 如果用户提供了自定义的 held_out_list，则使用用户的值
    if args.held_out_list is not None:
        default_held_out_list = args.held_out_list
        logger.info(f"Using custom held_out_list: {default_held_out_list}")
    else:
        logger.info(f"Using default held_out_list for forward_pred: {default_held_out_list}")

    for eval_mode in args.eval_modes:
        mode_name = mode_mapping.get(eval_mode, eval_mode)
        logger.info(f"\n{'='*60}")
        logger.info(f"Running evaluation: {eval_mode} -> {mode_name}")
        logger.info(f"{'='*60}")
        
        eval_kwargs = {
            "method_name": "ssl",
            "mode": mode_name,
            "is_aligned": True,
            "n_time_steps": n_time_steps,
            "target_regions": ["all"],
            "n_jobs": args.n_jobs,
            "save_path": os.path.join(args.save_path, eval_mode),
            "subtract": None,
            "onset_alignment": [],
            "held_out_list": default_held_out_list if mode_name == "forward_pred" else None,
        }

        try:
            results = co_smoothing_eval(model, accelerator, test_dataloader, test_dataset, n=1, **eval_kwargs)
            
            mode_save_path = eval_kwargs["save_path"]
            bps_path = os.path.join(mode_save_path, "bps.npy")
            r2_path = os.path.join(mode_save_path, "r2.npy")
            
            bps_per_neuron = np.load(bps_path) if os.path.exists(bps_path) else np.array([])
            r2_per_neuron = np.load(r2_path) if os.path.exists(r2_path) else np.array([[np.nan, np.nan]])
            
            all_results[eval_mode] = {
                'bps_per_neuron': bps_per_neuron,
                'r2_per_neuron': r2_per_neuron,
                'summary': results,
            }
            
            # 生成示例神经元可视化
            logger.info(f"Generating example neuron plots for mode: {eval_mode}")
            plot_example_neurons(test_dataset, bps_per_neuron, r2_per_neuron, eval_mode, mode_save_path, args.example_neurons)
            
            logger.info(f"Results for {eval_mode}:")
            logger.info(f"  Mean BPS: {np.nanmean(bps_per_neuron):.4f} +/- {np.nanstd(bps_per_neuron):.4f}")
            if r2_per_neuron.ndim > 1:
                logger.info(f"  Mean R2 PSTH: {np.nanmean(r2_per_neuron[:, 0]):.4f}")
                logger.info(f"  Mean R2 Trial: {np.nanmean(r2_per_neuron[:, 1]):.4f}")
                
        except Exception as e:
            logger.error(f"Error running mode {eval_mode}: {e}")
            import traceback
            traceback.print_exc()
            all_results[eval_mode] = {
                'bps_per_neuron': np.array([]),
                'r2_per_neuron': np.array([[np.nan, np.nan]]),
                'error': str(e),
            }

    # 生成综合总结图
    logger.info(f"\n{'='*60}")
    logger.info("Generating comprehensive summary...")
    logger.info(f"{'='*60}")
    plot_comprehensive_summary(all_results, args.save_path)

    # 保存最终摘要 JSON
    final_summary = {}
    for mode, data in all_results.items():
        bps = data.get('bps_per_neuron', np.array([]))
        r2 = data.get('r2_per_neuron', np.array([[np.nan, np.nan]]))
        
        final_summary[mode] = {
            'mean_bps': float(np.nanmean(bps)) if len(bps) > 0 else np.nan,
            'std_bps': float(np.nanstd(bps)) if len(bps) > 0 else np.nan,
            'median_bps': float(np.nanmedian(bps)) if len(bps) > 0 else np.nan,
            'n_valid_neurons': int(np.sum(np.isfinite(bps))) if len(bps) > 0 else 0,
            'total_neurons': int(len(bps)) if len(bps) > 0 else 0,
            'mean_r2_psth': float(np.nanmean(r2[:, 0])) if r2.size > 0 else np.nan,
            'mean_r2_trial': float(np.nanmean(r2[:, 1])) if r2.size > 0 else np.nan,
        }
    
    summary_path = os.path.join(args.save_path, "final_summary.json")
    with open(summary_path, "w") as f:
        json.dump(final_summary, f, indent=2)
    logger.info(f"Saved final summary to {summary_path}")

    # 打印最终摘要
    logger.info(f"\n{'='*60}")
    logger.info("FINAL EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    for mode, stats in final_summary.items():
        logger.info(f"\n{mode.replace('_', ' ').title()}:")
        logger.info(f"  Mean BPS: {stats['mean_bps']:.4f} +/- {stats['std_bps']:.4f}")
        logger.info(f"  Median BPS: {stats['median_bps']:.4f}")
        logger.info(f"  Valid neurons: {stats['n_valid_neurons']}/{stats['total_neurons']}")
        logger.info(f"  Mean R2 PSTH: {stats['mean_r2_psth']:.4f}")
        logger.info(f"  Mean R2 Trial: {stats['mean_r2_trial']:.4f}")
    
    logger.info(f"\nAll results saved to: {args.save_path}")
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
