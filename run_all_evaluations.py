"""
运行所有 4 种 masking 模式的评估脚本
生成: bps summary + single-neuron viz plots
"""
import sys
import os
import numpy as np

# 设置工作目录
work_dir = '/home/jqh/Workspace/IBL foundation model/MtM'
os.chdir(work_dir)
print('='*60)
print('工作目录:', work_dir)
print('='*60)

# 添加 src 路径
sys.path.append(os.path.join(work_dir, 'src'))

from src.utils.eval_utils import load_model_data_local, co_smoothing_eval

# ========================
# 1. 配置参数
# ========================

model_path = 'results/train/num_session_1/model_NDT1/method_ssl/mask_all/stitch_True/model_best.pt'
dataset_path = 'data/4b00df29-3769-43be-bb40-128b1cba6d35_aligned'

configs = {
    'model_config': 'src/configs/ndt1.yaml',
    'model_path': model_path,
    'trainer_config': 'src/configs/ssl_session_trainer.yaml',
    'dataset_path': dataset_path,
    'seed': 42,
}

# ========================
# 2. 加载模型和数据
# ========================
print("\n正在加载模型和数据...")
model, accelerator, dataset, dataloader = load_model_data_local(**configs)

# 查看数据集信息
print(f"\n数据集信息:")
print(f"  - Trial 数量: {len(dataset)}")
print(f"  - 神经元数量: {len(dataset['cluster_regions'][0])}")
print(f"  - 数据集列名: {dataset.column_names}")

# 调试：检查 behavior 变量的实际值
print(f"\n调试 - Behavior 变量值:")
choice_vals = np.array(dataset['choice'])
reward_vals = np.array(dataset['reward'])
block_vals = np.array(dataset['block'])
print(f"  - choice 唯一值: {np.unique(choice_vals)}")
print(f"  - reward 唯一值: {np.unique(reward_vals)}")
print(f"  - block 唯一值: {np.unique(block_vals)}")

# 获取脑区信息
regions = list(set(dataset['cluster_regions'][0]))
print(f"  - 脑区数量: {len(regions)}")
print(f"  - 脑区列表: {regions[:5]}..." if len(regions) > 5 else f"  - 脑区列表: {regions}")

# 确定时间步数（从 session_metadata 或 dataloader 推断）
n_time_steps = 60  # 默认值，从配置文件获取
print(f"  - 时间步数: {n_time_steps}")

# ========================
# 3. 定义评估任务
# ========================

tasks = {
    'co-smooth': {
        'mode': 'per_neuron',
        'target_regions': None,
        'held_out_list': None,
        'description': 'N-1 预测: mask 一个神经元，预测其余神经元',
    },
    'causal': {
        'mode': 'forward_pred',
        'target_regions': None,
        'held_out_list': list(range(n_time_steps // 2, n_time_steps)),  # 预测后半段
        'description': '因果预测: 使用历史时间步预测未来时间步',
    },
    'inter-region': {
        'mode': 'inter_region',
        'target_regions': ['all'],
        'held_out_list': None,
        'description': '跨脑区: mask 一个脑区，预测该脑区神经元',
    },
    'intra-region': {
        'mode': 'intra_region',
        'target_regions': ['all'],
        'held_out_list': None,
        'description': '区域内: mask 其他脑区，预测目标脑区神经元',
    },
}

# ========================
# 4. 运行评估
# ========================

results_summary = {}

for task_name, task_config in tasks.items():
    print(f"\n{'='*60}")
    print(f"任务: {task_name}")
    print(f"描述: {task_config['description']}")
    print('='*60)

    save_path = f'figs/eval/{task_name}'

    eval_configs = {
        'subtract': 'task',
        'onset_alignment': [n_time_steps // 3],  # 约 1/3 位置作为对齐点
        'method_name': task_name,
        'save_path': save_path,
        'mode': task_config['mode'],
        'n_time_steps': n_time_steps,
        'is_aligned': True,
        'target_regions': task_config['target_regions'],
        'held_out_list': task_config['held_out_list'],
        'n_jobs': 8,
    }

    # 运行评估
    try:
        result = co_smoothing_eval(model, accelerator, dataloader, dataset, **eval_configs)
        results_summary[task_name] = result
        print(f"\n✅ {task_name} 评估完成!")
        print(f"   输出目录: {save_path}")
    except Exception as e:
        print(f"\n❌ {task_name} 评估失败: {e}")
        import traceback
        traceback.print_exc()
        results_summary[task_name] = None

# ========================
# 5. 打印汇总结果
# ========================

print("\n" + "="*60)
print("评估结果汇总")
print("="*60)

for task_name, result in results_summary.items():
    print(f"\n📊 {task_name}:")
    if result:
        mode_key = tasks[task_name]['mode']
        print(f"   Mean BPS: {result.get(f'{mode_key}_mean_bps', 'N/A'):.4f}")
        print(f"   Std BPS:  {result.get(f'{mode_key}_std_bps', 'N/A'):.4f}")
        print(f"   Mean R² (PSTH): {result.get(f'{mode_key}_mean_r2_psth', 'N/A'):.4f}")
        print(f"   Mean R² (Trial): {result.get(f'{mode_key}_mean_r2_trial', 'N/A'):.4f}")
    else:
        print("   评估失败")

print("\n" + "="*60)
print("所有任务完成!")
print("="*60)
print(f"\n图表保存在: {os.path.join(work_dir, 'figs/eval/')}")
