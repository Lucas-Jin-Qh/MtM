"""
Run evaluation for all 4 masking modes
Generate: bps summary + single-neuron viz plots
"""
import sys
import os
import numpy as np

# Set working directory
work_dir = '/home/jqh/Workspace/IBL foundation model/MtM/src'
os.chdir(work_dir)
print('='*60)
print('Working directory:', work_dir)
print('='*60)

# Add src path (add parent directory so 'src' module can be found)
sys.path.append(os.path.dirname(work_dir))

from src.utils.eval_utils import load_model_data_local, co_smoothing_eval

# ========================
# 1. Configuration
# ========================

model_path = '../results/train/eid_4b00df29/num_session_1/model_NDT1/method_ssl/mask_all/stitch_True/model_best.pt'
dataset_path = '../data/4b00df29-3769-43be-bb40-128b1cba6d35_aligned'

# Extract eid from dataset_path for organizing eval results
dataset_eid = os.path.basename(dataset_path).split('_')[0]  # "4b00df29-3769-43be-bb40-128b1cba6d35_aligned" -> "4b00df29"

configs = {
    'model_config': 'configs/ndt1.yaml',
    'model_path': model_path,
    'trainer_config': 'configs/ssl_session_trainer.yaml',
    'dataset_path': dataset_path,
    'seed': 42,
}

# ========================
# 2. Load model and data
# ========================
print("\nLoading model and data...")
model, accelerator, dataset, dataloader = load_model_data_local(**configs)

# Print dataset info
print(f"\nDataset Info:")
print(f"  - Number of trials: {len(dataset)}")
print(f"  - Number of neurons: {len(dataset['cluster_regions'][0])}")
print(f"  - Dataset columns: {dataset.column_names}")

# Debug: check behavior variables actual values
print(f"\nDebug - Behavior variable values:")
choice_vals = np.array(dataset['choice'])
reward_vals = np.array(dataset['reward'])
block_vals = np.array(dataset['block'])
print(f"  - Choice unique values: {np.unique(choice_vals)}")
print(f"  - Reward unique values: {np.unique(reward_vals)}")
print(f"  - Block unique values: {np.unique(block_vals)}")

# Get brain regions info
regions = list(set(dataset['cluster_regions'][0]))
print(f"  - Number of brain regions: {len(regions)}")
print(f"  - Brain regions: {regions[:5]}..." if len(regions) > 5 else f"  - Brain regions: {regions}")

# Determine time steps (inferred from session_metadata or config)
n_time_steps = 60  # Default value from config
print(f"  - Time steps: {n_time_steps}")

# ========================
# 3. Define evaluation tasks
# ========================

tasks = {
    'co-smooth': {
        'mode': 'per_neuron',
        'target_regions': None,
        'held_out_list': None,
        'description': 'N-1 prediction: mask one neuron, predict remaining neurons',
    },
    'causal': {
        'mode': 'forward_pred',
        'target_regions': None,
        'held_out_list': list(range(n_time_steps // 2, n_time_steps)),  # Predict second half
        'description': 'Causal prediction: use historical time steps to predict future',
    },
    'inter-region': {
        'mode': 'inter_region',
        'target_regions': ['all'],
        'held_out_list': None,
        'description': 'Inter-region: mask one brain region, predict neurons in that region',
    },
    'intra-region': {
        'mode': 'intra_region',
        'target_regions': ['all'],
        'held_out_list': None,
        'description': 'Intra-region: mask other brain regions, predict target region neurons',
    },
}

# ========================
# 4. Run evaluation
# ========================

results_summary = {}

for task_name, task_config in tasks.items():
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"Description: {task_config['description']}")
    print('='*60)

    save_path = f'figs/eval/eid_{dataset_eid}/{task_name}'

    eval_configs = {
        'subtract': 'task',
        'onset_alignment': [n_time_steps // 3],  # ~1/3 position as alignment point
        'method_name': task_name,
        'save_path': save_path,
        'mode': task_config['mode'],
        'n_time_steps': n_time_steps,
        'is_aligned': True,
        'target_regions': task_config['target_regions'],
        'held_out_list': task_config['held_out_list'],
        'n_jobs': 8,
        # Custom var_value2label with numeric format
        'var_value2label': {
            'block': {(0.2,): "block: 0.20",
                      (0.5,): "block: 0.50",
                      (0.8,): "block: 0.80",
                      (0,): "block: 0.00",
                      (1,): "block: 1.00",
                      (2,): "block: 2.00"},
            'choice': {(-1.0,): "choice: -1.00",
                       (1.0,): "choice: 1.00",
                       (0,): "choice: 0.00",
                       (1,): "choice: 1.00"},
            'reward': {(0.,): "reward: 0.00",
                       (1.,): "reward: 1.00"},
        },
    }

    # Run evaluation
    try:
        result = co_smoothing_eval(model, accelerator, dataloader, dataset, **eval_configs)
        results_summary[task_name] = result
        print(f"\n✅ {task_name} evaluation completed!")
        print(f"   Output directory: {save_path}")
    except Exception as e:
        print(f"\n❌ {task_name} evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        results_summary[task_name] = None

# ========================
# 5. Print summary results
# ========================

print("\n" + "="*60)
print("Evaluation Results Summary")
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
        print("   Evaluation failed")

print("\n" + "="*60)
print("All tasks completed!")
print("="*60)
print(f"\nFigures saved to: {os.path.join(work_dir, 'figs/eval/')}")
