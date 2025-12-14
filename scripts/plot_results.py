"""
Plot training and evaluation results.
"""

import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import List, Dict, Optional


def plot_training_rewards(log_file: str, output_dir: str = "results/plots"):
    """
    Plot training reward convergence.
    
    Args:
        log_file: Path to training log JSON
        output_dir: Output directory for plots
    """
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    episode_rewards = data['episode_rewards']
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot raw rewards
    plt.plot(episode_rewards, alpha=0.3, label='Episode Reward')
    
    # Plot moving average
    window = 10
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_rewards)), moving_avg, 
                linewidth=2, label=f'{window}-Episode Moving Average')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Reward Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'training_rewards.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_baseline_comparison(baseline_file: str, output_dir: str = "results/plots"):
    """
    Plot baseline configuration comparison.
    
    Args:
        baseline_file: Path to baseline results JSON
        output_dir: Output directory
    """
    with open(baseline_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    configs = [r['config'] for r in results]
    read_iops = [r['metrics']['read_iops'] for r in results]
    throughput = [r['metrics']['read_throughput_mb'] for r in results]
    latency = [r['metrics']['avg_read_latency_ms'] for r in results]
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # IOPS
    axes[0].bar(configs, read_iops, color='steelblue')
    axes[0].set_xlabel('Configuration')
    axes[0].set_ylabel('Read IOPS')
    axes[0].set_title('Read IOPS by Configuration')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Throughput
    axes[1].bar(configs, throughput, color='seagreen')
    axes[1].set_xlabel('Configuration')
    axes[1].set_ylabel('Throughput (MB/s)')
    axes[1].set_title('Read Throughput by Configuration')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Latency
    axes[2].bar(configs, latency, color='coral')
    axes[2].set_xlabel('Configuration')
    axes[2].set_ylabel('Avg Latency (ms)')
    axes[2].set_title('Read Latency by Configuration')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'baseline_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_episode_logs(log_dir: str, output_dir: str = "results/plots"):
    """
    Plot action distribution from episode logs.
    
    Args:
        log_dir: Directory containing episode JSON logs
        output_dir: Output directory
    """
    # Load all episode logs
    log_files = sorted(glob.glob(os.path.join(log_dir, 'episode_*.json')))
    
    if not log_files:
        print(f"No episode logs found in {log_dir}")
        return
    
    print(f"Found {len(log_files)} episode logs")
    
    # Collect action statistics
    action_counts = {}
    
    for log_file in log_files[:20]:  # Use first 20 episodes
        with open(log_file, 'r') as f:
            steps = json.load(f)
        
        for step in steps:
            action = step['action']
            action_counts[action] = action_counts.get(action, 0) + 1
    
    if not action_counts:
        print("No actions found in logs")
        return
    
    # Plot action distribution
    actions = sorted(action_counts.keys())
    counts = [action_counts[a] for a in actions]
    
    plt.figure(figsize=(8, 6))
    plt.bar(actions, counts, color='mediumpurple')
    plt.xlabel('Action Index')
    plt.ylabel('Frequency')
    plt.title('Action Distribution (First 20 Episodes)')
    plt.xticks(actions)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'action_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def generate_all_plots(
    training_log: Optional[str] = None,
    baseline_log: Optional[str] = None,
    episode_log_dir: Optional[str] = None,
    output_dir: str = "results/plots"
):
    """
    Generate all available plots.
    
    Args:
        training_log: Path to training log
        baseline_log: Path to baseline log
        episode_log_dir: Directory with episode logs
        output_dir: Output directory
    """
    print("=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    print()
    
    if training_log and os.path.exists(training_log):
        print("Plotting training rewards...")
        plot_training_rewards(training_log, output_dir)
    
    if baseline_log and os.path.exists(baseline_log):
        print("Plotting baseline comparison...")
        plot_baseline_comparison(baseline_log, output_dir)
    
    if episode_log_dir and os.path.exists(episode_log_dir):
        print("Plotting action distribution...")
        plot_episode_logs(episode_log_dir, output_dir)
    
    print()
    print("=" * 70)
    print(f"All plots saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate result plots")
    parser.add_argument('--training-log', type=str, help='Training log JSON')
    parser.add_argument('--baseline-log', type=str, help='Baseline log JSON')
    parser.add_argument('--episode-log-dir', type=str, help='Episode logs directory')
    parser.add_argument('--output-dir', type=str, default='results/plots')
    
    args = parser.parse_args()
    
    # Auto-detect logs if not specified
    if args.training_log is None:
        logs = glob.glob('logs/rl_controlled/training_log_*.json')
        if logs:
            args.training_log = max(logs, key=os.path.getmtime)
            print(f"Auto-detected training log: {args.training_log}")
    
    if args.baseline_log is None:
        logs = glob.glob('logs/baseline/baseline_*.json')
        if logs:
            args.baseline_log = max(logs, key=os.path.getmtime)
            print(f"Auto-detected baseline log: {args.baseline_log}")
    
    if args.episode_log_dir is None:
        if os.path.exists('logs/training'):
            args.episode_log_dir = 'logs/training'
            print(f"Using episode log dir: {args.episode_log_dir}")
    
    generate_all_plots(
        training_log=args.training_log,
        baseline_log=args.baseline_log,
        episode_log_dir=args.episode_log_dir,
        output_dir=args.output_dir,
    )
