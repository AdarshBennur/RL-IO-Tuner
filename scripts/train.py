"""
Main training script for DQN agent.
Convenience wrapper for RL-controlled experiment.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.rl_controlled import run_rl_experiment


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train DQN agent for storage parameter optimization"
    )
    parser.add_argument(
        '--device',
        type=str,
        default='sda',
        help='Block device name (e.g., sda, nvme0n1)'
    )
    parser.add_argument(
        '--workload',
        type=str,
        default='rand_read',
        help='FIO workload (seq_read, rand_read, seq_write, rand_write)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=50,
        help='Number of training episodes'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=20,
        help='Maximum steps per episode'
    )
    parser.add_argument(
        '--fio-runtime',
        type=int,
        default=10,
        help='FIO benchmark runtime per step (seconds)'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs/training',
        help='Directory for training logs'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='results/models/dqn_agent.pth',
        help='Path to save trained model'
    )
    
    args = parser.parse_args()
    
    print("Starting DQN training...")
    print()
    
    run_rl_experiment(
        device=args.device,
        workload_name=args.workload,
        num_episodes=args.episodes,
        max_steps_per_episode=args.steps,
        fio_runtime=args.fio_runtime,
        output_dir=args.log_dir,
        model_save_path=args.model_path,
    )
