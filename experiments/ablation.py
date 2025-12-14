"""
Ablation study experiment.
Placeholder for testing different reward functions, state representations, etc.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.rl_controlled import run_rl_experiment
from env.reward import create_reward_function


def run_ablation_study(
    device: str = "sda",
    workload_name: str = "rand_read",
    num_episodes: int = 30,
):
    """
    Run ablation study to analyze component contributions.
    
    Future extensions:
    - Test different reward function weights
    - Compare state representations (with/without history)
    - Evaluate different DQN hyperparameters
    - Test action space discretization strategies
    
    Args:
        device: Target block device
        workload_name: FIO workload name
        num_episodes: Episodes per configuration
    """
    print("=" * 70)
    print("ABLATION STUDY (Placeholder)")
    print("=" * 70)
    print("This is a placeholder for future ablation experiments.")
    print()
    print("Potential ablation dimensions:")
    print("1. Reward function weights (throughput vs latency)")
    print("2. State representation (history length, normalization)")
    print("3. Action space granularity")
    print("4. DQN hyperparameters (learning rate, epsilon decay)")
    print("5. Target network update frequency")
    print()
    print("To run basic RL experiment, use:")
    print("  python experiments/rl_controlled.py")
    print("=" * 70)
    
    # Example: Run with different reward configurations
    # This is a template for future research
    
    reward_configs = [
        {'throughput_weight': 1.0, 'latency_weight': 0.5, 'name': 'balanced'},
        {'throughput_weight': 1.5, 'latency_weight': 0.3, 'name': 'throughput_focused'},
        {'throughput_weight': 0.7, 'latency_weight': 0.8, 'name': 'latency_focused'},
    ]
    
    print("\nAblation configurations defined:")
    for config in reward_configs:
        print(f"  - {config['name']}: throughput={config['throughput_weight']}, "
              f"latency={config['latency_weight']}")
    
    print("\nImplementation of full ablation study is left for future research.")
    print("Each configuration would be run with run_rl_experiment() and results compared.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument('--device', type=str, default='sda')
    parser.add_argument('--workload', type=str, default='rand_read')
    parser.add_argument('--episodes', type=int, default=30)
    
    args = parser.parse_args()
    
    run_ablation_study(
        device=args.device,
        workload_name=args.workload,
        num_episodes=args.episodes,
    )
