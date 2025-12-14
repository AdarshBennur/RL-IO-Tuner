"""
Evaluation script for trained DQN agent.
"""

import sys
import os
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.io_env import StorageIOEnv
from agent.dqn import DQNAgent
from benchmarks.run_fio import get_available_workloads


def evaluate_agent(
    model_path: str,
    device: str = "sda",
    workload_name: str = "rand_read",
    num_episodes: int = 5,
    max_steps: int = 20,
    fio_runtime: int = 10,
    output_dir: str = "logs/evaluation",
):
    """
    Evaluate trained DQN agent.
    
    Args:
        model_path: Path to trained model
        device: Block device
        workload_name: FIO workload
        num_episodes: Number of evaluation episodes
        max_steps: Max steps per episode
        fio_runtime: FIO runtime per step
        output_dir: Output directory
    """
    print("=" * 70)
    print("AGENT EVALUATION")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Workload: {workload_name}")
    print(f"Episodes: {num_episodes}")
    print()
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # Get workload
    workloads = get_available_workloads()
    if workload_name not in workloads:
        print(f"Error: Workload '{workload_name}' not found")
        return
    
    workload_path = workloads[workload_name]
    
    # Create environment
    env = StorageIOEnv(
        device=device,
        fio_workload=workload_path,
        fio_runtime=fio_runtime,
        normalize_state=True,
        history_length=3,
        log_dir=output_dir,
    )
    env.max_steps_per_episode = max_steps
    
    # Create and load agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        epsilon_start=0.0,  # No exploration during evaluation
    )
    
    agent.load(model_path)
    print(f"Loaded model from {model_path}")
    print()
    
    # Evaluation loop
    episode_rewards = []
    episode_metrics = []
    
    for episode in range(num_episodes):
        print(f"\n[Evaluation Episode {episode+1}/{num_episodes}]")
        
        state, info = env.reset()
        episode_reward = 0
        step_metrics = []
        
        while True:
            # Select action (greedy)
            action = agent.select_action(state, eval_mode=True)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Log
            print(f"  Step {info['step']}: "
                  f"action={action} (ra={env.READAHEAD_VALUES[action]}), "
                  f"reward={reward:.3f}")
            
            step_metrics.append(info['metrics'])
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_metrics.append(step_metrics)
        
        print(f"  Episode reward: {episode_reward:.3f}")
    
    # Compute statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Mean episode reward: {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"Min reward: {np.min(episode_rewards):.3f}")
    print(f"Max reward: {np.max(episode_rewards):.3f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "evaluation_results.json")
    
    with open(results_file, 'w') as f:
        json.dump({
            'model_path': model_path,
            'device': device,
            'workload': workload_name,
            'num_episodes': num_episodes,
            'episode_rewards': episode_rewards,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print("=" * 70)
    
    # Clean up
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained DQN agent")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model'
    )
    parser.add_argument('--device', type=str, default='sda')
    parser.add_argument('--workload', type=str, default='rand_read')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--fio-runtime', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='logs/evaluation')
    
    args = parser.parse_args()
    
    evaluate_agent(
        model_path=args.model,
        device=args.device,
        workload_name=args.workload,
        num_episodes=args.episodes,
        max_steps=args.steps,
        fio_runtime=args.fio_runtime,
        output_dir=args.output_dir,
    )
