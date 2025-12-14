"""
RL-controlled experiment: DQN agent learns optimal parameters.
"""

import sys
import os
import json
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.io_env import StorageIOEnv
from agent.dqn import DQNAgent
from benchmarks.run_fio import get_available_workloads


def run_rl_experiment(
    device: str = "sda",
    workload_name: str = "rand_read",
    num_episodes: int = 50,
    max_steps_per_episode: int = 20,
    fio_runtime: int = 10,
    output_dir: str = "logs/rl_controlled",
    model_save_path: str = "results/models/dqn_agent.pth",
):
    """
    Run RL-controlled experiment.
    
    Args:
        device: Target block device
        workload_name: FIO workload name
        num_episodes: Number of training episodes
        max_steps_per_episode: Max steps per episode
        fio_runtime: FIO runtime per step
        output_dir: Output directory
        model_save_path: Path to save trained model
    """
    print("=" * 70)
    print("RL-CONTROLLED EXPERIMENT: DQN Agent Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Workload: {workload_name}")
    print(f"Episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print(f"FIO runtime per step: {fio_runtime}s")
    print()
    
    # Get workload path
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
    env.max_steps_per_episode = max_steps_per_episode
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=(128, 128),
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.98,
        buffer_capacity=5000,
        batch_size=32,
        target_update_freq=5,
    )
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action space (read_ahead_kb): {env.READAHEAD_VALUES}")
    print()
    
    # Training loop
    episode_rewards = []
    episode_lengths = []
    training_start = time.time()
    
    for episode in range(num_episodes):
        episode_start = time.time()
        state, info = env.reset()
        episode_reward = 0
        step = 0
        
        print(f"\n[Episode {episode+1}/{num_episodes}]")
        print(f"  Epsilon: {agent.epsilon:.4f}")
        
        while True:
            # Select action
            action = agent.select_action(state, eval_mode=False)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train()
            
            # Update state
            state = next_state
            episode_reward += reward
            step += 1
            
            # Log step
            print(f"  Step {step}: action={action} (ra={env.READAHEAD_VALUES[action]}), "
                  f"reward={reward:.3f}, loss={loss:.4f if loss else 0:.4f}")
            
            if done:
                break
        
        # Episode complete
        episode_time = time.time() - episode_start
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Log episode summary
        print(f"  Episode reward: {episode_reward:.3f}")
        print(f"  Episode length: {step}")
        print(f"  Time: {episode_time:.1f}s")
        print(f"  Avg reward (last 10): {np.mean(episode_rewards[-10:]):.3f}")
        
        # Save checkpoint periodically
        if (episode + 1) % 10 == 0:
            checkpoint_path = model_save_path.replace('.pth', f'_ep{episode+1}.pth')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            agent.save(checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Training complete
    training_time = time.time() - training_start
    
    # Save final model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    agent.save(model_save_path)
    
    # Save training log
    log_file = os.path.join(output_dir, f"training_log_{int(time.time())}.json")
    with open(log_file, 'w') as f:
        json.dump({
            'device': device,
            'workload': workload_name,
            'num_episodes': num_episodes,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'training_time': training_time,
        }, f, indent=2)
    
    # Clean up
    env.close()
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total episodes: {num_episodes}")
    print(f"Total training time: {training_time:.1f}s")
    print(f"Average episode reward: {np.mean(episode_rewards):.3f}")
    print(f"Best episode reward: {np.max(episode_rewards):.3f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"\nModel saved: {model_save_path}")
    print(f"Training log: {log_file}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RL-controlled experiment")
    parser.add_argument('--device', type=str, default='sda', help='Block device')
    parser.add_argument('--workload', type=str, default='rand_read', help='FIO workload')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes')
    parser.add_argument('--steps', type=int, default=20, help='Max steps per episode')
    parser.add_argument('--fio-runtime', type=int, default=10, help='FIO runtime (s)')
    parser.add_argument('--output-dir', type=str, default='logs/rl_controlled')
    parser.add_argument('--model-path', type=str, default='results/models/dqn_agent.pth')
    
    args = parser.parse_args()
    
    run_rl_experiment(
        device=args.device,
        workload_name=args.workload,
        num_episodes=args.episodes,
        max_steps_per_episode=args.steps,
        fio_runtime=args.fio_runtime,
        output_dir=args.output_dir,
        model_save_path=args.model_path,
    )
