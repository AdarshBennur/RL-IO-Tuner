"""
Gymnasium environment for Linux storage parameter optimization.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Optional, Tuple, Any
import time
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.read_stats import DiskStatsReader
from system.apply_params import KernelParamController
from env.metrics import MetricsExtractor
from env.reward import RewardFunction


class StorageIOEnv(gym.Env):
    """
    Gymnasium environment for RL-based storage parameter tuning.
    
    State: Normalized disk I/O metrics with temporal history
    Action: Discrete kernel parameter values (read_ahead_kb)
    Reward: Performance-based (throughput - latency penalty)
    """
    
    # Discrete action space: readahead values in KB
    READAHEAD_VALUES = [128, 256, 512, 1024, 2048, 4096]
    
    def __init__(
        self,
        device: str = "sda",
        fio_workload: Optional[str] = None,
        fio_runtime: int = 10,
        normalize_state: bool = True,
        history_length: int = 3,
        reward_config: Optional[Dict] = None,
        log_dir: str = "logs/training",
    ):
        """
        Initialize storage I/O environment.
        
        Args:
            device: Block device name (e.g., 'sda', 'nvme0n1')
            fio_workload: Path to FIO config file (None for metric collection only)
            fio_runtime: FIO runtime in seconds
            normalize_state: Normalize metrics for state
            history_length: Number of historical observations in state
            reward_config: Reward function configuration
            log_dir: Directory for logging
        """
        super().__init__()
        
        self.device = device
        self.fio_workload = fio_workload
        self.fio_runtime = fio_runtime
        self.log_dir = log_dir
        
        # Initialize components
        self.stats_reader = DiskStatsReader(device)
        self.param_controller = KernelParamController(device)
        self.metrics_extractor = MetricsExtractor(
            normalize=normalize_state,
            history_length=history_length
        )
        
        reward_cfg = reward_config or {}
        self.reward_fn = RewardFunction(**reward_cfg)
        
        # Define spaces
        state_dim = self.metrics_extractor.get_state_dim()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        self.action_space = gym.spaces.Discrete(len(self.READAHEAD_VALUES))
        
        # Episode tracking
        self.current_step = 0
        self.max_steps_per_episode = 50
        self.episode_count = 0
        
        # Logging
        os.makedirs(log_dir, exist_ok=True)
        self.episode_log = []
    
    def _collect_metrics(self) -> Dict[str, float]:
        """
        Collect current disk metrics.
        
        Returns:
            Dictionary of metrics
        """
        # Wait for metrics to settle
        time.sleep(1)
        
        delta_metrics = self.stats_reader.get_delta_stats()
        instant_metrics = self.stats_reader.get_instant_stats()
        
        if delta_metrics is None or instant_metrics is None:
            # Return zero metrics on error
            return {key: 0.0 for key in MetricsExtractor.METRIC_NAMES}
        
        return {**delta_metrics, **instant_metrics}
    
    def _run_fio_workload(self) -> Dict[str, float]:
        """
        Run FIO workload and collect metrics.
        
        Returns:
            Performance metrics from FIO
        """
        if self.fio_workload is None:
            # No workload specified, just collect passive metrics
            return self._collect_metrics()
        
        # Import here to avoid circular dependency
        from benchmarks.run_fio import run_fio_benchmark
        
        # Run FIO benchmark
        fio_results = run_fio_benchmark(
            config_file=self.fio_workload,
            device=self.device,
            runtime=self.fio_runtime
        )
        
        if fio_results is None:
            print("Warning: FIO benchmark failed, using passive metrics")
            return self._collect_metrics()
        
        return fio_results
    
    def _apply_action(self, action: int) -> bool:
        """
        Apply action (set kernel parameter).
        
        Args:
            action: Action index
        
        Returns:
            True if successful
        """
        readahead_kb = self.READAHEAD_VALUES[action]
        success = self.param_controller.set_read_ahead_kb(readahead_kb)
        
        if not success:
            print(f"Warning: Failed to set read_ahead_kb={readahead_kb}")
        
        return success
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
        
        Returns:
            Tuple of (initial_state, info)
        """
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_count += 1
        self.episode_log = []
        
        # Reset metrics extractor
        self.metrics_extractor.reset()
        
        # Restore default parameters
        self.param_controller.restore_defaults()
        
        # Collect initial metrics
        initial_metrics = self._collect_metrics()
        initial_state = self.metrics_extractor.process_metrics(initial_metrics)
        
        info = {
            'episode': self.episode_count,
            'metrics': initial_metrics,
        }
        
        return initial_state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Action to take (index into READAHEAD_VALUES)
        
        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        self.current_step += 1
        
        # Apply action
        self._apply_action(action)
        
        # Run workload and collect metrics
        metrics = self._run_fio_workload()
        
        # Process metrics into state
        state = self.metrics_extractor.process_metrics(metrics)
        
        # Compute reward
        reward = self.reward_fn.compute(metrics)
        
        # Check termination
        terminated = False  # Task is continuing
        truncated = self.current_step >= self.max_steps_per_episode
        
        # Build info dictionary
        info = {
            'step': self.current_step,
            'episode': self.episode_count,
            'action': action,
            'readahead_kb': self.READAHEAD_VALUES[action],
            'metrics': metrics,
            'reward': reward,
        }
        
        # Log step
        self.episode_log.append(info)
        
        # Save episode log on completion
        if terminated or truncated:
            self._save_episode_log()
        
        return state, reward, terminated, truncated, info
    
    def _save_episode_log(self):
        """Save episode log to file."""
        import json
        
        log_file = os.path.join(
            self.log_dir,
            f"episode_{self.episode_count:04d}.json"
        )
        
        with open(log_file, 'w') as f:
            json.dump(self.episode_log, f, indent=2)
    
    def close(self):
        """Clean up environment."""
        # Restore defaults
        self.param_controller.restore_defaults()


if __name__ == "__main__":
    # Test the environment
    print("Testing StorageIOEnv")
    print("=" * 60)
    
    env = StorageIOEnv(
        device="sda",
        fio_workload=None,  # No FIO for testing
        normalize_state=True,
        history_length=3,
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Readahead values: {env.READAHEAD_VALUES}")
    print()
    
    # Run a few steps
    state, info = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Initial metrics: {info['metrics']}")
    print()
    
    for i in range(3):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {i+1}:")
        print(f"  Action: {action} (readahead_kb={env.READAHEAD_VALUES[action]})")
        print(f"  Reward: {reward:.3f}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        print()
        
        if terminated or truncated:
            break
    
    env.close()
    print("Environment test complete")
