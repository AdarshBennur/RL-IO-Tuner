"""
Reward function for storage optimization RL agent.
Modular design for research experimentation.
"""

from typing import Dict, Optional
import numpy as np


class RewardFunction:
    """
    Computes reward based on storage performance metrics.
    
    Base formula: reward = α * throughput - β * latency - γ * penalty
    
    Weights can be adjusted for different optimization objectives.
    """
    
    def __init__(
        self,
        throughput_weight: float = 1.0,
        latency_weight: float = 0.5,
        penalty_weight: float = 0.1,
        target_throughput: float = 100.0,  # MB/s
        target_latency: float = 10.0,      # ms
    ):
        """
        Initialize reward function.
        
        Args:
            throughput_weight: α - Weight for throughput component
            latency_weight: β - Weight for latency penalty
            penalty_weight: γ - Weight for constraint violations
            target_throughput: Baseline throughput for normalization
            target_latency: Baseline latency for normalization
        """
        self.throughput_weight = throughput_weight
        self.latency_weight = latency_weight
        self.penalty_weight = penalty_weight
        self.target_throughput = target_throughput
        self.target_latency = target_latency
    
    def compute(self, metrics: Dict[str, float], prev_metrics: Optional[Dict[str, float]] = None) -> float:
        """
        Compute reward from performance metrics.
        
        Args:
            metrics: Current performance metrics
            prev_metrics: Previous metrics for delta computation (optional)
        
        Returns:
            Scalar reward value
        """
        # Extract key metrics
        throughput = metrics.get('read_throughput_mb', 0.0) + metrics.get('write_throughput_mb', 0.0)
        read_latency = metrics.get('avg_read_latency_ms', 0.0)
        write_latency = metrics.get('avg_write_latency_ms', 0.0)
        avg_latency = (read_latency + write_latency) / 2.0
        utilization = metrics.get('utilization', 0.0)
        
        # Throughput component (normalized, higher is better)
        throughput_reward = (throughput / self.target_throughput) * self.throughput_weight
        
        # Latency penalty (normalized, lower is better)
        latency_penalty = (avg_latency / self.target_latency) * self.latency_weight
        
        # Constraint penalties
        penalty = 0.0
        
        # Penalty for very high utilization (potential bottleneck)
        if utilization > 90.0:
            penalty += (utilization - 90.0) / 10.0
        
        # Penalty for excessive latency
        if avg_latency > self.target_latency * 2.0:
            penalty += (avg_latency - self.target_latency * 2.0) / self.target_latency
        
        # Apply penalty weight
        penalty *= self.penalty_weight
        
        # Compute final reward
        reward = throughput_reward - latency_penalty - penalty
        
        return reward
    
    def compute_improvement_reward(
        self,
        metrics: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> float:
        """
        Compute reward based on improvement over baseline.
        
        Args:
            metrics: Current metrics
            baseline_metrics: Baseline metrics
        
        Returns:
            Improvement-based reward
        """
        current_reward = self.compute(metrics)
        baseline_reward = self.compute(baseline_metrics)
        
        return current_reward - baseline_reward


class AdaptiveRewardFunction(RewardFunction):
    """
    Adaptive reward function that adjusts weights based on performance.
    
    Extension for future research: dynamically balance throughput vs latency
    based on workload characteristics.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.performance_history = []
    
    def compute(self, metrics: Dict[str, float], prev_metrics: Optional[Dict[str, float]] = None) -> float:
        """
        Compute adaptive reward.
        
        Future: Adjust weights based on observed performance patterns.
        Currently uses base implementation.
        """
        # Store metrics for adaptation
        self.performance_history.append(metrics)
        
        # Keep history bounded
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        # Use base reward computation
        return super().compute(metrics, prev_metrics)
    
    def adapt_weights(self):
        """
        Adapt reward weights based on performance history.
        
        Placeholder for future research extension.
        """
        # Future: Implement adaptive logic
        # Example: If latency is consistently high, increase latency_weight
        pass


class TailLatencyRewardFunction(RewardFunction):
    """
    Reward function emphasizing tail latency (p95, p99).
    
    Extension point for workloads sensitive to latency variance.
    """
    
    def __init__(self, tail_weight: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.tail_weight = tail_weight
    
    def compute(
        self,
        metrics: Dict[str, float],
        prev_metrics: Optional[Dict[str, float]] = None,
        tail_latencies: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute reward with tail latency consideration.
        
        Args:
            metrics: Current metrics
            prev_metrics: Previous metrics
            tail_latencies: Dictionary with 'p95', 'p99' latencies (optional)
        
        Returns:
            Reward with tail latency penalty
        """
        base_reward = super().compute(metrics, prev_metrics)
        
        if tail_latencies is None:
            return base_reward
        
        # Add tail latency penalty
        p99_latency = tail_latencies.get('p99', 0.0)
        if p99_latency > self.target_latency * 3.0:
            tail_penalty = (p99_latency / self.target_latency) * self.tail_weight
            base_reward -= tail_penalty
        
        return base_reward


def create_reward_function(reward_type: str = "base", **kwargs) -> RewardFunction:
    """
    Factory function for creating reward functions.
    
    Args:
        reward_type: Type of reward function ('base', 'adaptive', 'tail_latency')
        **kwargs: Additional arguments for reward function
    
    Returns:
        RewardFunction instance
    """
    if reward_type == "base":
        return RewardFunction(**kwargs)
    elif reward_type == "adaptive":
        return AdaptiveRewardFunction(**kwargs)
    elif reward_type == "tail_latency":
        return TailLatencyRewardFunction(**kwargs)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


if __name__ == "__main__":
    # Test reward functions
    reward_fn = RewardFunction(
        throughput_weight=1.0,
        latency_weight=0.5,
        target_throughput=100.0,
        target_latency=10.0
    )
    
    # Good performance
    good_metrics = {
        'read_throughput_mb': 80.0,
        'write_throughput_mb': 40.0,
        'avg_read_latency_ms': 8.0,
        'avg_write_latency_ms': 12.0,
        'utilization': 60.0,
    }
    
    # Poor performance
    poor_metrics = {
        'read_throughput_mb': 40.0,
        'write_throughput_mb': 20.0,
        'avg_read_latency_ms': 25.0,
        'avg_write_latency_ms': 30.0,
        'utilization': 95.0,
    }
    
    print("Reward Function Test")
    print("=" * 60)
    print(f"Good performance reward: {reward_fn.compute(good_metrics):.3f}")
    print(f"Poor performance reward: {reward_fn.compute(poor_metrics):.3f}")
    print()
    
    # Test improvement reward
    improvement = reward_fn.compute_improvement_reward(good_metrics, poor_metrics)
    print(f"Improvement over baseline: {improvement:.3f}")
