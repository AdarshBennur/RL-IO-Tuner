"""
Metric extraction and normalization for RL state representation.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque


class MetricNormalizer:
    """
    Normalizes metrics for RL state space using running statistics.
    
    Uses Welford's online algorithm for stable mean/variance computation.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize normalizer.
        
        Args:
            window_size: Rolling window size for statistics
        """
        self.window_size = window_size
        self.metrics: Dict[str, deque] = {}
        self.counts: Dict[str, int] = {}
        self.means: Dict[str, float] = {}
        self.m2s: Dict[str, float] = {}  # Sum of squared differences
    
    def update(self, metric_name: str, value: float):
        """
        Update running statistics for a metric.
        
        Args:
            metric_name: Name of the metric
            value: New value
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = deque(maxlen=self.window_size)
            self.counts[metric_name] = 0
            self.means[metric_name] = 0.0
            self.m2s[metric_name] = 0.0
        
        self.metrics[metric_name].append(value)
        
        # Update Welford statistics
        self.counts[metric_name] += 1
        count = self.counts[metric_name]
        delta = value - self.means[metric_name]
        self.means[metric_name] += delta / count
        delta2 = value - self.means[metric_name]
        self.m2s[metric_name] += delta * delta2
    
    def normalize(self, metric_name: str, value: float, clip: float = 10.0) -> float:
        """
        Normalize a metric value using running statistics.
        
        Args:
            metric_name: Name of the metric
            value: Value to normalize
            clip: Maximum absolute value after normalization
        
        Returns:
            Normalized value (z-score)
        """
        if metric_name not in self.means:
            return 0.0
        
        mean = self.means[metric_name]
        count = self.counts[metric_name]
        
        if count < 2:
            return 0.0
        
        # Compute standard deviation
        variance = self.m2s[metric_name] / (count - 1)
        std = np.sqrt(variance)
        
        if std < 1e-8:
            return 0.0
        
        # Z-score normalization
        normalized = (value - mean) / std
        
        # Clip extreme values
        return np.clip(normalized, -clip, clip)
    
    def get_stats(self, metric_name: str) -> Optional[Dict[str, float]]:
        """
        Get statistics for a metric.
        
        Args:
            metric_name: Name of the metric
        
        Returns:
            Dictionary with mean, std, min, max
        """
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return None
        
        values = list(self.metrics[metric_name])
        count = self.counts[metric_name]
        variance = self.m2s[metric_name] / max(count - 1, 1)
        
        return {
            'mean': self.means[metric_name],
            'std': np.sqrt(variance),
            'min': min(values),
            'max': max(values),
            'count': count,
        }


class MetricsExtractor:
    """
    Extracts and prepares metrics for RL environment state.
    """
    
    # Metric names used in state representation
    METRIC_NAMES = [
        'read_iops',
        'write_iops',
        'read_throughput_mb',
        'write_throughput_mb',
        'avg_read_latency_ms',
        'avg_write_latency_ms',
        'utilization',
        'queue_depth',
    ]
    
    def __init__(self, normalize: bool = True, history_length: int = 3):
        """
        Initialize metrics extractor.
        
        Args:
            normalize: Whether to normalize metrics
            history_length: Number of historical observations to include in state
        """
        self.normalize = normalize
        self.history_length = history_length
        self.normalizer = MetricNormalizer() if normalize else None
        self.history: deque = deque(maxlen=history_length)
    
    def process_metrics(self, raw_metrics: Dict[str, float]) -> np.ndarray:
        """
        Process raw metrics into RL state vector.
        
        Args:
            raw_metrics: Raw metrics from disk stats reader
        
        Returns:
            State vector as numpy array
        """
        # Extract relevant metrics
        values = []
        for metric_name in self.METRIC_NAMES:
            value = raw_metrics.get(metric_name, 0.0)
            
            if self.normalize:
                self.normalizer.update(metric_name, value)
                normalized_value = self.normalizer.normalize(metric_name, value)
                values.append(normalized_value)
            else:
                values.append(value)
        
        # Convert to numpy array
        current_state = np.array(values, dtype=np.float32)
        
        # Add to history
        self.history.append(current_state)
        
        # Build state with history
        if len(self.history) < self.history_length:
            # Pad with zeros if not enough history
            padding = [np.zeros_like(current_state) for _ in range(self.history_length - len(self.history))]
            state_history = padding + list(self.history)
        else:
            state_history = list(self.history)
        
        # Flatten history into single vector
        state = np.concatenate(state_history)
        
        return state
    
    def get_state_dim(self) -> int:
        """
        Get dimension of state vector.
        
        Returns:
            State dimension
        """
        return len(self.METRIC_NAMES) * self.history_length
    
    def reset(self):
        """Reset history (call at episode start)."""
        self.history.clear()
    
    def get_metric_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all metrics.
        
        Returns:
            Dictionary mapping metric names to statistics
        """
        if not self.normalize:
            return {}
        
        stats = {}
        for metric_name in self.METRIC_NAMES:
            metric_stats = self.normalizer.get_stats(metric_name)
            if metric_stats:
                stats[metric_name] = metric_stats
        
        return stats


if __name__ == "__main__":
    # Test the extractor
    extractor = MetricsExtractor(normalize=True, history_length=3)
    
    print(f"State dimension: {extractor.get_state_dim()}")
    print()
    
    # Simulate some metrics
    for i in range(5):
        raw_metrics = {
            'read_iops': 100.0 + i * 10,
            'write_iops': 50.0 + i * 5,
            'read_throughput_mb': 200.0 + i * 20,
            'write_throughput_mb': 100.0 + i * 10,
            'avg_read_latency_ms': 5.0 + i * 0.5,
            'avg_write_latency_ms': 10.0 + i * 1.0,
            'utilization': 50.0 + i * 5,
            'queue_depth': 2.0,
        }
        
        state = extractor.process_metrics(raw_metrics)
        print(f"Step {i}: State shape = {state.shape}, Sample values = {state[:5]}")
    
    print("\nMetric statistics:")
    for metric, stats in extractor.get_metric_stats().items():
        print(f"  {metric}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
