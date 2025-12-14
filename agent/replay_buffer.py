"""
Experience replay buffer for DQN training.
"""

import numpy as np
from typing import Tuple, Optional
import random


class ReplayBuffer:
    """
    Circular buffer for storing and sampling experience tuples.
    
    Stores: (state, action, reward, next_state, done)
    """
    
    def __init__(self, capacity: int, state_dim: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            state_dim: Dimension of state vector
        """
        self.capacity = capacity
        self.state_dim = state_dim
        
        # Pre-allocate arrays for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        self.position = 0
        self.size = 0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Episode termination flag
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Optional[Tuple[np.ndarray, ...]]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) or None if buffer too small
        """
        if self.size < batch_size:
            return None
        
        # Random sampling
        indices = random.sample(range(self.size), batch_size)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return self.size
    
    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has enough experiences for training.
        
        Args:
            min_size: Minimum required size
        
        Returns:
            True if buffer size >= min_size
        """
        return self.size >= min_size


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized experience replay buffer.
    
    Extension point for future research: samples experiences based on TD error.
    Currently implements uniform sampling (same as base ReplayBuffer).
    """
    
    def __init__(self, capacity: int, state_dim: int, alpha: float = 0.6):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            state_dim: State dimension
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
        """
        super().__init__(capacity, state_dim)
        self.alpha = alpha
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add experience with maximum priority."""
        super().push(state, action, reward, next_state, done)
        
        # Assign maximum priority to new experience
        self.priorities[self.position - 1] = self.max_priority
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Optional[Tuple[np.ndarray, ...]]:
        """
        Sample batch with prioritization.
        
        Args:
            batch_size: Batch size
            beta: Importance sampling correction exponent
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        if self.size < batch_size:
            return None
        
        # Compute sampling probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)
        
        # Compute importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights.astype(np.float32),
        )
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: Indices of experiences
            priorities: New priority values (e.g., TD errors)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


if __name__ == "__main__":
    # Test replay buffer
    print("Testing ReplayBuffer")
    print("=" * 60)
    
    state_dim = 24  # Example state dimension
    buffer = ReplayBuffer(capacity=1000, state_dim=state_dim)
    
    # Add some experiences
    for i in range(150):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.random.randint(0, 6)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = (i % 50 == 49)  # Episode ends every 50 steps
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Ready for batch_size=32: {buffer.is_ready(32)}")
    print()
    
    # Sample a batch
    batch = buffer.sample(32)
    if batch is not None:
        states, actions, rewards, next_states, dones = batch
        print(f"Sampled batch:")
        print(f"  States shape: {states.shape}")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Rewards shape: {rewards.shape}")
        print(f"  Sample rewards: {rewards[:5]}")
        print(f"  Sample actions: {actions[:5]}")
