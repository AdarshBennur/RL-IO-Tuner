# RL-io-Tuner

**Dynamic Linux Storage Optimization Using Deep Reinforcement Learning**

A systems research project implementing a closed-loop reinforcement learning framework for real-time Linux kernel I/O parameter optimization. Uses Deep Q-Network (DQN) to learn optimal storage configurations based on workload characteristics measured via FIO benchmarks.

---

## Domain Classification

- **Primary:** Operating Systems / Systems Research
- **Secondary:** Reinforcement Learning
- **Application:** Storage / I/O Performance Optimization

---

## Overview

RL-io-Tuner implements an autonomous agent that:

1. **Observes** disk I/O metrics from `/proc/diskstats`
2. **Decides** kernel parameter values using a trained DQN
3. **Applies** parameters to `/sys/block/*/queue/`
4. **Executes** FIO workloads to measure performance
5. **Learns** from reward signals (throughput - latency penalty)
6. **Adapts** parameter selection over time

This is a **production-ready research framework** designed for academic publication and real Linux deployment.

---

## Architecture

```
┌─────────────────┐
│   DQN Agent     │
│  (PyTorch)      │
└────────┬────────┘
         │ Action (read_ahead_kb)
         ▼
┌─────────────────┐
│ Kernel Params   │
│ /sys/block/*    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│ FIO Benchmark   │─────▶│   Metrics    │
│ (workload)      │      │ /proc/stats  │
└─────────────────┘      └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │    Reward    │
                         │  Function    │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │Replay Buffer │
                         └──────────────┘
```

---

## Prerequisites

### System Requirements

- **OS:** Ubuntu 20.04+ (or similar Linux with `/proc/diskstats` and `/sys/block/`)
- **Permissions:** `sudo` access for writing to `/sys/block/*`
- **Python:** 3.8+

### Required Software

```bash
# Install FIO
sudo apt-get update
sudo apt-get install fio

# Install Python dependencies (see Installation below)
```

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd RL-io-Tuner

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Project Structure

```
RL-io-Tuner/
├── env/                    # RL Environment
│   ├── io_env.py          # Gymnasium environment
│   ├── metrics.py         # Metric extraction and normalization
│   └── reward.py          # Reward function
├── agent/                  # DQN Agent
│   ├── dqn.py             # DQN network and training
│   └── replay_buffer.py   # Experience replay
├── system/                 # Linux System Integration
│   ├── read_stats.py      # /proc/diskstats parser
│   ├── apply_params.py    # /sys/block/* writer
│   └── reset_params.sh    # Reset kernel defaults
├── benchmarks/             # FIO Workloads
│   ├── fio/               # FIO configs (seq/rand read/write)
│   └── run_fio.py         # FIO execution wrapper
├── experiments/            # Experiment Scripts
│   ├── baseline.py        # Static parameter baseline
│   ├── rl_controlled.py   # RL-driven experiment
│   └── ablation.py        # Ablation study placeholder
├── scripts/                # Main Entry Points
│   ├── train.py           # Train DQN agent
│   ├── evaluate.py        # Evaluate trained agent
│   └── plot_results.py    # Generate visualizations
├── logs/                   # Output Logs
├── results/                # Saved Models and Plots
└── README.md
```

---

## Usage

### 1. Run Baseline Experiment

Establish performance baseline with static kernel parameters:

```bash
sudo python experiments/baseline.py \
    --device sda \
    --workload rand_read \
    --runtime 30
```

**Output:** JSON log in `logs/baseline/` with performance for each configuration.

---

### 2. Train RL Agent

Train DQN agent to learn optimal parameters:

```bash
sudo python scripts/train.py \
    --device sda \
    --workload rand_read \
    --episodes 50 \
    --steps 20 \
    --fio-runtime 10
```

**Arguments:**

- `--device`: Block device (e.g., `sda`, `nvme0n1`)
- `--workload`: FIO workload type (`seq_read`, `rand_read`, `seq_write`, `rand_write`)
- `--episodes`: Number of training episodes
- `--steps`: Max steps per episode
- `--fio-runtime`: FIO benchmark duration per step (seconds)

**Output:**

- Trained model: `results/models/dqn_agent.pth`
- Training logs: `logs/training/training_log_*.json`
- Episode logs: `logs/training/episode_*.json`

---

### 3. Evaluate Trained Agent

Test the trained agent's performance:

```bash
sudo python scripts/evaluate.py \
    --model results/models/dqn_agent.pth \
    --device sda \
    --workload rand_read \
    --episodes 5
```

**Output:** Evaluation results in `logs/evaluation/evaluation_results.json`

---

### 4. Generate Visualizations

Create plots from training and evaluation logs:

```bash
python scripts/plot_results.py \
    --training-log logs/rl_controlled/training_log_*.json \
    --baseline-log logs/baseline/baseline_*.json \
    --episode-log-dir logs/training \
    --output-dir results/plots
```

**Generates:**

- `training_rewards.png` - Reward convergence over episodes
- `baseline_comparison.png` - Performance across static configs
- `action_distribution.png` - Agent's action selection patterns

---

## Key Parameters

### Tunable Kernel Parameters

| Parameter | Path | Range | Description |
|-----------|------|-------|-------------|
| `read_ahead_kb` | `/sys/block/{device}/queue/read_ahead_kb` | 0-16384 | Readahead window size (KB) |
| `nr_requests` | `/sys/block/{device}/queue/nr_requests` | 1-10000 | I/O scheduler queue depth |
| `scheduler` | `/sys/block/{device}/queue/scheduler` | various | I/O scheduler algorithm |

**Note:** Currently, the primary tunable is `read_ahead_kb`. Others can be added by extending `system/apply_params.py`.

### Action Space

Discrete values for `read_ahead_kb` (in KB):

```
[128, 256, 512, 1024, 2048, 4096]
```

### State Space

Normalized vector containing:

- Read/Write IOPS
- Read/Write throughput (MB/s)
- Average read/write latency (ms)
- Device utilization (%)
- Queue depth
- Historical observations (configurable window)

### Reward Function

```
reward = α × throughput - β × latency - γ × penalty
```

Where:

- **α** (throughput_weight): Reward for high throughput
- **β** (latency_weight): Penalty for high latency
- **γ** (penalty_weight): Penalty for constraint violations

---

## Important Notes

### Permissions

**Most operations require `sudo`** to modify kernel parameters:

```bash
sudo python scripts/train.py ...
```

### Device Names

Ensure correct device naming:

- **HDD/SSD:** `sda`, `sdb`, etc.
- **NVMe:** `nvme0n1`, `nvme1n1`, etc.

Check available devices:

```bash
ls -1 /sys/block/
```

### Reset Parameters

To restore kernel defaults:

```bash
sudo ./system/reset_params.sh sda
```

To restore from backup:

```bash
sudo ./system/reset_params.sh sda --restore-backup
```

---

## FIO Workload Profiles

| Workload | Type | Block Size | Use Case |
|----------|------|------------|----------|
| `seq_read` | Sequential Read | 128K | Streaming, large file access |
| `rand_read` | Random Read | 4K | Database, key-value stores |
| `seq_write` | Sequential Write | 128K | Logging, video recording |
| `rand_write` | Random Write | 4K | Transactional workloads |

---

## Extending the Framework

### Add New Parameters

1. Update `system/apply_params.py` with new parameter methods
2. Modify action space in `env/io_env.py`
3. Update state representation in `env/metrics.py` if needed

### Custom Reward Functions

Create new reward classes in `env/reward.py`:

```python
class CustomReward(RewardFunction):
    def compute(self, metrics, prev_metrics=None):
        # Your custom logic
        return reward
```

### Different Workloads

Add custom FIO configs to `benchmarks/fio/`:

```ini
[global]
ioengine=libaio
direct=1
...

[custom-workload]
rw=randrw
bs=8k
...
```

---

## Troubleshooting

### "Permission denied" errors

- Run with `sudo`
- Check file permissions on `/sys/block/{device}/queue/`

### "Device not found"

- Verify device name: `ls /sys/block/`
- Update `--device` argument

### FIO not installed

```bash
sudo apt-get install fio
```

### "No module named 'torch'"

- Activate virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

---

## Output Files

### Logs

- `logs/baseline/` - Baseline experiment results
- `logs/training/` - Episode-by-episode RL training logs
- `logs/rl_controlled/` - RL experiment summaries
- `logs/evaluation/` - Agent evaluation results

### Models

- `results/models/` - Trained DQN models (`.pth` files)

### Plots

- `results/plots/` - Visualization outputs (`.png` files)

---

## Research Extensions

This framework is designed for extensibility:

1. **Multi-parameter optimization** - Tune multiple kernel parameters simultaneously
2. **Workload detection** - Automatically identify workload type and adapt
3. **Prioritized experience replay** - Use TD-error for importance sampling
4. **Continuous action spaces** - Replace discrete actions with DDPG/SAC
5. **Transfer learning** - Train on one workload, transfer to another
6. **Multi-objective optimization** - Balance throughput, latency, and energy

---

## Citation

If you use this framework in academic research, please cite:

```
@software{rl_io_tuner,
  title={RL-io-Tuner: Dynamic Linux Storage Optimization Using Deep Reinforcement Learning},
  author={Adarsh Bennur},
  year={2025},
  url={https://github.com/AdarshBennur/RL-io-Tuner}
}
```

---

**Last Updated:** December 2024
