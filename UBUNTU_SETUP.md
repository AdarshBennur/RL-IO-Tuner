# Ubuntu Setup Guide for RL-io-Tuner

This guide provides step-by-step commands to set up and run the RL-IO-Tuner project on Ubuntu from scratch.

---

## Prerequisites

- Ubuntu 20.04+ (Server or Desktop)
- Sudo privileges
- Internet connection

---

## Step 1: Install System Dependencies

```bash
# Update package list
sudo apt-get update

# Install FIO (benchmark tool)
sudo apt-get install -y fio

# Install Python 3.8+ and virtual environment tools
sudo apt-get install -y python3 python3-venv python3-pip
```

---

## Step 2: Clone Repository and Setup Environment

```bash
# Clone the repository
git clone https://github.com/AdarshBennur/RL-IO-Tuner.git

# Navigate to project directory
cd RL-IO-Tuner

# Create Python virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install the project in development mode (important!)
pip install -e .
```

---

## Step 3: Identify Your Block Device

```bash
# List all available block devices
ls -1 /sys/block/

# Common device names:
# - HDD/SSD: sda, sdb, sdc, etc.
# - NVMe: nvme0n1, nvme1n1, etc.
# 
# Note the device name for use in subsequent commands
```

---

## Step 4: Run Baseline Experiment

Establish performance baseline with static kernel parameters:

```bash
# Replace 'vda' with your actual device name
sudo PYTHONPATH=/home/adarsh/RL-IO-Tuner ./venv/bin/python3 experiments/baseline.py \
    --device vda \
    --workload rand_read \
    --runtime 30
```

**Output:** Results saved to `logs/baseline/baseline_*.json`

---

## Step 5: Train the RL Agent

Train the DQN agent to learn optimal I/O parameters:

```bash
# Replace 'vda' with your actual device name
# Set PYTHONPATH to ensure modules are found with sudo
sudo PYTHONPATH=/home/adarsh/RL-IO-Tuner ./venv/bin/python3 scripts/train.py \
    --device vda \
    --workload rand_read \
    --episodes 50 \
    --steps 20 \
    --fio-runtime 10
```

**Note:** Training duration depends on episodes, steps, and FIO runtime. For example:

- 50 episodes × 20 steps × 10s FIO runtime = ~2-3 hours

**Output:**

- Trained model: `results/models/dqn_agent.pth`
- Training logs: `logs/training/training_log_*.json`
- Episode logs: `logs/training/episode_*.json`

---

## Step 6: Evaluate Trained Agent

Test the trained agent's performance:

```bash
# Replace 'vda' with your actual device name
sudo PYTHONPATH=/home/adarsh/RL-IO-Tuner ./venv/bin/python3 scripts/evaluate.py \
    --model results/models/dqn_agent.pth \
    --device vda \
    --workload rand_read \
    --episodes 5
```

**Output:** Evaluation results in `logs/evaluation/evaluation_results.json`

---

## Step 7: Generate Visualizations

Create plots from training and baseline data:

```bash
# First, check the actual log filenames
ls logs/rl_controlled/
ls logs/baseline/

# Generate plots (replace wildcards with actual filenames if needed)
python3 scripts/plot_results.py \
    --training-log logs/rl_controlled/training_log_*.json \
    --baseline-log logs/baseline/baseline_*.json \
    --episode-log-dir logs/training \
    --output-dir results/plots

# View generated plots
ls results/plots/
```

**Generated plots:**

- `training_rewards.png` - Reward convergence
- `baseline_comparison.png` - Performance comparison
- `action_distribution.png` - Agent's action patterns

---

## Step 8: Reset Kernel Parameters (Optional)

Restore original kernel parameter values:

```bash
# Replace 'sda' with your actual device name
sudo ./system/reset_params.sh sda
```

---

## Alternative Workloads

You can experiment with different FIO workloads:

```bash
# Sequential Read
sudo PYTHONPATH=/home/adarsh/RL-IO-Tuner ./venv/bin/python3 scripts/train.py --device vda --workload seq_read --episodes 50 --steps 20 --fio-runtime 10

# Random Write
sudo PYTHONPATH=/home/adarsh/RL-IO-Tuner ./venv/bin/python3 scripts/train.py --device vda --workload rand_write --episodes 50 --steps 20 --fio-runtime 10

# Sequential Write
sudo PYTHONPATH=/home/adarsh/RL-IO-Tuner ./venv/bin/python3 scripts/train.py --device vda --workload seq_write --episodes 50 --steps 20 --fio-runtime 10
```

**Available workloads:**

- `seq_read` - Sequential Read (128K blocks)
- `rand_read` - Random Read (4K blocks)
- `seq_write` - Sequential Write (128K blocks)
- `rand_write` - Random Write (4K blocks)

---

## Quick Start (Minimal Test)

For a quick test run with shorter training:

```bash
# Quick baseline (5-minute runtime)
sudo PYTHONPATH=/home/adarsh/RL-IO-Tuner ./venv/bin/python3 experiments/baseline.py --device vda --workload rand_read --runtime 5

# Quick training (10 episodes, 5 steps, 5s FIO)
sudo PYTHONPATH=/home/adarsh/RL-IO-Tuner ./venv/bin/python3 scripts/train.py --device vda --workload rand_read --episodes 10 --steps 5 --fio-runtime 5

# Quick evaluation
sudo PYTHONPATH=/home/adarsh/RL-IO-Tuner ./venv/bin/python3 scripts/evaluate.py --model results/models/dqn_agent.pth --device vda --workload rand_read --episodes 3
```

---

## Troubleshooting

### Permission Errors

```bash
# Ensure you're using sudo with PYTHONPATH set for kernel parameter modifications
sudo PYTHONPATH=/home/adarsh/RL-IO-Tuner ./venv/bin/python3 scripts/train.py ...
```

### Device Not Found

```bash
# Verify device exists
ls /sys/block/
# Update --device argument accordingly
```

### FIO Not Installed

```bash
sudo apt-get install -y fio
```

### Python Module Not Found

```bash
# Activate virtual environment
source venv/bin/activate
# Reinstall dependencies
pip install -r requirements.txt
```

### Virtual Environment Issues

```bash
# Deactivate current environment
deactivate
# Remove and recreate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## SSH Setup (For Remote Access)

If running on a headless Ubuntu server and accessing via SSH:

```bash
# On Ubuntu server
sudo apt-get install -y openssh-server
sudo systemctl start ssh
sudo systemctl enable ssh

# Get server IP address
ip addr show

# From macOS/client
ssh username@server-ip-address

# Copy results back to local machine
scp username@server-ip:~/RL-IO-Tuner/results/plots/*.png ~/Desktop/
```

---

## Directory Structure After Setup

```
RL-io-Tuner/
├── venv/                   # Virtual environment (created)
├── logs/                   # Generated logs
│   ├── baseline/          # Baseline experiment logs
│   ├── training/          # Episode-by-episode training logs
│   ├── rl_controlled/     # RL experiment summaries
│   └── evaluation/        # Evaluation results
├── results/               # Generated outputs
│   ├── models/            # Trained DQN models (.pth)
│   └── plots/             # Visualizations (.png)
└── ...
```

---

## Important Notes

1. **Always use `sudo`** for training and evaluation scripts (kernel parameter access required)
2. **Device name matters** - use the exact name from `/sys/block/`
3. **Training takes time** - plan accordingly based on episodes × steps × FIO runtime
4. **Virtual environment** - always activate before running Python scripts
5. **Backup important data** - this modifies kernel I/O parameters

---

**Last Updated:** December 2024
