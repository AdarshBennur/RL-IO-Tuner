"""
Baseline experiment: Static kernel parameters.
Tests multiple fixed configurations to establish performance baseline.
"""

import sys
import os
import json
import time
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.apply_params import KernelParamController
from benchmarks.run_fio import run_fio_benchmark, get_available_workloads


def run_baseline_experiment(
    device: str = "sda",
    workload_name: str = "rand_read",
    runtime: int = 30,
    output_dir: str = "logs/baseline"
):
    """
    Run baseline experiment with static parameter configurations.
    
    Args:
        device: Target block device
        workload_name: FIO workload name
        runtime: Benchmark runtime per configuration
        output_dir: Output directory for results
    """
    print("=" * 70)
    print("BASELINE EXPERIMENT: Static Kernel Parameters")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Workload: {workload_name}")
    print(f"Runtime: {runtime}s per configuration")
    print()
    
    # Get workload path
    workloads = get_available_workloads()
    if workload_name not in workloads:
        print(f"Error: Workload '{workload_name}' not found")
        print(f"Available: {list(workloads.keys())}")
        return
    
    workload_path = workloads[workload_name]
    
    # Define baseline configurations to test
    configs = [
        {'read_ahead_kb': 128, 'name': 'default'},
        {'read_ahead_kb': 256, 'name': 'low'},
        {'read_ahead_kb': 512, 'name': 'medium'},
        {'read_ahead_kb': 1024, 'name': 'high'},
        {'read_ahead_kb': 2048, 'name': 'very_high'},
    ]
    
    # Initialize controller
    controller = KernelParamController(device)
    
    # Store results
    results = []
    
    # Test each configuration
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testing configuration: {config['name']}")
        print(f"  read_ahead_kb = {config['read_ahead_kb']}")
        
        # Apply configuration
        success = controller.set_read_ahead_kb(config['read_ahead_kb'])
        if not success:
            print(f"  Failed to apply configuration, skipping...")
            continue
        
        # Wait for configuration to take effect
        time.sleep(2)
        
        # Run benchmark
        print(f"  Running FIO benchmark...")
        metrics = run_fio_benchmark(workload_path, device, runtime)
        
        if metrics is None:
            print(f"  Benchmark failed, skipping...")
            continue
        
        # Store results
        result = {
            'config': config['name'],
            'read_ahead_kb': config['read_ahead_kb'],
            'metrics': metrics,
        }
        results.append(result)
        
        # Print key metrics
        print(f"  Results:")
        print(f"    Read IOPS:      {metrics.get('read_iops', 0):.2f}")
        print(f"    Write IOPS:     {metrics.get('write_iops', 0):.2f}")
        print(f"    Throughput:     {metrics.get('read_throughput_mb', 0):.2f} MB/s (R), "
              f"{metrics.get('write_throughput_mb', 0):.2f} MB/s (W)")
        print(f"    Avg Latency:    {metrics.get('avg_read_latency_ms', 0):.3f} ms (R), "
              f"{metrics.get('avg_write_latency_ms', 0):.3f} ms (W)")
    
    # Restore defaults
    print("\nRestoring default parameters...")
    controller.restore_defaults()
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"baseline_{workload_name}_{int(time.time())}.json")
    
    with open(output_file, 'w') as f:
        json.dump({
            'device': device,
            'workload': workload_name,
            'runtime': runtime,
            'results': results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if results:
        print(f"\nBest configuration by read IOPS:")
        best_iops = max(results, key=lambda x: x['metrics'].get('read_iops', 0))
        print(f"  Config: {best_iops['config']} (read_ahead_kb={best_iops['read_ahead_kb']})")
        print(f"  Read IOPS: {best_iops['metrics'].get('read_iops', 0):.2f}")
        
        print(f"\nBest configuration by throughput:")
        best_throughput = max(results, key=lambda x: x['metrics'].get('read_throughput_mb', 0))
        print(f"  Config: {best_throughput['config']} (read_ahead_kb={best_throughput['read_ahead_kb']})")
        print(f"  Throughput: {best_throughput['metrics'].get('read_throughput_mb', 0):.2f} MB/s")
    
    print("=" * 70)
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run baseline experiment")
    parser.add_argument('--device', type=str, default='sda', help='Block device name')
    parser.add_argument('--workload', type=str, default='rand_read', help='FIO workload name')
    parser.add_argument('--runtime', type=int, default=30, help='Runtime per config (seconds)')
    parser.add_argument('--output-dir', type=str, default='logs/baseline', help='Output directory')
    
    args = parser.parse_args()
    
    run_baseline_experiment(
        device=args.device,
        workload=args.workload,
        runtime=args.runtime,
        output_dir=args.output_dir
    )
