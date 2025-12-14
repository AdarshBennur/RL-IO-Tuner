"""
FIO benchmark runner and results parser.
"""

import subprocess
import json
import os
from typing import Dict, Optional


def run_fio_benchmark(
    config_file: str,
    device: str = "sda",
    runtime: Optional[int] = None,
    output_format: str = "json"
) -> Optional[Dict[str, float]]:
    """
    Run FIO benchmark and parse results.
    
    Args:
        config_file: Path to FIO configuration file
        device: Target device (used to replace 'filename' in config)
        runtime: Override runtime in seconds (optional)
        output_format: Output format ('json' or 'normal')
    
    Returns:
        Dictionary of performance metrics, or None on error
    """
    if not os.path.exists(config_file):
        print(f"Error: FIO config file not found: {config_file}")
        return None
    
    # Construct FIO command
    # Note: Assumes test file is in /tmp/fio-test-${device}
    test_file = f"/tmp/fio-test-{device}"
    
    cmd = [
        "fio",
        config_file,
        f"--filename={test_file}",
        f"--output-format={output_format}",
    ]
    
    if runtime is not None:
        cmd.append(f"--runtime={runtime}")
    
    try:
        # Run FIO
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=runtime + 30 if runtime else 300,  # Add 30s buffer
        )
        
        if result.returncode != 0:
            print(f"FIO failed with exit code {result.returncode}")
            print(f"Error: {result.stderr}")
            return None
        
        # Parse results
        if output_format == "json":
            return _parse_fio_json(result.stdout)
        else:
            return _parse_fio_text(result.stdout)
    
    except subprocess.TimeoutExpired:
        print(f"FIO benchmark timeout")
        return None
    except Exception as e:
        print(f"Error running FIO: {e}")
        return None


def _parse_fio_json(json_output: str) -> Optional[Dict[str, float]]:
    """
    Parse FIO JSON output.
    
    Args:
        json_output: JSON string from FIO
    
    Returns:
        Dictionary of metrics
    """
    try:
        data = json.loads(json_output)
        
        # FIO JSON has structure: {"jobs": [...], ...}
        if "jobs" not in data or len(data["jobs"]) == 0:
            print("Error: No jobs in FIO output")
            return None
        
        # Use first job (or aggregate if multiple)
        job = data["jobs"][0]
        
        # Extract metrics
        read_stats = job.get("read", {})
        write_stats = job.get("write", {})
        
        metrics = {
            # IOPS
            'read_iops': read_stats.get('iops', 0.0),
            'write_iops': write_stats.get('iops', 0.0),
            
            # Bandwidth (convert from KB/s to MB/s)
            'read_throughput_mb': read_stats.get('bw', 0.0) / 1024.0,
            'write_throughput_mb': write_stats.get('bw', 0.0) / 1024.0,
            
            # Latency (convert from nanoseconds to milliseconds)
            'avg_read_latency_ms': read_stats.get('lat_ns', {}).get('mean', 0.0) / 1e6,
            'avg_write_latency_ms': write_stats.get('lat_ns', {}).get('mean', 0.0) / 1e6,
            
            # Latency percentiles
            'p50_read_latency_ms': read_stats.get('clat_ns', {}).get('percentile', {}).get('50.000000', 0.0) / 1e6,
            'p95_read_latency_ms': read_stats.get('clat_ns', {}).get('percentile', {}).get('95.000000', 0.0) / 1e6,
            'p99_read_latency_ms': read_stats.get('clat_ns', {}).get('percentile', {}).get('99.000000', 0.0) / 1e6,
            
            # Utilization (estimated from I/O percentage)
            'utilization': job.get('usr_cpu', 0.0) + job.get('sys_cpu', 0.0),
            
            # Queue depth (average)
            'queue_depth': float(job.get('iodepth_level', {}).get('1', 0)),
        }
        
        return metrics
    
    except json.JSONDecodeError as e:
        print(f"Error parsing FIO JSON: {e}")
        return None
    except KeyError as e:
        print(f"Missing key in FIO JSON: {e}")
        return None


def _parse_fio_text(text_output: str) -> Optional[Dict[str, float]]:
    """
    Parse FIO text output.
    
    Args:
        text_output: Text output from FIO
    
    Returns:
        Dictionary of metrics (simplified)
    """
    # Basic text parsing (less reliable than JSON)
    # This is a fallback option
    metrics = {
        'read_iops': 0.0,
        'write_iops': 0.0,
        'read_throughput_mb': 0.0,
        'write_throughput_mb': 0.0,
        'avg_read_latency_ms': 0.0,
        'avg_write_latency_ms': 0.0,
        'utilization': 0.0,
        'queue_depth': 0.0,
    }
    
    # Simple regex parsing would go here
    # For production use, prefer JSON format
    
    return metrics


def get_available_workloads(benchmarks_dir: str = "benchmarks/fio") -> Dict[str, str]:
    """
    Get available FIO workload configurations.
    
    Args:
        benchmarks_dir: Directory containing FIO configs
    
    Returns:
        Dictionary mapping workload names to file paths
    """
    workloads = {}
    
    if not os.path.exists(benchmarks_dir):
        return workloads
    
    for filename in os.listdir(benchmarks_dir):
        if filename.endswith('.fio'):
            name = filename.replace('.fio', '')
            workloads[name] = os.path.join(benchmarks_dir, filename)
    
    return workloads


if __name__ == "__main__":
    import sys
    
    # Test FIO runner
    if len(sys.argv) < 2:
        print("Usage: python run_fio.py <config_file> [device] [runtime]")
        print("\nAvailable workloads:")
        for name, path in get_available_workloads().items():
            print(f"  {name}: {path}")
        sys.exit(1)
    
    config_file = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "sda"
    runtime = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    print(f"Running FIO benchmark:")
    print(f"  Config: {config_file}")
    print(f"  Device: {device}")
    print(f"  Runtime: {runtime}s")
    print("-" * 60)
    
    results = run_fio_benchmark(config_file, device, runtime)
    
    if results:
        print("\nResults:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.2f}")
    else:
        print("\nFailed to run benchmark")
        print("\nNote: This requires FIO to be installed:")
        print("  sudo apt-get install fio")
