"""
Linux disk I/O statistics reader.
Parses /proc/diskstats to extract real-time disk performance metrics.
"""

import time
from typing import Dict, Optional


class DiskStatsReader:
    """
    Reads and parses /proc/diskstats for disk I/O metrics.
    
    /proc/diskstats format (space-separated):
    major minor name reads reads_merged sectors_read ms_reading writes writes_merged 
    sectors_written ms_writing ios_in_progress ms_io total_weighted_ms ...
    """
    
    DISKSTATS_PATH = "/proc/diskstats"
    SECTOR_SIZE = 512  # bytes
    
    def __init__(self, device: str = "sda"):
        """
        Initialize diskstats reader.
        
        Args:
            device: Block device name (e.g., 'sda', 'nvme0n1')
        """
        self.device = device
        self.last_stats = None
        self.last_time = None
    
    def read_raw_stats(self) -> Optional[Dict[str, int]]:
        """
        Read raw statistics from /proc/diskstats for the target device.
        
        Returns:
            Dictionary with raw counter values, or None on error
        """
        try:
            with open(self.DISKSTATS_PATH, 'r') as f:
                for line in f:
                    fields = line.split()
                    if len(fields) >= 14 and fields[2] == self.device:
                        return {
                            'reads_completed': int(fields[3]),
                            'reads_merged': int(fields[4]),
                            'sectors_read': int(fields[5]),
                            'time_reading_ms': int(fields[6]),
                            'writes_completed': int(fields[7]),
                            'writes_merged': int(fields[8]),
                            'sectors_written': int(fields[9]),
                            'time_writing_ms': int(fields[10]),
                            'ios_in_progress': int(fields[11]),
                            'time_io_ms': int(fields[12]),
                            'weighted_time_io_ms': int(fields[13]),
                        }
            return None
        except (IOError, ValueError) as e:
            print(f"Error reading diskstats: {e}")
            return None
    
    def get_delta_stats(self) -> Optional[Dict[str, float]]:
        """
        Get delta statistics since last call (rates and throughput).
        
        Returns:
            Dictionary with computed metrics:
            - read_iops: Reads per second
            - write_iops: Writes per second
            - read_throughput_mb: Read MB/s
            - write_throughput_mb: Write MB/s
            - avg_read_latency_ms: Average read latency
            - avg_write_latency_ms: Average write latency
            - utilization: Device utilization percentage
        """
        current_stats = self.read_raw_stats()
        current_time = time.time()
        
        if current_stats is None:
            return None
        
        # First call - store baseline and return zeros
        if self.last_stats is None:
            self.last_stats = current_stats
            self.last_time = current_time
            return {
                'read_iops': 0.0,
                'write_iops': 0.0,
                'read_throughput_mb': 0.0,
                'write_throughput_mb': 0.0,
                'avg_read_latency_ms': 0.0,
                'avg_write_latency_ms': 0.0,
                'utilization': 0.0,
            }
        
        # Compute deltas
        time_delta = current_time - self.last_time
        if time_delta < 0.001:  # Avoid division by zero
            return None
        
        delta = {}
        for key in current_stats:
            delta[key] = current_stats[key] - self.last_stats[key]
        
        # Compute metrics
        metrics = {}
        
        # IOPS
        metrics['read_iops'] = delta['reads_completed'] / time_delta
        metrics['write_iops'] = delta['writes_completed'] / time_delta
        
        # Throughput (MB/s)
        metrics['read_throughput_mb'] = (delta['sectors_read'] * self.SECTOR_SIZE) / (time_delta * 1024 * 1024)
        metrics['write_throughput_mb'] = (delta['sectors_written'] * self.SECTOR_SIZE) / (time_delta * 1024 * 1024)
        
        # Average latency (ms)
        if delta['reads_completed'] > 0:
            metrics['avg_read_latency_ms'] = delta['time_reading_ms'] / delta['reads_completed']
        else:
            metrics['avg_read_latency_ms'] = 0.0
        
        if delta['writes_completed'] > 0:
            metrics['avg_write_latency_ms'] = delta['time_writing_ms'] / delta['writes_completed']
        else:
            metrics['avg_write_latency_ms'] = 0.0
        
        # Utilization (percentage)
        # time_io_ms is cumulative time device was busy
        metrics['utilization'] = min(100.0, (delta['time_io_ms'] / (time_delta * 1000)) * 100)
        
        # Update last values
        self.last_stats = current_stats
        self.last_time = current_time
        
        return metrics
    
    def get_instant_stats(self) -> Optional[Dict[str, float]]:
        """
        Get instantaneous statistics (current queue depth).
        
        Returns:
            Dictionary with instant metrics
        """
        current_stats = self.read_raw_stats()
        if current_stats is None:
            return None
        
        return {
            'queue_depth': float(current_stats['ios_in_progress']),
        }


def get_disk_metrics(device: str = "sda") -> Optional[Dict[str, float]]:
    """
    Convenience function to get comprehensive disk metrics.
    
    Args:
        device: Block device name
    
    Returns:
        Combined delta and instant metrics
    """
    reader = DiskStatsReader(device)
    
    # Get baseline
    reader.read_raw_stats()
    time.sleep(0.1)  # Short sampling period
    
    delta = reader.get_delta_stats()
    instant = reader.get_instant_stats()
    
    if delta is None or instant is None:
        return None
    
    return {**delta, **instant}


if __name__ == "__main__":
    # Test the reader
    import sys
    
    device = sys.argv[1] if len(sys.argv) > 1 else "sda"
    print(f"Monitoring device: {device}")
    print("-" * 60)
    
    reader = DiskStatsReader(device)
    
    try:
        while True:
            metrics = reader.get_delta_stats()
            if metrics:
                print(f"Read IOPS: {metrics['read_iops']:.2f}, "
                      f"Write IOPS: {metrics['write_iops']:.2f}, "
                      f"Throughput: {metrics['read_throughput_mb']:.2f} MB/s (R) "
                      f"{metrics['write_throughput_mb']:.2f} MB/s (W), "
                      f"Util: {metrics['utilization']:.1f}%")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped monitoring")
