"""
Kernel storage parameter controller.
Safely writes to /sys/block/* to tune Linux I/O subsystem parameters.
"""

import os
from typing import Dict, Optional, Tuple


class KernelParamController:
    """
    Controls Linux kernel storage parameters via /sys/block interface.
    
    Supports tuning:
    - read_ahead_kb: Readahead window size
    - nr_requests: I/O scheduler queue depth
    - scheduler: I/O scheduler algorithm
    """
    
    SYSFS_BASE = "/sys/block"
    
    # Parameter bounds and valid values
    PARAM_BOUNDS = {
        'read_ahead_kb': (0, 16384),  # 0 to 16 MB
        'nr_requests': (1, 10000),
    }
    
    VALID_SCHEDULERS = ['noop', 'deadline', 'cfq', 'bfq', 'kyber', 'mq-deadline', 'none']
    
    def __init__(self, device: str = "sda"):
        """
        Initialize parameter controller.
        
        Args:
            device: Block device name (e.g., 'sda', 'nvme0n1')
        """
        self.device = device
        self.device_path = os.path.join(self.SYSFS_BASE, device)
        self.queue_path = os.path.join(self.device_path, "queue")
        
        # Store original values for rollback
        self.original_params: Dict[str, str] = {}
    
    def _read_param(self, param_name: str) -> Optional[str]:
        """
        Read current parameter value.
        
        Args:
            param_name: Parameter name (e.g., 'read_ahead_kb')
        
        Returns:
            Current value as string, or None on error
        """
        param_path = os.path.join(self.queue_path, param_name)
        try:
            with open(param_path, 'r') as f:
                return f.read().strip()
        except (IOError, PermissionError) as e:
            print(f"Error reading {param_name}: {e}")
            return None
    
    def _write_param(self, param_name: str, value: str) -> bool:
        """
        Write parameter value.
        
        Args:
            param_name: Parameter name
            value: Value to write
        
        Returns:
            True on success, False on error
        """
        param_path = os.path.join(self.queue_path, param_name)
        
        # Store original value on first write
        if param_name not in self.original_params:
            original = self._read_param(param_name)
            if original is not None:
                self.original_params[param_name] = original
        
        try:
            with open(param_path, 'w') as f:
                f.write(str(value))
            return True
        except (IOError, PermissionError) as e:
            print(f"Error writing {param_name}={value}: {e}")
            print(f"Note: You may need sudo privileges to modify kernel parameters")
            return False
    
    def set_read_ahead_kb(self, value: int) -> bool:
        """
        Set readahead window size.
        
        Args:
            value: Readahead size in KB
        
        Returns:
            True on success
        """
        param_name = 'read_ahead_kb'
        
        # Validate bounds
        min_val, max_val = self.PARAM_BOUNDS[param_name]
        if not (min_val <= value <= max_val):
            print(f"Error: {param_name}={value} out of bounds [{min_val}, {max_val}]")
            return False
        
        return self._write_param(param_name, str(value))
    
    def set_nr_requests(self, value: int) -> bool:
        """
        Set I/O scheduler queue depth.
        
        Args:
            value: Queue depth
        
        Returns:
            True on success
        """
        param_name = 'nr_requests'
        
        # Validate bounds
        min_val, max_val = self.PARAM_BOUNDS[param_name]
        if not (min_val <= value <= max_val):
            print(f"Error: {param_name}={value} out of bounds [{min_val}, {max_val}]")
            return False
        
        return self._write_param(param_name, str(value))
    
    def set_scheduler(self, scheduler: str) -> bool:
        """
        Set I/O scheduler algorithm.
        
        Args:
            scheduler: Scheduler name (e.g., 'mq-deadline', 'bfq')
        
        Returns:
            True on success
        """
        # Read available schedulers
        available = self._read_param('scheduler')
        if available is None:
            return False
        
        # Parse available schedulers (format: "noop deadline [cfq]")
        available_list = available.replace('[', '').replace(']', '').split()
        
        if scheduler not in available_list:
            print(f"Error: Scheduler '{scheduler}' not available. Available: {available_list}")
            return False
        
        return self._write_param('scheduler', scheduler)
    
    def get_current_params(self) -> Dict[str, str]:
        """
        Get all current parameter values.
        
        Returns:
            Dictionary of parameter names to values
        """
        params = {}
        for param in ['read_ahead_kb', 'nr_requests', 'scheduler']:
            value = self._read_param(param)
            if value is not None:
                params[param] = value
        return params
    
    def restore_defaults(self) -> bool:
        """
        Restore original parameter values.
        
        Returns:
            True if all parameters restored successfully
        """
        success = True
        for param_name, original_value in self.original_params.items():
            if not self._write_param(param_name, original_value):
                success = False
                print(f"Failed to restore {param_name} to {original_value}")
        return success
    
    def apply_config(self, config: Dict[str, int]) -> Tuple[bool, Dict[str, str]]:
        """
        Apply a configuration dictionary.
        
        Args:
            config: Dictionary mapping parameter names to values
                   e.g., {'read_ahead_kb': 512, 'nr_requests': 128}
        
        Returns:
            Tuple of (success, resulting_params)
        """
        success = True
        
        if 'read_ahead_kb' in config:
            if not self.set_read_ahead_kb(config['read_ahead_kb']):
                success = False
        
        if 'nr_requests' in config:
            if not self.set_nr_requests(config['nr_requests']):
                success = False
        
        if 'scheduler' in config:
            if not self.set_scheduler(config['scheduler']):
                success = False
        
        return success, self.get_current_params()


if __name__ == "__main__":
    import sys
    
    device = sys.argv[1] if len(sys.argv) > 1 else "sda"
    print(f"Kernel parameter controller for device: {device}")
    print("-" * 60)
    
    controller = KernelParamController(device)
    
    # Show current parameters
    current = controller.get_current_params()
    print("Current parameters:")
    for param, value in current.items():
        print(f"  {param}: {value}")
    
    # Example: Set readahead to 1024 KB
    print("\nSetting read_ahead_kb to 1024...")
    if controller.set_read_ahead_kb(1024):
        print("Success!")
        print(f"New value: {controller._read_param('read_ahead_kb')}")
    
    # Restore defaults
    print("\nRestoring defaults...")
    if controller.restore_defaults():
        print("Defaults restored!")
        print(f"Restored read_ahead_kb: {controller._read_param('read_ahead_kb')}")
