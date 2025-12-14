#!/bin/bash

# Reset kernel storage parameters to defaults
# Usage: ./reset_params.sh [device]

DEVICE=${1:-sda}
SYSFS_BASE="/sys/block/${DEVICE}/queue"
BACKUP_FILE="/tmp/kernel_params_${DEVICE}_backup.txt"

echo "==================================================================="
echo "Kernel Storage Parameter Reset Script"
echo "Device: $DEVICE"
echo "==================================================================="

# Check if device exists
if [ ! -d "/sys/block/${DEVICE}" ]; then
    echo "Error: Device ${DEVICE} not found in /sys/block/"
    echo "Available devices:"
    ls -1 /sys/block/
    exit 1
fi

# Function to get current value
get_param() {
    local param=$1
    local path="${SYSFS_BASE}/${param}"
    if [ -r "$path" ]; then
        cat "$path"
    else
        echo "N/A"
    fi
}

# Function to set parameter
set_param() {
    local param=$1
    local value=$2
    local path="${SYSFS_BASE}/${param}"
    
    if [ -w "$path" ]; then
        echo "$value" > "$path" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "  ✓ Set ${param} = ${value}"
        else
            echo "  ✗ Failed to set ${param}"
        fi
    else
        echo "  ⚠ Cannot write to ${param} (requires sudo)"
    fi
}

# Save current parameters if backup doesn't exist
if [ ! -f "$BACKUP_FILE" ]; then
    echo "Saving current parameters to backup file..."
    echo "read_ahead_kb=$(get_param read_ahead_kb)" > "$BACKUP_FILE"
    echo "nr_requests=$(get_param nr_requests)" >> "$BACKUP_FILE"
    echo "scheduler=$(get_param scheduler | tr -d '[]')" >> "$BACKUP_FILE"
    echo "Backup saved to: $BACKUP_FILE"
    echo ""
fi

# Option 1: Restore from backup
if [ "$2" == "--restore-backup" ]; then
    echo "Restoring from backup file..."
    if [ -f "$BACKUP_FILE" ]; then
        source "$BACKUP_FILE"
        set_param "read_ahead_kb" "$read_ahead_kb"
        set_param "nr_requests" "$nr_requests"
        # Note: scheduler restoration may fail if scheduler not available
        echo "$scheduler" > "${SYSFS_BASE}/scheduler" 2>/dev/null || echo "  ⚠ Could not restore scheduler"
    else
        echo "Error: No backup file found at $BACKUP_FILE"
        exit 1
    fi
# Option 2: Reset to common defaults
else
    echo "Resetting to common default values..."
    echo "(Use --restore-backup to restore from saved backup)"
    echo ""
    
    # Common Linux defaults
    set_param "read_ahead_kb" "128"
    set_param "nr_requests" "128"
    
    # Scheduler: Try to set mq-deadline if available, otherwise keep current
    available_schedulers=$(get_param scheduler)
    echo "  Available schedulers: $available_schedulers"
    if echo "$available_schedulers" | grep -q "mq-deadline"; then
        echo "mq-deadline" > "${SYSFS_BASE}/scheduler" 2>/dev/null || echo "  ⚠ Could not set scheduler"
    fi
fi

echo ""
echo "==================================================================="
echo "Current Parameter Values:"
echo "==================================================================="
echo "  read_ahead_kb: $(get_param read_ahead_kb)"
echo "  nr_requests:   $(get_param nr_requests)"
echo "  scheduler:     $(get_param scheduler)"
echo "==================================================================="
echo ""
echo "Note: If you see permission errors, run with sudo:"
echo "  sudo ./reset_params.sh $DEVICE"
echo "==================================================================="
