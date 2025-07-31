#!/usr/bin/env python
"""Monitor EEGPT linear probe training progress."""

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


def check_process():
    """Check if training process is running."""
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        return "train_tuab_probe.py" in result.stdout
    except Exception:
        return False


def get_latest_log_file():
    """Find the latest lightning logs directory."""
    logs_dir = Path("lightning_logs")
    if not logs_dir.exists():
        return None

    versions = list(logs_dir.glob("version_*/events.out.tfevents.*"))
    if not versions:
        return None

    return max(versions, key=lambda p: p.stat().st_mtime)


def check_gpu_usage():
    """Check GPU usage on macOS."""
    try:
        # Check if Metal Performance Shaders are being used
        result = subprocess.run(
            ["ps", "-A", "-o", "pid,comm,%cpu,%mem"], capture_output=True, text=True
        )
        lines = result.stdout.split("\n")
        for line in lines:
            if "python" in line.lower() and float(line.split()[-2]) > 50:
                return f"Python process using high CPU: {line.strip()}"
        return "No high CPU usage detected"
    except Exception as e:
        return f"Error checking GPU: {e}"


def monitor():
    """Monitor training progress."""
    print("🔍 EEGPT Linear Probe Training Monitor")
    print("=" * 60)

    while True:
        # Check if process is running
        is_running = check_process()
        status = "✅ RUNNING" if is_running else "❌ NOT RUNNING"

        # Check latest log file
        log_file = get_latest_log_file()
        log_info = "No logs found"
        if log_file:
            size = log_file.stat().st_size / 1024  # KB
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            log_info = f"{log_file.parent.name} - {size:.1f}KB - Updated: {mtime:%H:%M:%S}"

        # Check GPU/CPU usage
        gpu_info = check_gpu_usage()

        # Clear screen and print status
        os.system("clear" if os.name == "posix" else "cls")
        print(f"🔍 EEGPT Training Monitor - {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("=" * 60)
        print(f"Process Status: {status}")
        print(f"Latest Log: {log_info}")
        print(f"System: {gpu_info}")
        print("=" * 60)
        print("\nPress Ctrl+C to exit")

        time.sleep(5)  # Update every 5 seconds


if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")
