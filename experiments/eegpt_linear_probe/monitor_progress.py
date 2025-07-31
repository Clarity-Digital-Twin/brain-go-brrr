#!/usr/bin/env python
"""Monitor training progress by tailing the latest log file."""

import subprocess
from pathlib import Path

# Find the latest log file
log_dir = Path("logs")
log_files = sorted(log_dir.glob("training_*.log"), key=lambda x: x.stat().st_mtime)

if not log_files:
    print("No training logs found!")
    exit(1)

latest_log = log_files[-1]
print(f"Monitoring: {latest_log}")
print("=" * 60)

# Tail the log file
cmd = ["tail", "-f", str(latest_log)]
subprocess.run(cmd)
