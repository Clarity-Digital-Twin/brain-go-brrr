#!/usr/bin/env python3
"""Autonomous training monitor - runs every 30 minutes automatically."""

import subprocess
import time
import os
import sys
from datetime import datetime
from pathlib import Path

def run_monitor_check():
    """Run comprehensive monitoring check."""
    print(f"\n{'='*80}")
    print(f"AUTONOMOUS MONITOR CHECK - {datetime.now()}")
    print(f"{'='*80}")
    
    # Check tmux session
    try:
        result = subprocess.run(['tmux', 'has-session', '-t', 'eegpt_training'], 
                              capture_output=True)
        if result.returncode != 0:
            print("❌ TRAINING CRASHED - SESSION NOT FOUND!")
            return False
        print("✅ Training session ACTIVE")
    except Exception as e:
        print(f"❌ Error checking tmux: {e}")
        return False
    
    # Get training output
    try:
        result = subprocess.run(['tmux', 'capture-pane', '-t', 'eegpt_training', '-p'],
                              capture_output=True, text=True)
        output_lines = result.stdout.strip().split('\n')[-30:]
        
        # Check for critical issues
        issues = []
        for line in output_lines:
            if 'nan' in line.lower() or 'inf' in line.lower():
                issues.append(f"⚠️ NaN/Inf detected: {line}")
            if 'error' in line.lower() or 'exception' in line.lower():
                issues.append(f"❌ Error found: {line}")
            if 'scanning' in line.lower() or 'found' in line.lower() and '.edf' in line:
                issues.append(f"❌ File scanning detected: {line}")
        
        if issues:
            print("\n🚨 CRITICAL ISSUES FOUND:")
            for issue in issues:
                print(issue)
        else:
            print("✅ No critical issues in recent output")
            
        # Show last few lines
        print("\n📊 Recent output:")
        print('-'*40)
        for line in output_lines[-10:]:
            print(line)
            
    except Exception as e:
        print(f"❌ Error getting tmux output: {e}")
    
    # Check GPU
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                               '--format=csv,noheader,nounits'],
                              capture_output=True, text=True)
        gpu_info = result.stdout.strip().split(', ')
        gpu_util = int(gpu_info[0])
        gpu_mem = int(gpu_info[1])
        
        print(f"\n🖥️ GPU Status: {gpu_util}% utilization, {gpu_mem} MB memory")
        
        if gpu_util < 50:
            print("⚠️ WARNING: Low GPU utilization!")
            
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
    
    # Check latest log
    try:
        log_pattern = Path("logs/eegpt_training_*/training.log")
        log_files = list(Path().glob(str(log_pattern)))
        if log_files:
            latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
            
            # Check for epochs
            with open(latest_log) as f:
                content = f.read()
                epoch_lines = [l for l in content.split('\n') if 'Epoch' in l and 'loss' in l]
                if epoch_lines:
                    print(f"\n📈 Training progress:")
                    for line in epoch_lines[-5:]:
                        print(line)
                        
            # Check cache status
            cache_lines = [l for l in content.split('\n') if 'Loaded TUAB' in l and 'cache' in l]
            if cache_lines:
                print(f"\n💾 Cache status: {cache_lines[-1]}")
                
    except Exception as e:
        print(f"❌ Error checking logs: {e}")
    
    print(f"\n{'='*80}")
    return True

def main():
    """Run monitoring loop."""
    print("🤖 AUTONOMOUS MONITOR STARTED")
    print("Will check every 30 minutes...")
    
    check_count = 0
    while True:
        check_count += 1
        print(f"\n\n{'#'*80}")
        print(f"CHECK #{check_count}")
        
        is_healthy = run_monitor_check()
        
        if not is_healthy:
            print("\n🚨 TRAINING APPEARS TO HAVE FAILED!")
            print("Manual intervention required!")
            # Could add email/notification here
            
        print(f"\nNext check in 30 minutes at {datetime.now().strftime('%H:%M')}")
        print("Press Ctrl+C to stop monitoring")
        
        # Wait 30 minutes
        time.sleep(1800)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Monitor stopped by user")
        sys.exit(0)