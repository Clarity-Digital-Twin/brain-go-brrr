#!/usr/bin/env python3
"""Monitor extraction progress in real-time."""

import time
import subprocess
from pathlib import Path

def count_edf_files(directory):
    """Count EDF files in a directory."""
    try:
        result = subprocess.run(
            ["find", str(directory), "-name", "*.edf"],
            capture_output=True, text=True, check=True
        )
        return len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
    except:
        return 0

def get_log_progress():
    """Get latest progress from extraction log."""
    try:
        result = subprocess.run(
            ["tail", "-1", "extraction_retry.log"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except:
        return "No log data"

def main():
    """Monitor extraction progress."""
    train_dir = Path("data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf/train")
    eval_dir = Path("data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf/eval")
    
    print("🚀 Monitoring extraction progress...")
    print("Press Ctrl+C to stop monitoring\n")
    
    target_train = 2717
    target_eval = 276
    
    try:
        while True:
            train_count = count_edf_files(train_dir)
            eval_count = count_edf_files(eval_dir)
            latest_log = get_log_progress()
            
            train_pct = (train_count / target_train) * 100 if target_train > 0 else 0
            eval_pct = (eval_count / target_eval) * 100 if target_eval > 0 else 0
            
            print(f"\r📊 Train: {train_count:,}/{target_train:,} ({train_pct:.1f}%) | "
                  f"Eval: {eval_count:,}/{target_eval:,} ({eval_pct:.1f}%) | "
                  f"Latest: {latest_log[-50:]}", end="", flush=True)
            
            if train_count >= target_train and eval_count >= target_eval:
                print(f"\n\n🎉 EXTRACTION COMPLETE!")
                print(f"✅ Train files: {train_count:,}")
                print(f"✅ Eval files: {eval_count:,}")
                break
                
            time.sleep(5)
            
    except KeyboardInterrupt:
        print(f"\n\n📊 Current progress:")
        print(f"Train files: {train_count:,}/{target_train:,} ({train_pct:.1f}%)")
        print(f"Eval files: {eval_count:,}/{target_eval:,} ({eval_pct:.1f}%)")

if __name__ == "__main__":
    main()