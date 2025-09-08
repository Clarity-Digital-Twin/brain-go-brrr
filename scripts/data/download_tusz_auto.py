#!/usr/bin/env python3
"""Download TUSZ dataset with automatic authentication using pexpect.

Handles password input automatically and shows progress.
"""

import os
import sys
from pathlib import Path

import pexpect
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATASET = "tusz"
VERSION = "v2.0.3"  # nosec:hardcoded-path - TUSZ version is fixed by TUH
REMOTE_PATH = f"data/tuh_eeg/tuh_eeg_seizure/{VERSION}/"
LOCAL_PATH = Path("data/datasets/tusz/")
USERNAME = os.getenv("TUH_USERNAME", "nedc-tuh-eeg")
PASSWORD = os.getenv("TUH_PASSWORD")

if not PASSWORD:
    print("❌ Error: TUH_PASSWORD not found in .env file")
    sys.exit(1)

# Create directory
LOCAL_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print(f"Starting TUSZ {VERSION} download")
print("This will download ~40GB of seizure detection data")
print("Download is resumable - safe to interrupt with Ctrl+C")
print("=" * 60)
print()

# Build rsync command
rsync_cmd = (
    f"rsync -auxvL --progress --partial "
    f"{USERNAME}@www.isip.piconepress.com:{REMOTE_PATH} "
    f"{LOCAL_PATH}/"
)

print(f"Running: {rsync_cmd}")
print()

try:
    # Spawn rsync process
    child = pexpect.spawn(rsync_cmd, timeout=30, encoding='utf-8')

    # Handle password prompt
    index = child.expect(['password:', pexpect.EOF, pexpect.TIMEOUT])

    if index == 0:
        print("🔐 Authenticating...")
        child.sendline(PASSWORD)
        print("✅ Authentication sent")
        print()
        print("📥 Download starting (this may take a while)...")
        print("-" * 60)

        # Stream output to console
        child.logfile = sys.stdout
        child.expect(pexpect.EOF, timeout=None)  # No timeout for download

    elif index == 1:
        print("Connection closed unexpectedly")
        sys.exit(1)
    else:
        print("Timeout waiting for password prompt")
        sys.exit(1)

    # Check exit status
    child.close()
    exit_status = child.exitstatus

    if exit_status == 0:
        print()
        print("=" * 60)
        print("✅ TUSZ download completed successfully!")
        print("=" * 60)

        # Create completion marker
        (LOCAL_PATH / ".download_complete").touch()

        # Show statistics
        print()
        print("Dataset statistics:")
        print("-" * 20)

        edf_files = list(LOCAL_PATH.rglob("*.edf"))
        csv_files = list(LOCAL_PATH.rglob("*.csv"))
        txt_files = list(LOCAL_PATH.rglob("*.txt"))

        print(f"Total EDF files: {len(edf_files)}")
        print(f"Total CSV annotations: {len(csv_files)}")
        print(f"Total TXT files: {len(txt_files)}")

        # Calculate size
        total_size = sum(f.stat().st_size for f in LOCAL_PATH.rglob("*") if f.is_file())
        print(f"Total size: {total_size / (1024**3):.2f} GB")

    else:
        print()
        print("=" * 60)
        print(f"⚠️ Download failed with exit code: {exit_status}")
        print("Run this script again to resume")
        print("=" * 60)
        sys.exit(exit_status)

except KeyboardInterrupt:
    print("\n\n" + "=" * 60)
    print("⚠️ Download interrupted by user")
    print("Run this script again to resume from where it left off")
    print("=" * 60)
    sys.exit(1)

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check your internet connection")
    print("2. Verify credentials in .env file")
    print("3. Try manual rsync with password input")
    sys.exit(1)
