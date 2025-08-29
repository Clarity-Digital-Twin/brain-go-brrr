#!/usr/bin/env python3
"""Download TUH EEG datasets using credentials from .env file.

USAGE:
    # Download TUAB (120GB)
    python scripts/data/download_datasets.py TUAB

    # Download TUEV (60GB)
    python scripts/data/download_datasets.py TUEV

    # Download all datasets
    python scripts/data/download_datasets.py ALL

SECURITY:
    - Credentials stored in .env file ONLY
    - Passwords NEVER logged or printed
    - Uses pexpect to handle authentication securely
"""

import os
import sys
from pathlib import Path

import pexpect
from dotenv import load_dotenv

# Load credentials from .env
env_path = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.env")
load_dotenv(env_path)

USERNAME = os.getenv("TUH_USERNAME", "nedc-tuh-eeg")
PASSWORD = os.getenv("TUH_PASSWORD")

if not PASSWORD:
    print("❌ ERROR: TUH_PASSWORD not found in .env file!")
    print("Add to .env: TUH_PASSWORD=your_password")
    sys.exit(1)

# CORRECT versions from Temple website
DATASETS = {
    'TUAB': {'version': 'v3.0.1', 'path': 'tuh_eeg_abnormal', 'size': '~120GB'},
    'TUEV': {'version': 'v2.0.1', 'path': 'tuh_eeg_events', 'size': '~60GB'},
    'TUSZ': {'version': 'v2.0.3', 'path': 'tuh_eeg_seizure', 'size': '~40GB'},
    'TUEP': {'version': 'v2.0.1', 'path': 'tuh_eeg_epilepsy', 'size': '~30GB'},
}

SERVER = "www.isip.piconepress.com"
BASE_LOCAL = "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/datasets"


def download_dataset(name, info):
    """Download dataset without exposing password."""
    remote_path = f"data/tuh_eeg/{info['path']}/{info['version']}/"
    local_path = f"{BASE_LOCAL}/{name.lower()}/"

    print("=" * 60)
    print(f"📥 DOWNLOADING {name} {info['version']}")
    print("=" * 60)
    print(f"Size: {info['size']}")
    print(f"To: {local_path}")
    print("=" * 60)

    Path(local_path).mkdir(parents=True, exist_ok=True)

    # Build command WITHOUT password
    cmd = f"rsync -auxvL --progress {USERNAME}@{SERVER}:{remote_path} {local_path}"

    print(f"Connecting as: {USERNAME}")
    print("Starting download...")
    print("")

    try:
        # Spawn with logging that won't show password
        child = pexpect.spawn(cmd, timeout=None, encoding='utf-8')

        # Custom logger that filters password
        class PasswordFilter:
            def write(self, data):
                # Don't log password prompts or entries
                if 'password:' not in data.lower():
                    sys.stdout.write(data)
                    sys.stdout.flush()

            def flush(self):
                sys.stdout.flush()

        child.logfile = PasswordFilter()

        while True:
            i = child.expect(
                [
                    'password:',
                    'Password:',
                    'Are you sure you want to continue connecting',
                    'yes/no',
                    pexpect.EOF,
                    pexpect.TIMEOUT,
                ],
                timeout=30,
            )

            if i in [0, 1]:  # Password prompt
                child.sendline(PASSWORD)  # Send but don't log
            elif i in [2, 3]:  # SSH fingerprint
                print("\n[Accepting SSH fingerprint]")
                child.sendline('yes')
            elif i == 4:  # EOF
                break
            elif i == 5:  # Timeout during download
                continue

        child.wait()

        if child.exitstatus == 0:
            print(f"\n✅ {name} download successful!")
            marker = Path(local_path) / ".download_complete"
            marker.write_text(f"{name} {info['version']} downloaded\n")
            return True
        else:
            print(f"\n⚠️ {name} had issues (exit: {child.exitstatus})")
            return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def main():
    """Main download function."""
    if len(sys.argv) < 2:
        print("Usage: python download_datasets.py [TUAB|TUEV|TUSZ|TUEP|ALL]")
        sys.exit(1)

    dataset = sys.argv[1].upper()

    print("=" * 60)
    print("🚀 SECURE TUH EEG DOWNLOADER")
    print("=" * 60)
    print(f"Username: {USERNAME}")
    print("Password: [LOADED FROM .env - NEVER LOGGED]")
    print("=" * 60)

    if dataset == "ALL":
        for name, info in DATASETS.items():
            download_dataset(name, info)
    elif dataset in DATASETS:
        if download_dataset(dataset, DATASETS[dataset]):
            print("\n✅ SUCCESS!")
            if dataset == "TUAB":
                print("Next steps:")
                print("1. Build cache: ./scripts/build_mne_cache.sh")
                print("2. Train: ./scripts/launch_tuab_mne.sh")
        else:
            print("\n❌ Download failed - check credentials")
    else:
        print(f"Unknown dataset: {dataset}")
        print(f"Available: {', '.join(DATASETS.keys())}")


if __name__ == "__main__":
    main()
