#!/bin/bash
# CI check to ensure launch scripts pass correct arguments to training scripts
# Prevents argument mismatches like --cache_dir vs --cache-dir

set -e

echo "Checking script argument consistency..."

# Check TUEV launch script
echo "Validating TUEV paper parity launch script..."
if grep -q -- '--cache_dir' experiments/eegpt_linear_probe/scripts/launch_tuev_paper_parity.sh; then
    echo "ERROR: launch_tuev_paper_parity.sh uses --cache_dir but train_tuev_mne.py expects --cache-dir"
    exit 1
fi

# Check TUAB launch script if exists
if [ -f "experiments/eegpt_linear_probe/scripts/launch_tuab_mne.sh" ]; then
    echo "Validating TUAB launch script..."
    if grep -q -- '--cache_dir' experiments/eegpt_linear_probe/scripts/launch_tuab_mne.sh; then
        echo "ERROR: launch_tuab_mne.sh uses --cache_dir but train_tuab_mne.py expects --cache-dir"
        exit 1
    fi
fi

# Check that all scripts using argparse have consistent naming
echo "Checking argparse consistency..."
for script in experiments/eegpt_linear_probe/train_*.py; do
    if [ -f "$script" ]; then
        echo "  Checking $script..."
        # Ensure hyphens used consistently in argparse
        if grep -q "parser.add_argument.*'--cache_dir'" "$script"; then
            echo "ERROR: $script uses underscore in --cache_dir argument"
            exit 1
        fi
    fi
done

echo "✓ All script arguments are consistent"
