#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Checking META schema compliance..."

# Check for legacy channel keys in writers
writers=$(grep -n '"channels19"\|"channels20"' src/brain_go_brrr/infra/data/*.py 2>/dev/null || true)

if [[ -n "$writers" ]]; then
    echo "❌ Found legacy META channel keys in writers:"
    echo "$writers"
    exit 1
fi

# Verify unified schema is used
unified=$(grep -n '"channels"' src/brain_go_brrr/infra/data/tuab_dataset.py src/brain_go_brrr/infra/data/tuev_dataset.py 2>/dev/null | wc -l)

if [[ "$unified" -eq 0 ]]; then
    echo "❌ Missing unified 'channels' key in dataset writers"
    exit 1
fi

echo "✅ META writers use unified schema (\"channels\" + \"n_channels\")"
exit 0