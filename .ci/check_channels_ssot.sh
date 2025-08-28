#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Checking channels SSOT compliance..."

# Check datasets don't define their own STANDARD_CHANNELS
hits=$(grep -n 'STANDARD_CHANNELS\s*=' src/brain_go_brrr/infra/data/*.py 2>/dev/null | grep -v 'channels\.py' || true)

if [[ -n "$hits" ]]; then
    echo "❌ Found dataset-local STANDARD_CHANNELS:"
    echo "$hits"
    exit 1
fi

# Verify SSOT channels are referenced
tuab_ref=$(grep -n 'CHANNELS_TUAB_19' src/brain_go_brrr/infra/data/tuab_dataset.py 2>/dev/null | wc -l)
tuev_ref=$(grep -n 'CHANNELS_TUEV_20' src/brain_go_brrr/infra/data/tuev_dataset.py 2>/dev/null | wc -l)

if [[ "$tuab_ref" -eq 0 ]] || [[ "$tuev_ref" -eq 0 ]]; then
    echo "❌ Datasets not referencing channels from SSOT"
    echo "TUAB references: $tuab_ref, TUEV references: $tuev_ref"
    exit 1
fi

echo "✅ Datasets reference channels from SSOT"
exit 0
