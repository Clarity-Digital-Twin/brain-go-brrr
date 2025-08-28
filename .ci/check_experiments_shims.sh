#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Checking experiments datasets are shims..."

# Check for class/def in experiments datasets (using extended regex)
hits=$(grep -E -n '^(class|def)\s+' experiments/eegpt_linear_probe/datasets/*.py 2>/dev/null | grep -v "__all__\|warnings\|import" || true)

if [[ -n "$hits" ]]; then
    echo "❌ Experiments datasets contain implementations:"
    echo "$hits"
    exit 1
fi

# Check src doesn't import from experiments
hits2=$(grep -r 'from\s+experiments' src --include="*.py" 2>/dev/null || true)

if [[ -n "$hits2" ]]; then
    echo "❌ src imports from experiments:"
    echo "$hits2"
    exit 1
fi

echo "✅ Experiments datasets are pure shims; src does not import experiments"
exit 0