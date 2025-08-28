#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Checking for Lightning imports..."

hits=$(grep -r -E '(^|\s)from\s+lightning|(^|\s)import\s+lightning|pytorch_lightning' src experiments --include="*.py" 2>/dev/null || true)

if [[ -n "$hits" ]]; then
    echo "❌ Found Lightning imports:"
    echo "$hits"
    exit 1
fi

echo "✅ No Lightning imports found"
exit 0