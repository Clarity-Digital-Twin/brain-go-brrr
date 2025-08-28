#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Checking for sys.path.insert usage..."

hits=$(grep -r 'sys\.path\.insert' src experiments --include="*.py" 2>/dev/null | grep -v "^#" || true)

if [[ -n "$hits" ]]; then
    echo "❌ Found sys.path.insert:"
    echo "$hits"
    exit 1
fi

echo "✅ No sys.path.insert found"
exit 0
