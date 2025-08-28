#!/usr/bin/env bash
set -euo pipefail

echo "Checking for legacy compat_coerce in production code..."

# Check for actual code usage, not comments or docstrings
# Look for compat_coerce as a parameter or variable, not in strings
if grep -r "compat_coerce\s*=" src/ --include="*.py" 2>/dev/null || \
   grep -r "compat_coerce\s*:" src/ --include="*.py" 2>/dev/null || \
   grep -r "\.compat_coerce" src/ --include="*.py" 2>/dev/null; then
  echo ""
  echo "❌ FAILURE: legacy compat_coerce code found in production!"
  echo "Production code must not use compat_coerce. Remove all references."
  exit 1
fi

echo "✅ SUCCESS: No legacy compat_coerce in production code"
exit 0
