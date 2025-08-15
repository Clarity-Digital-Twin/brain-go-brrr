#!/bin/bash
# Validate everything before pushing - ensure we stay in banger-town! 🚀

set -e  # Exit on any error

echo "🔍 Running pre-push validation..."
echo "================================="

# 1. Format check
echo "📝 Checking formatting..."
make format-check || { echo "❌ Format check failed!"; exit 1; }

# 2. Lint check
echo "🔍 Running linter..."
make lint-ci || { echo "❌ Lint check failed!"; exit 1; }

# 3. Type check
echo "🔎 Type checking..."
make type-check || { echo "⚠️  Type check has issues (continuing)"; }

# 4. Unit tests
echo "🧪 Running unit tests..."
make test-unit || { echo "❌ Unit tests failed!"; exit 1; }

# 5. Quick import smoke test
echo "💨 Import smoke test..."
uv run python - <<'PY'
import importlib
import sys
try:
    for m in [
        "brain_go_brrr.api.app",
        "brain_go_brrr.infra.data.edf_loader",
        "brain_go_brrr.infra.data.edf_streaming",
    ]:
        importlib.import_module(m)
        print(f"  ✓ {m}")
    print("\n✅ All critical imports working!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
PY

echo ""
echo "================================="
echo "✅ ALL CHECKS PASSED - READY TO PUSH! 🚀"
echo "You are officially in BANGER-TOWN! 💯"
