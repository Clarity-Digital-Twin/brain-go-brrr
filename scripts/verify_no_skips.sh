#!/bin/bash
# Enterprise-level guard against bogus skips

echo "🔍 Checking for unauthorized pytest.skip usage..."

# Find any @pytest.mark.skip that's NOT in integration tests
BOGUS_SKIPS=$(grep -r "@pytest\.mark\.skip\b" tests --include="*.py" | grep -v "integration" | wc -l)

if [ "$BOGUS_SKIPS" -gt 0 ]; then
    echo "❌ FOUND UNAUTHORIZED SKIPS!"
    grep -r "@pytest\.mark\.skip\b" tests --include="*.py" | grep -v "integration"
    exit 1
fi

# Find any pytest.skip() calls in test bodies (not fixtures)
RUNTIME_SKIPS=$(grep -r "pytest\.skip(" tests --include="*.py" | grep -v "fixture" | grep -v "conftest" | wc -l)

if [ "$RUNTIME_SKIPS" -gt 0 ]; then
    echo "⚠️  Found runtime skips (should be fixtures or markers):"
    grep -r "pytest\.skip(" tests --include="*.py" | grep -v "fixture" | grep -v "conftest"
fi

echo "✅ No unauthorized skips found!"

# Verify test counts
echo ""
echo "📊 Test Statistics:"
echo "Unit tests: $(uv run pytest tests --co -q -m 'not integration' 2>/dev/null | grep "test session" -A1 | tail -1 | cut -d' ' -f1) collected"
echo "Integration: $(uv run pytest tests --co -q -m 'integration' 2>/dev/null | grep "test session" -A1 | tail -1 | cut -d' ' -f1) collected"

# Verify deselection is working
echo ""
echo "🎯 Deselection Check:"
uv run pytest tests -q --tb=no 2>&1 | tail -1 | grep -E "deselected|skipped"

if [ $? -eq 0 ]; then
    echo "✅ Deselection working correctly!"
else
    echo "❌ Issue with deselection!"
fi