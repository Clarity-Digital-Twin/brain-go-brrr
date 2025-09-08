#!/bin/bash
# P2 Technical Debt Verification Script
# Auto-generated from P2_TECHNICAL_DEBT.md

echo "🔍 P2 Technical Debt Verification"
echo "================================="
echo ""

FAILURES=0
FIXED=0

# Check 768-dim removal
echo -n "1. 768-dim tolerance removal: "
if command -v rg &> /dev/null && rg -q "768" src/brain_go_brrr/domain/abnormal/detector.py 2>/dev/null; then
    echo "❌ Still present"
    ((FAILURES++))
else
    echo "✅ Removed"
    ((FIXED++))
fi

# Check EEGPTProbe usage
echo -n "2. EEGPTProbe migration: "
if command -v rg &> /dev/null && rg -q "from.*eegpt_probe_unified import EEGPTProbe" src/brain_go_brrr/application 2>/dev/null; then
    echo "❌ Still using deprecated class"
    ((FAILURES++))
else
    echo "✅ Migrated"
    ((FIXED++))
fi

# Check duplicate CachePort
echo -n "3. Duplicate CachePort: "
if command -v rg &> /dev/null; then
    CACHE_PORTS=$(rg "^class\s+CachePort" src 2>/dev/null | wc -l)
else
    CACHE_PORTS=$(grep -r "^class CachePort" src 2>/dev/null | wc -l)
fi
if [ "$CACHE_PORTS" -gt 1 ]; then
    echo "❌ Multiple definitions found"
    ((FAILURES++))
else
    echo "✅ Single definition"
    ((FIXED++))
fi

# Check eegpt_compat re-export
echo -n "4. eegpt_compat re-export: "
if grep -q "from .eegpt_compat import" src/brain_go_brrr/infra/ml_models/__init__.py 2>/dev/null; then
    echo "❌ Still re-exported"
    ((FAILURES++))
else
    echo "✅ Cleaned"
    ((FIXED++))
fi

# Check services redirect
echo -n "5. Services redirect: "
if command -v rg &> /dev/null && rg -q "services.yasa_adapter" src tests 2>/dev/null; then
    echo "❌ Old imports exist"
    ((FAILURES++))
else
    echo "✅ All migrated"
    ((FIXED++))
fi

# Check sys.path hacks
echo -n "6. sys.path hacks: "
if command -v rg &> /dev/null && rg -q "sys\.path\.(insert|append)" --type py experiments/ 2>/dev/null; then
    echo "❌ Found in Python files"
    ((FAILURES++))
else
    echo "✅ None in Python"
    ((FIXED++))
fi

# Check Lightning imports
echo -n "7. PyTorch Lightning: "
if command -v rg &> /dev/null && rg -q "import\s+lightning|from\s+lightning" src experiments 2>/dev/null; then
    echo "❌ Lightning imports found"
    ((FAILURES++))
else
    echo "✅ No Lightning"
    ((FIXED++))
fi

# Check Redis aliases
echo -n "8. Redis alias cleanup: "
if command -v rg &> /dev/null && rg -q "as RedisCache" tests/ 2>/dev/null; then
    echo "❌ Aliases present"
    ((FAILURES++))
else
    echo "✅ Direct imports"
    ((FIXED++))
fi

# Check for security tools
echo -n "9. Security scanning: "
if grep -q "pip-audit" pyproject.toml 2>/dev/null; then
    echo "✅ pip-audit installed"
    ((FIXED++))
else
    echo "⚠️  pip-audit not installed (run: uv add --dev pip-audit)"
    ((FAILURES++))
fi

# Summary
echo ""
echo "================================="
echo "Summary: $FIXED/9 items completed"
echo ""

if [ "$FAILURES" -eq 0 ]; then
    echo "🎉 All P2 Sprint 1 items complete!"
    echo "Next: Enable blocking CI check"
    exit 0
elif [ "$FIXED" -ge 5 ]; then
    echo "📊 Good progress! $FAILURES items remaining"
    echo "Continue with Sprint 2"
    exit 0
else
    echo "⚠️  $FAILURES P2 items need attention"
    echo "Start with Sprint 1 quick wins"
    exit 1
fi