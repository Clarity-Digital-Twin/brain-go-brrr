#!/usr/bin/env bash
set -euo pipefail

# Check for parallel implementations between experiments/ and src/
# This prevents the architecture drift disaster from happening again

echo "🔍 Checking for parallel implementations..."

EXIT_CODE=0

# 1. Check for sys.path.insert hacks
echo -n "Checking for sys.path.insert hacks... "
if grep -r "sys\.path\.insert" experiments/ --include="*.py" 2>/dev/null | grep -v "^#"; then
    echo "❌ FOUND"
    echo "ERROR: sys.path.insert found in experiments! Use proper imports from brain_go_brrr instead."
    EXIT_CODE=1
else
    echo "✅ clean"
fi

# 2. Check for duplicate preprocessing implementations
echo -n "Checking for duplicate preprocessing... "
PREPROCESSING_FILES=$(find experiments/ -name "*preprocess*.py" -o -name "*normali*.py" 2>/dev/null | grep -v __pycache__ || true)
for file in $PREPROCESSING_FILES; do
    # Check if file has actual implementation (>100 lines) not just imports
    LINE_COUNT=$(wc -l < "$file" 2>/dev/null || echo 0)
    if [ "$LINE_COUNT" -gt 100 ]; then
        # Check if it imports from brain_go_brrr
        if ! grep -q "from brain_go_brrr" "$file"; then
            echo "❌ FOUND"
            echo "ERROR: $file has $LINE_COUNT lines but doesn't import from brain_go_brrr!"
            EXIT_CODE=1
            break
        fi
    fi
done
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ clean"
fi

# 3. Check dataset files are thin shims (<50 lines)
echo -n "Checking datasets are thin shims... "
DATASET_FILES=$(find experiments/ -path "*/datasets/*.py" 2>/dev/null | grep -v __pycache__ || true)
for file in $DATASET_FILES; do
    LINE_COUNT=$(wc -l < "$file" 2>/dev/null || echo 0)
    if [ "$LINE_COUNT" -gt 50 ]; then
        echo "❌ BLOATED"
        echo "ERROR: $file has $LINE_COUNT lines! Datasets in experiments/ should be thin shims (<50 lines)."
        echo "Move the implementation to src/brain_go_brrr/infra/data/ and import from there."
        EXIT_CODE=1
        break
    fi
done
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ clean"
fi

# 4. Check for duplicate model implementations
echo -n "Checking for duplicate models... "
MODEL_FILES=$(find experiments/ -name "*model*.py" -o -name "*wrapper*.py" 2>/dev/null | grep -v __pycache__ || true)
for file in $MODEL_FILES; do
    # Check for class definitions that aren't importing from brain_go_brrr
    if grep -q "class.*Model\|class.*Wrapper" "$file" 2>/dev/null; then
        if ! grep -q "from brain_go_brrr" "$file"; then
            echo "❌ FOUND"
            echo "ERROR: $file defines models but doesn't import from brain_go_brrr!"
            EXIT_CODE=1
            break
        fi
    fi
done
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ clean"
fi

# 5. Check import ratio (should import more from src than from experiments)
echo -n "Checking import sources... "
SRC_IMPORTS=$(grep -r "from brain_go_brrr" experiments/ --include="*.py" 2>/dev/null | wc -l || echo 0)
EXP_IMPORTS=$(grep -r "from experiments" experiments/ --include="*.py" 2>/dev/null | grep -v "^#" | wc -l || echo 0)

if [ "$EXP_IMPORTS" -gt 0 ] && [ "$SRC_IMPORTS" -lt "$EXP_IMPORTS" ]; then
    echo "❌ BAD RATIO"
    echo "ERROR: More imports from experiments ($EXP_IMPORTS) than from src ($SRC_IMPORTS)!"
    echo "Experiments should primarily import from brain_go_brrr, not from itself."
    EXIT_CODE=1
else
    echo "✅ good ratio ($SRC_IMPORTS src vs $EXP_IMPORTS exp)"
fi

# Final result
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: No parallel implementations detected"
else
    echo "❌ FAILURE: Parallel implementations found - this is how the AUROC=0.50 disaster happened!"
    echo ""
    echo "REMEMBER: experiments/ should ONLY contain:"
    echo "  - Training loops (<200 lines)"
    echo "  - Config files"
    echo "  - Thin dataset shims (<50 lines) that import from src/"
    echo ""
    echo "EVERYTHING ELSE belongs in src/brain_go_brrr/"
fi

exit $EXIT_CODE