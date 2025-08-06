#!/bin/bash
# Run the same tests as GitHub Actions Nightly Unit Tests

echo "=== NIGHTLY UNIT TEST SUITE ==="
echo "Running the same tests as GitHub Actions..."
echo ""

# Run tests with the same flags as CI
uv run pytest tests \
    -n auto \
    --cov=brain_go_brrr \
    --cov-config=.coveragerc \
    --dist=loadfile \
    -m "not slow and not integration and not external" \
    --junitxml=test-results.xml \
    -v

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ ALL NIGHTLY TESTS PASSED!"
    echo "GitHub Actions nightly run should succeed."
else
    echo ""
    echo "❌ TESTS FAILED!"
    echo "Fix the failures before the nightly run."
    exit 1
fi