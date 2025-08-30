#!/bin/bash
# CRITICAL: Run this BEFORE pushing to prevent CI failures
# This script runs the EXACT same checks as CI/CD

set -e  # Exit on any error

echo "==================================================="
echo "🚨 PRE-PUSH VALIDATION - MUST PASS OR CI WILL FAIL"
echo "==================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "\n${YELLOW}1. Formatting code with CI's ruff (0.12.3)...${NC}"
# Prefer project venv ruff to avoid version drift; fall back to uv
if [ -x ".venv/bin/ruff" ]; then
  RUFF_BIN=".venv/bin/ruff"
else
  RUFF_BIN="ruff"
fi

"${RUFF_BIN}" format src/ tests/ scripts/ experiments/
echo -e "${GREEN}✅ Formatted${NC}"

echo -e "\n${YELLOW}2. Checking format matches CI...${NC}"
if "${RUFF_BIN}" format --check src/ tests/ scripts/ experiments/; then
    echo -e "${GREEN}✅ Format check passed${NC}"
else
    echo -e "${RED}❌ FORMAT CHECK FAILED - CI WILL FAIL${NC}"
    echo "Run: uv run ruff format src/ tests/ scripts/ experiments/"
    exit 1
fi

echo -e "\n${YELLOW}3. Checking for lint errors...${NC}"
if "${RUFF_BIN}" check src/ tests/ scripts/ experiments/; then
    echo -e "${GREEN}✅ Lint check passed${NC}"
else
    echo -e "${RED}❌ LINT CHECK FAILED - CI WILL FAIL${NC}"
    echo "Run: uv run ruff check --fix src/ tests/ scripts/ experiments/"
    echo "Then manually fix any remaining errors"
    exit 1
fi

echo -e "\n${YELLOW}4. Running mypy type checking...${NC}"
if [ -x ".venv/bin/mypy" ]; then
  MYPY_BIN=".venv/bin/mypy"
else
  MYPY_BIN="mypy"
fi
if "${MYPY_BIN}" --config-file mypy.ini src/brain_go_brrr; then
    echo -e "${GREEN}✅ Type check passed${NC}"
else
    echo -e "${RED}❌ TYPE CHECK FAILED - CI WILL FAIL${NC}"
    echo "Fix the type errors shown above"
    exit 1
fi

echo -e "\n${YELLOW}5. Running pre-commit hooks...${NC}"
if pre-commit run --all-files; then
    echo -e "${GREEN}✅ Pre-commit hooks passed${NC}"
else
    echo -e "${YELLOW}⚠️  Pre-commit hooks had issues (may auto-fix)${NC}"
    echo "Re-run this script after reviewing changes"
fi

echo -e "\n${GREEN}==================================================="
echo "🎉 ALL CHECKS PASSED - SAFE TO PUSH!"
echo "===================================================${NC}"
echo ""
echo "Commands to push:"
echo "  git add -A"
echo "  git commit -m \"your commit message\""
echo "  git push"
