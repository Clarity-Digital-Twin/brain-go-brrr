# CI/CD Alignment Guide

## 🎯 Purpose
Ensure local development environment exactly matches CI/CD pipeline to prevent "works on my machine" issues.

## ⚠️ Common CI Failures & Solutions

### 1. **Quote Style Issues**
- **Problem**: CI expects double quotes, local uses single quotes
- **Solution**: Run `make format` before committing

### 2. **Missing Trailing Newline**
- **Problem**: Files missing newline at EOF
- **Solution**: Ruff formatter adds them automatically

### 3. **Whitespace on Blank Lines (W293)**
- **Problem**: Blank lines containing spaces/tabs
- **Solution**: Ruff formatter removes them

## 🔧 Makefile Targets for CI Alignment

| Target | Purpose | When to Use |
|--------|---------|-------------|
| `make format` | Apply ruff formatter (writes files) | Before every commit |
| `make lint-ci` | Run linter WITHOUT auto-fix (like CI) | Test CI compliance |
| `make pre-push` | Complete pre-push validation | Before `git push` |
| `make check-all` | Full CI pipeline locally | Final validation |

## 📋 Pre-Push Checklist

```bash
# 1. Format all code
make format

# 2. Commit formatting changes
git add -A
git commit -m "style: apply ruff formatting"

# 3. Run CI-style checks (no auto-fix)
make lint-ci  # Must pass without errors
make type-fast # Type checking

# 4. Run pre-push validation
make pre-push  # Comprehensive check

# 5. If all pass, push
git push
```

## 🚫 What NOT to Do

1. **Don't run `make lint` before pushing** - it auto-fixes, CI doesn't
2. **Don't ignore formatting changes** - commit them!
3. **Don't skip `make pre-push`** - it catches CI issues

## 🔄 Quick Fix When CI Fails

If CI fails with style issues:

```bash
# Pull latest
git pull

# Fix all style issues
make format

# Check it matches CI
make lint-ci

# Commit fixes
git add -A
git commit -m "style: fix CI linting issues"

# Push
git push
```

## 🎯 Golden Rule

**Always run `make pre-push` before pushing!**

This ensures:
- ✅ Code is formatted (double quotes, trailing newlines)
- ✅ No uncommitted formatting changes
- ✅ Linting passes WITHOUT auto-fix (like CI)
- ✅ Type checking passes
- ✅ CI will be green

## 📝 Makefile Updates

We've added these targets for CI alignment:

```makefile
lint-ci: ## Run linter exactly as CI does (no auto-fix)
	@echo "$(CYAN)Running CI-style lint check...$(NC)"
	$(RUFF) check $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)CI lint check passed!$(NC)"

pre-push: ## Run before pushing to ensure CI will pass
	@echo "$(CYAN)Running pre-push checks...$(NC)"
	$(MAKE) format
	@echo "$(YELLOW)Checking for uncommitted formatting changes...$(NC)"
	@git diff --exit-code || (echo "$(RED)Error: Formatting changes detected. Please commit them.$(NC)" && exit 1)
	$(MAKE) lint-ci
	$(MAKE) type-fast
	@echo "$(GREEN)Ready to push! CI should pass.$(NC)"
```

## 🔍 Debugging CI Failures

Check GitHub Actions logs for exact error:

1. **Quote style**: Look for `Q000` errors
2. **Trailing newline**: Look for `W292` errors  
3. **Blank line whitespace**: Look for `W293` errors

Then fix locally with `make format` and commit.

---

*Last updated: August 2025*