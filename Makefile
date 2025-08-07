# Brain Go Brrr - Modern Python ML Project Makefile
# 2025 Best Practices with uv, ruff, and modern tooling

.PHONY: help install dev-install test test-fast test-cov lint format type-check quality check pre-commit clean build docs serve-docs train preprocess evaluate serve notebook mlflow dvc-setup docker-build docker-run benchmark profile

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
MAGENTA := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
NC := \033[0m # No Color

# Project variables
PROJECT_NAME := brain-go-brrr
SRC_DIR := src/brain_go_brrr
TEST_DIR := tests
DOCS_DIR := docs
CONFIG_DIR := config
NOTEBOOKS_DIR := notebooks

# Python and uv settings
PYTHON_VERSION := 3.11
UV := uv
PYTHON := $(UV) run python
PIP := $(UV) pip
PYTEST := $(UV) run pytest
RUFF := $(UV) run ruff

# Pytest options - can be overridden via environment
PYTEST_BASE_OPTS ?= -v
PYTEST_NO_PLUGINS ?= -p no:pytest_benchmark -p no:xdist
PYTEST_PARALLEL ?= -p xdist -n auto
MYPY := $(UV) run mypy
JUPYTER := $(UV) run jupyter
PRE_COMMIT := $(UV) run pre-commit

# Default target
help: ## Show this help message
	@echo "$(CYAN)Brain Go Brrr - Modern ML Project Commands$(NC)"
	@echo "$(YELLOW)================================================$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BLUE)Development Workflow:$(NC)"
	@echo "  1. $(GREEN)make install$(NC)     - Install dependencies"
	@echo "  2. $(GREEN)make dev-install$(NC) - Install dev dependencies"
	@echo "  3. $(GREEN)make pre-commit$(NC)  - Setup pre-commit hooks"
	@echo "  4. $(GREEN)make test$(NC)        - Run tests"
	@echo "  5. $(GREEN)make check$(NC)       - Run all quality checks"

##@ Dependencies

install: ## Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	$(UV) sync --no-dev

dev-install: ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	$(UV) sync
	@echo "$(GREEN)Dependencies installed successfully!$(NC)"

update: ## Update dependencies
	@echo "$(GREEN)Updating dependencies...$(NC)"
	$(UV) lock --upgrade

##@ Development

dev-setup: dev-install pre-commit ## Complete development setup
	@echo "$(GREEN)Development environment setup complete!$(NC)"

pre-commit: ## Setup pre-commit hooks
	@echo "$(GREEN)Setting up pre-commit hooks...$(NC)"
	$(PRE_COMMIT) install
	$(PRE_COMMIT) install --hook-type commit-msg
	@echo "$(GREEN)Pre-commit hooks installed!$(NC)"

notebook: ## Start Jupyter Lab
	@echo "$(GREEN)Starting Jupyter Lab...$(NC)"
	$(JUPYTER) lab --notebook-dir=$(NOTEBOOKS_DIR)

##@ Code Quality

lint: ## Run linting with ruff (with auto-fix)
	@echo "$(GREEN)Running linting checks...$(NC)"
	$(RUFF) check $(SRC_DIR) $(TEST_DIR) --fix
	@echo "$(GREEN)Linting complete!$(NC)"

lint-ci: ## Run linter exactly as CI does (no auto-fix)
	@echo "$(CYAN)Running CI-style lint check...$(NC)"
	$(RUFF) check $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)CI lint check passed!$(NC)"

format: ## Format code with ruff
	@echo "$(GREEN)Formatting code...$(NC)"
	$(RUFF) format $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)Code formatted!$(NC)"

type-check: ## Run full strict type checking (CI/pre-commit)
	@echo "$(CYAN)Running full type checks...$(NC)"
	@rm -rf .mypy_cache_strict 2>/dev/null || true
	$(MYPY) --config-file mypy.ini src/brain_go_brrr
	@echo "$(GREEN)Type checking complete!$(NC)"

type-fast: ## Fast type checking for development (no hangs)
	@echo "$(CYAN)Running fast type checks...$(NC)"
	$(MYPY) --config-file mypy-fast.ini src/brain_go_brrr
	@echo "$(GREEN)Fast type checking complete!$(NC)"

type-strict: ## Strictest type checking (catches everything)
	@echo "$(CYAN)Running strict type checks...$(NC)"
	$(MYPY) --config-file mypy.ini src/brain_go_brrr
	@echo "$(GREEN)Strict type checking complete!$(NC)"

type-critical: ## Type check critical modules only
	@echo "$(CYAN)Type checking critical modules...$(NC)"
	$(MYPY) --config-file mypy.ini \
		src/brain_go_brrr/data/tuab_cached_dataset.py \
		src/brain_go_brrr/models/eegpt_* \
		src/brain_go_brrr/tasks/enhanced_abnormality_detection.py
	@echo "$(GREEN)Critical modules type checked!$(NC)"

type-check-file: ## Check specific file: make type-check-file FILE=path/to/file.py
	@echo "$(CYAN)Type checking $(FILE)...$(NC)"
	$(MYPY) $(FILE)
	@echo "$(GREEN)File type checking complete!$(NC)"

quality: lint format type-check ## Run all code quality checks
	@echo "$(GREEN)All quality checks complete!$(NC)"

check: test-fast quality ## Run all tests and quality checks
	@echo "$(GREEN)All checks passed!$(NC)"

##@ Testing

test: ## Run fast tests only (excludes slow, external, gpu) with parallel execution
	@echo "$(GREEN)Running fast tests with parallel execution...$(NC)"
	$(PYTEST) $(TEST_DIR) $(PYTEST_BASE_OPTS) -m "not slow and not external and not gpu" --ignore=tests/benchmarks -n 4 --no-cov

test-unit: ## Run unit tests only (fast)
	@echo "$(GREEN)Running unit tests...$(NC)"
	$(PYTEST) tests/unit $(PYTEST_BASE_OPTS) -q $(PYTEST_NO_PLUGINS)

test-unit-cov: ## Run unit tests with coverage (excludes MNE modules)
	@echo "$(GREEN)Running unit tests with coverage...$(NC)"
	$(PYTEST) tests/unit -m "not integration and not slow" \
		--cov=brain_go_brrr \
		--cov-config=.coveragerc \
		--cov-report=term-missing:skip-covered \
		--cov-report=html \
		--no-cov-on-fail \
		--timeout=600
	@echo "$(CYAN)Coverage report: htmlcov/index.html$(NC)"

test-fast-cov: ## Run ONLY fast tests with coverage for quick feedback
	@echo "$(CYAN)Running FAST tests with coverage (no benchmarks, no slow tests)...$(NC)"
	$(PYTEST) tests/unit tests/api \
		-m "not integration and not slow and not benchmark" \
		--cov=brain_go_brrr \
		--cov-config=.coveragerc \
		--cov-report=term-missing:skip-covered \
		--tb=short \
		-q

test-integration: ## Run integration tests without coverage (includes MNE/YASA)
	@echo "$(GREEN)Running integration tests (no coverage)...$(NC)"
	$(PYTEST) tests/integration -m "not slow" --no-cov -v
	@echo "$(GREEN)Integration tests complete!$(NC)"

test-slow: ## Run slow tests (nightly CI only)
	@echo "$(YELLOW)Running slow tests (this may take a while)...$(NC)"
	$(PYTEST) -m "slow" --no-cov -v
	@echo "$(GREEN)Slow tests complete!$(NC)"

test-perf: ## Run performance benchmarks
	@echo "$(GREEN)Running performance benchmarks...$(NC)"
	$(PYTEST) tests/benchmarks $(PYTEST_BASE_OPTS) -m perf -p pytest_benchmark

test-parallel: ## Run tests in parallel (with xdist)
	@echo "$(GREEN)Running tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR) $(PYTEST_BASE_OPTS) $(PYTEST_PARALLEL) --ignore=tests/benchmarks

test-fast: ## Run tests in parallel without coverage (fastest)
	@echo "$(GREEN)Running tests in parallel (fast mode)...$(NC)"
	$(PYTEST) $(TEST_DIR) $(PYTEST_BASE_OPTS) -m "not slow and not external and not gpu" --ignore=tests/benchmarks -n 4 --no-cov

test-cov: ## Run tests with coverage (single process, longer timeout)
	@echo "$(GREEN)Running tests with coverage (single process, ~2-3 minutes)...$(NC)"
	@echo "$(YELLOW)Note: Using single process for accurate coverage. This takes longer than parallel tests.$(NC)"
	$(PYTEST) $(TEST_DIR) \
		--cov=brain_go_brrr \
		--cov-report=term-missing:skip-covered \
		--cov-report= \
		--no-cov-on-fail \
		-m "not slow and not integration and not external" \
		--tb=short \
		--timeout=600
	@echo "$(CYAN)Generating HTML coverage report...$(NC)"
	@$(UV) run coverage html
	@echo "$(GREEN)Coverage report generated at: htmlcov/index.html$(NC)"

test-cov-parallel: ## Run tests with coverage in parallel (requires combine step)
	@echo "$(GREEN)Running tests with coverage in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR) -n auto \
		--cov=brain_go_brrr \
		--cov-config=.coveragerc \
		--dist=loadfile \
		-m "not slow and not integration and not external"
	@echo "$(CYAN)Combining coverage data...$(NC)"
	@$(UV) run coverage combine
	@$(UV) run coverage html
	@echo "$(GREEN)Coverage report generated at: htmlcov/index.html$(NC)"

coverage-report: ## Display coverage report summary
	@echo "$(GREEN)Coverage Summary:$(NC)"
	@$(UV) run coverage report --skip-covered | head -20
	@echo ""
	@echo "$(CYAN)Full HTML coverage report available at: htmlcov/index.html$(NC)"

cov: ## Quick coverage check - shows TOTAL coverage percentage
	@echo "$(GREEN)Running quick coverage check...$(NC)"
	@$(PYTEST) tests \
		--cov=brain_go_brrr \
		--cov-report=term \
		-m "not slow and not external and not gpu and not integration" \
		--tb=short \
		--timeout=300 \
		-q | grep -E "TOTAL.*[0-9]+%" || echo "No coverage data found"

test-ci: ## Run tests for CI with coverage and XML report
	@echo "$(GREEN)Running CI test suite with coverage...$(NC)"
	$(PYTEST) $(TEST_DIR) -n auto \
		--cov=brain_go_brrr \
		--cov-config=.coveragerc \
		--dist=loadfile \
		-m "not slow and not integration and not external" \
		--junitxml=test-results.xml
	@$(UV) run coverage combine
	@$(UV) run coverage xml
	@$(UV) run coverage report
	@echo "$(GREEN)CI test results: test-results.xml, coverage.xml$(NC)"

# Duplicate removed - see line 157 for test-integration target

test-all: ## Run ALL tests including slow/external/gpu
	@echo "$(YELLOW)Running ALL tests (including slow/external)...$(NC)"
	$(PYTEST) $(TEST_DIR) $(PYTEST_BASE_OPTS) -m "" --ignore=tests/benchmarks

test-all-cov: ## Run ALL tests with coverage report (excludes slow benchmarks)
	@echo "$(GREEN)Running all tests with full coverage (excluding slow benchmarks)...$(NC)"
	$(PYTEST) $(TEST_DIR) \
		--cov=brain_go_brrr \
		--cov-report=term-missing \
		--cov-report= \
		--no-cov-on-fail \
		-m "not benchmark" \
		--ignore=tests/benchmarks
	@echo "$(CYAN)Generating HTML coverage report...$(NC)"
	@$(UV) run coverage html
	@echo "$(GREEN)Coverage report generated at: htmlcov/index.html$(NC)"

test-benchmarks: ## Run benchmark tests WITHOUT coverage (fast)
	@echo "$(YELLOW)Running benchmark tests without coverage...$(NC)"
	$(PYTEST) tests/benchmarks -m "benchmark or slow" --benchmark-only -v

test-benchmarks-strict: ## Run benchmarks with strict CI thresholds
	@echo "$(RED)Running benchmarks with STRICT CI thresholds...$(NC)"
	CI_BENCHMARKS=1 $(PYTEST) tests/benchmarks -m "benchmark or slow" --benchmark-only -v

test-watch: ## Run tests in watch mode
	@echo "$(GREEN)Running tests in watch mode...$(NC)"
	$(PYTEST) $(TEST_DIR) -f -m "not slow"

benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running benchmarks...$(NC)"
	$(PYTEST) $(TEST_DIR) --benchmark-only

##@ Machine Learning

train: ## Train EEGPT model
	@echo "$(GREEN)Starting model training...$(NC)"
	$(PYTHON) -m brain_go_brrr.cli train

preprocess: ## Preprocess EEG data
	@echo "$(GREEN)Preprocessing EEG data...$(NC)"
	$(PYTHON) -m brain_go_brrr.cli preprocess

evaluate: ## Evaluate trained model
	@echo "$(GREEN)Evaluating model...$(NC)"
	$(PYTHON) -m brain_go_brrr.cli evaluate

serve: ## Serve model via API
	@echo "$(GREEN)Starting model server...$(NC)"
	$(PYTHON) -m brain_go_brrr.cli serve

##@ MLOps

mlflow: ## Start MLflow tracking server
	@echo "$(GREEN)Starting MLflow server...$(NC)"
	$(UV) run mlflow server --host 0.0.0.0 --port 5000

dvc-setup: ## Initialize DVC for data versioning
	@echo "$(GREEN)Setting up DVC...$(NC)"
	$(UV) run dvc init
	@echo "$(GREEN)DVC initialized!$(NC)"

dvc-status: ## Check DVC status
	@echo "$(GREEN)Checking DVC status...$(NC)"
	$(UV) run dvc status

dvc-push: ## Push data to remote storage
	@echo "$(GREEN)Pushing data to remote storage...$(NC)"
	$(UV) run dvc push

dvc-pull: ## Pull data from remote storage
	@echo "$(GREEN)Pulling data from remote storage...$(NC)"
	$(UV) run dvc pull

##@ Documentation

docs: ## Build documentation
	@echo "$(GREEN)Building documentation...$(NC)"
	$(UV) run mkdocs build

docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Starting documentation server...$(NC)"
	$(UV) run mkdocs serve

docs-deploy: ## Deploy documentation to GitHub Pages
	@echo "$(GREEN)Deploying documentation...$(NC)"
	$(UV) run mkdocs gh-deploy

##@ Building & Packaging

build: ## Build package
	@echo "$(GREEN)Building package...$(NC)"
	$(UV) build

publish: ## Publish package to PyPI
	@echo "$(GREEN)Publishing package...$(NC)"
	$(UV) publish

##@ Docker

docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -t $(PROJECT_NAME):latest .

docker-run: ## Run Docker container
	@echo "$(GREEN)Running Docker container...$(NC)"
	docker run -it --rm -p 8000:8000 $(PROJECT_NAME):latest

docker-dev: ## Run Docker container in development mode
	@echo "$(GREEN)Running Docker container in dev mode...$(NC)"
	docker run -it --rm -v $(PWD):/app -p 8000:8000 $(PROJECT_NAME):latest

##@ Profiling & Debugging

profile: ## Run profiling on training
	@echo "$(GREEN)Running profiling...$(NC)"
	$(PYTHON) -m cProfile -o profile_output.prof -m brain_go_brrr.cli train
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile_output.prof'); p.sort_stats('cumulative').print_stats(20)"

debug: ## Run with debugger
	@echo "$(GREEN)Starting debug session...$(NC)"
	$(PYTHON) -m pdb -m brain_go_brrr.cli train

##@ Utilities

clean: ## Clean build artifacts and cache
	@echo "$(GREEN)Cleaning build artifacts...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .tox/
	@echo "$(GREEN)Cleanup complete!$(NC)"

reset: clean ## Reset environment completely
	@echo "$(GREEN)Resetting environment...$(NC)"
	rm -rf .venv/
	$(UV) sync

version: ## Show version information
	@echo "$(GREEN)Version Information:$(NC)"
	@$(PYTHON) -m brain_go_brrr.cli version
	@echo "Python: $(shell python --version)"
	@echo "UV: $(shell uv --version)"

env-info: ## Show environment information
	@echo "$(GREEN)Environment Information:$(NC)"
	@echo "Python: $(shell python --version)"
	@echo "UV: $(shell uv --version)"
	@echo "Platform: $(shell uname -s)"
	@echo "Architecture: $(shell uname -m)"
	@$(PYTHON) -c "import sys; print(f'Python Path: {sys.executable}')"

##@ Git Hooks

pre-push-old: test quality ## Old pre-push target (deprecated)
	@echo "$(GREEN)Running pre-push checks...$(NC)"
	@echo "$(GREEN)All pre-push checks passed!$(NC)"

ci: ## Run CI pipeline locally
	@echo "$(GREEN)Running CI pipeline...$(NC)"
	$(MAKE) clean
	$(MAKE) dev-install
	$(MAKE) check
	$(MAKE) build
	@echo "$(GREEN)CI pipeline completed successfully!$(NC)"

check-all: ## Run all quality checks (for CI/CD)
	@echo "$(GREEN)Running all quality checks...$(NC)"
	$(MAKE) format
	$(MAKE) lint-ci
	$(MAKE) type-fast
	$(MAKE) test-ci
	@echo "$(GREEN)All checks passed!$(NC)"

pre-push: ## Run before pushing to ensure CI will pass
	@echo "$(CYAN)Running pre-push checks...$(NC)"
	$(MAKE) format
	@echo "$(YELLOW)Checking for uncommitted formatting changes...$(NC)"
	@git diff --exit-code || (echo "$(RED)Error: Formatting changes detected. Please commit them.$(NC)" && exit 1)
	$(MAKE) lint-ci
	$(MAKE) type-fast
	@echo "$(GREEN)Ready to push! CI should pass.$(NC)"

##@ Examples

example-train: ## Run training example
	@echo "$(GREEN)Running training example...$(NC)"
	$(PYTHON) -m brain_go_brrr.cli train --debug

example-preprocess: ## Run preprocessing example
	@echo "$(GREEN)Running preprocessing example...$(NC)"
	$(PYTHON) -m brain_go_brrr.cli preprocess data/raw data/processed --debug

# Help target should be first
.DEFAULT_GOAL := help
