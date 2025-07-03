# Comprehensive Test Automation Makefile
# Task 9.7: Automated test execution pipeline

.PHONY: help test test-unit test-integration test-performance test-edge test-snowflake
.PHONY: test-parallel test-sequential test-coverage test-benchmarks
.PHONY: coverage coverage-html coverage-xml coverage-json
.PHONY: clean clean-cache clean-results setup install-deps
.PHONY: pre-commit install-hooks lint format check
.PHONY: ci-test ci-coverage ci-benchmarks

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
TEST_RUNNER := scripts/run_tests.py

help: ## Show this help message
	@echo "🎯 Comprehensive Test Automation Commands"
	@echo "=========================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Environment Setup
setup: ## Set up development environment
	@echo "🔧 Setting up development environment..."
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install pytest pytest-cov pytest-xdist pytest-html pytest-json-report
	$(PIP) install coverage[toml] pytest-mock pre-commit
	$(PIP) install pandas numpy
	@echo "✅ Development environment ready!"

install-deps: ## Install all dependencies
	@echo "📦 Installing dependencies..."
	$(PIP) install -r requirements.txt || echo "No requirements.txt found"
	$(PIP) install pytest pytest-cov coverage pandas numpy
	@echo "✅ Dependencies installed!"

# Test Execution Commands
test: ## Run all test suites with parallel execution
	@echo "🚀 Running all test suites..."
	$(PYTHON) $(TEST_RUNNER) --suite all

test-unit: ## Run unit tests only
	@echo "🔬 Running unit tests..."
	$(PYTHON) $(TEST_RUNNER) --suite unit

test-integration: ## Run integration tests only
	@echo "🔗 Running integration tests..."
	$(PYTHON) $(TEST_RUNNER) --suite integration

test-performance: ## Run performance tests only
	@echo "⚡ Running performance tests..."
	$(PYTHON) $(TEST_RUNNER) --suite performance

test-edge: ## Run edge case tests only
	@echo "🎯 Running edge case tests..."
	$(PYTHON) $(TEST_RUNNER) --suite edge

test-snowflake: ## Run Snowflake integration tests only
	@echo "❄️  Running Snowflake integration tests..."
	$(PYTHON) $(TEST_RUNNER) --suite snowflake

test-parallel: ## Run all tests with maximum parallelization
	@echo "⚡ Running tests in parallel mode..."
	$(PYTHON) $(TEST_RUNNER) --suite all

test-sequential: ## Run all tests sequentially
	@echo "📝 Running tests sequentially..."
	$(PYTHON) $(TEST_RUNNER) --suite all --no-parallel

test-fast-fail: ## Run tests with fast fail on first failure
	@echo "💥 Running tests with fast fail..."
	$(PYTHON) $(TEST_RUNNER) --suite all --fast-fail

# Coverage Commands
coverage: ## Run comprehensive coverage analysis
	@echo "📊 Running coverage analysis..."
	$(PYTHON) $(TEST_RUNNER) --coverage-only

coverage-html: ## Generate HTML coverage report
	@echo "📄 Generating HTML coverage report..."
	$(PYTHON) -m pytest tests/ --cov=row_id_generator --cov-report=html:coverage_reports/html
	@echo "📁 Open coverage_reports/html/index.html to view report"

coverage-xml: ## Generate XML coverage report
	@echo "📋 Generating XML coverage report..."
	$(PYTHON) -m pytest tests/ --cov=row_id_generator --cov-report=xml:coverage_reports/coverage.xml

coverage-json: ## Generate JSON coverage report
	@echo "🔢 Generating JSON coverage report..."
	$(PYTHON) -m pytest tests/ --cov=row_id_generator --cov-report=json:coverage_reports/coverage.json

test-coverage: ## Run tests with coverage reporting
	@echo "🔍 Running tests with coverage..."
	$(PYTHON) -m pytest tests/ --cov=row_id_generator --cov-report=term-missing --cov-report=html:coverage_reports/html

# Performance and Benchmarks
test-benchmarks: ## Run performance benchmarks
	@echo "🏁 Running performance benchmarks..."
	$(PYTHON) $(TEST_RUNNER) --benchmarks-only

benchmark-detailed: ## Run detailed performance analysis
	@echo "📈 Running detailed performance analysis..."
	$(PYTHON) -m pytest tests/test_performance.py -v --tb=short

# Code Quality Commands
lint: ## Run code linting
	@echo "🧹 Running code linting..."
	flake8 row_id_generator/ tests/ --max-line-length=88 --extend-ignore=E203,W503 || echo "Install flake8 to run linting"

format: ## Format code with black and isort
	@echo "✨ Formatting code..."
	black row_id_generator/ tests/ --line-length=88 || echo "Install black to format code"
	isort row_id_generator/ tests/ --profile black || echo "Install isort to sort imports"

check: ## Run all code quality checks
	@echo "✅ Running code quality checks..."
	$(MAKE) lint
	$(MAKE) format

# Pre-commit Hooks
install-hooks: ## Install pre-commit hooks
	@echo "🪝 Installing pre-commit hooks..."
	pre-commit install || echo "Install pre-commit to use hooks"

pre-commit: ## Run pre-commit hooks manually
	@echo "🔍 Running pre-commit hooks..."
	pre-commit run --all-files || echo "Install pre-commit to use hooks"

# CI/CD Commands
ci-test: ## Run tests in CI/CD mode (no parallel, with coverage)
	@echo "🤖 Running CI/CD test suite..."
	$(PYTHON) $(TEST_RUNNER) --suite all --no-parallel

ci-coverage: ## Run coverage analysis for CI/CD
	@echo "📊 Running CI/CD coverage analysis..."
	$(PYTHON) -m pytest tests/ --cov=row_id_generator --cov-report=xml:coverage.xml --cov-report=term

ci-benchmarks: ## Run performance benchmarks for CI/CD
	@echo "⚡ Running CI/CD performance benchmarks..."
	$(PYTHON) -m pytest tests/test_performance.py::TestBenchmarkBaselines --junit-xml=benchmark-results.xml

ci-all: ## Run complete CI/CD pipeline
	@echo "🚀 Running complete CI/CD pipeline..."
	$(MAKE) ci-test
	$(MAKE) ci-coverage
	$(MAKE) ci-benchmarks

# Cleanup Commands
clean: ## Clean up test artifacts and cache files
	@echo "🧹 Cleaning up..."
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf coverage_reports/
	rm -rf test_results/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "✅ Cleanup completed!"

clean-cache: ## Clean only cache files
	@echo "🗑️  Cleaning cache files..."
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-results: ## Clean test results and coverage reports
	@echo "📁 Cleaning test results..."
	$(PYTHON) $(TEST_RUNNER) --cleanup 7
	rm -rf coverage_reports/
	rm -rf test_results/

# Quick Commands
quick: ## Run quick test suite for development
	@echo "⚡ Running quick test suite..."
	$(PYTHON) tests/test_runner.py --quick

quick-unit: ## Run quick unit tests
	@echo "🔬 Running quick unit tests..."
	$(PYTHON) -m pytest tests/test_core.py::TestBasicHashingFunctionality -v

quick-integration: ## Run quick integration tests
	@echo "🔗 Running quick integration tests..."
	$(PYTHON) -m pytest tests/test_integration_simple.py::TestSimpleIntegrationWorkflows::test_simple_workflow -v

# Verification Commands
verify: ## Verify test environment and dependencies
	@echo "🔍 Verifying test environment..."
	@$(PYTHON) --version
	@$(PYTHON) -c "import pytest; print(f'pytest: {pytest.__version__}')" || echo "❌ pytest not installed"
	@$(PYTHON) -c "import pandas; print(f'pandas: {pandas.__version__}')" || echo "❌ pandas not installed"
	@$(PYTHON) -c "import numpy; print(f'numpy: {numpy.__version__}')" || echo "❌ numpy not installed"
	@echo "📁 Test files:"
	@find tests/ -name "test_*.py" -type f | wc -l | xargs echo "   Found test files:"
	@echo "✅ Environment verification completed!"

# Documentation Commands
test-docs: ## Generate test documentation
	@echo "📚 Generating test documentation..."
	@echo "# Test Suite Documentation" > TEST_DOCUMENTATION.md
	@echo "" >> TEST_DOCUMENTATION.md
	@echo "## Available Test Suites" >> TEST_DOCUMENTATION.md
	@echo "- **unit**: Core functionality tests" >> TEST_DOCUMENTATION.md
	@echo "- **integration**: End-to-end workflow tests" >> TEST_DOCUMENTATION.md
	@echo "- **performance**: Load and performance tests" >> TEST_DOCUMENTATION.md
	@echo "- **edge**: Edge case and boundary tests" >> TEST_DOCUMENTATION.md
	@echo "- **snowflake**: Snowflake integration tests" >> TEST_DOCUMENTATION.md
	@echo "" >> TEST_DOCUMENTATION.md
	@echo "## Usage Examples" >> TEST_DOCUMENTATION.md
	@echo "\`\`\`bash" >> TEST_DOCUMENTATION.md
	@echo "make test          # Run all tests" >> TEST_DOCUMENTATION.md
	@echo "make test-unit     # Run unit tests only" >> TEST_DOCUMENTATION.md
	@echo "make coverage      # Generate coverage report" >> TEST_DOCUMENTATION.md
	@echo "\`\`\`" >> TEST_DOCUMENTATION.md
	@echo "📄 Documentation generated: TEST_DOCUMENTATION.md"

# Status Commands
status: ## Show current test environment status
	@echo "📊 Test Environment Status"
	@echo "=========================="
	@echo "🐍 Python: $(shell $(PYTHON) --version)"
	@echo "📦 Test files: $(shell find tests/ -name 'test_*.py' | wc -l)"
	@echo "📁 Test results: $(shell find test_results/ -name '*.xml' 2>/dev/null | wc -l) XML files" || echo "📁 Test results: 0 XML files"
	@echo "📊 Coverage reports: $(shell find coverage_reports/ -name '*.xml' 2>/dev/null | wc -l) XML files" || echo "📊 Coverage reports: 0 XML files"
	@echo "🔧 Dependencies:"
	@$(PYTHON) -c "import pytest; print(f'  ✅ pytest: {pytest.__version__}')" 2>/dev/null || echo "  ❌ pytest: not installed"
	@$(PYTHON) -c "import coverage; print(f'  ✅ coverage: {coverage.__version__}')" 2>/dev/null || echo "  ❌ coverage: not installed"
	@$(PYTHON) -c "import pandas; print(f'  ✅ pandas: {pandas.__version__}')" 2>/dev/null || echo "  ❌ pandas: not installed" 