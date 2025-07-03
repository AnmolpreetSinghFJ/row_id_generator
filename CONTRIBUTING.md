# Contributing to Row ID Generator

Thank you for your interest in contributing to Row ID Generator! We welcome contributions from the community and appreciate your help in making this project better.

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [support@example.com](mailto:support@example.com).

## How to Contribute

### Reporting Issues

- **Bug Reports**: Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
- **Feature Requests**: Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md)
- **Questions**: Use [GitHub Discussions](https://github.com/alakob/row_id_generator/discussions)

Before creating an issue, please:
1. Search existing issues to avoid duplicates
2. Include as much detail as possible
3. Provide reproducible examples when reporting bugs

### Development Setup

#### Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) for fast Python package management. Install uv first:

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip (if you already have pip)
pip install uv
```

#### Setup Steps

1. **Fork and Clone**
   ```bash
   git clone https://github.com/alakob/row_id_generator.git
   cd row_id_generator
   ```

2. **Create Virtual Environment**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   uv pip install -e ".[dev]"
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

### Making Changes

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Follow Coding Standards**
   - Use [Black](https://black.readthedocs.io/) for code formatting
   - Follow [PEP 8](https://pep8.org/) style guidelines
   - Add type hints to all functions
   - Write comprehensive docstrings

3. **Code Quality Checks**
   ```bash
   # Format code
   black .
   isort .
   
   # Lint code
   flake8 .
   
   # Type checking
   mypy .
   
   # Run all checks
   pre-commit run --all-files
   ```

### Testing

1. **Write Tests**
   - Add unit tests for new functionality
   - Add integration tests for complex features
   - Maintain or improve code coverage
   - Use descriptive test names

2. **Run Tests**
   ```bash
   # Run all tests
   pytest
   
   # Run with coverage
   pytest --cov=row_id_generator --cov-report=html
   
   # Run specific test categories
   pytest -m unit
   pytest -m integration
   pytest -m performance
   ```

3. **Test Categories**
   - `@pytest.mark.unit` - Fast, isolated unit tests
   - `@pytest.mark.integration` - Integration tests with external systems
   - `@pytest.mark.performance` - Performance and benchmark tests
   - `@pytest.mark.slow` - Long-running tests

### Documentation

1. **Update Documentation**
   - Update docstrings for new/changed functions
   - Update README.md if needed
   - Add examples for new features
   - Update CHANGELOG.md

2. **Build Documentation Locally**
   ```bash
   cd docs
   make html
   # Open docs/_build/html/index.html
   ```

### Pull Request Process

1. **Before Creating PR**
   - Ensure all tests pass
   - Update documentation
   - Add changelog entry
   - Rebase on latest main branch

2. **Creating the PR**
   - Use descriptive title and description
   - Reference related issues
   - Include screenshots/examples if applicable
   - Ensure CI checks pass

3. **PR Review Process**
   - Maintainers will review your code
   - Address feedback promptly
   - Keep discussions constructive
   - Be patient - reviews take time

### Commit Message Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

**Examples:**
```
feat(core): add hash collision detection
fix(utils): handle null values in column selection
docs(readme): update installation instructions
test(integration): add snowflake connection tests
```

## Development Guidelines

### Code Style

- **Line Length**: 88 characters (Black default)
- **Imports**: Use `isort` for import organization
- **Type Hints**: Required for all public functions
- **Docstrings**: Use Google-style docstrings

### Architecture Principles

- **Modularity**: Keep functions focused and single-purpose
- **Error Handling**: Use appropriate exceptions with clear messages
- **Performance**: Consider memory usage and processing time
- **Observability**: Add logging and metrics for important operations
- **Testing**: Write testable code with clear dependencies

### Performance Considerations

- Use vectorized pandas operations where possible
- Consider memory usage for large DataFrames
- Add progress bars for long-running operations
- Profile code for performance bottlenecks

### Security Guidelines

- Never commit credentials or sensitive data
- Validate all user inputs
- Use secure defaults
- Follow security best practices for data handling

## Development Workflow with uv

### Adding New Dependencies

```bash
# Add a new production dependency
uv add package-name

# Add a new development dependency
uv add --dev package-name

# Add with version constraints
uv add "package-name>=1.0.0,<2.0.0"
```

### Virtual Environment Management

```bash
# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install from lockfile (if exists)
uv pip sync requirements.lock

# Update all dependencies
uv pip compile requirements.in --upgrade
```

### Performance Benefits

Using `uv` provides significant performance improvements:
- **10-100x faster** than pip for dependency resolution
- **Faster installs** with better caching
- **Better dependency resolution** with conflict detection
- **Consistent environments** with lockfile support

## Release Process

1. **Version Bumping**
   - Follow [Semantic Versioning](https://semver.org/)
   - Update version in `row_id_generator/__init__.py`
   - Update CHANGELOG.md

2. **Release Checklist**
   - [ ] All tests pass
   - [ ] Documentation updated
   - [ ] Changelog updated
   - [ ] Version bumped
   - [ ] Tag created
   - [ ] PyPI release

## Getting Help

- 💬 [GitHub Discussions](https://github.com/alakob/row_id_generator/discussions)
- 📖 [Documentation](https://github.com/alakob/row_id_generator/blob/main/docs/)
- 🐛 [Issues](https://github.com/alakob/row_id_generator/issues)

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- GitHub contributor graphs

Thank you for contributing to Row ID Generator! 🎉 