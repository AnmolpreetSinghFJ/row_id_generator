# Row ID Generator

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/alakob/row_id_generator/workflows/Tests/badge.svg)](https://github.com/alakob/row_id_generator/actions)
[![Coverage](https://codecov.io/gh/alakob/row_id_generator/branch/main/graph/badge.svg)](https://codecov.io/gh/alakob/row_id_generator)

A Python package for generating unique, stable row IDs for Pandas DataFrames before loading into Snowflake databases. Features intelligent column selection, deterministic SHA-256 hashing, and comprehensive observability for production data pipelines.

## 🚀 Features

- **🔍 Intelligent Column Selection**: Automatically selects optimal columns based on uniqueness and completeness
- **🔒 Deterministic Hashing**: SHA-256 based row IDs that are consistent across runs
- **❄️ Snowflake Integration**: Seamless integration with Snowflake data loading workflows
- **⚡ High Performance**: Vectorized operations with memory-efficient batch processing
- **📊 Full Observability**: Built-in monitoring, metrics, alerting, and dashboards
- **🛠️ Production Ready**: Comprehensive error handling, validation, and recovery mechanisms
- **📈 Scalable**: Handles DataFrames from thousands to millions of rows
- **🔧 Flexible**: Extensive configuration options and multiple API levels

## 📦 Installation

### Quick Install from PyPI

```bash
uv add row-id-generator
```

### Alternative with pip

```bash
pip install row-id-generator
```

### Install from GitHub (Latest Development Version)

```bash
# Using uv (recommended)
uv pip install git+https://github.com/alakob/row_id_generator.git

# Using pip
pip install git+https://github.com/alakob/row_id_generator.git
```

### Install specific GitHub branch or tag

```bash
# Install from specific branch
uv pip install git+https://github.com/alakob/row_id_generator.git@main

# Install from specific tag
uv pip install git+https://github.com/alakob/row_id_generator.git@v1.0.0

# Using pip
pip install git+https://github.com/alakob/row_id_generator.git@main
```

### Development Installation

```bash
# Clone and install for development
git clone https://github.com/alakob/row_id_generator.git
cd row_id_generator
uv sync --dev
```

Or with pip:
```bash
git clone https://github.com/alakob/row_id_generator.git
cd row_id_generator
pip install -e ".[dev]"
```

### With Optional Dependencies

```bash
# Install with Snowflake integration
uv add "row-id-generator[snowflake]"

# Install with observability features
uv add "row-id-generator[observability]"

# Install with all optional dependencies
uv add "row-id-generator[all]"

# From GitHub with extras
uv pip install "git+https://github.com/alakob/row_id_generator.git[all]"
```

## 🔥 Quick Start

### Basic Usage

```python
import pandas as pd
from row_id_generator import generate_unique_row_ids

# Create sample data
df = pd.DataFrame({
    'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com'],
    'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown'],
    'age': [28, 34, 29],
    'department': ['Engineering', 'Marketing', 'Engineering']
})

# Generate row IDs with intelligent defaults
result_df = generate_unique_row_ids(df)

print(result_df)
#                email           name  age  department                                             row_id
# 0   alice@example.com  Alice Johnson   28  Engineering  a1b2c3d4e5f6...  (64-character SHA-256 hash)
# 1     bob@example.com      Bob Smith   34    Marketing  f6e5d4c3b2a1...
# 2  charlie@example.com Charlie Brown   29  Engineering  9f8e7d6c5b4a...
```

### Advanced Configuration

```python
from row_id_generator import generate_unique_row_ids

# Custom configuration with detailed control
result_df = generate_unique_row_ids(
    df=df,
    columns=['email', 'name'],           # Manually specify columns
    id_column_name='custom_row_id',      # Custom ID column name
    uniqueness_threshold=0.98,           # Higher uniqueness requirement
    separator='::',                      # Custom value separator
    show_progress=True,                  # Progress bar for large datasets
    enable_monitoring=True,              # Detailed logging and metrics
    enable_quality_checks=True,          # Data quality validation
    return_audit_trail=True              # Return detailed processing info
)

# Access comprehensive results
if isinstance(result_df, dict):
    df_with_ids = result_df['result_dataframe']
    audit_trail = result_df['audit_trail']
    column_selection = result_df['column_selection']
    
    print('\nDataFrame with IDs:')
    print(df_with_ids)
    
    print(f'\nSelected columns: {column_selection["selected_columns"]}')
    print(f'Column selection quality: {column_selection["overall_quality_score"]:.4f}')
    print(f'Selection method: {column_selection["selection_method"]}')
    print(f'Session ID: {result_df["session_id"]}')
```

## 🎯 API Reference

### Core Functions

#### `generate_unique_row_ids()`

The main entry point for row ID generation with comprehensive options.

```python
def generate_unique_row_ids(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    id_column_name: str = 'row_id',
    uniqueness_threshold: float = 0.95,
    separator: str = '|',
    enable_monitoring: bool = True,
    enable_quality_checks: bool = True,
    show_progress: bool = True,
    show_warnings: bool = True,
    enable_enhanced_lineage: bool = False,
    return_audit_trail: bool = False,
    **kwargs
) -> Union[pd.DataFrame, Dict[str, Any]]
```

**Parameters:**
- `df`: Input pandas DataFrame
- `columns`: Optional list of columns to use (auto-selected if None)
- `id_column_name`: Name for the generated ID column
- `uniqueness_threshold`: Minimum uniqueness ratio for column selection (0.0-1.0)
- `separator`: String separator for concatenating values
- `enable_monitoring`: Enable comprehensive observability features
- `enable_quality_checks`: Enable data quality validation
- `show_progress`: Display progress bars for large datasets
- `return_audit_trail`: Return detailed processing information

#### Performance Variants

```python
from row_id_generator import generate_row_ids_simple, generate_row_ids_fast

# Minimal overhead for simple use cases
df_simple = generate_row_ids_simple(df)

# Maximum performance for large datasets
df_fast = generate_row_ids_fast(df, columns=['email'])
```

### Utility Functions

#### Column Selection

```python
from row_id_generator import select_columns_for_hashing

# Intelligent column selection
selected_columns = select_columns_for_hashing(
    df=df,
    manual_columns=None,
    uniqueness_threshold=0.95,
    include_email=True
)
```

#### Data Preprocessing

```python
from row_id_generator import (
    prepare_data_for_hashing,
    normalize_string_data,
    handle_null_values,
    standardize_datetime,
    normalize_numeric_data
)

# Prepare data for consistent hashing
processed_df = prepare_data_for_hashing(df, columns=['email', 'name'])

# Individual preprocessing functions
clean_strings = normalize_string_data(df['name'], case_conversion='lower')
handled_nulls = handle_null_values(df['description'], strategy='replace')
standard_dates = standardize_datetime(df['created_at'], target_timezone='UTC')
normal_numbers = normalize_numeric_data(df['amount'], precision=2)
```

### Observable API (Production Monitoring)

For production environments requiring comprehensive observability:

```python
from row_id_generator import (
    ObservableHashingEngine,
    create_observable_engine,
    create_minimal_observable_engine,
    create_full_observable_engine
)

# Create observable engine with full monitoring
engine = create_full_observable_engine('config/observability.yaml')

# Generate row IDs with full observability
result_df, selected_columns, audit_trail = engine.generate_unique_row_ids(
    df=df,
    show_progress=True
)

# Get comprehensive system health report
health_report = engine.get_system_health_report()
print(f"CPU Usage: {health_report['system_resources']['cpu_percent']:.1f}%")
print(f"Active Alerts: {len(health_report['active_alerts'])}")

# Export metrics for external monitoring systems
prometheus_metrics = engine.export_metrics(format_type="prometheus")
json_metrics = engine.export_metrics(format_type="json")

# Generate performance dashboard
dashboard_html = engine.generate_performance_dashboard()
```

## 🔗 Snowflake Integration

### Complete Workflow

```python
import pandas as pd
from row_id_generator import generate_unique_row_ids
from row_id_generator.core import prepare_for_snowflake, load_to_snowflake

# 1. Generate row IDs
df_with_ids = generate_unique_row_ids(df)

# 2. Prepare for Snowflake (handle special characters, etc.)
snowflake_ready_df = prepare_for_snowflake(df_with_ids, 'users_table')

# 3. Load to Snowflake
connection_params = {
    'account': 'your_account.region',
    'user': 'your_username',
    'password': 'your_password',
    'warehouse': 'your_warehouse',
    'database': 'your_database',
    'schema': 'your_schema'
}

success, rows_loaded = load_to_snowflake(
    df_with_ids=snowflake_ready_df,
    connection_params=connection_params,
    table_name='users_table',
    if_exists='append'
)

print(f"Successfully loaded {rows_loaded} rows: {success}")
```

## ⚡ Performance & Scalability

### Memory-Efficient Processing

```python
from row_id_generator import create_optimized_row_id_function

# Create optimized function for large datasets
optimized_generator = create_optimized_row_id_function(
    max_memory_mb=500,      # Memory limit
    enable_chunking=True,   # Process in chunks
    enable_streaming=True   # Stream concatenation
)

# Process large DataFrame efficiently
large_df = pd.read_csv('large_dataset.csv')  # 10M+ rows
result_df = optimized_generator(
    df=large_df,
    chunk_size=50000
)
```

### Performance Benchmarks

| DataFrame Size | Processing Time | Memory Usage | Throughput    |
|---------------|----------------|--------------|---------------|
| 1K rows       | 0.01s          | 10 MB        | 100K rows/s   |
| 10K rows      | 0.08s          | 25 MB        | 125K rows/s   |
| 100K rows     | 0.6s           | 150 MB       | 167K rows/s   |
| 1M rows       | 5.2s           | 800 MB       | 192K rows/s   |
| 10M rows      | 58s            | 6 GB         | 172K rows/s   |

## 🔧 Configuration

### Environment Variables

Set these in your `.env` file or environment:

```bash
# Logging
ROWID_LOG_LEVEL=INFO
ROWID_LOG_FORMAT=structured

# Performance
ROWID_DEFAULT_BATCH_SIZE=10000
ROWID_MAX_MEMORY_MB=1000
ROWID_ENABLE_PARALLEL=true

# Monitoring
ROWID_ENABLE_METRICS=true
ROWID_METRICS_RETENTION_HOURS=168  # 7 days
```

### Configuration File

Create `config.yaml` for advanced settings:

```yaml
hashing:
  algorithm: sha256
  separator: "|"
  encoding: utf-8

column_selection:
  uniqueness_threshold: 0.95
  include_email: true
  max_columns: 10

performance:
  batch_size: 10000
  max_memory_mb: 1000
  enable_parallel: true
  chunk_size_auto: true

monitoring:
  enable_detailed_logging: true
  metrics_retention_hours: 168
  enable_alerts: true
  dashboard_auto_refresh: 30

quality_checks:
  enable_validation: true
  max_null_ratio: 0.1
  min_uniqueness: 0.8
```

## 🧪 Testing & Validation

### Built-in Validation

```python
from row_id_generator.utils import (
    validate_dataframe_input,
    analyze_dataframe_quality,
    get_column_quality_score
)

# Validate input DataFrame
try:
    validate_dataframe_input(df)
    print("✅ DataFrame validation passed")
except ValueError as e:
    print(f"❌ Validation failed: {e}")

# Analyze data quality
quality_metrics = analyze_dataframe_quality(df)
print(f"Overall quality score: {quality_metrics.get_summary_report()['overall_score']}")

# Check individual column quality
for column in df.columns:
    score = get_column_quality_score(df, column)
    print(f"{column}: {score['score']:.2f} ({score['grade']})")
```

### Error Handling

```python
from row_id_generator.core import (
    RowIDGenerationError,
    DataValidationError,
    HashGenerationError
)

try:
    result_df = generate_unique_row_ids(problematic_df)
except DataValidationError as e:
    print(f"Data validation failed: {e.message}")
    print(f"Context: {e.context}")
    print(f"Suggestions: {e.suggestions}")
except HashGenerationError as e:
    print(f"Hash generation failed: {e.message}")
    # Implement retry logic or fallback
except RowIDGenerationError as e:
    print(f"General error: {e.message}")
    # Log error and notify monitoring systems
```

## 📚 Examples

### Real-World Use Cases

#### Customer Data Processing

```python
# Customer data with email deduplication
customer_df = pd.DataFrame({
    'email': ['alice@corp.com', 'bob@startup.io', 'charlie@agency.co'],
    'first_name': ['Alice', 'Bob', 'Charlie'],
    'last_name': ['Johnson', 'Smith', 'Brown'],
    'company': ['Corp Inc', 'Startup IO', 'Agency Co'],
    'signup_date': pd.to_datetime(['2024-01-15', '2024-01-16', '2024-01-17'])
})

# Generate stable customer IDs (auto-selects email as primary identifier)
customer_df_with_ids = generate_unique_row_ids(
    customer_df,
    id_column_name='customer_id'
)
```

#### Transaction Data Processing

```python
# Financial transaction data
transaction_df = pd.DataFrame({
    'transaction_id': ['T001', 'T002', 'T003'],
    'user_email': ['alice@corp.com', 'bob@startup.io', 'alice@corp.com'],
    'amount': [99.99, 149.50, 79.99],
    'currency': ['USD', 'USD', 'EUR'],
    'timestamp': pd.to_datetime(['2024-01-15 10:30', '2024-01-15 11:15', '2024-01-15 14:22'])
})

# Generate row IDs for transaction records
transaction_df_with_ids = generate_unique_row_ids(
    transaction_df,
    columns=['transaction_id', 'user_email', 'timestamp'],
    uniqueness_threshold=0.99  # High uniqueness for financial data
)
```

#### Product Catalog Processing

```python
# Product catalog with hierarchical data
product_df = pd.DataFrame({
    'sku': ['LAPTOP-001', 'PHONE-002', 'TABLET-003'],
    'name': ['Business Laptop', 'Smartphone Pro', 'Tablet Air'],
    'category': ['Electronics', 'Electronics', 'Electronics'],
    'subcategory': ['Computers', 'Mobile', 'Computers'],
    'price': [999.99, 699.99, 399.99],
    'brand': ['TechCorp', 'PhoneCo', 'TechCorp']
})

# Generate product IDs with custom separator
product_df_with_ids = generate_unique_row_ids(
    product_df,
    columns=['sku', 'brand'],
    separator='::',
    id_column_name='product_hash'
)
```

## 🔍 Troubleshooting

### Common Issues

#### Low Uniqueness Warning

```python
# If you get uniqueness warnings, try:
result_df = generate_unique_row_ids(
    df,
    uniqueness_threshold=0.8,  # Lower threshold
    columns=['email', 'name', 'phone']  # Add more columns
)
```

#### Memory Issues with Large DataFrames

```python
# For memory-constrained environments:
optimized_generator = create_optimized_row_id_function(
    max_memory_mb=200,
    enable_chunking=True,
    enable_streaming=True
)

result_df = optimized_generator(
    df=large_df,
    chunk_size=1000  # Smaller chunks
)
```

#### Performance Optimization

```python
# For maximum speed:
result_df = generate_row_ids_fast(
    df,
    columns=['primary_key']  # Use fewer, high-quality columns
)
```

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/alakob/row_id_generator.git
cd row_id_generator

# Install development dependencies with uv (recommended)
uv sync --dev

# Alternative with pip
pip install -e ".[dev]"

# Run tests
uv run pytest

# Run quality checks
uv run black .
uv run flake8 .
uv run mypy .

# Generate coverage report
uv run pytest --cov=row_id_generator --cov-report=html
```

### Running Tests

```bash
# Full test suite
uv run pytest

# Specific test categories
uv run pytest tests/test_core.py              # Core functionality
uv run pytest tests/test_utils.py             # Utility functions
uv run pytest tests/test_observability.py     # Monitoring features
uv run pytest tests/test_integration.py       # Integration tests

# Performance tests
uv run pytest tests/test_performance.py -v

# Coverage with detailed report
uv run pytest --cov=row_id_generator --cov-report=term-missing
```

## 📖 Documentation

- **[API Reference](docs/api.md)** - Complete API documentation
- **[User Guide](docs/guide.md)** - Comprehensive usage guide
- **[Performance Guide](docs/performance.md)** - Optimization techniques
- **[Observability Guide](docs/observability.md)** - Monitoring and alerting
- **[Examples](examples/)** - Real-world usage examples
- **[Changelog](CHANGELOG.md)** - Version history
- **[Migration Guide](docs/migration.md)** - Upgrading between versions

## 📋 Requirements

- **Python**: 3.9+ (3.11+ recommended for best performance)
- **pandas**: >= 2.2.0
- **numpy**: >= 1.24.0
- **psutil**: >= 5.8.0 (for system monitoring)
- **tqdm**: >= 4.65.0 (for progress bars)
- **snowflake-connector-python**: >= 3.0.0 (for Snowflake integration)

### Optional Dependencies

```bash
# For observability features
uv add "row-id-generator[observability]"

# For development
uv add "row-id-generator[dev]"

# For performance testing
uv add "row-id-generator[performance]"

# All optional dependencies
uv add "row-id-generator[all]"
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **📧 Email**: support@example.com
- **🐛 Bug Reports**: [GitHub Issues](https://github.com/alakob/row_id_generator/issues)
- **💬 Questions**: [GitHub Discussions](https://github.com/alakob/row_id_generator/discussions)
- **📖 Documentation**: [GitHub Docs](https://github.com/alakob/row_id_generator/blob/main/docs/)

## 🙏 Acknowledgments

- Built with [pandas](https://pandas.pydata.org/) for powerful data manipulation
- Integrated with [Snowflake](https://www.snowflake.com/) for cloud data warehousing
- Inspired by the critical need for deterministic row identification in modern data pipelines
- Thanks to the open-source community for continuous feedback and contributions

---

**⭐ Star this repository if it helps your data pipeline!** 