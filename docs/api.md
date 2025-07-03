# API Reference

This document provides comprehensive API reference for the row-id-generator package. All functions and classes are documented with their signatures, parameters, return values, and examples.

## Table of Contents

- [Core Functions](#core-functions)
- [Utility Functions](#utility-functions)
- [Observable API](#observable-api)
- [Data Types and Classes](#data-types-and-classes)
- [Exceptions](#exceptions)
- [Configuration](#configuration)

---

## Core Functions

### `generate_unique_row_ids()`

The primary function for generating unique row IDs with comprehensive options.

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
- `df` (pd.DataFrame): Input pandas DataFrame
- `columns` (Optional[List[str]]): List of column names to use for hashing. If None, columns are automatically selected based on uniqueness and completeness
- `id_column_name` (str): Name for the generated row ID column. Default: 'row_id'
- `uniqueness_threshold` (float): Minimum uniqueness ratio (0.0-1.0) for automatic column selection. Default: 0.95
- `separator` (str): String separator for concatenating column values. Default: '|'
- `enable_monitoring` (bool): Enable comprehensive observability features. Default: True
- `enable_quality_checks` (bool): Enable data quality validation before processing. Default: True
- `show_progress` (bool): Display progress bars for large datasets. Default: True
- `show_warnings` (bool): Show warnings for data quality issues. Default: True
- `enable_enhanced_lineage` (bool): Enable detailed data lineage tracking. Default: False
- `return_audit_trail` (bool): Return detailed processing information. Default: False

**Returns:**
- `pd.DataFrame`: DataFrame with added row ID column (if return_audit_trail=False)
- `Dict[str, Any]`: Dictionary containing dataframe, audit_trail, and selected_columns (if return_audit_trail=True)

**Raises:**
- `DataValidationError`: If input DataFrame fails validation
- `RowIDGenerationError`: If row ID generation fails
- `ValueError`: If parameters are invalid

**Example:**
```python
import pandas as pd
from row_id_generator import generate_unique_row_ids

df = pd.DataFrame({
    'email': ['alice@example.com', 'bob@example.com'],
    'name': ['Alice Johnson', 'Bob Smith'],
    'age': [28, 34]
})

# Basic usage
result_df = generate_unique_row_ids(df)

# Advanced usage with audit trail
result = generate_unique_row_ids(
    df,
    columns=['email', 'name'],
    uniqueness_threshold=0.98,
    return_audit_trail=True
)
df_with_ids = result['dataframe']
audit_trail = result['audit_trail']
```

### `generate_row_ids_simple()`

Simplified version for basic use cases with minimal overhead.

```python
def generate_row_ids_simple(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    id_column_name: str = 'row_id'
) -> pd.DataFrame
```

**Parameters:**
- `df` (pd.DataFrame): Input pandas DataFrame
- `columns` (Optional[List[str]]): Columns to use for hashing
- `id_column_name` (str): Name for the row ID column

**Returns:**
- `pd.DataFrame`: DataFrame with added row ID column

**Example:**
```python
# Minimal overhead for simple use cases
result_df = generate_row_ids_simple(df, columns=['email'])
```

### `generate_row_ids_fast()`

High-performance version optimized for large datasets.

```python
def generate_row_ids_fast(
    df: pd.DataFrame,
    columns: List[str],
    id_column_name: str = 'row_id'
) -> pd.DataFrame
```

**Parameters:**
- `df` (pd.DataFrame): Input pandas DataFrame
- `columns` (List[str]): Required. Columns to use for hashing
- `id_column_name` (str): Name for the row ID column

**Returns:**
- `pd.DataFrame`: DataFrame with added row ID column

**Example:**
```python
# Maximum performance for large datasets
result_df = generate_row_ids_fast(large_df, columns=['primary_key'])
```

---

## Utility Functions

### Column Selection

#### `select_columns_for_hashing()`

Intelligently selects optimal columns for hashing based on data quality metrics.

```python
def select_columns_for_hashing(
    df: pd.DataFrame,
    manual_columns: Optional[List[str]] = None,
    uniqueness_threshold: float = 0.95,
    include_email: bool = True,
    max_columns: int = 10
) -> List[str]
```

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame to analyze
- `manual_columns` (Optional[List[str]]): Manually specified columns (overrides auto-selection)
- `uniqueness_threshold` (float): Minimum uniqueness ratio for column inclusion
- `include_email` (bool): Prioritize email columns in selection
- `max_columns` (int): Maximum number of columns to select

**Returns:**
- `List[str]`: List of selected column names

**Example:**
```python
from row_id_generator import select_columns_for_hashing

selected_cols = select_columns_for_hashing(
    df,
    uniqueness_threshold=0.9,
    include_email=True
)
```

### Data Preprocessing

#### `prepare_data_for_hashing()`

Preprocesses DataFrame columns for consistent hashing.

```python
def prepare_data_for_hashing(
    df: pd.DataFrame,
    columns: List[str]
) -> pd.DataFrame
```

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `columns` (List[str]): Columns to preprocess

**Returns:**
- `pd.DataFrame`: Processed DataFrame with normalized data

**Example:**
```python
from row_id_generator import prepare_data_for_hashing

processed_df = prepare_data_for_hashing(df, ['email', 'name'])
```

#### `normalize_string_data()`

Normalizes string data for consistent hashing.

```python
def normalize_string_data(
    series: pd.Series,
    case_conversion: str = 'lower',
    strip_whitespace: bool = True,
    remove_special_chars: bool = False
) -> pd.Series
```

**Parameters:**
- `series` (pd.Series): Input string series
- `case_conversion` (str): 'lower', 'upper', or 'none'
- `strip_whitespace` (bool): Remove leading/trailing whitespace
- `remove_special_chars` (bool): Remove special characters

**Returns:**
- `pd.Series`: Normalized string series

#### `handle_null_values()`

Handles null values in data for consistent hashing.

```python
def handle_null_values(
    series: pd.Series,
    strategy: str = 'replace',
    replacement_value: str = '__NULL__'
) -> pd.Series
```

**Parameters:**
- `series` (pd.Series): Input series with potential null values
- `strategy` (str): 'replace', 'drop', or 'keep'
- `replacement_value` (str): Value to replace nulls with

**Returns:**
- `pd.Series`: Series with handled null values

#### `standardize_datetime()`

Standardizes datetime values for consistent hashing.

```python
def standardize_datetime(
    series: pd.Series,
    target_timezone: str = 'UTC',
    date_format: str = '%Y-%m-%d %H:%M:%S'
) -> pd.Series
```

**Parameters:**
- `series` (pd.Series): Input datetime series
- `target_timezone` (str): Target timezone for standardization
- `date_format` (str): Output date format string

**Returns:**
- `pd.Series`: Standardized datetime series

#### `normalize_numeric_data()`

Normalizes numeric data for consistent hashing.

```python
def normalize_numeric_data(
    series: pd.Series,
    precision: int = 2,
    handle_infinity: bool = True
) -> pd.Series
```

**Parameters:**
- `series` (pd.Series): Input numeric series
- `precision` (int): Decimal precision for rounding
- `handle_infinity` (bool): Handle infinite values

**Returns:**
- `pd.Series`: Normalized numeric series

### Validation Functions

#### `validate_dataframe_input()`

Validates input DataFrame for processing.

```python
def validate_dataframe_input(
    df: pd.DataFrame,
    min_rows: int = 1,
    max_rows: Optional[int] = None
) -> None
```

**Parameters:**
- `df` (pd.DataFrame): DataFrame to validate
- `min_rows` (int): Minimum required rows
- `max_rows` (Optional[int]): Maximum allowed rows

**Raises:**
- `DataValidationError`: If validation fails

#### `validate_columns_parameter()`

Validates column parameter input.

```python
def validate_columns_parameter(
    columns: Optional[List[str]],
    df_columns: List[str]
) -> None
```

**Parameters:**
- `columns` (Optional[List[str]]): Columns to validate
- `df_columns` (List[str]): Available DataFrame columns

**Raises:**
- `ValueError`: If columns are invalid

#### `validate_id_column_name()`

Validates row ID column name.

```python
def validate_id_column_name(
    id_column_name: str,
    existing_columns: List[str]
) -> None
```

**Parameters:**
- `id_column_name` (str): Proposed ID column name
- `existing_columns` (List[str]): Existing DataFrame columns

**Raises:**
- `ValueError`: If ID column name conflicts

### Quality Analysis

#### `analyze_dataframe_quality()`

Performs comprehensive data quality analysis.

```python
def analyze_dataframe_quality(
    df: pd.DataFrame,
    include_recommendations: bool = True
) -> DataQualityMetrics
```

**Parameters:**
- `df` (pd.DataFrame): DataFrame to analyze
- `include_recommendations` (bool): Include improvement recommendations

**Returns:**
- `DataQualityMetrics`: Comprehensive quality metrics object

#### `get_column_quality_score()`

Calculates quality score for individual columns.

```python
def get_column_quality_score(
    df: pd.DataFrame,
    column: str
) -> Dict[str, Any]
```

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `column` (str): Column to analyze

**Returns:**
- `Dict[str, Any]`: Quality metrics including score, grade, and details

---

## Observable API

For production environments requiring comprehensive monitoring and observability.

### Engine Creation

#### `create_observable_engine()`

Creates a configurable observable engine.

```python
def create_observable_engine(
    config_path: str,
    enable_logging: bool = True,
    enable_metrics: bool = True
) -> ObservableHashingEngine
```

#### `create_minimal_observable_engine()`

Creates a minimal observable engine for basic monitoring.

```python
def create_minimal_observable_engine() -> ObservableHashingEngine
```

#### `create_full_observable_engine()`

Creates a full-featured observable engine with all monitoring capabilities.

```python
def create_full_observable_engine(
    config_path: Optional[str] = None
) -> ObservableHashingEngine
```

### ObservableHashingEngine Class

#### Methods

##### `generate_unique_row_ids()`

Generate row IDs with full observability.

```python
def generate_unique_row_ids(
    self,
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    show_progress: bool = True,
    **kwargs
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]
```

**Returns:**
- `Tuple[pd.DataFrame, List[str], Dict[str, Any]]`: Result DataFrame, selected columns, audit trail

##### `get_system_health_report()`

Get comprehensive system health metrics.

```python
def get_system_health_report(self) -> Dict[str, Any]
```

**Returns:**
- `Dict[str, Any]`: System health metrics including resource usage, active alerts, and performance indicators

##### `get_session_summary()`

Get summary of current session operations.

```python
def get_session_summary(self) -> Dict[str, Any]
```

**Returns:**
- `Dict[str, Any]`: Session metrics including operation count and success rate

##### `export_metrics()`

Export metrics in various formats.

```python
def export_metrics(
    self,
    format_type: str = "json"
) -> Union[str, Dict[str, Any]]
```

**Parameters:**
- `format_type` (str): Export format ('json', 'prometheus', 'csv')

**Returns:**
- Metrics in requested format

##### `generate_performance_dashboard()`

Generate HTML performance dashboard.

```python
def generate_performance_dashboard(self) -> str
```

**Returns:**
- `str`: HTML dashboard content

**Example:**
```python
from row_id_generator import create_full_observable_engine

# Create observable engine
engine = create_full_observable_engine()

# Generate row IDs with monitoring
result_df, selected_columns, audit_trail = engine.generate_unique_row_ids(
    df=df,
    show_progress=True
)

# Get system health
health = engine.get_system_health_report()
print(f"CPU Usage: {health['system_resources']['cpu_percent']}%")

# Export metrics
prometheus_metrics = engine.export_metrics("prometheus")
```

---

## Data Types and Classes

### DataQualityMetrics

Comprehensive data quality analysis results.

```python
class DataQualityMetrics:
    def __init__(self, df: pd.DataFrame):
        # Initialization logic
        
    def get_summary_report(self) -> Dict[str, Any]:
        """Get summary quality report."""
        
    def get_column_analysis(self, column: str) -> Dict[str, Any]:
        """Get detailed column analysis."""
        
    def get_recommendations(self) -> List[str]:
        """Get quality improvement recommendations."""
```

### Performance Classes

#### `PerformanceBaseline`

Tracks performance baseline metrics.

```python
class PerformanceBaseline:
    def __init__(self, operation_type: str, baseline_time: float):
        self.operation_type = operation_type
        self.baseline_time = baseline_time
```

#### `SessionMetrics`

Tracks session-level metrics.

```python
class SessionMetrics:
    def __init__(self):
        self.operation_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
```

### Monitoring Classes

#### `ProcessStage`

Enumeration of processing stages.

```python
class ProcessStage(Enum):
    INITIALIZATION = "initialization"
    COLUMN_SELECTION = "column_selection"
    DATA_PREPARATION = "data_preparation"
    HASH_GENERATION = "hash_generation"
    VALIDATION = "validation"
    COMPLETION = "completion"
```

#### `HashingEvent`

Individual hashing operation event.

```python
class HashingEvent:
    def __init__(
        self,
        event_type: HashingEventType,
        timestamp: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.event_type = event_type
        self.timestamp = timestamp
        self.metadata = metadata or {}
```

---

## Exceptions

### `RowIDGenerationError`

Base exception for row ID generation errors.

```python
class RowIDGenerationError(Exception):
    """Base exception for row ID generation errors."""
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.suggestions = suggestions or []
```

### `DataValidationError`

Exception for data validation failures.

```python
class DataValidationError(RowIDGenerationError):
    """Raised when input data fails validation."""
    pass
```

### `HashGenerationError`

Exception for hash generation failures.

```python
class HashGenerationError(RowIDGenerationError):
    """Raised when hash generation fails."""
    pass
```

### `ConfigurationError`

Exception for configuration issues.

```python
class ConfigurationError(RowIDGenerationError):
    """Raised when configuration is invalid."""
    pass
```

**Example Error Handling:**
```python
from row_id_generator import (
    generate_unique_row_ids,
    DataValidationError,
    HashGenerationError,
    RowIDGenerationError
)

try:
    result_df = generate_unique_row_ids(df)
except DataValidationError as e:
    print(f"Data validation failed: {e.message}")
    print(f"Context: {e.context}")
    print(f"Suggestions: {e.suggestions}")
except HashGenerationError as e:
    print(f"Hash generation failed: {e.message}")
    # Implement retry logic
except RowIDGenerationError as e:
    print(f"General error: {e.message}")
    # Log and notify monitoring systems
```

---

## Configuration

### Environment Variables

```bash
# Logging Configuration
ROWID_LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR
ROWID_LOG_FORMAT=structured             # structured, simple

# Performance Configuration
ROWID_DEFAULT_BATCH_SIZE=10000          # Default batch size for processing
ROWID_MAX_MEMORY_MB=1000               # Maximum memory usage in MB
ROWID_ENABLE_PARALLEL=true             # Enable parallel processing

# Monitoring Configuration
ROWID_ENABLE_METRICS=true              # Enable metrics collection
ROWID_METRICS_RETENTION_HOURS=168      # Metrics retention (7 days)
ROWID_ENABLE_ALERTS=true               # Enable alerting system
```

### Configuration Classes

#### `HashingConfig`

Configuration for hashing parameters.

```python
@dataclass
class HashingConfig:
    algorithm: str = "sha256"
    separator: str = "|"
    encoding: str = "utf-8"
    enable_caching: bool = True
```

#### `ColumnSelectionConfig`

Configuration for column selection.

```python
@dataclass
class ColumnSelectionConfig:
    uniqueness_threshold: float = 0.95
    include_email: bool = True
    max_columns: int = 10
    prioritize_complete_columns: bool = True
```

#### `PerformanceConfig`

Configuration for performance optimization.

```python
@dataclass
class PerformanceConfig:
    batch_size: int = 10000
    max_memory_mb: int = 1000
    enable_parallel: bool = True
    chunk_size_auto: bool = True
    progress_update_interval: int = 1000
```

#### `MonitoringConfig`

Configuration for monitoring and observability.

```python
@dataclass
class MonitoringConfig:
    enable_detailed_logging: bool = True
    metrics_retention_hours: int = 168
    enable_alerts: bool = True
    dashboard_auto_refresh: int = 30
    export_prometheus: bool = False
```

#### `QualityConfig`

Configuration for data quality checks.

```python
@dataclass
class QualityConfig:
    enable_validation: bool = True
    max_null_ratio: float = 0.1
    min_uniqueness: float = 0.8
    require_email_validation: bool = True
```

**Example Configuration Usage:**
```python
from row_id_generator import (
    HashingConfig,
    ColumnSelectionConfig,
    PerformanceConfig,
    create_observable_engine
)

# Create custom configuration
hashing_config = HashingConfig(separator="::", enable_caching=True)
column_config = ColumnSelectionConfig(uniqueness_threshold=0.98)
performance_config = PerformanceConfig(batch_size=5000)

# Use with observable engine
engine = create_observable_engine(
    "config.yaml",
    hashing_config=hashing_config,
    column_config=column_config,
    performance_config=performance_config
)
```

---

## Version Information

- **Package Version**: Check with `row_id_generator.__version__`
- **API Version**: 2.0.0
- **Compatibility**: Python 3.9+
- **Dependencies**: pandas >= 2.2.0, numpy >= 1.24.0

For the latest API updates and changes, see the [Changelog](../CHANGELOG.md).

---

*This API reference is automatically generated and maintained. For questions or issues, please refer to the [GitHub Issues](https://github.com/alakob/row_id_generator/issues).* 