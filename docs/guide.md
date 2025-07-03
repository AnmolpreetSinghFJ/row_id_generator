# User Guide

A comprehensive guide to using the row-id-generator package effectively in various scenarios and environments.

## Table of Contents

- [Getting Started](#getting-started)
- [Basic Usage Patterns](#basic-usage-patterns)
- [Advanced Configuration](#advanced-configuration)
- [Production Deployment](#production-deployment)
- [Performance Optimization](#performance-optimization)
- [Monitoring and Observability](#monitoring-and-observability)
- [Common Use Cases](#common-use-cases)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Getting Started

### Installation Options

#### From PyPI (Stable Release)

```bash
# Using uv (recommended)
uv add row-id-generator

# Using pip
pip install row-id-generator
```

#### From GitHub (Latest Development)

```bash
# Latest development version
uv add git+https://github.com/alakob/row_id_generator.git

# Using pip
pip install git+https://github.com/alakob/row_id_generator.git
```

#### Specific Version or Branch

```bash
# Specific branch
uv add git+https://github.com/alakob/row_id_generator.git@main

# Specific tag/version
uv add git+https://github.com/alakob/row_id_generator.git@v1.0.0

# With pip
pip install git+https://github.com/alakob/row_id_generator.git@main
```

#### Development Installation

```bash
# Clone and setup for development
git clone https://github.com/alakob/row_id_generator.git
cd row_id_generator
uv sync --dev

# Or with pip
pip install -e ".[dev]"
```

#### With Optional Dependencies

```bash
# Snowflake integration
uv add "row-id-generator[snowflake]"

# Observability features
uv add "row-id-generator[observability]"

# All optional dependencies
uv add "row-id-generator[all]"

# From GitHub with extras
uv add "git+https://github.com/alakob/row_id_generator.git[all]"
```

### Your First Row ID Generation

```python
import pandas as pd
from row_id_generator import generate_unique_row_ids

# Create sample data
df = pd.DataFrame({
    'email': ['user1@example.com', 'user2@example.com'],
    'name': ['Alice Johnson', 'Bob Smith'],
    'department': ['Engineering', 'Marketing']
})

# Generate row IDs
result_df = generate_unique_row_ids(df)
print(result_df)
```

That's it! The package automatically:
- Selects the best columns for uniqueness
- Generates deterministic SHA-256 hashes
- Adds a `row_id` column to your DataFrame

---

## Basic Usage Patterns

### Pattern 1: Automatic Column Selection

Let the package choose the best columns based on data quality:

```python
# The package analyzes your data and selects optimal columns
result_df = generate_unique_row_ids(df)
```

**When to use:**
- You're not sure which columns provide the best uniqueness
- You want the package to optimize for data quality
- You have many columns and want automated selection

### Pattern 2: Manual Column Selection

Specify exactly which columns to use:

```python
result_df = generate_unique_row_ids(
    df,
    columns=['email', 'user_id'],  # Explicit column choice
    id_column_name='customer_hash'  # Custom column name
)
```

**When to use:**
- You know which columns provide business uniqueness
- You need consistent column selection across runs
- You want to exclude certain sensitive columns

### Pattern 3: Performance-Optimized

Use simplified versions for better performance:

```python
from row_id_generator import generate_row_ids_fast

# Maximum performance for large datasets
result_df = generate_row_ids_fast(
    large_df,
    columns=['primary_key'],  # Must specify columns
)
```

**When to use:**
- Processing millions of rows
- Performance is more important than features
- You have well-defined primary keys

---

## Advanced Configuration

### Data Quality Control

```python
result_df = generate_unique_row_ids(
    df,
    uniqueness_threshold=0.98,      # Higher uniqueness requirement
    enable_quality_checks=True,     # Enable data validation
    show_warnings=True              # Show data quality warnings
)
```

### Custom Processing Options

```python
result_df = generate_unique_row_ids(
    df,
    separator='::',                 # Custom value separator
    enable_monitoring=True,         # Enable observability
    show_progress=True,            # Show progress bars
    return_audit_trail=True        # Get detailed processing info
)

# Handle different return types
if isinstance(result_df, dict):
    df_with_ids = result_df['dataframe']
    audit_info = result_df['audit_trail']
    selected_columns = result_df['selected_columns']
else:
    df_with_ids = result_df
```

### Environment-Specific Configuration

```python
# Development: Full monitoring and validation
dev_result = generate_unique_row_ids(
    df,
    enable_monitoring=True,
    enable_quality_checks=True,
    show_progress=True,
    show_warnings=True
)

# Production: Optimized for performance
prod_result = generate_unique_row_ids(
    df,
    columns=['known_unique_column'],
    enable_monitoring=False,
    enable_quality_checks=False,
    show_progress=False
)
```

---

## Production Deployment

### Observable Engine for Production

For production environments, use the Observable API:

```python
from row_id_generator import create_full_observable_engine

# Create production-ready engine
engine = create_full_observable_engine('config/production.yaml')

# Process data with full observability
result_df, selected_columns, audit_trail = engine.generate_unique_row_ids(
    df=df,
    show_progress=True
)

# Monitor system health
health_report = engine.get_system_health_report()
if health_report['system_resources']['cpu_percent'] > 80:
    print("High CPU usage detected")

# Export metrics to monitoring systems
prometheus_metrics = engine.export_metrics("prometheus")
```

### Configuration Management

Create a production configuration file:

```yaml
# config/production.yaml
hashing:
  algorithm: sha256
  separator: "|"
  encoding: utf-8

column_selection:
  uniqueness_threshold: 0.95
  include_email: true
  max_columns: 10

performance:
  batch_size: 50000
  max_memory_mb: 2000
  enable_parallel: true

monitoring:
  enable_detailed_logging: true
  metrics_retention_hours: 168
  enable_alerts: true
  dashboard_auto_refresh: 30

quality_checks:
  enable_validation: true
  max_null_ratio: 0.1
  min_uniqueness: 0.85
```

### Error Handling and Recovery

```python
from row_id_generator import (
    generate_unique_row_ids,
    DataValidationError,
    HashGenerationError,
    RowIDGenerationError
)

def safe_row_id_generation(df, max_retries=3):
    """Production-ready row ID generation with error handling."""
    for attempt in range(max_retries):
        try:
            result_df = generate_unique_row_ids(
                df,
                enable_quality_checks=True,
                show_warnings=False  # Don't spam logs in production
            )
            return result_df, None
            
        except DataValidationError as e:
            if attempt < max_retries - 1:
                # Try with lower quality requirements
                result_df = generate_unique_row_ids(
                    df,
                    uniqueness_threshold=0.8,
                    enable_quality_checks=False
                )
                return result_df, f"Used fallback configuration: {e.message}"
            else:
                return None, f"Data validation failed: {e.message}"
                
        except HashGenerationError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                return None, f"Hash generation failed: {e.message}"
                
        except Exception as e:
            return None, f"Unexpected error: {str(e)}"
    
    return None, "Max retries exceeded"

# Usage
result_df, error_message = safe_row_id_generation(df)
if error_message:
    logger.warning(f"Row ID generation issue: {error_message}")
```

---

## Performance Optimization

### Memory-Efficient Processing

For large datasets, use chunked processing:

```python
from row_id_generator import create_optimized_row_id_function

# Create memory-optimized processor
optimized_processor = create_optimized_row_id_function(
    max_memory_mb=1000,      # Limit memory usage
    enable_chunking=True,    # Process in chunks
    enable_streaming=True    # Stream operations
)

# Process large DataFrame efficiently
large_df = pd.read_csv('massive_dataset.csv')  # 10M+ rows
result_df = optimized_processor(
    df=large_df,
    columns=['email', 'user_id'],
    chunk_size=50000  # Process 50k rows at a time
)
```

### Performance Monitoring

```python
import time
from row_id_generator import generate_unique_row_ids

# Time your operations
start_time = time.time()
result_df = generate_unique_row_ids(df, show_progress=True)
processing_time = time.time() - start_time

rows_per_second = len(df) / processing_time
print(f"Processed {len(df):,} rows in {processing_time:.2f}s ({rows_per_second:,.0f} rows/sec)")
```

### Batch Processing Patterns

```python
def process_data_in_batches(file_paths, batch_size=100000):
    """Process multiple files in batches."""
    all_results = []
    
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        
        # Process in chunks if file is large
        if len(df) > batch_size:
            chunks = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]
            
            chunk_results = []
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)} of {file_path}")
                chunk_result = generate_unique_row_ids(
                    chunk,
                    show_progress=False  # Disable for chunk processing
                )
                chunk_results.append(chunk_result)
            
            # Combine chunks
            file_result = pd.concat(chunk_results, ignore_index=True)
        else:
            file_result = generate_unique_row_ids(df)
        
        all_results.append(file_result)
    
    return pd.concat(all_results, ignore_index=True)
```

---

## Monitoring and Observability

### Basic Monitoring Setup

```python
from row_id_generator import create_minimal_observable_engine

# Create engine with basic monitoring
engine = create_minimal_observable_engine()

# Process data with monitoring
result_df, selected_columns, audit_trail = engine.generate_unique_row_ids(df)

# Check processing metrics
session_summary = engine.get_session_summary()
print(f"Operations: {session_summary['operation_count']}")
print(f"Success rate: {session_summary['success_rate']:.1%}")
```

### Advanced Monitoring and Alerting

```python
from row_id_generator import create_full_observable_engine

# Create full monitoring engine
engine = create_full_observable_engine()

# Process with comprehensive monitoring
result_df, selected_columns, audit_trail = engine.generate_unique_row_ids(df)

# Get detailed health report
health_report = engine.get_system_health_report()

# Check for performance issues
cpu_usage = health_report['system_resources']['cpu_percent']
memory_usage = health_report['system_resources']['memory_percent']
active_alerts = health_report.get('active_alerts', [])

if cpu_usage > 80:
    print(f"⚠️ High CPU usage: {cpu_usage:.1f}%")
if memory_usage > 85:
    print(f"⚠️ High memory usage: {memory_usage:.1f}%")
if active_alerts:
    print(f"🚨 Active alerts: {len(active_alerts)}")

# Export metrics for external monitoring
metrics = engine.export_metrics("json")
prometheus_metrics = engine.export_metrics("prometheus")

# Generate performance dashboard
dashboard_html = engine.generate_performance_dashboard()
with open('performance_dashboard.html', 'w') as f:
    f.write(dashboard_html)
```

### Integration with External Monitoring

```python
import requests
import json

def send_metrics_to_datadog(engine, api_key):
    """Send metrics to Datadog."""
    metrics = engine.export_metrics("json")
    
    payload = {
        'series': [
            {
                'metric': 'rowid.operations.count',
                'points': [[time.time(), metrics['operation_count']]],
                'tags': ['environment:production']
            },
            {
                'metric': 'rowid.performance.throughput',
                'points': [[time.time(), metrics['throughput']]],
                'tags': ['environment:production']
            }
        ]
    }
    
    response = requests.post(
        'https://api.datadoghq.com/api/v1/series',
        headers={'DD-API-KEY': api_key},
        data=json.dumps(payload)
    )
    return response.status_code == 202
```

---

## Common Use Cases

### Use Case 1: Customer Data Deduplication

```python
# Customer data with potential duplicates
customer_df = pd.DataFrame({
    'email': ['alice@corp.com', 'bob@startup.io', 'alice@corp.com'],
    'first_name': ['Alice', 'Bob', 'Alice'],
    'last_name': ['Johnson', 'Smith', 'Johnson'],
    'phone': ['+1-555-0001', '+1-555-0002', '+1-555-0001']
})

# Generate stable customer IDs for deduplication
customer_df_with_ids = generate_unique_row_ids(
    customer_df,
    columns=['email'],  # Email is primary identifier
    id_column_name='customer_id'
)

# Identify duplicates
duplicates = customer_df_with_ids.duplicated(subset=['customer_id'], keep=False)
print(f"Found {duplicates.sum()} duplicate records")
```

### Use Case 2: Financial Transaction Processing

```python
# Financial transaction data requiring high uniqueness
transaction_df = pd.DataFrame({
    'transaction_id': ['T001', 'T002', 'T003'],
    'user_email': ['alice@corp.com', 'bob@startup.io', 'charlie@agency.co'],
    'amount': [99.99, 149.50, 79.99],
    'timestamp': pd.to_datetime(['2024-01-15 10:30', '2024-01-15 11:15', '2024-01-15 14:22'])
})

# Generate transaction hashes with high uniqueness requirement
transaction_df_with_ids = generate_unique_row_ids(
    transaction_df,
    columns=['transaction_id', 'user_email', 'timestamp'],
    uniqueness_threshold=0.99,  # Very high uniqueness for financial data
    enable_quality_checks=True,  # Strict validation
    id_column_name='transaction_hash'
)
```

### Use Case 3: Product Catalog Management

```python
# Product catalog with hierarchical data
product_df = pd.DataFrame({
    'sku': ['LAPTOP-001', 'PHONE-002', 'TABLET-003'],
    'name': ['Business Laptop', 'Smartphone Pro', 'Tablet Air'],
    'brand': ['TechCorp', 'PhoneCo', 'TechCorp'],
    'category': ['Electronics', 'Electronics', 'Electronics'],
    'price': [999.99, 699.99, 399.99]
})

# Generate product hashes for catalog management
product_df_with_ids = generate_unique_row_ids(
    product_df,
    columns=['sku', 'brand'],  # Business-relevant uniqueness
    separator='::',  # Custom separator for readability
    id_column_name='product_hash'
)
```

### Use Case 4: Data Pipeline Integration

```python
def etl_pipeline_with_row_ids(source_file, target_table):
    """ETL pipeline with row ID generation."""
    
    # Extract
    df = pd.read_csv(source_file)
    print(f"Extracted {len(df):,} rows from {source_file}")
    
    # Transform: Add row IDs
    df_with_ids = generate_unique_row_ids(
        df,
        enable_monitoring=True,
        show_progress=True,
        return_audit_trail=True
    )
    
    # Handle audit trail
    if isinstance(df_with_ids, dict):
        df_transformed = df_with_ids['dataframe']
        audit_trail = df_with_ids['audit_trail']
        print(f"Selected columns: {df_with_ids['selected_columns']}")
        print(f"Processing time: {audit_trail.get('processing_time', 0):.2f}s")
    else:
        df_transformed = df_with_ids
    
    # Load to Snowflake
    from row_id_generator.core import load_to_snowflake
    
    connection_params = {
        'account': os.getenv('SNOWFLAKE_ACCOUNT'),
        'user': os.getenv('SNOWFLAKE_USER'),
        'password': os.getenv('SNOWFLAKE_PASSWORD'),
        'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
        'database': os.getenv('SNOWFLAKE_DATABASE'),
        'schema': os.getenv('SNOWFLAKE_SCHEMA')
    }
    
    success, rows_loaded = load_to_snowflake(
        df_transformed,
        connection_params,
        table_name=target_table,
        if_exists='append'
    )
    
    if success:
        print(f"✅ Successfully loaded {rows_loaded:,} rows to {target_table}")
    else:
        print(f"❌ Failed to load data to {target_table}")
    
    return success
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Low Uniqueness Warning

**Problem:** You get warnings about low uniqueness in selected columns.

**Solution:**
```python
# Option 1: Lower the uniqueness threshold
result_df = generate_unique_row_ids(
    df,
    uniqueness_threshold=0.8  # Lower from default 0.95
)

# Option 2: Add more columns manually
result_df = generate_unique_row_ids(
    df,
    columns=['email', 'name', 'phone', 'address']  # More columns
)

# Option 3: Preprocess data to improve uniqueness
df_cleaned = df.dropna().drop_duplicates()
result_df = generate_unique_row_ids(df_cleaned)
```

#### Issue 2: Memory Issues with Large DataFrames

**Problem:** Out of memory errors when processing large datasets.

**Solution:**
```python
# Use memory-optimized processing
from row_id_generator import create_optimized_row_id_function

optimized_generator = create_optimized_row_id_function(
    max_memory_mb=500,  # Limit memory usage
    enable_chunking=True,
    enable_streaming=True
)

result_df = optimized_generator(
    df=large_df,
    columns=['primary_key'],
    chunk_size=10000  # Smaller chunks
)
```

#### Issue 3: Performance is Too Slow

**Problem:** Processing takes too long for your use case.

**Solution:**
```python
# Use fast variant with minimal features
from row_id_generator import generate_row_ids_fast

result_df = generate_row_ids_fast(
    df,
    columns=['primary_key'],  # Must specify columns
    # No monitoring, validation, or progress bars
)

# Or disable expensive features
result_df = generate_unique_row_ids(
    df,
    enable_monitoring=False,
    enable_quality_checks=False,
    show_progress=False
)
```

#### Issue 4: Column Selection is Inconsistent

**Problem:** Automatic column selection varies between runs.

**Solution:**
```python
# Always specify columns manually for consistency
result_df = generate_unique_row_ids(
    df,
    columns=['email', 'user_id'],  # Explicit columns
    # This ensures consistent column selection
)
```

### Debugging and Diagnostics

#### Enable Detailed Logging

```python
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('row_id_generator')

# Generate with detailed logging
result_df = generate_unique_row_ids(
    df,
    enable_monitoring=True,
    show_warnings=True
)
```

#### Analyze Data Quality Issues

```python
from row_id_generator import analyze_dataframe_quality, get_column_quality_score

# Comprehensive data quality analysis
quality_metrics = analyze_dataframe_quality(df)
summary = quality_metrics.get_summary_report()

print(f"Overall quality score: {summary['overall_score']:.2f}")
print(f"Recommendations: {quality_metrics.get_recommendations()}")

# Check individual columns
for column in df.columns:
    score = get_column_quality_score(df, column)
    print(f"{column}: {score['score']:.2f} ({score['grade']})")
```

---

## Best Practices

### 1. Choose the Right Function for Your Use Case

```python
# For simple, one-off processing
result_df = generate_row_ids_simple(df, columns=['email'])

# For most production use cases
result_df = generate_unique_row_ids(df, enable_monitoring=True)

# For high-performance, large-scale processing
result_df = generate_row_ids_fast(df, columns=['primary_key'])

# For production with full observability
engine = create_full_observable_engine()
result_df, cols, audit = engine.generate_unique_row_ids(df)
```

### 2. Always Validate Your Data

```python
from row_id_generator import validate_dataframe_input

# Validate before processing
try:
    validate_dataframe_input(df, min_rows=1)
    result_df = generate_unique_row_ids(df)
except Exception as e:
    print(f"Validation failed: {e}")
    # Handle error appropriately
```

### 3. Use Appropriate Column Selection Strategy

```python
# For customer data: prioritize email and user IDs
customer_result = generate_unique_row_ids(
    customer_df,
    columns=['email', 'customer_id']
)

# For transaction data: include time-based uniqueness
transaction_result = generate_unique_row_ids(
    transaction_df,
    columns=['transaction_id', 'user_id', 'timestamp']
)

# For product data: use SKU and variant information
product_result = generate_unique_row_ids(
    product_df,
    columns=['sku', 'variant_id', 'brand']
)
```

### 4. Monitor Performance in Production

```python
# Set up monitoring for production
engine = create_full_observable_engine()

# Process with monitoring
start_time = time.time()
result_df, cols, audit = engine.generate_unique_row_ids(df)
processing_time = time.time() - start_time

# Log metrics
throughput = len(df) / processing_time
print(f"Throughput: {throughput:,.0f} rows/sec")

# Alert on performance degradation
if throughput < 10000:  # Below threshold
    print("⚠️ Performance degradation detected")
```

### 5. Handle Errors Gracefully

```python
def robust_row_id_generation(df, fallback_columns=None):
    """Robust row ID generation with fallback strategies."""
    try:
        # Try with full validation
        return generate_unique_row_ids(
            df,
            enable_quality_checks=True,
            uniqueness_threshold=0.95
        )
    except Exception as e:
        print(f"Primary method failed: {e}")
        
        try:
            # Fallback 1: Lower quality requirements
            return generate_unique_row_ids(
                df,
                uniqueness_threshold=0.8,
                enable_quality_checks=False
            )
        except Exception as e:
            print(f"Fallback 1 failed: {e}")
            
            if fallback_columns:
                try:
                    # Fallback 2: Use specific columns
                    return generate_unique_row_ids(
                        df,
                        columns=fallback_columns,
                        enable_quality_checks=False
                    )
                except Exception as e:
                    print(f"Fallback 2 failed: {e}")
            
            # Final fallback: Add row numbers
            df_copy = df.copy()
            df_copy['row_id'] = range(len(df_copy))
            return df_copy
```

### 6. Document Your Configuration

```python
# Document your production configuration
PRODUCTION_CONFIG = {
    'uniqueness_threshold': 0.95,
    'enable_monitoring': True,
    'enable_quality_checks': True,
    'show_progress': False,  # Disable in production
    'columns': ['email', 'user_id'],  # Business-specific
    'id_column_name': 'customer_hash',
    'separator': '|'
}

# Use consistent configuration
result_df = generate_unique_row_ids(df, **PRODUCTION_CONFIG)
```

---

For additional help and advanced topics, see:
- [API Reference](api.md) - Complete API documentation
- [Performance Guide](performance.md) - Optimization techniques
- [Examples](../examples/) - Real-world usage examples 