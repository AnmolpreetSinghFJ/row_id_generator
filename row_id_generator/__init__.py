"""
Row ID Generator: A high-performance library for generating unique, deterministic row IDs.

This package provides utilities for generating SHA-256 based row identifiers from pandas DataFrame data,
with support for column selection, data preprocessing, and comprehensive monitoring.
"""

__version__ = "1.0.0"
__author__ = "Row ID Generator Team"
__email__ = "support@example.com"

# Import main functionality for easy access
from .core import (
    generate_unique_row_ids,
    generate_row_hash,
    generate_row_ids_vectorized,
    generate_row_ids_simple,
    generate_row_ids_fast,
    generate_row_ids_with_audit,
    HashingEngine,
    HashingObserver,
    prepare_for_snowflake,
    load_to_snowflake,
    create_optimized_row_id_function
)
from .utils import (
    select_columns_for_hashing,
    prepare_data_for_hashing,
    prepare_data_for_hashing_with_dtype_preservation,
    restore_original_dtypes,
    normalize_string_data,
    handle_null_values,
    standardize_datetime,
    normalize_numeric_data,
    get_column_quality_score,
    analyze_dataframe_quality,
    DataQualityMetrics
)

# Import observability-integrated functionality
from .observable import (
    ObservableHashingEngine,
    create_observable_engine,
    create_minimal_observable_engine,
    create_full_observable_engine
)

# CLI functionality
from . import cli

# Define what gets imported with "from row_id_generator import *"
__all__ = [
    # Core functionality
    'generate_unique_row_ids',
    'generate_row_hash',
    'generate_row_ids_vectorized',
    'generate_row_ids_simple',
    'generate_row_ids_fast',
    'generate_row_ids_with_audit',
    'HashingEngine',
    'HashingObserver',
    'select_columns_for_hashing',
    'prepare_data_for_hashing',
    'prepare_data_for_hashing_with_dtype_preservation',
    'restore_original_dtypes',
    'normalize_string_data',
    'handle_null_values',
    'standardize_datetime',
    'normalize_numeric_data',
    'get_column_quality_score',
    'analyze_dataframe_quality',
    'DataQualityMetrics',
    
    # Observable functionality
    'ObservableHashingEngine',
    'create_observable_engine',
    'create_minimal_observable_engine',
    'create_full_observable_engine',
    
    # Snowflake integration
    'prepare_for_snowflake',
    'load_to_snowflake',
    
    # Performance optimization
    'create_optimized_row_id_function',
    
    # CLI module
    'cli'
] 