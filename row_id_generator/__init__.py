"""
Row ID Generator Package

A Python package for generating unique, stable row IDs for Pandas DataFrames 
before loading into Snowflake databases.

Now includes comprehensive observability features for monitoring, metrics,
alerting, and dashboard capabilities, plus a command-line interface.
"""

__version__ = "1.0.0"
__author__ = "Row ID Generator Team"
__email__ = "support@example.com"

# Import main functionality for easy access
from .core import (
    generate_unique_row_ids, 
    generate_row_ids_simple, 
    generate_row_ids_fast,
    create_optimized_row_id_function
)
from .utils import (
    select_columns_for_hashing, 
    prepare_data_for_hashing,
    normalize_string_data,
    handle_null_values,
    standardize_datetime,
    normalize_numeric_data
)

# Import observability-integrated functionality
from .observable import (
    ObservableHashingEngine,
    ObservabilityMetrics,
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
    'generate_row_ids_simple',
    'generate_row_ids_fast',
    'create_optimized_row_id_function',
    'select_columns_for_hashing', 
    'prepare_data_for_hashing',
    'normalize_string_data',
    'handle_null_values',
    'standardize_datetime',
    'normalize_numeric_data',
    
    # Observable functionality
    'ObservableHashingEngine',
    'ObservabilityMetrics',
    'create_observable_engine',
    'create_minimal_observable_engine',
    'create_full_observable_engine',
    
    # CLI module
    'cli'
] 