# Examples

This directory contains example scripts demonstrating how to use the row-id-generator package in various scenarios.

## Available Examples

### `basic_usage.py`
Demonstrates the basic functionality of the package:
- Creating a sample DataFrame
- Generating row IDs with default settings
- Using custom configuration options
- Understanding the output format

**Run the example:**
```bash
cd examples
python basic_usage.py
```

## Planned Examples

The following examples will be added as the package features are implemented:

### `snowflake_integration.py` *(Coming Soon)*
- Connecting to Snowflake
- Loading DataFrames with row IDs
- Error handling and recovery
- Performance optimization tips

### `large_dataframe_processing.py` *(Coming Soon)*
- Handling large DataFrames efficiently
- Memory management techniques
- Progress monitoring
- Performance benchmarks

### `column_selection_examples.py` *(Coming Soon)*
- Automatic column selection scenarios
- Manual column specification
- Handling edge cases (all nulls, low uniqueness)
- Email column prioritization

### `data_preprocessing_examples.py` *(Coming Soon)*
- String normalization
- Datetime standardization
- Numeric data handling
- NULL value management

### `performance_tuning.py` *(Coming Soon)*
- Optimization techniques
- Memory usage monitoring
- Parallel processing
- Benchmarking different approaches

### `error_handling_examples.py` *(Coming Soon)*
- Common error scenarios
- Graceful error recovery
- Validation techniques
- Debugging tips

## Running Examples

### Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) for fast Python package management. Install uv first if you haven't already:

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Setup and Execution

1. **Install the package in development mode:**
   ```bash
   # Create virtual environment
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install package with development dependencies
   uv pip install -e ".[dev]"
   ```

2. **Navigate to examples directory:**
   ```bash
   cd examples
   ```

3. **Run individual examples:**
   ```bash
   python basic_usage.py
   python snowflake_integration.py
   # etc.
   ```

## Example Data

Some examples use sample datasets. These are either:
- Generated programmatically within the script
- Small, representative datasets included in the examples
- Instructions for downloading public datasets

## Contributing Examples

We welcome additional examples! If you have a use case that would benefit others:

1. Create a new Python file in this directory
2. Include comprehensive comments and docstrings
3. Add error handling and clear output messages
4. Update this README with a description
5. Submit a pull request

### Example Template

```python
"""
Description of what this example demonstrates.

This example shows how to...
"""

import pandas as pd
from row_id_generator import generate_unique_row_ids

def main():
    """Main example function."""
    print("Example Name - Description")
    print("=" * 40)
    
    # Your example code here
    
    print("Example completed successfully! 🎉")

if __name__ == "__main__":
    main()
```

## Support

If you have questions about any examples or need help adapting them to your use case:

- 💬 [GitHub Discussions](https://github.com/alakob/row_id_generator/discussions)
- 🐛 [Issues](https://github.com/alakob/row_id_generator/issues)
- 📧 Email: [support@example.com](mailto:support@example.com) 