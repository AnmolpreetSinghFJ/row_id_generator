#!/usr/bin/env python3
"""
Command-line interface for row-id-generator package.

Provides basic functionality for generating row IDs from CSV files.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, List

import pandas as pd

from . import __version__
from .core import generate_unique_row_ids
from .utils import select_columns_for_hashing


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate unique, stable row IDs for Pandas DataFrames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.csv --output output.csv
  %(prog)s input.csv --columns email,user_id --output output.csv
  %(prog)s input.csv --quality-threshold 0.9 --output output.csv
  %(prog)s --version
        """
    )
    
    # Version
    parser.add_argument(
        '--version', 
        action='version', 
        version=f'row-id-generator {__version__}'
    )
    
    # Input file
    parser.add_argument(
        'input_file',
        nargs='?',
        help='Input CSV file path'
    )
    
    # Output file
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output CSV file path (default: adds _with_row_ids to input filename)'
    )
    
    # Column selection
    parser.add_argument(
        '-c', '--columns',
        type=str,
        help='Comma-separated list of columns to use for hashing (auto-selected if not specified)'
    )
    
    # Quality threshold
    parser.add_argument(
        '-q', '--quality-threshold',
        type=float,
        default=0.8,
        help='Minimum uniqueness threshold for column selection (default: 0.8)'
    )
    
    # Row ID column name
    parser.add_argument(
        '--row-id-column',
        type=str,
        default='row_id',
        help='Name for the generated row ID column (default: row_id)'
    )
    
    # Performance options
    parser.add_argument(
        '--chunk-size',
        type=int,
        help='Process data in chunks of this size for large files'
    )
    
    # Verbose output
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Show column analysis
    parser.add_argument(
        '--analyze-columns',
        action='store_true',
        help='Show column uniqueness analysis and exit'
    )
    
    return parser


def load_data(file_path: str, chunk_size: Optional[int] = None) -> pd.DataFrame:
    """Load data from CSV file."""
    try:
        if chunk_size:
            # For large files, load in chunks
            chunks = pd.read_csv(file_path, chunksize=chunk_size)
            return pd.concat(chunks, ignore_index=True)
        else:
            return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)


def save_data(df: pd.DataFrame, output_path: str, verbose: bool = False) -> None:
    """Save DataFrame to CSV file."""
    try:
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"✅ Successfully saved {len(df)} rows to '{output_path}'")
    except Exception as e:
        print(f"Error saving file '{output_path}': {e}", file=sys.stderr)
        sys.exit(1)


def get_output_filename(input_path: str, output_path: Optional[str]) -> str:
    """Generate output filename if not provided."""
    if output_path:
        return output_path
    
    path = Path(input_path)
    stem = path.stem
    suffix = path.suffix
    return str(path.parent / f"{stem}_with_row_ids{suffix}")


def analyze_columns_command(df: pd.DataFrame, threshold: float) -> None:
    """Run column analysis and display results."""
    print("📊 Column Uniqueness Analysis")
    print("=" * 50)
    
    for column in df.columns:
        unique_count = df[column].nunique()
        total_count = len(df)
        uniqueness = unique_count / total_count if total_count > 0 else 0
        
        status = "✅" if uniqueness >= threshold else "❌"
        print(f"{status} {column:20} | {uniqueness:6.2%} | {unique_count:,}/{total_count:,} unique")
    
    print("\nRecommended columns for hashing:")
    try:
        selected_cols = select_columns_for_hashing(df, uniqueness_threshold=threshold)
        if selected_cols:
            for col in selected_cols:
                print(f"  • {col}")
        else:
            print("  • No columns meet the uniqueness threshold")
            print(f"  • Consider lowering threshold below {threshold:.1%}")
    except Exception as e:
        print(f"  • Error in column selection: {e}")


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Show help if no input file provided
    if not args.input_file:
        parser.print_help()
        sys.exit(1)
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    # Load data
    if args.verbose:
        print(f"📂 Loading data from '{args.input_file}'...")
    
    df = load_data(args.input_file, args.chunk_size)
    
    if args.verbose:
        print(f"📊 Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Column analysis mode
    if args.analyze_columns:
        analyze_columns_command(df, args.quality_threshold)
        return
    
    # Parse columns if provided
    columns_to_use: Optional[List[str]] = None
    if args.columns:
        columns_to_use = [col.strip() for col in args.columns.split(',')]
        # Validate columns exist
        missing_cols = [col for col in columns_to_use if col not in df.columns]
        if missing_cols:
            print(f"Error: Columns not found in data: {missing_cols}", file=sys.stderr)
            print(f"Available columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)
    
    # Generate row IDs
    if args.verbose:
        print("🔨 Generating row IDs...")
    
    try:
        if columns_to_use:
            result_df = generate_unique_row_ids(
                df, 
                columns=columns_to_use,
                uniqueness_threshold=args.quality_threshold,
                row_id_column=args.row_id_column
            )
        else:
            result_df = generate_unique_row_ids(
                df,
                uniqueness_threshold=args.quality_threshold,
                row_id_column=args.row_id_column
            )
        
        if args.verbose:
            print(f"✅ Generated row IDs for {len(result_df)} rows")
    
    except Exception as e:
        print(f"Error generating row IDs: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Save results
    output_path = get_output_filename(args.input_file, args.output)
    
    if args.verbose:
        print(f"💾 Saving results to '{output_path}'...")
    
    save_data(result_df, output_path, args.verbose)
    
    if not args.verbose:
        print(f"Row IDs generated and saved to '{output_path}'")


if __name__ == '__main__':
    main() 