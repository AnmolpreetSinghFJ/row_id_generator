#!/usr/bin/env python3
"""
Row ID Generator - Column Introspection Example
===============================================

This example demonstrates how to determine which columns were used for row ID generation
and proposes enhancements for better introspection capabilities.
"""

import pandas as pd
import json
import hashlib
from row_id_generator import generate_unique_row_ids, select_columns_for_hashing

def demonstrate_current_introspection():
    """Show current column introspection capabilities"""
    print("=== CURRENT COLUMN INTROSPECTION CAPABILITIES ===\n")
    
    # Create sample data
    df = pd.DataFrame({
        'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com'],
        'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown'],
        'age': [28, 34, 29],
        'department': ['Engineering', 'Marketing', 'Engineering'],
        'salary': [75000, 85000, 72000],
        'join_date': pd.to_datetime(['2022-01-15', '2021-03-22', '2023-06-10'])
    })
    
    print("📊 Sample Data:")
    print(df)
    
    # Method 1: Using audit trail
    print("\n📋 METHOD 1: Audit Trail Information")
    result_with_audit = generate_unique_row_ids(df, return_audit_trail=True, show_progress=False)
    if isinstance(result_with_audit, dict):
        column_info = result_with_audit['column_selection']
        print(f"  ✅ Selected columns: {column_info['selected_columns']}")
        print(f"  🔄 Selection method: {column_info['selection_method']}")
        print(f"  🆔 Session ID: {result_with_audit['session_id']}")
        
        # Show configuration used
        config = result_with_audit['audit_trail']['configuration']
        print(f"  ⚙️  Configuration used:")
        print(f"     - Separator: '{config.get('separator', 'N/A')}'")
        print(f"     - Uniqueness threshold: {config.get('uniqueness_threshold', 'N/A')}")
        print(f"     - ID column name: '{config.get('id_column_name', 'N/A')}'")
        
        # Show first row ID
        result_df = result_with_audit['result_dataframe']
        first_row_id = result_df.iloc[0]['row_id']
        print(f"  🔑 First row ID: {first_row_id[:24]}...")
    
    # Method 2: Column selection analysis
    print("\n🔍 METHOD 2: Column Selection Analysis")
    selected_columns = select_columns_for_hashing(df, uniqueness_threshold=0.95)
    print(f"  📋 Auto-selected columns: {selected_columns}")
    print(f"  📊 Total columns available: {len(df.columns)}")
    print(f"  ✅ Columns used: {len(selected_columns)}")
    print(f"  📈 Selection efficiency: {len(selected_columns)/len(df.columns)*100:.1f}%")
    
    return result_with_audit

def create_column_metadata_hash(columns, separator="|"):
    """Create a hash representing the column configuration used"""
    column_config = {
        'columns': sorted(columns),  # Sort for consistency
        'separator': separator,
        'count': len(columns)
    }
    config_string = json.dumps(column_config, sort_keys=True)
    return hashlib.md5(config_string.encode()).hexdigest()[:8]

def demonstrate_enhanced_introspection():
    """Demonstrate enhanced column introspection capabilities"""
    print("\n=== ENHANCED COLUMN INTROSPECTION CAPABILITIES ===\n")
    
    # Create sample data
    df = pd.DataFrame({
        'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com'],
        'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown'],
        'age': [28, 34, 29],
        'department': ['Engineering', 'Marketing', 'Engineering']
    })
    
    # Generate row IDs with detailed audit trail
    result = generate_unique_row_ids(df, return_audit_trail=True, show_progress=False)
    if isinstance(result, dict):
        columns_used = result['column_selection']['selected_columns']
        # Use default separator if not in audit trail
        separator = result['audit_trail']['configuration'].get('separator', '|')
        
        print("🔍 ENHANCED INTROSPECTION METHODS:")
        
        # Method 1: Column metadata hash
        column_hash = create_column_metadata_hash(columns_used, separator)
        print(f"\n1. 📋 Column Configuration Hash: {column_hash}")
        print(f"   - Represents: {columns_used} with separator '{separator}'")
        print(f"   - Use case: Quick identification of column sets")
        
        # Method 2: Deterministic recreation
        print(f"\n2. 🔄 Deterministic Recreation:")
        print(f"   - Original columns: {columns_used}")
        recreated_columns = select_columns_for_hashing(df, uniqueness_threshold=0.95)
        print(f"   - Recreated columns: {recreated_columns}")
        print(f"   - Match: {'✅ Yes' if columns_used == recreated_columns else '❌ No'}")
        
        # Method 3: Column signature
        column_signature = f"cols={len(columns_used)}_sep={separator}_hash={column_hash}"
        print(f"\n3. 🏷️  Column Signature: {column_signature}")
        print(f"   - Compact representation of column configuration")
        
        # Method 4: Reverse lookup simulation
        print(f"\n4. 🔍 Reverse Lookup Simulation:")
        
        # Create a "database" of common column sets
        common_column_sets = {
            create_column_metadata_hash(['email'], '|'): ['email'],
            create_column_metadata_hash(['email', 'name'], '|'): ['email', 'name'],
            create_column_metadata_hash(['email', 'name', 'age', 'department'], '|'): ['email', 'name', 'age', 'department'],
            create_column_metadata_hash(['id', 'email'], '|'): ['id', 'email'],
        }
        
        current_hash = create_column_metadata_hash(columns_used, separator)
        if current_hash in common_column_sets:
            print(f"   ✅ Found match: {common_column_sets[current_hash]}")
        else:
            print(f"   ❌ No match found for hash: {current_hash}")
            print(f"   💡 Would need to store: {columns_used}")

def demonstrate_production_introspection():
    """Demonstrate production-ready introspection with metadata storage"""
    print("\n=== PRODUCTION-READY INTROSPECTION ===\n")
    
    # Create sample data
    df = pd.DataFrame({
        'user_id': ['U001', 'U002', 'U003'],
        'email': ['alice@corp.com', 'bob@startup.io', 'charlie@agency.co'],
        'full_name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown'],
        'registration_date': pd.to_datetime(['2024-01-15', '2024-01-16', '2024-01-17'])
    })
    
    # Generate row IDs with comprehensive metadata
    result = generate_unique_row_ids(df, return_audit_trail=True, show_progress=False)
    if isinstance(result, dict):
        # Extract comprehensive metadata
        metadata = {
            'columns_used': result['column_selection']['selected_columns'],
            'selection_method': result['column_selection']['selection_method'],
            'configuration': result['audit_trail']['configuration'],
            'session_id': result['session_id'],
            'timestamp': result['audit_trail']['audit_metadata'].get('generation_timestamp', 'N/A'),
            'data_fingerprint': result['audit_trail']['data_fingerprint'],
            'column_hash': create_column_metadata_hash(
                result['column_selection']['selected_columns'],
                result['audit_trail']['configuration'].get('separator', '|')
            )
        }
        
        print("🏗️  PRODUCTION METADATA STORAGE:")
        print(json.dumps(metadata, indent=2, default=str))
        
        # Demonstrate lookup capabilities
        print(f"\n🔍 LOOKUP CAPABILITIES:")
        print(f"   - Column hash: {metadata['column_hash']}")
        print(f"   - Can recreate: {metadata['columns_used']}")
        print(f"   - With separator: '{metadata['configuration'].get('separator', '|')}'")
        print(f"   - Data fingerprint: {metadata['data_fingerprint']['structure_fingerprint'][:16]}...")
        
        # Show the row IDs generated
        result_df = result['result_dataframe']
        print(f"\n📋 GENERATED ROW IDS:")
        for i, row in result_df.iterrows():
            print(f"   Row {i+1}: {row['row_id'][:24]}... (using {len(metadata['columns_used'])} columns)")

def propose_enhancements():
    """Propose enhancements for better column introspection"""
    print("\n=== PROPOSED ENHANCEMENTS ===\n")
    
    enhancements = [
        {
            'title': '🏷️ Column Metadata Embedding',
            'description': 'Embed column metadata in row ID as prefix',
            'example': 'col4_fd450ad8f11094e6... (4 columns used)',
            'benefits': ['Immediate column count visibility', 'No external metadata needed'],
            'drawbacks': ['Slightly longer IDs', 'Still no specific column names']
        },
        {
            'title': '🗂️ Column Registry',
            'description': 'Maintain a registry of column sets with short IDs',
            'example': 'Registry: {A1: [email,name,age,dept], B2: [email,name]}',
            'benefits': ['Compact representation', 'Easy reverse lookup'],
            'drawbacks': ['Requires persistent storage', 'Registry management']
        },
        {
            'title': '📊 Enhanced Audit Trail',
            'description': 'Store detailed column metadata in audit trail',
            'example': 'Current implementation (already available!)',
            'benefits': ['Comprehensive tracking', 'Full reversibility'],
            'drawbacks': ['Requires return_audit_trail=True']
        },
        {
            'title': '🔍 Introspection Functions',
            'description': 'Add dedicated functions for column introspection',
            'example': 'get_columns_from_row_id(), analyze_column_usage()',
            'benefits': ['Purpose-built API', 'Easy integration'],
            'drawbacks': ['Additional API surface']
        }
    ]
    
    for i, enhancement in enumerate(enhancements, 1):
        print(f"{i}. {enhancement['title']}")
        print(f"   📝 {enhancement['description']}")
        print(f"   💡 Example: {enhancement['example']}")
        print(f"   ✅ Benefits: {', '.join(enhancement['benefits'])}")
        print(f"   ⚠️  Drawbacks: {', '.join(enhancement['drawbacks'])}")
        print()

if __name__ == "__main__":
    print("🔍 Row ID Generator - Column Introspection Demo")
    print("=" * 50)
    
    # Demonstrate current capabilities
    result = demonstrate_current_introspection()
    
    # Show enhanced capabilities
    demonstrate_enhanced_introspection()
    
    # Show production-ready approach
    demonstrate_production_introspection()
    
    # Propose future enhancements
    propose_enhancements()
    
    print("\n🎯 SUMMARY:")
    print("✅ Current: Audit trail provides complete column information")
    print("✅ Enhanced: Column hashes enable quick identification")
    print("✅ Production: Comprehensive metadata storage and lookup")
    print("🔮 Future: Additional introspection APIs and embedding options") 