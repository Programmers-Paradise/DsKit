"""
Demo: Data Validation & Profiling
=================================
This demo showcases data validation and profiling functions in dskit.
"""

from dskit import validate_schema, duplicate_summary, basic_profile
import pandas as pd
import numpy as np


def create_sample_data():
    """Create sample dataset for validation demos"""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'user_id': list(range(n)),
        'name': [f'User_{i}' for i in range(n)],
        'age': np.random.randint(18, 70, n),
        'income': np.random.uniform(25000, 150000, n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'score': np.where(np.random.random(n) > 0.9, np.nan, np.random.uniform(0, 100, n))
    })


def create_duplicate_data():
    """Create dataset with duplicates for demo"""
    return pd.DataFrame({
        'id': [1, 2, 2, 3, 4, 4, 4, 5],
        'value': ['a', 'b', 'b', 'c', 'd', 'd', 'd', 'e'],
        'score': [10, 20, 20, 30, 40, 40, 40, 50]
    })


def demo_validate_schema():
    """Demo 1: Schema Validation"""
    print("=" * 60)
    print("DEMO 1: Schema Validation")
    print("=" * 60)
    
    df = create_sample_data()
    
    print("\n📊 Dataset columns:", list(df.columns))
    print(f"   Shape: {df.shape}")
    
    print("\n🔧 Validating schema...")
    print("   Required columns: ['user_id', 'name', 'email']")
    print("   Expected dtypes: {'age': 'int', 'income': 'int'}")
    
    result = validate_schema(
        df,
        required_columns=['user_id', 'name', 'email'],  # 'email' doesn't exist
        expected_dtypes={'age': 'int', 'income': 'int'}  # income is float
    )
    
    print(f"\n✓ Validation Results:")
    print(f"   Is Valid: {result['is_valid']}")
    print(f"   Missing Columns: {result['missing_columns']}")
    print(f"   Dtype Mismatches: {result['dtype_mismatches']}")


def demo_duplicate_summary():
    """Demo 2: Duplicate Summary"""
    print("\n" + "=" * 60)
    print("DEMO 2: Duplicate Summary")
    print("=" * 60)
    
    df = create_duplicate_data()
    
    print("\n📊 Sample data with duplicates:")
    print(df)
    
    print("\n🔧 Analyzing duplicates (all columns)...")
    result = duplicate_summary(df)
    
    print(f"\n✓ Duplicate Analysis:")
    print(f"   Total Rows: {result['total_rows']}")
    print(f"   Duplicates: {result['total_duplicates']}")
    print(f"   Duplicate %: {result['duplicate_percentage']}%")
    print(f"   Unique Rows: {result['unique_rows']}")
    
    print("\n🔧 Analyzing duplicates by 'id' column only...")
    result_by_id = duplicate_summary(df, subset=['id'])
    
    print(f"\n✓ Duplicate Analysis (by ID):")
    print(f"   Duplicates: {result_by_id['total_duplicates']}")
    print(f"   By Subset: {result_by_id['by_subset']}")


def demo_basic_profile():
    """Demo 3: Basic Profile"""
    print("\n" + "=" * 60)
    print("DEMO 3: Basic Data Profile")
    print("=" * 60)
    
    df = create_sample_data()
    
    print("\n🔧 Generating data profile...")
    profile = basic_profile(df)
    
    print(f"\n✓ Data Profile:")
    print(f"   Rows: {profile['rows']}")
    print(f"   Columns: {profile['cols']}")
    print(f"   Memory: {profile['memory_usage_mb']:.4f} MB")
    print(f"   Missing Cells: {profile['missing_cells']}")
    print(f"   Missing %: {profile['missing_percentage']}%")
    print(f"   Dtypes: {profile['dtypes']}")
    
    print("\n📊 Numeric Summary (age column):")
    if 'age' in profile['numeric_summary']:
        age_stats = profile['numeric_summary']['age']
        print(f"   Mean: {age_stats['mean']:.2f}")
        print(f"   Std: {age_stats['std']:.2f}")
        print(f"   Min: {age_stats['min']:.2f}")
        print(f"   Max: {age_stats['max']:.2f}")


if __name__ == "__main__":
    print("\n" + "📋" * 30)
    print("DATA VALIDATION DEMO".center(60))
    print("📋" * 30 + "\n")
    
    demo_validate_schema()
    demo_duplicate_summary()
    demo_basic_profile()
    
    print("\n" + "✅" * 30)
    print("ALL DEMOS COMPLETED".center(60))
    print("✅" * 30 + "\n")
