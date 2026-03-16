"""
Demo: Data I/O Operations
=========================
This demo showcases all data input/output functions in dskit.
"""

from dskit import load, read_folder, save
import pandas as pd
import os

def demo_basic_loading():
    """Demo 1: Load data from various file formats"""
    print("=" * 60)
    print("DEMO 1: Loading Data from Different Formats")
    print("=" * 60)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Akash', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 40, 45],
        'salary': [50000, 60000, 70000, 80000, 90000]
    })
    
    # Save in different formats
    os.makedirs('temp_data', exist_ok=True)
    sample_data.to_csv('temp_data/sample.csv', index=False)
    sample_data.to_excel('temp_data/sample.xlsx', index=False)
    sample_data.to_json('temp_data/sample.json', orient='records')

    # Load from CSV
    print("\n📁 Loading CSV file...")
    df_csv = load('temp_data/sample.csv')
    print(f"✓ Loaded {len(df_csv)} rows from CSV")
    print(df_csv.head())

    # Load from Excel
    print("\n📊 Loading Excel file...")
    df_excel = load('temp_data/sample.xlsx')
    print(f"✓ Loaded {len(df_excel)} rows from Excel")

    # Load from JSON
    print("\n📋 Loading JSON file...")
    df_json = load('temp_data/sample.json')
    print(f"✓ Loaded {len(df_json)} rows from JSON")

    # Load from Parquet (Safely wrapped to prevent crashes)
    print("\n🗂️ Loading Parquet file...")
    try:
        # We attempt to save AND load here so we catch the missing engine error
        sample_data.to_parquet('temp_data/sample.parquet', index=False)
        df_parquet = load('temp_data/sample.parquet')
        print(f"✓ Loaded {len(df_parquet)} rows from Parquet")
    except ImportError:
        print("⚠️ Skipped Parquet demo: 'pyarrow' or 'fastparquet' not installed.")
    except Exception as e:
        print(f"⚠️ Skipped Parquet demo: {str(e)}")


def demo_folder_loading():
    """Demo 2: Batch load multiple files from folder"""
    print("\n" + "=" * 60)
    print("DEMO 2: Batch Loading from Folder")
    print("=" * 60)

    # Create multiple CSV files
    os.makedirs('temp_data/batch', exist_ok=True)

    for i in range(3):
        df = pd.DataFrame({
            'id': range(i*10, (i+1)*10),
            'value': range(100, 110)
        })
        df.to_csv(f'temp_data/batch/file_{i+1}.csv', index=False)

    print("\n📂 Loading all CSV files from folder...")
    dfs = read_folder('temp_data/batch', file_type='csv')
    print(f"✓ Loaded {len(dfs)} files")

    for i, df in enumerate(dfs, 1):
        print(f"  File {i}: {len(df)} rows")


def demo_save_operations():
    """Demo 3: Save data in various formats"""
    print("\n" + "=" * 60)
    print("DEMO 3: Saving Data")
    print("=" * 60)

    # Create sample data
    df = pd.DataFrame({
        'x': range(1, 6),
        'y': [10, 20, 30, 40, 50]
    })

    os.makedirs('temp_data/output', exist_ok=True)

    # Save as CSV
    print("\n💾 Saving as CSV...")
    save(df, 'temp_data/output/result.csv')
    print("✓ Saved to result.csv")

    # Save as Excel
    print("\n💾 Saving as Excel...")
    save(df, 'temp_data/output/result.xlsx')
    print("✓ Saved to result.xlsx")

    # Save as JSON
    print("\n💾 Saving as JSON...")
    save(df, 'temp_data/output/result.json')
    print("✓ Saved to result.json")

    # Save as Parquet (Safely wrapped to prevent crashes)
    print("\n💾 Saving as Parquet...")
    try:
        save(df, 'temp_data/output/result.parquet')
        print("✓ Saved to result.parquet")
    except ImportError:
        print("⚠️ Skipped Parquet save: 'pyarrow' or 'fastparquet' not installed.")
    except Exception as e:
        print(f"⚠️ Skipped Parquet save: {str(e)}")


def cleanup():
    """Clean up temporary files"""
    import shutil
    if os.path.exists('temp_data'):
        shutil.rmtree('temp_data')
        print("\n🧹 Cleaned up temporary files")


if __name__ == "__main__":
    print("\n" + "🚀" * 30)
    print("DATA I/O OPERATIONS DEMO".center(60))
    print("🚀" * 30 + "\n")

    try:
        demo_basic_loading()
        demo_folder_loading()
        demo_save_operations()
    finally:
        cleanup()

    print("\n" + "✅" * 30)
    print("ALL DEMOS COMPLETED".center(60))
    print("✅" * 30 + "\n")