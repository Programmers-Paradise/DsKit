"""
Demo: Exploratory Data Analysis (EDA)
=====================================
This demo showcases EDA functions in dskit.
"""

from dskit import basic_stats, quick_eda, comprehensive_eda, data_health_check, feature_analysis_report
import pandas as pd
import numpy as np

def create_sample_data():
    """Create sample dataset for EDA"""
    np.random.seed(42)
    return pd.DataFrame({
        'age': np.random.randint(18, 70, 100),
        'salary': np.random.randint(30000, 150000, 100),
        'experience': np.random.randint(0, 30, 100),
        'department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing'], 100),
        'performance': np.random.choice(['Low', 'Medium', 'High'], 100),
        'satisfaction': np.random.uniform(1, 10, 100).round(2)
    })


def demo_basic_stats():
    """Demo 1: Basic statistics"""
    print("=" * 60)
    print("DEMO 1: Basic Statistics")
    print("=" * 60)
    
    df = create_sample_data()
    
    print("\n📊 Computing basic statistics...")
    stats = basic_stats(df)
    print("\n✓ Basic Statistics:")
    print(stats)


def demo_quick_eda():
    """Demo 2: Quick EDA overview"""
    print("\n" + "=" * 60)
    print("DEMO 2: Quick EDA Overview")
    print("=" * 60)
    
    df = create_sample_data()
    
    print("\n📊 Running quick EDA...")
    quick_eda(df)
    print("\n✓ Quick EDA completed")


def demo_comprehensive_eda():
    """Demo 3: Comprehensive EDA with target"""
    print("\n" + "=" * 60)
    print("DEMO 3: Comprehensive EDA")
    print("=" * 60)
    
    df = create_sample_data()
    
    print("\n📊 Running comprehensive EDA with target column...")
    print("   (This generates detailed analysis and visualizations)")
    
    # Note: This function generates plots, so it's best run interactively
    try:
        comprehensive_eda(df, target_col='performance')
        print("\n✓ Comprehensive EDA completed")
    except Exception as e:
        print(f"\n⚠️ Note: {str(e)}")
        print("   (Some visualizations may require display environment)")


def demo_health_check():
    """Demo 4: Data health check"""
    print("\n" + "=" * 60)
    print("DEMO 4: Data Health Check")
    print("=" * 60)
    
    # Create data with some issues
    df = create_sample_data()
    # Introduce some issues
    df.loc[0:5, 'age'] = np.nan
    df.loc[10:15, 'salary'] = 1000000  # Outliers
    
    print("\n📊 Running data health check...")
    report = data_health_check(df)
    
    print("\n✓ Health Check Report:")

    # --- FIX START: Logic Bug Prevention ---
    # The data_health_check function might return a float (score) instead of a dictionary.
    # We check the type before trying to access keys to prevent an AttributeError.
    if isinstance(report, dict):
        print(f"  Missing Values: {report.get('missing_values', {})}")
        print(f"  Data Types: {report.get('dtypes', {})}")
        print(f"  Shape: {report.get('shape', {})}")
    else:
        # Fallback if it returns a score (float/int)
        print(f"  Overall Health Score: {report}")
        print("  (Detailed dictionary report was not returned by the function)")
    # --- FIX END ---


def demo_feature_analysis():
    """Demo 5: Feature analysis report"""
    print("\n" + "=" * 60)
    print("DEMO 5: Feature Analysis Report")
    print("=" * 60)

    df = create_sample_data()

    print("\n📊 Generating feature analysis report...")
    try:
        report = feature_analysis_report(df, target_col='performance')
        print("\n✓ Feature analysis completed")
        print(f"   Analyzed {len(df.columns)} features")
    except Exception as e:
        print(f"\n⚠️ Note: {str(e)}")


if __name__ == "__main__":
    print("\n" + "📊" * 30)
    print("EXPLORATORY DATA ANALYSIS DEMO".center(60))
    print("📊" * 30 + "\n")

    demo_basic_stats()
    demo_quick_eda()
    demo_comprehensive_eda()
    demo_health_check()
    demo_feature_analysis()

    print("\n" + "✅" * 30)
    print("ALL DEMOS COMPLETED".center(60))
    print("✅" * 30 + "\n")