import pandas as pd
from .cleaning import missing_summary, outlier_summary
from .visualization import plot_histograms, plot_correlation_heatmap, plot_missingness

def basic_stats(df):
    """
    Create a summary of essential statistics.
    """
    desc = df.describe(include='all').transpose()
    desc['missing_count'] = df.isnull().sum()
    desc['missing_percent'] = 100 * df.isnull().sum() / len(df)
    return desc

def quick_eda(df):
    """
    Generate a complete EDA report.
    """
    print("=== Basic Statistics ===")
    print(basic_stats(df))
    print("\n=== Missing Values Summary ===")
    print(missing_summary(df))
    print("\n=== Outlier Summary ===")
    print(outlier_summary(df))
    
    print("\nGenerating Plots...")
    try:
        plot_missingness(df)
        plot_histograms(df)
        plot_correlation_heatmap(df)
    except Exception as e:
        print(f"Error generating plots: {e}")
