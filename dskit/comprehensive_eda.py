import pandas as pd
import numpy as np
import warnings
from .cleaning import missing_summary, outlier_summary
from .visualization import plot_histograms, plot_correlation_heatmap, plot_missingness
from .advanced_visualization import (plot_missing_patterns_advanced, plot_outliers_advanced, 
                                   plot_target_distribution, plot_feature_vs_target)
try:
    import pandas_profiling as pp
except ImportError:
    try:
        from ydata_profiling import ProfileReport as pp
    except ImportError:
        pp = None

def comprehensive_eda(df, target_col=None, sample_size=None):
    """
    Comprehensive EDA with detailed analysis and visualizations.
    """
    # Sample data if too large
    if sample_size and len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
        print(f"Using sample of {sample_size} rows for analysis.")
    else:
        df_sample = df
    
    # Initialize task variable
    task = None
    
    print("=" * 60)
    print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Basic Info
    print("\n1. DATASET OVERVIEW")
    print("-" * 30)
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    
    # Missing Values Analysis
    print("\n2. MISSING VALUES ANALYSIS")
    print("-" * 30)
    missing_info = missing_summary(df_sample)
    if not missing_info.empty:
        print(missing_info)
        print("\nMissing data visualizations:")
        plot_missing_patterns_advanced(df_sample)
    else:
        print("No missing values found!")
    
    # Data Quality Issues
    print("\n3. DATA QUALITY ISSUES")
    print("-" * 30)
    
    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    # Constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        print(f"Constant columns: {constant_cols}")
    
    # High cardinality columns
    high_card_cols = [col for col in df.select_dtypes(include=['object']).columns 
                     if df[col].nunique() > 0.9 * len(df)]
    if high_card_cols:
        print(f"High cardinality columns (>90% unique): {high_card_cols}")
    
    # Numeric Statistics
    print("\n4. NUMERIC FEATURES ANALYSIS")
    print("-" * 30)
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df_sample[numeric_cols].describe())
        
        # Outliers
        print("\nOutlier Analysis:")
        outlier_info = outlier_summary(df_sample)
        if not outlier_info.empty:
            print(outlier_info)
            plot_outliers_advanced(df_sample)
        
        # Distributions
        print("\nDistribution Analysis:")
        plot_histograms(df_sample)
        
        # Correlations
        print("\nCorrelation Analysis:")
        plot_correlation_heatmap(df_sample)
    
    # Categorical Features
    print("\n5. CATEGORICAL FEATURES ANALYSIS")
    print("-" * 30)
    cat_cols = df_sample.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        print(f"\n{col}:")
        print(f"  Unique values: {df_sample[col].nunique()}")
        print(f"  Top 5 values:")
        print(df_sample[col].value_counts().head().to_dict())
    
    # Target Analysis (if provided)
    if target_col and target_col in df.columns:
        print(f"\n6. TARGET VARIABLE ANALYSIS ({target_col})")
        print("-" * 30)
        
        # Target distribution
        if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() < 20:
            task = 'classification'
            print("Target distribution:")
            print(df[target_col].value_counts())
            plot_target_distribution(df_sample, target_col, task)
        else:
            task = 'regression'
            print(f"Target statistics:")
            print(df[target_col].describe())
            plot_target_distribution(df_sample, target_col, task)
        
        # Feature vs Target relationships
        print("\nFeature vs Target Relationships:")
        for col in numeric_cols[:5]:  # Top 5 numeric features
            if col != target_col:
                plot_feature_vs_target(df_sample, col, target_col, task)
    
    # Data Insights
    print("\n7. KEY INSIGHTS & RECOMMENDATIONS")
    print("-" * 30)
    
    insights = []
    
    if not missing_info.empty:
        high_missing = missing_info[missing_info['Missing %'] > 30]
        if not high_missing.empty:
            insights.append(f"⚠️  High missing values in: {list(high_missing.index)}")
    
    if duplicates > 0:
        insights.append(f"⚠️  Found {duplicates} duplicate rows - consider removing")
    
    if constant_cols:
        insights.append(f"⚠️  Constant columns found: {constant_cols} - consider removing")
    
    if high_card_cols:
        insights.append(f"⚠️  High cardinality columns: {high_card_cols} - consider encoding strategies")
    
    if len(numeric_cols) > 0:
        # Check for skewed distributions
        skewed_cols = []
        for col in numeric_cols:
            if abs(df_sample[col].skew()) > 2:
                skewed_cols.append(col)
        if skewed_cols:
            insights.append(f"📊 Highly skewed columns: {skewed_cols} - consider transformations")
    
    if target_col and task == 'classification':
        # Check for class imbalance
        target_dist = df[target_col].value_counts(normalize=True)
        if target_dist.min() < 0.05:  # Less than 5% of smallest class
            insights.append("⚖️  Class imbalance detected - consider resampling techniques")
    
    if not insights:
        insights.append("✅ Data looks relatively clean!")
    
    for insight in insights:
        print(insight)
    
    print(f"\n{'='*60}")
    print("EDA COMPLETE")
    print(f"{'='*60}")

def generate_pandas_profile(df, output_file='profile_report.html'):
    """
    Generate automated EDA report using pandas-profiling.
    """
    if pp is None:
        print("pandas-profiling not installed. Please install it using:")
        print("pip install pandas-profiling")
        return
    
    print("Generating comprehensive profile report...")
    
    if hasattr(pp, 'ProfileReport'):
        # ydata-profiling format
        profile = pp(df, title="Dataset Profile Report", explorative=True)
    else:
        # pandas-profiling format
        profile = pp.ProfileReport(df, title="Dataset Profile Report", explorative=True)
    
    profile.to_file(output_file)
    print(f"Profile report saved as '{output_file}'")
    
    return profile

def data_health_check(df):
    """
    Quick data health check with scores.
    """
    print("DATA HEALTH CHECK")
    print("=" * 30)
    
    total_score = 0
    max_score = 0
    
    # Missing values score (0-20)
    max_score += 20
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    if missing_pct == 0:
        missing_score = 20
    elif missing_pct < 5:
        missing_score = 15
    elif missing_pct < 15:
        missing_score = 10
    elif missing_pct < 30:
        missing_score = 5
    else:
        missing_score = 0
    
    total_score += missing_score
    print(f"Missing Values: {missing_pct:.1f}% - Score: {missing_score}/20")
    
    # Duplicates score (0-15)
    max_score += 15
    duplicate_pct = (df.duplicated().sum() / len(df)) * 100
    if duplicate_pct == 0:
        duplicate_score = 15
    elif duplicate_pct < 1:
        duplicate_score = 12
    elif duplicate_pct < 5:
        duplicate_score = 8
    elif duplicate_pct < 10:
        duplicate_score = 4
    else:
        duplicate_score = 0
    
    total_score += duplicate_score
    print(f"Duplicates: {duplicate_pct:.1f}% - Score: {duplicate_score}/15")
    
    # Data types score (0-15)
    max_score += 15
    total_cols = len(df.columns)
    object_cols = len(df.select_dtypes(include=['object']).columns)
    type_score = max(0, 15 - (object_cols / total_cols * 10))
    
    total_score += type_score
    print(f"Data Types: {object_cols}/{total_cols} object columns - Score: {type_score:.0f}/15")
    
    # Consistency score (0-25)
    max_score += 25
    consistency_issues = 0
    
    # Check for consistent naming
    inconsistent_naming = sum(1 for col in df.columns if ' ' in col or col != col.lower())
    consistency_issues += inconsistent_naming
    
    # Check for constant columns
    constant_cols = sum(1 for col in df.columns if df[col].nunique() <= 1)
    consistency_issues += constant_cols
    
    consistency_score = max(0, 25 - consistency_issues * 5)
    total_score += consistency_score
    print(f"Consistency Issues: {consistency_issues} - Score: {consistency_score}/25")
    
    # Outliers score (0-25)
    max_score += 25
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        outlier_cols = 0
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
            if len(outliers) > 0.05 * len(df):  # More than 5% outliers
                outlier_cols += 1
        
        outlier_score = max(0, 25 - (outlier_cols / len(numeric_cols) * 25))
    else:
        outlier_score = 25  # No numeric columns, no outlier issues
    
    total_score += outlier_score
    print(f"Outliers: {outlier_cols}/{len(numeric_cols)} columns with significant outliers - Score: {outlier_score:.0f}/25")
    
    # Final score
    final_score = (total_score / max_score) * 100
    print(f"\nOVERALL DATA HEALTH SCORE: {final_score:.1f}/100")
    
    if final_score >= 90:
        grade = "A+ (Excellent)"
    elif final_score >= 80:
        grade = "A (Very Good)"
    elif final_score >= 70:
        grade = "B (Good)"
    elif final_score >= 60:
        grade = "C (Acceptable)"
    elif final_score >= 50:
        grade = "D (Poor)"
    else:
        grade = "F (Very Poor)"
    
    print(f"GRADE: {grade}")
    
    return final_score

def feature_analysis_report(df, target_col=None):
    """
    Detailed feature analysis report.
    """
    print("FEATURE ANALYSIS REPORT")
    print("=" * 40)
    
    # Feature summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns
    
    print(f"Total Features: {len(df.columns)}")
    print(f"  - Numeric: {len(numeric_cols)}")
    print(f"  - Categorical: {len(cat_cols)}")
    print(f"  - Datetime: {len(datetime_cols)}")
    
    # Feature quality analysis
    feature_quality = pd.DataFrame(index=df.columns)
    feature_quality['Type'] = df.dtypes
    feature_quality['Unique_Values'] = df.nunique()
    feature_quality['Unique_Ratio'] = df.nunique() / len(df)
    feature_quality['Missing_Count'] = df.isnull().sum()
    feature_quality['Missing_Ratio'] = df.isnull().sum() / len(df)
    
    # Information content score
    feature_quality['Info_Score'] = 0
    for col in df.columns:
        score = 10  # Base score
        
        # Penalize high missing values
        missing_ratio = feature_quality.loc[col, 'Missing_Ratio']
        if missing_ratio > 0.5:
            score -= 5
        elif missing_ratio > 0.2:
            score -= 2
        
        # Penalize constant or near-constant features
        unique_ratio = feature_quality.loc[col, 'Unique_Ratio']
        if unique_ratio < 0.01:
            score -= 5
        elif unique_ratio < 0.05:
            score -= 2
        
        # Bonus for numeric features
        if col in numeric_cols:
            score += 2
        
        feature_quality.loc[col, 'Info_Score'] = max(0, score)
    
    print(f"\nTop 10 Features by Information Score:")
    top_features = feature_quality.sort_values('Info_Score', ascending=False).head(10)
    print(top_features[['Type', 'Unique_Values', 'Missing_Ratio', 'Info_Score']])
    
    print(f"\nBottom 5 Features (Consider Removing):")
    bottom_features = feature_quality.sort_values('Info_Score', ascending=True).head(5)
    print(bottom_features[['Type', 'Unique_Values', 'Missing_Ratio', 'Info_Score']])
    
    return feature_quality