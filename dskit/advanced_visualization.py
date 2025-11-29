import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    px = None
    go = None
    make_subplots = None

def plot_feature_importance(model, feature_names=None, top_n=20):
    """
    Plot feature importance for tree-based models.
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute.")
        return
    
    importances = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(importances))]
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()

def plot_target_distribution(df, target_col, task='classification'):
    """
    Plot target variable distribution.
    """
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found.")
        return
    
    plt.figure(figsize=(10, 6))
    
    if task == 'classification':
        value_counts = df[target_col].value_counts()
        plt.subplot(1, 2, 1)
        value_counts.plot(kind='bar')
        plt.title('Target Distribution (Count)')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
        plt.title('Target Distribution (Percentage)')
    else:
        plt.hist(df[target_col].dropna(), bins=30, alpha=0.7)
        plt.title('Target Distribution')
        plt.xlabel(target_col)
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def plot_feature_vs_target(df, feature_col, target_col, task='classification'):
    """
    Plot relationship between a feature and target.
    """
    if feature_col not in df.columns or target_col not in df.columns:
        print("Feature or target column not found.")
        return
    
    plt.figure(figsize=(12, 5))
    
    if task == 'classification':
        plt.subplot(1, 2, 1)
        for target_val in df[target_col].unique():
            subset = df[df[target_col] == target_val]
            plt.hist(subset[feature_col].dropna(), alpha=0.6, label=f'{target_col}={target_val}', bins=20)
        plt.xlabel(feature_col)
        plt.ylabel('Frequency')
        plt.title(f'{feature_col} by {target_col}')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, x=target_col, y=feature_col)
        plt.title(f'{feature_col} by {target_col} (Boxplot)')
    else:
        plt.scatter(df[feature_col], df[target_col], alpha=0.6)
        plt.xlabel(feature_col)
        plt.ylabel(target_col)
        plt.title(f'{feature_col} vs {target_col}')
        
        # Add trend line
        z = np.polyfit(df[feature_col].dropna(), df[target_col].dropna(), 1)
        p = np.poly1d(z)
        plt.plot(df[feature_col], p(df[feature_col]), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.show()

def plot_correlation_advanced(df, method='pearson', threshold=0.5):
    """
    Advanced correlation plot with clustering and filtering.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("No numeric columns for correlation.")
        return
    
    corr = numeric_df.corr(method=method)
    
    # Filter correlations above threshold
    mask = np.abs(corr) < threshold
    
    # Create clustered heatmap
    plt.figure(figsize=(12, 10))
    sns.clustermap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                   center=0, square=True, cbar_kws={'shrink': 0.8})
    plt.title(f'Correlation Heatmap ({method.title()}) - Above {threshold}')
    plt.show()

def plot_missing_patterns_advanced(df):
    """
    Advanced missing data visualization with patterns.
    """
    if df.isnull().sum().sum() == 0:
        print("No missing data to visualize.")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Missing data heatmap
    plt.subplot(2, 2, 1)
    sns.heatmap(df.isnull(), cbar=True, cmap='viridis')
    plt.title('Missing Data Heatmap')
    
    # Missing data bar chart
    plt.subplot(2, 2, 2)
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=True)
    missing_counts.plot(kind='barh')
    plt.title('Missing Data Count by Column')
    plt.xlabel('Missing Count')
    
    # Missing data percentage
    plt.subplot(2, 2, 3)
    missing_pct = (df.isnull().sum() / len(df) * 100)
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=True)
    missing_pct.plot(kind='barh', color='orange')
    plt.title('Missing Data Percentage by Column')
    plt.xlabel('Missing Percentage (%)')
    
    # Co-occurrence of missing values
    plt.subplot(2, 2, 4)
    missing_df = df.isnull()
    if len(missing_df.columns) > 1:
        corr_missing = missing_df.corr()
        sns.heatmap(corr_missing, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
        plt.title('Missing Data Co-occurrence')
    
    plt.tight_layout()
    plt.show()

def plot_outliers_advanced(df, method='iqr'):
    """
    Advanced outlier visualization.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("No numeric columns for outlier analysis.")
        return
    
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, col in enumerate(numeric_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Boxplot
        plt.boxplot(df[col].dropna())
        plt.title(f'{col} - Outliers')
        plt.ylabel(col)
        
        # Mark outliers
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        else:  # zscore
            mean = df[col].mean()
            std = df[col].std()
            z_scores = np.abs((df[col] - mean) / std)
            outliers = df[z_scores > 3][col]
        
        plt.text(0.02, 0.98, f'Outliers: {len(outliers)}', transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    plt.show()

def plot_interactive_scatter(df, x_col, y_col, color_col=None, size_col=None):
    """
    Create interactive scatter plot using Plotly.
    """
    if px is None:
        print("Plotly not installed. Please install it using 'pip install plotly'")
        return
    
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col,
                     hover_data=df.columns.tolist(),
                     title=f'{x_col} vs {y_col}')
    fig.show()

def plot_distribution_comparison(df, cols, by_group=None):
    """
    Compare distributions of multiple columns.
    """
    if len(cols) == 0:
        print("No columns provided.")
        return
    
    n_cols = min(3, len(cols))
    n_rows = (len(cols) + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, col in enumerate(cols):
        if col not in df.columns:
            continue
            
        plt.subplot(n_rows, n_cols, i + 1)
        
        if by_group and by_group in df.columns:
            for group in df[by_group].unique():
                subset = df[df[by_group] == group]
                plt.hist(subset[col].dropna(), alpha=0.6, label=f'{by_group}={group}', bins=20)
            plt.legend()
        else:
            plt.hist(df[col].dropna(), bins=30, alpha=0.7)
        
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()