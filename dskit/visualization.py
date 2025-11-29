import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_missingness(df):
    """
    Visualizes missing data patterns.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Data Heatmap")
    plt.show()

def plot_histograms(df, bins=30):
    """
    Plots histograms for all numeric features.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("No numeric columns to plot.")
        return

    df[numeric_cols].hist(bins=bins, figsize=(15, 10), layout=(len(numeric_cols)//3 + 1, 3))
    plt.tight_layout()
    plt.show()

def plot_boxplots(df):
    """
    Plots boxplots for numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("No numeric columns to plot.")
        return

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols):
        plt.subplot(len(numeric_cols)//3 + 1, 3, i+1)
        sns.boxplot(y=df[col])
        plt.title(col)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df):
    """
    Plots correlation heatmap for numeric variables.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("No numeric columns for correlation.")
        return

    plt.figure(figsize=(12, 8))
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

def plot_pairplot(df, hue=None):
    """
    Generates a pairplot.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for pairplot.")
        return
        
    sns.pairplot(df, hue=hue)
    plt.show()
