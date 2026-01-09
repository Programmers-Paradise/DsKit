import pandas as pd
import numpy as np
import re

def fix_dtypes(df):
    """
    Auto-detects and converts column types.
    """
    df = df.copy()
    for col in df.columns:
        # Try converting to numeric
        try:
            df[col] = pd.to_numeric(df[col])
            continue
        except (ValueError, TypeError):
            pass

        # Try converting to datetime
        try:
            df[col] = pd.to_datetime(df[col])
            continue
        except (ValueError, TypeError):
            pass
            
        # Convert object to category if low cardinality
        if df[col].dtype == 'object':
            if df[col].nunique() / len(df) < 0.5: # Heuristic
                df[col] = df[col].astype('category')
                
    return df

def rename_columns_auto(df):
    """
    Cleans column names: lowercase, replace spaces with underscores, remove special chars.
    """
    df = df.copy()
    new_cols = []
    for col in df.columns:
        clean_col = str(col).strip().lower()
        clean_col = clean_col.replace(' ', '_')
        clean_col = re.sub(r'[^a-z0-9_]', '', clean_col)
        new_cols.append(clean_col)
    df.columns = new_cols
    return df

def replace_specials(df, chars_to_remove=r'[@#%$]', replacement=''):
    """
    Removes or replaces special characters from text columns.
    """
    df = df.copy()
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str).str.replace(chars_to_remove, replacement, regex=True)
    return df

def missing_summary(df):
    """
    Returns a summary of missing values.
    """
    missing = df.isnull().sum()
    missing_percent = 100 * df.isnull().sum() / len(df)
    summary = pd.concat([missing, missing_percent], axis=1, keys=['Missing Count', 'Missing %'])
    return summary[summary['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)

def fill_missing(df, strategy='auto', fill_value=None):
    """
    Fills missing values.
    strategy: 'auto', 'mean', 'median', 'mode', 'ffill', 'bfill', 'constant'
    """
    df = df.copy()
    
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
            
        col_strategy = strategy
        if strategy == 'auto':
            if pd.api.types.is_numeric_dtype(df[col]):
                col_strategy = 'mean'
            else:
                col_strategy = 'mode'
        
        if col_strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        elif col_strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif col_strategy == 'mode':
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
        elif col_strategy == 'ffill':
            df[col] = df[col].fillna(method='ffill')
        elif col_strategy == 'bfill':
            df[col] = df[col].fillna(method='bfill')
        elif col_strategy == 'constant' and fill_value is not None:
            df[col] = df[col].fillna(fill_value)
            
    return df

def outlier_summary(df, method='iqr', threshold=1.5):
    """
    Returns a summary of outliers.
    """
    summary = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (threshold * IQR)
            upper_bound = Q3 + (threshold * IQR)
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:
                continue
            z_scores = (df[col] - mean) / std
            outliers = df[np.abs(z_scores) > 3]
        
        if not outliers.empty:
            summary[col] = len(outliers)
            
    return pd.Series(summary, name='Outlier Count').sort_values(ascending=False)

def remove_outliers(df, method='iqr', threshold=1.5):
    """
    Removes rows with outliers.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (threshold * IQR)
            upper_bound = Q3 + (threshold * IQR)
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            if std==0:
                continue
            z_scores = (df[col] - mean) / std
            df = df[np.abs(z_scores) <= 3]
            
    return df

def simple_nlp_clean(df, text_cols=None):
    """
    Basic text cleaning: lowercase, remove punctuation, remove extra spaces.
    """
    df = df.copy()
    if text_cols is None:
        text_cols = df.select_dtypes(include=['object', 'string']).columns
        
    for col in text_cols:
        # Lowercase
        df[col] = df[col].astype(str).str.lower()
        # Remove punctuation
        df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
        # Remove extra spaces
        df[col] = df[col].str.replace(r'\s+', ' ', regex=True).str.strip()
        
    return df
