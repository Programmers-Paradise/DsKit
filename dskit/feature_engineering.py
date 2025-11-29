import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
import warnings

def create_polynomial_features(df, degree=2, interaction_only=False, include_bias=False):
    """
    Creates polynomial and interaction features.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("No numeric columns found for polynomial features.")
        return df
    
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, 
                              include_bias=include_bias)
    
    # Apply only to numeric columns
    numeric_data = df[numeric_cols]
    poly_features = poly.fit_transform(numeric_data)
    
    # Create feature names
    feature_names = poly.get_feature_names_out(numeric_cols)
    
    # Create new dataframe with polynomial features
    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
    
    # Combine with non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        result_df = pd.concat([poly_df, df[non_numeric_cols]], axis=1)
    else:
        result_df = poly_df
    
    return result_df

def create_date_features(df, date_cols=None):
    """
    Extracts date features like year, month, day, weekday, etc.
    """
    df = df.copy()
    
    if date_cols is None:
        date_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns
    
    for col in date_cols:
        if col not in df.columns:
            continue
            
        # Convert to datetime if not already
        df[col] = pd.to_datetime(df[col])
        
        # Extract features
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
        df[f'{col}_weekday'] = df[col].dt.weekday
        df[f'{col}_quarter'] = df[col].dt.quarter
        df[f'{col}_is_weekend'] = (df[col].dt.weekday >= 5).astype(int)
        
        # Drop original date column
        df = df.drop(col, axis=1)
    
    return df

def create_binning_features(df, numeric_cols=None, n_bins=5, strategy='quantile'):
    """
    Creates binned versions of numeric features.
    """
    df = df.copy()
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
            
        try:
            df[f'{col}_binned'] = pd.cut(df[col], bins=n_bins, labels=False)
        except Exception as e:
            print(f"Could not bin column {col}: {e}")
    
    return df

def select_features_univariate(X, y, k=10, task='classification'):
    """
    Selects top k features using univariate statistical tests.
    """
    if task == 'classification':
        score_func = f_classif
    else:
        score_func = f_regression
    
    selector = SelectKBest(score_func=score_func, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    if hasattr(X, 'columns'):
        selected_features = X.columns[selector.get_support()].tolist()
    else:
        selected_features = [f'feature_{i}' for i in range(X_selected.shape[1])]
    
    return X_selected, selected_features, selector

def select_features_rfe(X, y, n_features=10, task='classification'):
    """
    Recursive Feature Elimination for feature selection.
    """
    if task == 'classification':
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    else:
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
    
    selector = RFE(estimator, n_features_to_select=n_features)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    if hasattr(X, 'columns'):
        selected_features = X.columns[selector.get_support()].tolist()
    else:
        selected_features = [f'feature_{i}' for i in range(X_selected.shape[1])]
    
    return X_selected, selected_features, selector

def apply_pca(df, n_components=None, variance_threshold=0.95):
    """
    Applies Principal Component Analysis for dimensionality reduction.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("No numeric columns found for PCA.")
        return df
    
    # Determine number of components
    if n_components is None:
        pca_temp = PCA()
        pca_temp.fit(df[numeric_cols])
        cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= variance_threshold) + 1
    
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(df[numeric_cols])
    
    # Create new dataframe
    pca_cols = [f'PC_{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(pca_features, columns=pca_cols, index=df.index)
    
    # Combine with non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        result_df = pd.concat([pca_df, df[non_numeric_cols]], axis=1)
    else:
        result_df = pca_df
    
    print(f"PCA reduced {len(numeric_cols)} features to {n_components} components")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_[:5]}...")  # Show first 5
    
    return result_df, pca

def create_aggregation_features(df, group_col, agg_cols=None, agg_funcs=['mean', 'std', 'min', 'max']):
    """
    Creates aggregation features grouped by a categorical column.
    """
    df = df.copy()
    
    if group_col not in df.columns:
        print(f"Group column '{group_col}' not found.")
        return df
    
    if agg_cols is None:
        agg_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove group_col from agg_cols if present
    agg_cols = [col for col in agg_cols if col != group_col]
    
    if len(agg_cols) == 0:
        print("No numeric columns found for aggregation.")
        return df
    
    # Create aggregations
    for col in agg_cols:
        for func in agg_funcs:
            agg_name = f'{col}_{func}_by_{group_col}'
            try:
                df[agg_name] = df.groupby(group_col)[col].transform(func)
            except Exception as e:
                print(f"Could not create aggregation {agg_name}: {e}")
    
    return df

def create_target_encoding(df, categorical_cols, target_col, smoothing=1.0):
    """
    Creates target-encoded features for categorical variables.
    """
    df = df.copy()
    
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found.")
        return df
    
    global_mean = df[target_col].mean()
    
    for col in categorical_cols:
        if col not in df.columns:
            continue
            
        # Calculate target mean for each category
        target_means = df.groupby(col)[target_col].agg(['mean', 'count'])
        
        # Apply smoothing
        smoothed_means = (target_means['mean'] * target_means['count'] + 
                         global_mean * smoothing) / (target_means['count'] + smoothing)
        
        # Map back to dataframe
        df[f'{col}_target_encoded'] = df[col].map(smoothed_means).fillna(global_mean)
    
    return df