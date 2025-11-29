import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

class TimeSeriesUtils:
    """
    Utilities for time series data analysis and feature engineering.
    """
    
    def __init__(self):
        self.is_fitted = False
        self.date_column = None
        self.freq = None
    
    def detect_time_column(self, df):
        """Automatically detect time/date columns."""
        time_columns = []
        
        for col in df.columns:
            if df[col].dtype in ['datetime64[ns]', 'datetime64']:
                time_columns.append(col)
            elif df[col].dtype == 'object':
                # Try to parse as datetime
                sample = df[col].dropna().head(100)
                try:
                    pd.to_datetime(sample, infer_datetime_format=True)
                    time_columns.append(col)
                except (ValueError, TypeError):
                    continue
        
        return time_columns
    
    def create_time_features(self, df, date_col, drop_original=False):
        """Create comprehensive time-based features."""
        df = df.copy()
        
        # Ensure datetime format
        if df[date_col].dtype != 'datetime64[ns]':
            df[date_col] = pd.to_datetime(df[date_col])
        
        dt = df[date_col].dt
        
        # Basic time features
        df[f'{date_col}_year'] = dt.year
        df[f'{date_col}_month'] = dt.month
        df[f'{date_col}_day'] = dt.day
        df[f'{date_col}_dayofweek'] = dt.dayofweek
        df[f'{date_col}_dayofyear'] = dt.dayofyear
        df[f'{date_col}_week'] = dt.isocalendar().week
        df[f'{date_col}_quarter'] = dt.quarter
        df[f'{date_col}_hour'] = dt.hour
        df[f'{date_col}_minute'] = dt.minute
        
        # Derived features
        df[f'{date_col}_is_weekend'] = (dt.dayofweek >= 5).astype(int)
        df[f'{date_col}_is_month_start'] = dt.is_month_start.astype(int)
        df[f'{date_col}_is_month_end'] = dt.is_month_end.astype(int)
        df[f'{date_col}_is_quarter_start'] = dt.is_quarter_start.astype(int)
        df[f'{date_col}_is_quarter_end'] = dt.is_quarter_end.astype(int)
        df[f'{date_col}_is_year_start'] = dt.is_year_start.astype(int)
        df[f'{date_col}_is_year_end'] = dt.is_year_end.astype(int)
        
        # Cyclical features (useful for ML)
        df[f'{date_col}_month_sin'] = np.sin(2 * np.pi * dt.month / 12)
        df[f'{date_col}_month_cos'] = np.cos(2 * np.pi * dt.month / 12)
        df[f'{date_col}_day_sin'] = np.sin(2 * np.pi * dt.day / 31)
        df[f'{date_col}_day_cos'] = np.cos(2 * np.pi * dt.day / 31)
        df[f'{date_col}_dayofweek_sin'] = np.sin(2 * np.pi * dt.dayofweek / 7)
        df[f'{date_col}_dayofweek_cos'] = np.cos(2 * np.pi * dt.dayofweek / 7)
        
        # Season encoding
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        df[f'{date_col}_season'] = dt.month.apply(get_season)
        
        # Time since epoch (for trend analysis)
        df[f'{date_col}_timestamp'] = df[date_col].astype(np.int64) // 10**9
        
        if drop_original:
            df = df.drop(columns=[date_col])
        
        return df
    
    def create_lag_features(self, df, value_col, date_col, lags=[1, 2, 3, 7, 30]):
        """Create lag features for time series."""
        df = df.copy()
        df = df.sort_values(date_col)
        
        for lag in lags:
            df[f'{value_col}_lag_{lag}'] = df[value_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df, value_col, date_col, windows=[7, 14, 30]):
        """Create rolling window features."""
        df = df.copy()
        df = df.sort_values(date_col)
        
        for window in windows:
            df[f'{value_col}_rolling_mean_{window}'] = df[value_col].rolling(window=window).mean()
            df[f'{value_col}_rolling_std_{window}'] = df[value_col].rolling(window=window).std()
            df[f'{value_col}_rolling_min_{window}'] = df[value_col].rolling(window=window).min()
            df[f'{value_col}_rolling_max_{window}'] = df[value_col].rolling(window=window).max()
            df[f'{value_col}_rolling_median_{window}'] = df[value_col].rolling(window=window).median()
        
        return df
    
    def create_expanding_features(self, df, value_col, date_col):
        """Create expanding window features."""
        df = df.copy()
        df = df.sort_values(date_col)
        
        df[f'{value_col}_expanding_mean'] = df[value_col].expanding().mean()
        df[f'{value_col}_expanding_std'] = df[value_col].expanding().std()
        df[f'{value_col}_expanding_min'] = df[value_col].expanding().min()
        df[f'{value_col}_expanding_max'] = df[value_col].expanding().max()
        
        return df
    
    def detect_seasonality(self, series, freq='D'):
        """Detect seasonality patterns in time series."""
        try:
            from scipy import signal
            from scipy.fft import fft, fftfreq
            
            # Remove NaN values
            clean_series = series.dropna()
            if len(clean_series) < 50:
                return {'has_seasonality': False, 'reason': 'insufficient_data'}
            
            # Detrend the series
            detrended = signal.detrend(clean_series.values)
            
            # Perform FFT
            fft_values = fft(detrended)
            frequencies = fftfreq(len(detrended))
            
            # Find dominant frequencies
            power_spectrum = np.abs(fft_values) ** 2
            peak_indices = signal.find_peaks(power_spectrum[1:len(power_spectrum)//2])[0] + 1
            
            if len(peak_indices) > 0:
                dominant_freq = frequencies[peak_indices[np.argmax(power_spectrum[peak_indices])]]
                period = 1 / abs(dominant_freq) if dominant_freq != 0 else None
                
                return {
                    'has_seasonality': True,
                    'dominant_period': period,
                    'strength': float(np.max(power_spectrum[peak_indices]) / np.sum(power_spectrum))
                }
            else:
                return {'has_seasonality': False, 'reason': 'no_dominant_peaks'}
                
        except ImportError:
            return {'has_seasonality': False, 'reason': 'scipy_not_available'}
        except Exception as e:
            return {'has_seasonality': False, 'reason': str(e)}
    
    def create_difference_features(self, df, value_col, date_col, periods=[1, 7, 30]):
        """Create differencing features."""
        df = df.copy()
        df = df.sort_values(date_col)
        
        for period in periods:
            df[f'{value_col}_diff_{period}'] = df[value_col].diff(periods=period)
            df[f'{value_col}_pct_change_{period}'] = df[value_col].pct_change(periods=period)
        
        return df

def create_interaction_features(df, feature_pairs=None, interaction_types=['multiply', 'add', 'subtract', 'divide']):
    """Create interaction features between numerical columns."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if feature_pairs is None:
        # Generate pairs from numeric columns
        from itertools import combinations
        feature_pairs = list(combinations(numeric_cols[:10], 2))  # Limit to first 10 to avoid explosion
    
    for col1, col2 in feature_pairs:
        if col1 in df.columns and col2 in df.columns:
            if 'multiply' in interaction_types:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            if 'add' in interaction_types:
                df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
            if 'subtract' in interaction_types:
                df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
            if 'divide' in interaction_types and (df[col2] != 0).all():
                df[f'{col1}_div_{col2}'] = df[col1] / df[col2]
    
    return df

def create_statistical_features(df, group_col, value_cols=None, stats=['mean', 'std', 'min', 'max', 'count', 'median']):
    """Create statistical features grouped by a categorical column."""
    if value_cols is None:
        value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df = df.copy()
    
    for col in value_cols:
        if col == group_col:
            continue
            
        for stat in stats:
            feature_name = f'{col}_{stat}_by_{group_col}'
            
            if stat == 'count':
                df[feature_name] = df.groupby(group_col)[col].transform('count')
            elif stat == 'mean':
                df[feature_name] = df.groupby(group_col)[col].transform('mean')
            elif stat == 'std':
                df[feature_name] = df.groupby(group_col)[col].transform('std')
            elif stat == 'min':
                df[feature_name] = df.groupby(group_col)[col].transform('min')
            elif stat == 'max':
                df[feature_name] = df.groupby(group_col)[col].transform('max')
            elif stat == 'median':
                df[feature_name] = df.groupby(group_col)[col].transform('median')
            elif stat == 'sum':
                df[feature_name] = df.groupby(group_col)[col].transform('sum')
            elif stat == 'var':
                df[feature_name] = df.groupby(group_col)[col].transform('var')
    
    return df

def create_frequency_encoding(df, categorical_cols=None):
    """Create frequency encoding for categorical variables."""
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    df = df.copy()
    
    for col in categorical_cols:
        if col in df.columns:
            freq_map = df[col].value_counts().to_dict()
            df[f'{col}_frequency'] = df[col].map(freq_map)
            
            # Also create relative frequency
            total_count = len(df)
            df[f'{col}_relative_frequency'] = df[f'{col}_frequency'] / total_count
    
    return df

def create_rank_features(df, numeric_cols=None, methods=['dense', 'min']):
    """Create rank-based features."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df = df.copy()
    
    for col in numeric_cols:
        for method in methods:
            df[f'{col}_rank_{method}'] = df[col].rank(method=method)
            # Normalized rank (0-1)
            df[f'{col}_rank_{method}_norm'] = df[f'{col}_rank_{method}'] / len(df)
    
    return df

def create_clustering_features(df, n_clusters=5, random_state=42):
    """Create cluster-based features using KMeans."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for clustering")
        return df
    
    df = df.copy()
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df['cluster'] = clusters
    df['cluster_distance_to_centroid'] = np.min(kmeans.transform(X_scaled), axis=1)
    
    return df

def create_anomaly_features(df, contamination=0.1):
    """Create anomaly detection features."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("No numeric columns for anomaly detection")
        return df
    
    df = df.copy()
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomaly_labels = iso_forest.fit_predict(X_scaled)
    anomaly_scores = iso_forest.decision_function(X_scaled)
    
    df['is_anomaly'] = (anomaly_labels == -1).astype(int)
    df['anomaly_score'] = anomaly_scores
    
    return df

class FeatureTransformer:
    """
    Advanced feature transformation utilities.
    """
    
    def __init__(self):
        self.transformers = {}
        self.is_fitted = False
    
    def apply_log_transform(self, df, columns=None, add_constant=1):
        """Apply log transformation to handle skewed data."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        df = df.copy()
        
        for col in columns:
            if (df[col] > 0).all():
                df[f'{col}_log'] = np.log(df[col])
            else:
                df[f'{col}_log1p'] = np.log1p(df[col] + add_constant)
        
        return df
    
    def apply_sqrt_transform(self, df, columns=None):
        """Apply square root transformation."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        df = df.copy()
        
        for col in columns:
            if (df[col] >= 0).all():
                df[f'{col}_sqrt'] = np.sqrt(df[col])
        
        return df
    
    def apply_box_cox_transform(self, df, columns=None):
        """Apply Box-Cox transformation."""
        try:
            from scipy import stats
        except ImportError:
            print("SciPy required for Box-Cox transformation")
            return df
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        df = df.copy()
        
        for col in columns:
            if (df[col] > 0).all():
                try:
                    transformed_data, lambda_val = stats.boxcox(df[col])
                    df[f'{col}_boxcox'] = transformed_data
                    self.transformers[f'{col}_boxcox_lambda'] = lambda_val
                except Exception as e:
                    print(f"Could not apply Box-Cox to {col}: {e}")
        
        return df
    
    def create_ratio_features(self, df, base_columns=None):
        """Create ratio features between numeric columns."""
        if base_columns is None:
            base_columns = df.select_dtypes(include=[np.number]).columns[:5]  # Limit to prevent explosion
        
        df = df.copy()
        
        for i, col1 in enumerate(base_columns):
            for col2 in base_columns[i+1:]:
                # Avoid division by zero
                if (df[col2] != 0).all():
                    df[f'{col1}_to_{col2}_ratio'] = df[col1] / df[col2]
        
        return df