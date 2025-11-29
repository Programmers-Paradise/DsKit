import pandas as pd
import numpy as np
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
    from sklearn.preprocessing import PowerTransformer, LabelEncoder, OrdinalEncoder
    from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
    from sklearn.feature_selection import SelectKBest, SelectPercentile, RFE, RFECV
    from sklearn.feature_selection import VarianceThreshold, SelectFromModel
    from sklearn.decomposition import PCA, TruncatedSVD, FastICA
    from sklearn.manifold import TSNE
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some sklearn features may not be available: {e}")
    SKLEARN_AVAILABLE = False
    # Create dummy classes to prevent errors
    class DummyTransformer:
        def fit_transform(self, X): return X
        def transform(self, X): return X
        def fit(self, X, y=None): return self
    
    StandardScaler = MinMaxScaler = RobustScaler = DummyTransformer
    QuantileTransformer = PowerTransformer = DummyTransformer

from scipy.stats import zscore, iqr
from scipy import stats
import warnings

class AdvancedPreprocessor:
    """
    Advanced preprocessing pipeline with state management.
    """
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.transformers = {}
        self.feature_selectors = {}
        self.is_fitted = False
        
    def fit_transform_scaling(self, df, method='standard', columns=None):
        """Advanced scaling with multiple methods."""
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
            
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'quantile': QuantileTransformer(output_distribution='normal'),
            'power': PowerTransformer(method='yeo-johnson')
        }
        
        if method not in scalers:
            raise ValueError(f"Method must be one of {list(scalers.keys())}")
            
        scaler = scalers[method]
        self.scalers[method] = scaler
        
        df[columns] = scaler.fit_transform(df[columns])
        return df
    
    def advanced_imputation(self, df, method='iterative', n_neighbors=5):
        """Advanced imputation methods."""
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Numeric imputation
        if len(numeric_cols) > 0:
            if method == 'iterative':
                imputer = IterativeImputer(random_state=42, max_iter=10)
            elif method == 'knn':
                imputer = KNNImputer(n_neighbors=n_neighbors)
            else:
                imputer = SimpleImputer(strategy=method)
                
            self.imputers['numeric'] = imputer
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        # Categorical imputation
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            self.imputers['categorical'] = cat_imputer
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
        
        return df
    
    def detect_and_handle_outliers(self, df, method='isolation_forest', contamination=0.1):
        """Advanced outlier detection and handling."""
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM
        from sklearn.covariance import EllipticEnvelope
        
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return df, []
        
        X = df[numeric_cols].values
        outlier_indices = []
        
        if method == 'isolation_forest':
            detector = IsolationForest(contamination=contamination, random_state=42)
            outliers = detector.fit_predict(X)
            outlier_indices = np.where(outliers == -1)[0]
            
        elif method == 'one_class_svm':
            detector = OneClassSVM(nu=contamination)
            outliers = detector.fit_predict(X)
            outlier_indices = np.where(outliers == -1)[0]
            
        elif method == 'elliptic_envelope':
            detector = EllipticEnvelope(contamination=contamination, random_state=42)
            outliers = detector.fit_predict(X)
            outlier_indices = np.where(outliers == -1)[0]
            
        elif method == 'local_outlier_factor':
            from sklearn.neighbors import LocalOutlierFactor
            detector = LocalOutlierFactor(contamination=contamination)
            outliers = detector.fit_predict(X)
            outlier_indices = np.where(outliers == -1)[0]
        
        return df.drop(outlier_indices), outlier_indices
    
    def smart_feature_selection(self, X, y, method='auto', k=10, task='classification'):
        """Intelligent feature selection."""
        if method == 'auto':
            # Use multiple methods and combine
            methods = ['variance', 'univariate', 'rfe']
            selected_features = set(X.columns)
            
            for m in methods:
                X_sel, features, _ = self.smart_feature_selection(X, y, method=m, k=k, task=task)
                selected_features &= set(features)
            
            return X[list(selected_features)], list(selected_features), None
        
        elif method == 'variance':
            selector = VarianceThreshold(threshold=0.01)
            X_selected = selector.fit_transform(X)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == 'univariate':
            if task == 'classification':
                from sklearn.feature_selection import f_classif
                selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
            else:
                from sklearn.feature_selection import f_regression
                selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
            
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == 'rfe':
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            
            if task == 'classification':
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                
            selector = RFE(estimator, n_features_to_select=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
        
        elif method == 'embedded':
            from sklearn.feature_selection import SelectFromModel
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            
            if task == 'classification':
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                
            selector = SelectFromModel(estimator, max_features=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
        
        self.feature_selectors[method] = selector
        return X[selected_features] if hasattr(X, 'columns') else X_selected, selected_features, selector

def advanced_data_validation(df):
    """Comprehensive data validation."""
    validation_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024*1024),
        'issues': []
    }
    
    # Check for completely empty columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        validation_report['issues'].append(f"Completely empty columns: {empty_cols}")
    
    # Check for columns with only one unique value
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        validation_report['issues'].append(f"Constant columns: {constant_cols}")
    
    # Check for high cardinality categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    high_card_cols = [col for col in cat_cols if df[col].nunique() > len(df) * 0.8]
    if high_card_cols:
        validation_report['issues'].append(f"High cardinality categorical columns: {high_card_cols}")
    
    # Check for potential datetime columns stored as strings
    potential_dates = []
    for col in df.select_dtypes(include=['object']).columns:
        sample = df[col].dropna().head(10).astype(str)
        date_patterns = 0
        for val in sample:
            if any(char.isdigit() for char in val) and any(sep in val for sep in ['-', '/', '.']):
                date_patterns += 1
        if date_patterns > len(sample) * 0.5:
            potential_dates.append(col)
    
    if potential_dates:
        validation_report['issues'].append(f"Potential date columns stored as text: {potential_dates}")
    
    # Check for suspicious numeric ranges
    numeric_issues = []
    for col in df.select_dtypes(include=[np.number]).columns:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max > 1e6 or col_min < -1e6:
            numeric_issues.append(f"{col}: extreme values ({col_min:.2e} to {col_max:.2e})")
        
        # Check for potential categorical variables stored as numbers
        if df[col].nunique() < 20 and df[col].dtype in ['int64', 'float64']:
            if all(df[col].dropna().apply(lambda x: x == int(x))):
                numeric_issues.append(f"{col}: might be categorical (only {df[col].nunique()} unique integers)")
    
    if numeric_issues:
        validation_report['issues'].append(f"Numeric data issues: {numeric_issues}")
    
    # Data quality score
    total_possible_issues = 5
    actual_issues = len(validation_report['issues'])
    validation_report['quality_score'] = max(0, (total_possible_issues - actual_issues) / total_possible_issues * 100)
    
    return validation_report

def smart_type_inference(df):
    """Intelligent data type inference and conversion."""
    df = df.copy()
    type_changes = {}
    
    for col in df.columns:
        original_type = df[col].dtype
        
        # Skip if already optimal type
        if original_type in ['category', 'datetime64[ns]']:
            continue
        
        # Try datetime conversion
        if original_type == 'object':
            try:
                # Sample a few values to check if they're dates
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    pd.to_datetime(sample, infer_datetime_format=True)
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    type_changes[col] = f"{original_type} -> datetime64[ns]"
                    continue
            except (ValueError, TypeError):
                pass
        
        # Try numeric conversion
        if original_type == 'object':
            try:
                # Check if it's numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                non_null_ratio = numeric_series.notna().sum() / len(df[col])
                
                if non_null_ratio > 0.8:  # If 80%+ can be converted to numeric
                    df[col] = numeric_series
                    type_changes[col] = f"{original_type} -> {df[col].dtype}"
                    continue
            except (ValueError, TypeError):
                pass
        
        # Convert to category if low cardinality
        if original_type == 'object':
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.1 and df[col].nunique() < 50:
                df[col] = df[col].astype('category')
                type_changes[col] = f"{original_type} -> category"
        
        # Optimize integer types
        if 'int' in str(original_type):
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype('uint8')
                    type_changes[col] = f"{original_type} -> uint8"
                elif col_max < 65535:
                    df[col] = df[col].astype('uint16')
                    type_changes[col] = f"{original_type} -> uint16"
            else:
                if col_min >= -128 and col_max <= 127:
                    df[col] = df[col].astype('int8')
                    type_changes[col] = f"{original_type} -> int8"
                elif col_min >= -32768 and col_max <= 32767:
                    df[col] = df[col].astype('int16')
                    type_changes[col] = f"{original_type} -> int16"
    
    return df, type_changes

def data_profiler(df, sample_size=None):
    """Comprehensive data profiling."""
    if sample_size and len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df
    
    profile = {
        'dataset_info': {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024*1024),
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': df.duplicated().sum() / len(df) * 100
        },
        'column_profiles': {}
    }
    
    for col in df_sample.columns:
        col_profile = {
            'dtype': str(df_sample[col].dtype),
            'missing_count': df_sample[col].isnull().sum(),
            'missing_percentage': df_sample[col].isnull().sum() / len(df_sample) * 100,
            'unique_count': df_sample[col].nunique(),
            'unique_percentage': df_sample[col].nunique() / len(df_sample) * 100
        }
        
        if df_sample[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            # Numeric profile
            col_profile.update({
                'mean': df_sample[col].mean(),
                'std': df_sample[col].std(),
                'min': df_sample[col].min(),
                'max': df_sample[col].max(),
                'median': df_sample[col].median(),
                'q1': df_sample[col].quantile(0.25),
                'q3': df_sample[col].quantile(0.75),
                'skewness': df_sample[col].skew(),
                'kurtosis': df_sample[col].kurtosis(),
                'zeros_count': (df_sample[col] == 0).sum(),
                'negative_count': (df_sample[col] < 0).sum()
            })
            
            # Detect distribution
            col_profile['suspected_distribution'] = detect_distribution(df_sample[col].dropna())
            
        elif df_sample[col].dtype == 'object' or df_sample[col].dtype.name == 'category':
            # Categorical profile
            value_counts = df_sample[col].value_counts()
            col_profile.update({
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'top_5_values': value_counts.head().to_dict()
            })
            
            # Text analysis for object columns
            if df_sample[col].dtype == 'object':
                text_lengths = df_sample[col].astype(str).str.len()
                col_profile.update({
                    'avg_text_length': text_lengths.mean(),
                    'min_text_length': text_lengths.min(),
                    'max_text_length': text_lengths.max()
                })
        
        profile['column_profiles'][col] = col_profile
    
    return profile

def detect_distribution(series):
    """Detect the most likely statistical distribution."""
    if len(series) < 30:
        return "insufficient_data"
    
    # Test common distributions
    distributions = {
        'normal': stats.normaltest,
        'uniform': lambda x: stats.kstest(x, 'uniform'),
        'exponential': lambda x: stats.kstest(x, 'expon')
    }
    
    results = {}
    for name, test_func in distributions.items():
        try:
            if name == 'normal':
                _, p_value = test_func(series)
            else:
                _, p_value = test_func(series)
            results[name] = p_value
        except:
            results[name] = 0
    
    # Return distribution with highest p-value (best fit)
    best_fit = max(results, key=results.get)
    return best_fit if results[best_fit] > 0.05 else "unknown"

def memory_optimizer(df):
    """Optimize DataFrame memory usage."""
    start_memory = df.memory_usage(deep=True).sum() / (1024*1024)
    
    optimized_df = df.copy()
    
    # Optimize object columns
    for col in optimized_df.select_dtypes(include=['object']).columns:
        unique_ratio = optimized_df[col].nunique() / len(optimized_df)
        
        if unique_ratio < 0.5:  # Convert to category if less than 50% unique
            optimized_df[col] = optimized_df[col].astype('category')
    
    # Optimize integer columns
    for col in optimized_df.select_dtypes(include=['int64']).columns:
        col_min = optimized_df[col].min()
        col_max = optimized_df[col].max()
        
        if col_min >= 0:
            if col_max < 255:
                optimized_df[col] = optimized_df[col].astype('uint8')
            elif col_max < 65535:
                optimized_df[col] = optimized_df[col].astype('uint16')
            elif col_max < 4294967295:
                optimized_df[col] = optimized_df[col].astype('uint32')
        else:
            if col_min > -128 and col_max < 127:
                optimized_df[col] = optimized_df[col].astype('int8')
            elif col_min > -32768 and col_max < 32767:
                optimized_df[col] = optimized_df[col].astype('int16')
            elif col_min > -2147483648 and col_max < 2147483647:
                optimized_df[col] = optimized_df[col].astype('int32')
    
    # Optimize float columns
    for col in optimized_df.select_dtypes(include=['float64']).columns:
        optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
    
    end_memory = optimized_df.memory_usage(deep=True).sum() / (1024*1024)
    memory_reduction = (start_memory - end_memory) / start_memory * 100
    
    print(f"Memory usage reduced by {memory_reduction:.2f}%")
    print(f"Original: {start_memory:.2f} MB -> Optimized: {end_memory:.2f} MB")
    
    return optimized_df