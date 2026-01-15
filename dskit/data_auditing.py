import pandas as pd
import numpy as np
import warnings

class DataAuditor:
    """
    Comprehensive data auditing and quality assessment.
    """
    
    def __init__(self):
        self.audit_results = {}
        self.recommendations = []
    
    def comprehensive_audit(self, df, target_col=None):
        """Perform comprehensive data audit."""
        print("🔍 Starting Comprehensive Data Audit...")
        
        audit_results = {
            'basic_info': self._audit_basic_info(df),
            'data_quality': self._audit_data_quality(df),
            'statistical_properties': self._audit_statistical_properties(df),
            'relationships': self._audit_relationships(df, target_col),
            'potential_issues': self._audit_potential_issues(df),
            'recommendations': []
        }
        
        # Generate recommendations based on findings
        audit_results['recommendations'] = self._generate_recommendations(audit_results, target_col)
        
        self.audit_results = audit_results
        return audit_results
    
    def _audit_basic_info(self, df):
        """Audit basic dataset information."""
        return {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
            'column_types': df.dtypes.value_counts().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': df.duplicated().sum() / len(df) * 100
        }
    
    def _audit_data_quality(self, df):
        """Audit data quality issues."""
        quality_issues = {}
        
        for col in df.columns:
            col_issues = {
                'missing_count': df[col].isnull().sum(),
                'missing_percentage': df[col].isnull().sum() / len(df) * 100,
                'unique_count': df[col].nunique(),
                'unique_percentage': df[col].nunique() / len(df) * 100
            }
            
            # Check for constant columns
            if df[col].nunique() <= 1:
                col_issues['is_constant'] = True
            
            # Check for high cardinality categorical
            if df[col].dtype in ['object', 'category'] and df[col].nunique() > len(df) * 0.8:
                col_issues['high_cardinality'] = True
            
            # Check for potential ID columns
            if df[col].nunique() == len(df) and 'id' in col.lower():
                col_issues['potential_id_column'] = True
            
            quality_issues[col] = col_issues
        
        return quality_issues
    
    def _audit_statistical_properties(self, df):
        """Audit statistical properties of numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats_audit = {}
        
        for col in numeric_cols:
            col_stats = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'skewness': float(df[col].skew()),
                'kurtosis': float(df[col].kurtosis()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'range': float(df[col].max() - df[col].min()),
                'coefficient_of_variation': float(df[col].std() / df[col].mean()) if df[col].mean() != 0 else float('inf')
            }
            
            # Detect distribution characteristics
            col_stats['is_highly_skewed'] = abs(col_stats['skewness']) > 2
            col_stats['has_extreme_values'] = col_stats['coefficient_of_variation'] > 3
            col_stats['is_likely_binary'] = df[col].nunique() == 2 and set(df[col].unique()).issubset({0, 1, True, False})
            
            # Outlier detection using IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            col_stats['outlier_count'] = len(outliers)
            col_stats['outlier_percentage'] = len(outliers) / len(df) * 100
            
            stats_audit[col] = col_stats
        
        return stats_audit
    
    def _audit_relationships(self, df, target_col):
        """Audit relationships between variables."""
        relationships = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            # Correlation analysis
            corr_matrix = df[numeric_cols].corr()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:
                        high_corr_pairs.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': float(corr_val)
                        })
            
            relationships['high_correlations'] = high_corr_pairs
            
            # Target correlation (if target is provided and numeric)
            if target_col and target_col in numeric_cols:
                target_correlations = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)
                relationships['target_correlations'] = target_correlations.head(10).to_dict()
        
        return relationships
    
    def _audit_potential_issues(self, df):
        """Identify potential data issues."""
        issues = []
        
        # Check for columns with suspicious names
        suspicious_patterns = ['unnamed', 'column', 'field', 'var']
        for col in df.columns:
            if any(pattern in col.lower() for pattern in suspicious_patterns):
                issues.append(f"Suspicious column name: {col}")
        
        # Check for date columns stored as strings
        for col in df.select_dtypes(include=['object']).columns:
            sample = df[col].dropna().head(20)
            date_like = 0
            for val in sample:
                val_str = str(val)
                if any(sep in val_str for sep in ['-', '/', '.']) and any(c.isdigit() for c in val_str):
                    date_like += 1
            
            if date_like > len(sample) * 0.7:
                issues.append(f"Column '{col}' might contain dates stored as strings")
        
        # Check for numeric columns stored as strings
        for col in df.select_dtypes(include=['object']).columns:
            sample = df[col].dropna().head(100)
            try:
                pd.to_numeric(sample, errors='raise')
                issues.append(f"Column '{col}' contains numeric data stored as strings")
            except (ValueError, TypeError):
                pass
        
        # Check for potential categorical variables stored as numbers
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].nunique() < 20 and df[col].dtype in ['int64', 'float64']:
                if (df[col].dropna() % 1 == 0).all():  # All integers
                    issues.append(f"Column '{col}' might be categorical (only {df[col].nunique()} unique integers)")
        
        return issues
    
    def _generate_recommendations(self, audit_results, target_col):
        """Generate actionable recommendations based on audit results."""
        recommendations = []
        
        # Memory optimization recommendations
        memory_mb = audit_results['basic_info']['memory_usage_mb']
        if memory_mb > 100:
            recommendations.append({
                'category': 'memory_optimization',
                'priority': 'medium',
                'issue': f'Dataset uses {memory_mb:.1f} MB of memory',
                'recommendation': 'Consider memory optimization techniques like downcasting dtypes or using categorical types'
            })
        
        # Duplicate handling
        duplicate_pct = audit_results['basic_info']['duplicate_percentage']
        if duplicate_pct > 1:
            recommendations.append({
                'category': 'data_quality',
                'priority': 'high',
                'issue': f'{duplicate_pct:.1f}% duplicate rows found',
                'recommendation': 'Remove duplicate rows or investigate if they are legitimate'
            })
        
        # Missing data recommendations
        for col, quality_info in audit_results['data_quality'].items():
            missing_pct = quality_info['missing_percentage']
            if missing_pct > 50:
                recommendations.append({
                    'category': 'missing_data',
                    'priority': 'high',
                    'issue': f"Column '{col}' has {missing_pct:.1f}% missing values",
                    'recommendation': 'Consider dropping this column or using advanced imputation techniques'
                })
            elif missing_pct > 10:
                recommendations.append({
                    'category': 'missing_data',
                    'priority': 'medium',
                    'issue': f"Column '{col}' has {missing_pct:.1f}% missing values",
                    'recommendation': 'Apply appropriate imputation strategy (mean, median, mode, or advanced methods)'
                })
        
        # Statistical issues recommendations
        for col, stats_info in audit_results.get('statistical_properties', {}).items():
            if stats_info.get('is_highly_skewed', False):
                recommendations.append({
                    'category': 'statistical_properties',
                    'priority': 'medium',
                    'issue': f"Column '{col}' is highly skewed (skewness: {stats_info['skewness']:.2f})",
                    'recommendation': 'Consider log transformation, square root transformation, or Box-Cox transformation'
                })
            
            if stats_info.get('outlier_percentage', 0) > 5:
                recommendations.append({
                    'category': 'outliers',
                    'priority': 'medium',
                    'issue': f"Column '{col}' has {stats_info['outlier_percentage']:.1f}% outliers",
                    'recommendation': 'Investigate outliers and consider removal or winsorization'
                })
        
        # Correlation recommendations
        high_corr = audit_results.get('relationships', {}).get('high_correlations', [])
        if high_corr:
            for corr_pair in high_corr[:3]:  # Top 3
                recommendations.append({
                    'category': 'multicollinearity',
                    'priority': 'medium',
                    'issue': f"High correlation ({corr_pair['correlation']:.2f}) between '{corr_pair['var1']}' and '{corr_pair['var2']}'",
                    'recommendation': 'Consider removing one of the highly correlated variables or using PCA'
                })
        
        return recommendations
    
    def print_audit_report(self):
        """Print comprehensive audit report."""
        if not self.audit_results:
            print("No audit results available. Run comprehensive_audit() first.")
            return
        
        print("📊 DATA AUDIT REPORT")
        print("=" * 50)
        
        # Basic Info
        basic = self.audit_results['basic_info']
        print(f"📋 Dataset Shape: {basic['shape']}")
        print(f"💾 Memory Usage: {basic['memory_usage_mb']:.2f} MB")
        print(f"🔄 Duplicate Rows: {basic['duplicate_rows']} ({basic['duplicate_percentage']:.2f}%)")
        
        # Data Quality Summary
        print(f"\n📈 DATA QUALITY SUMMARY")
        print("-" * 30)
        
        total_cols = len(self.audit_results['data_quality'])
        high_missing_cols = sum(1 for col_info in self.audit_results['data_quality'].values() 
                               if col_info['missing_percentage'] > 10)
        constant_cols = sum(1 for col_info in self.audit_results['data_quality'].values() 
                           if col_info.get('is_constant', False))
        
        print(f"Total Columns: {total_cols}")
        print(f"Columns with >10% Missing: {high_missing_cols}")
        print(f"Constant Columns: {constant_cols}")
        
        # Top Recommendations
        print(f"\n⚠️  TOP RECOMMENDATIONS")
        print("-" * 30)
        
        high_priority_recs = [rec for rec in self.audit_results['recommendations'] 
                             if rec['priority'] == 'high']
        
        if high_priority_recs:
            for i, rec in enumerate(high_priority_recs[:5], 1):
                print(f"{i}. {rec['issue']}")
                print(f"   → {rec['recommendation']}")
                print()
        else:
            print("✅ No high-priority issues found!")
        
        # Statistical Insights
        if 'statistical_properties' in self.audit_results:
            print(f"\n📊 STATISTICAL INSIGHTS")
            print("-" * 30)
            
            skewed_cols = [col for col, stats in self.audit_results['statistical_properties'].items() 
                          if stats.get('is_highly_skewed', False)]
            outlier_cols = [col for col, stats in self.audit_results['statistical_properties'].items() 
                           if stats.get('outlier_percentage', 0) > 5]
            
            print(f"Highly Skewed Columns: {len(skewed_cols)}")
            if skewed_cols[:3]:
                print(f"  Examples: {', '.join(skewed_cols[:3])}")
            
            print(f"Columns with >5% Outliers: {len(outlier_cols)}")
            if outlier_cols[:3]:
                print(f"  Examples: {', '.join(outlier_cols[:3])}")

class DataSampler:
    """
    Advanced data sampling techniques.
    """
    
    def __init__(self):
        pass
    
    def stratified_sample(self, df, target_col, n_samples=1000, random_state=42):
        """Create stratified sample maintaining class distribution."""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Calculate proportions
        proportions = df[target_col].value_counts(normalize=True)
        
        samples = []
        for value, prop in proportions.items():
            value_df = df[df[target_col] == value]
            n_value_samples = max(1, int(n_samples * prop))
            
            if len(value_df) >= n_value_samples:
                sample = value_df.sample(n=n_value_samples, random_state=random_state)
            else:
                sample = value_df  # Take all available samples
            
            samples.append(sample)
        
        return pd.concat(samples, ignore_index=True)
    
    def time_aware_sample(self, df, date_col, n_samples=1000, method='recent'):
        """Sample data with time awareness."""
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found")
        
        # Ensure datetime format
        df_copy = df.copy()
        if df_copy[date_col].dtype != 'datetime64[ns]':
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        
        df_copy = df_copy.sort_values(date_col)
        
        if method == 'recent':
            # Sample most recent data
            return df_copy.tail(n_samples)
        elif method == 'uniform':
            # Sample uniformly across time periods
            return df_copy.sample(n=min(n_samples, len(df_copy)), random_state=42)
        elif method == 'seasonal':
            # Sample from each season/period
            df_copy['month'] = df_copy[date_col].dt.month
            samples_per_month = max(1, n_samples // 12)
            
            monthly_samples = []
            for month in range(1, 13):
                month_data = df_copy[df_copy['month'] == month]
                if not month_data.empty:
                    sample_size = min(samples_per_month, len(month_data))
                    monthly_samples.append(month_data.sample(sample_size, random_state=42))
            
            result = pd.concat(monthly_samples, ignore_index=True) if monthly_samples else df_copy.head(0)
            return result.drop('month', axis=1)
    
    def balanced_sample(self, df, target_col, method='undersample', random_state=42):
        """Create balanced sample for imbalanced datasets."""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        value_counts = df[target_col].value_counts()
        
        if method == 'undersample':
            # Undersample to smallest class
            min_samples = value_counts.min()
            samples = []
            
            for value in value_counts.index:
                value_df = df[df[target_col] == value]
                sample = value_df.sample(n=min_samples, random_state=random_state)
                samples.append(sample)
            
            return pd.concat(samples, ignore_index=True)
        
        elif method == 'oversample':
            # Simple oversample to largest class (with replacement)
            max_samples = value_counts.max()
            samples = []
            
            for value in value_counts.index:
                value_df = df[df[target_col] == value]
                if len(value_df) < max_samples:
                    sample = value_df.sample(n=max_samples, replace=True, random_state=random_state)
                else:
                    sample = value_df
                samples.append(sample)
            
            return pd.concat(samples, ignore_index=True)

def create_synthetic_data(df, n_samples=1000, method='gaussian_copula'):
    """Generate synthetic data that mimics the original dataset."""
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
    except ImportError:
        print("Scikit-learn required for synthetic data generation")
        return None
    
    if method == 'gaussian_copula':
        # Simple Gaussian copula approach
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("No numeric columns found for synthetic data generation")
            return None
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[numeric_cols].fillna(df[numeric_cols].mean()))
        
        # Fit multivariate normal distribution
        mean = np.mean(X_scaled, axis=0)
        cov = np.cov(X_scaled.T)
        
        # Generate synthetic samples
        synthetic_scaled = np.random.multivariate_normal(mean, cov, n_samples)
        
        # Transform back to original scale
        synthetic_data = scaler.inverse_transform(synthetic_scaled)
        
        # Create DataFrame
        synthetic_df = pd.DataFrame(synthetic_data, columns=numeric_cols)
        
        # Add categorical columns (sample from original)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            synthetic_df[col] = np.random.choice(df[col].dropna().values, n_samples)
        
        return synthetic_df
    
    else:
        print(f"Method '{method}' not implemented")
        return None


def validate_schema(df, expected_dtypes=None, required_columns=None):
    """
    Validate DataFrame schema against expected dtypes and required columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    expected_dtypes : dict, optional
        Dictionary mapping column names to expected dtype strings.
        Example: {'age': 'int64', 'name': 'object'}
    required_columns : list, optional
        List of column names that must be present.
    
    Returns
    -------
    dict
        Validation results with keys:
        - 'is_valid': bool, True if all checks pass
        - 'missing_columns': list of missing required columns
        - 'dtype_mismatches': dict of {column: {'expected': x, 'actual': y}}
        - 'extra_columns': list of columns not in expected_dtypes (if provided)
    
    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    >>> validate_schema(df, required_columns=['a', 'b', 'c'])
    {'is_valid': False, 'missing_columns': ['c'], ...}
    """
    result = {
        'is_valid': True,
        'missing_columns': [],
        'dtype_mismatches': {},
        'extra_columns': []
    }
    
    # Check required columns
    if required_columns is not None:
        missing = [col for col in required_columns if col not in df.columns]
        result['missing_columns'] = missing
        if missing:
            result['is_valid'] = False
    
    # Check dtype mismatches
    if expected_dtypes is not None:
        for col, expected_dtype in expected_dtypes.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                # Allow flexible matching (e.g., 'int64' matches 'int')
                if not (expected_dtype in actual_dtype or actual_dtype in expected_dtype):
                    result['dtype_mismatches'][col] = {
                        'expected': expected_dtype,
                        'actual': actual_dtype
                    }
                    result['is_valid'] = False
        
        # Identify extra columns not in expected schema
        expected_cols = set(expected_dtypes.keys())
        actual_cols = set(df.columns)
        result['extra_columns'] = list(actual_cols - expected_cols)
    
    return result


def duplicate_summary(df, subset=None):
    """
    Summarize duplicate rows in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    subset : list, optional
        Column names to consider for duplicate detection.
        If None, all columns are used.
    
    Returns
    -------
    dict
        Summary with keys:
        - 'total_duplicates': int, count of duplicate rows
        - 'duplicate_percentage': float, percentage of duplicates
        - 'total_rows': int, total number of rows
        - 'unique_rows': int, number of unique rows
        - 'by_subset': dict (only if subset provided), duplicates per subset column
    
    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 1, 2], 'b': [1, 1, 2]})
    >>> duplicate_summary(df)
    {'total_duplicates': 1, 'duplicate_percentage': 33.33, ...}
    """
    total_rows = len(df)
    
    if total_rows == 0:
        return {
            'total_duplicates': 0,
            'duplicate_percentage': 0.0,
            'total_rows': 0,
            'unique_rows': 0
        }
    
    duplicates = df.duplicated(subset=subset)
    total_duplicates = duplicates.sum()
    
    result = {
        'total_duplicates': int(total_duplicates),
        'duplicate_percentage': round(total_duplicates / total_rows * 100, 2),
        'total_rows': total_rows,
        'unique_rows': total_rows - int(total_duplicates)
    }
    
    # If subset is provided, also show duplicates per column in subset
    if subset is not None:
        by_subset = {}
        for col in subset:
            if col in df.columns:
                col_duplicates = df.duplicated(subset=[col]).sum()
                by_subset[col] = int(col_duplicates)
        result['by_subset'] = by_subset
    
    return result


def basic_profile(df):
    """
    Generate a basic profile summary of a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to profile.
    
    Returns
    -------
    dict
        Profile with keys:
        - 'rows': int, number of rows
        - 'cols': int, number of columns  
        - 'dtypes': dict, count of each dtype
        - 'memory_usage_mb': float, memory usage in MB
        - 'missing_cells': int, total missing values
        - 'missing_percentage': float, overall missing percentage
        - 'numeric_summary': dict, describe() output for numeric columns
    
    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    >>> profile = basic_profile(df)
    >>> profile['rows']
    3
    """
    rows, cols = df.shape
    total_cells = rows * cols
    missing_cells = df.isnull().sum().sum()
    
    # Get dtype counts
    dtype_counts = df.dtypes.value_counts()
    dtypes = {str(k): int(v) for k, v in dtype_counts.items()}
    
    # Numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_summary = df[numeric_cols].describe().to_dict()
    else:
        numeric_summary = {}
    
    result = {
        'rows': rows,
        'cols': cols,
        'dtypes': dtypes,
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 ** 2), 4),
        'missing_cells': int(missing_cells),
        'missing_percentage': round(missing_cells / total_cells * 100, 2) if total_cells > 0 else 0.0,
        'numeric_summary': numeric_summary
    }
    
    return result