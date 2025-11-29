import pandas as pd
import numpy as np
import sqlite3
import warnings
from contextlib import contextmanager
from typing import Dict, List, Optional, Union

class DatabaseConnector:
    """
    Universal database connector with support for multiple database types.
    """
    
    def __init__(self):
        self.connections = {}
        self.supported_databases = ['sqlite', 'mysql', 'postgresql', 'oracle', 'sqlserver']
    
    def connect(self, db_type: str, connection_params: dict, connection_name: str = 'default'):
        """Connect to various database types."""
        
        if db_type not in self.supported_databases:
            raise ValueError(f"Unsupported database type. Supported: {self.supported_databases}")
        
        try:
            if db_type == 'sqlite':
                conn = sqlite3.connect(connection_params.get('database', ':memory:'))
            
            elif db_type == 'mysql':
                try:
                    import mysql.connector
                    conn = mysql.connector.connect(**connection_params)
                except ImportError:
                    print("MySQL connector not available. Install with: pip install mysql-connector-python")
                    return None
            
            elif db_type == 'postgresql':
                try:
                    import psycopg2
                    conn = psycopg2.connect(**connection_params)
                except ImportError:
                    print("PostgreSQL connector not available. Install with: pip install psycopg2-binary")
                    return None
            
            elif db_type == 'oracle':
                try:
                    import cx_Oracle
                    conn = cx_Oracle.connect(**connection_params)
                except ImportError:
                    print("Oracle connector not available. Install with: pip install cx_Oracle")
                    return None
            
            elif db_type == 'sqlserver':
                try:
                    import pyodbc
                    conn_string = f"DRIVER={{SQL Server}};SERVER={connection_params['server']};DATABASE={connection_params['database']};UID={connection_params['username']};PWD={connection_params['password']}"
                    conn = pyodbc.connect(conn_string)
                except ImportError:
                    print("SQL Server connector not available. Install with: pip install pyodbc")
                    return None
            
            self.connections[connection_name] = {
                'connection': conn,
                'type': db_type,
                'params': connection_params
            }
            
            print(f"✅ Connected to {db_type} database as '{connection_name}'")
            return conn
            
        except Exception as e:
            print(f"❌ Failed to connect to {db_type}: {e}")
            return None
    
    @contextmanager
    def get_connection(self, connection_name: str = 'default'):
        """Context manager for database connections."""
        if connection_name not in self.connections:
            raise ValueError(f"Connection '{connection_name}' not found")
        
        conn = self.connections[connection_name]['connection']
        try:
            yield conn
        finally:
            pass  # Don't close connection in context manager
    
    def execute_query(self, query: str, connection_name: str = 'default', params: tuple = None) -> pd.DataFrame:
        """Execute query and return results as DataFrame."""
        if connection_name not in self.connections:
            raise ValueError(f"Connection '{connection_name}' not found")
        
        conn = self.connections[connection_name]['connection']
        
        try:
            if params:
                result = pd.read_sql_query(query, conn, params=params)
            else:
                result = pd.read_sql_query(query, conn)
            return result
        except Exception as e:
            print(f"Query execution failed: {e}")
            return pd.DataFrame()
    
    def insert_dataframe(self, df: pd.DataFrame, table_name: str, connection_name: str = 'default', 
                        if_exists: str = 'append', index: bool = False) -> bool:
        """Insert DataFrame into database table."""
        if connection_name not in self.connections:
            raise ValueError(f"Connection '{connection_name}' not found")
        
        conn = self.connections[connection_name]['connection']
        
        try:
            df.to_sql(table_name, conn, if_exists=if_exists, index=index)
            print(f"✅ Inserted {len(df)} rows into {table_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to insert data: {e}")
            return False
    
    def get_table_info(self, table_name: str, connection_name: str = 'default') -> dict:
        """Get information about a database table."""
        db_type = self.connections[connection_name]['type']
        
        if db_type == 'sqlite':
            info_query = f"PRAGMA table_info({table_name})"
        elif db_type in ['mysql', 'postgresql']:
            info_query = f"DESCRIBE {table_name}"
        else:
            print(f"Table info not implemented for {db_type}")
            return {}
        
        try:
            info_df = self.execute_query(info_query, connection_name)
            return info_df.to_dict('records')
        except Exception as e:
            print(f"Failed to get table info: {e}")
            return {}
    
    def list_tables(self, connection_name: str = 'default') -> List[str]:
        """List all tables in the database."""
        db_type = self.connections[connection_name]['type']
        
        if db_type == 'sqlite':
            query = "SELECT name FROM sqlite_master WHERE type='table'"
        elif db_type == 'mysql':
            query = "SHOW TABLES"
        elif db_type == 'postgresql':
            query = "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        else:
            print(f"List tables not implemented for {db_type}")
            return []
        
        try:
            tables_df = self.execute_query(query, connection_name)
            return tables_df.iloc[:, 0].tolist()
        except Exception as e:
            print(f"Failed to list tables: {e}")
            return []
    
    def close_connection(self, connection_name: str = 'default'):
        """Close database connection."""
        if connection_name in self.connections:
            self.connections[connection_name]['connection'].close()
            del self.connections[connection_name]
            print(f"Closed connection '{connection_name}'")

class DataProfiler:
    """
    Advanced data profiling and statistical analysis.
    """
    
    def __init__(self):
        self.profile_cache = {}
    
    def comprehensive_profile(self, df: pd.DataFrame, include_correlations: bool = True, 
                            include_distributions: bool = True, sample_size: int = None) -> dict:
        """Generate comprehensive data profile."""
        
        # Sample data if too large
        if sample_size and len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
            print(f"📊 Profiling sample of {sample_size} rows")
        else:
            df_sample = df
        
        profile = {
            'dataset_info': self._profile_dataset_info(df_sample),
            'column_profiles': self._profile_columns(df_sample),
            'data_types': self._profile_data_types(df_sample),
            'missing_patterns': self._profile_missing_patterns(df_sample)
        }
        
        if include_correlations:
            profile['correlations'] = self._profile_correlations(df_sample)
        
        if include_distributions:
            profile['distributions'] = self._profile_distributions(df_sample)
        
        # Cache the profile
        profile_id = f"profile_{hash(str(df.shape))}"
        self.profile_cache[profile_id] = profile
        
        return profile
    
    def _profile_dataset_info(self, df: pd.DataFrame) -> dict:
        """Profile basic dataset information."""
        return {
            'shape': df.shape,
            'memory_usage': {
                'total_mb': df.memory_usage(deep=True).sum() / (1024**2),
                'per_column': df.memory_usage(deep=True).to_dict()
            },
            'dtypes_summary': df.dtypes.value_counts().to_dict(),
            'duplicates': {
                'count': df.duplicated().sum(),
                'percentage': df.duplicated().sum() / len(df) * 100
            }
        }
    
    def _profile_columns(self, df: pd.DataFrame) -> dict:
        """Profile each column individually."""
        column_profiles = {}
        
        for col in df.columns:
            col_profile = {
                'dtype': str(df[col].dtype),
                'non_null_count': df[col].count(),
                'null_count': df[col].isnull().sum(),
                'null_percentage': df[col].isnull().sum() / len(df) * 100,
                'unique_count': df[col].nunique(),
                'unique_percentage': df[col].nunique() / len(df) * 100
            }
            
            # Numeric column profiling
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numeric_stats = self._profile_numeric_column(df[col])
                col_profile.update(numeric_stats)
            
            # Categorical column profiling
            elif df[col].dtype in ['object', 'category']:
                categorical_stats = self._profile_categorical_column(df[col])
                col_profile.update(categorical_stats)
            
            # Datetime column profiling
            elif 'datetime' in str(df[col].dtype):
                datetime_stats = self._profile_datetime_column(df[col])
                col_profile.update(datetime_stats)
            
            column_profiles[col] = col_profile
        
        return column_profiles
    
    def _profile_numeric_column(self, series: pd.Series) -> dict:
        """Profile numeric column."""
        series_clean = series.dropna()
        
        if len(series_clean) == 0:
            return {'error': 'No non-null numeric values'}
        
        stats = {
            'min': float(series_clean.min()),
            'max': float(series_clean.max()),
            'mean': float(series_clean.mean()),
            'median': float(series_clean.median()),
            'std': float(series_clean.std()),
            'variance': float(series_clean.var()),
            'skewness': float(series_clean.skew()),
            'kurtosis': float(series_clean.kurtosis()),
            'range': float(series_clean.max() - series_clean.min())
        }
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            stats[f'percentile_{p}'] = float(series_clean.quantile(p/100))
        
        # Additional insights
        stats['is_integer'] = series_clean.apply(lambda x: x.is_integer()).all()
        stats['has_negatives'] = (series_clean < 0).any()
        stats['has_zeros'] = (series_clean == 0).any()
        
        # Outlier analysis using IQR
        Q1 = series_clean.quantile(0.25)
        Q3 = series_clean.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series_clean[(series_clean < lower_bound) | (series_clean > upper_bound)]
        stats['outlier_count'] = len(outliers)
        stats['outlier_percentage'] = len(outliers) / len(series_clean) * 100
        
        return stats
    
    def _profile_categorical_column(self, series: pd.Series) -> dict:
        """Profile categorical column."""
        series_clean = series.dropna()
        
        if len(series_clean) == 0:
            return {'error': 'No non-null categorical values'}
        
        value_counts = series_clean.value_counts()
        
        stats = {
            'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
            'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
            'least_frequent_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
            'cardinality': len(value_counts),
            'cardinality_percentage': len(value_counts) / len(series_clean) * 100
        }
        
        # Top categories
        stats['top_categories'] = value_counts.head(10).to_dict()
        
        # String analysis (if applicable)
        if series_clean.dtype == 'object':
            string_lengths = series_clean.astype(str).str.len()
            stats['avg_length'] = float(string_lengths.mean())
            stats['min_length'] = int(string_lengths.min())
            stats['max_length'] = int(string_lengths.max())
            
            # Check for potential patterns
            stats['has_digits'] = series_clean.astype(str).str.contains(r'\d').any()
            stats['has_special_chars'] = series_clean.astype(str).str.contains(r'[^a-zA-Z0-9\s]').any()
        
        return stats
    
    def _profile_datetime_column(self, series: pd.Series) -> dict:
        """Profile datetime column."""
        series_clean = series.dropna()
        
        if len(series_clean) == 0:
            return {'error': 'No non-null datetime values'}
        
        stats = {
            'min_date': series_clean.min().isoformat(),
            'max_date': series_clean.max().isoformat(),
            'date_range_days': (series_clean.max() - series_clean.min()).days,
        }
        
        # Extract time components
        stats['years_span'] = list(series_clean.dt.year.unique())
        stats['months_present'] = list(series_clean.dt.month.unique())
        stats['weekdays_present'] = list(series_clean.dt.dayofweek.unique())
        
        # Check for patterns
        stats['has_time_component'] = (series_clean.dt.time != pd.Timestamp('00:00:00').time()).any()
        
        return stats
    
    def _profile_data_types(self, df: pd.DataFrame) -> dict:
        """Analyze data type optimization opportunities."""
        recommendations = {}
        
        for col in df.columns:
            recommendations[col] = {
                'current_dtype': str(df[col].dtype),
                'current_memory_mb': df[col].memory_usage(deep=True) / (1024**2),
                'optimization_suggestions': []
            }
            
            # Integer optimization
            if df[col].dtype in ['int64', 'float64']:
                if df[col].isnull().sum() == 0:  # No nulls
                    min_val = df[col].min()
                    max_val = df[col].max()
                    
                    if min_val >= 0 and max_val <= 255:
                        recommendations[col]['optimization_suggestions'].append('uint8')
                    elif min_val >= -128 and max_val <= 127:
                        recommendations[col]['optimization_suggestions'].append('int8')
                    elif min_val >= 0 and max_val <= 65535:
                        recommendations[col]['optimization_suggestions'].append('uint16')
                    elif min_val >= -32768 and max_val <= 32767:
                        recommendations[col]['optimization_suggestions'].append('int16')
            
            # Float optimization
            if df[col].dtype == 'float64':
                # Check if can be converted to float32
                recommendations[col]['optimization_suggestions'].append('float32')
            
            # Categorical optimization
            if df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    recommendations[col]['optimization_suggestions'].append('category')
        
        return recommendations
    
    def _profile_missing_patterns(self, df: pd.DataFrame) -> dict:
        """Analyze missing data patterns."""
        missing_info = {
            'total_missing': df.isnull().sum().sum(),
            'missing_percentage': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
            'columns_with_missing': df.columns[df.isnull().any()].tolist(),
            'missing_by_column': df.isnull().sum().to_dict()
        }
        
        # Missing patterns across rows
        if len(df.columns) > 1:
            missing_patterns = df.isnull().apply(lambda x: ''.join(x.astype(int).astype(str)), axis=1)
            pattern_counts = missing_patterns.value_counts().head(10)
            missing_info['common_missing_patterns'] = pattern_counts.to_dict()
        
        return missing_info
    
    def _profile_correlations(self, df: pd.DataFrame) -> dict:
        """Analyze correlations between numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'error': 'Less than 2 numeric columns for correlation analysis'}
        
        corr_matrix = df[numeric_cols].corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': float(corr_val)
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'strong_correlations': strong_correlations,
            'max_correlation': float(corr_matrix.abs().unstack().max()),
            'avg_correlation': float(corr_matrix.abs().unstack().mean())
        }
    
    def _profile_distributions(self, df: pd.DataFrame, max_bins: int = 50) -> dict:
        """Analyze distributions of numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        distributions = {}
        
        for col in numeric_cols:
            series_clean = df[col].dropna()
            
            if len(series_clean) == 0:
                continue
            
            # Create histogram
            hist, bin_edges = np.histogram(series_clean, bins=min(max_bins, len(series_clean.unique())))
            
            distributions[col] = {
                'histogram': {
                    'counts': hist.tolist(),
                    'bin_edges': bin_edges.tolist()
                },
                'distribution_type': self._infer_distribution_type(series_clean)
            }
        
        return distributions
    
    def _infer_distribution_type(self, series: pd.Series) -> str:
        """Simple distribution type inference."""
        skew = series.skew()
        kurtosis = series.kurtosis()
        
        if abs(skew) < 0.5 and abs(kurtosis - 3) < 1:
            return 'approximately_normal'
        elif skew > 1:
            return 'right_skewed'
        elif skew < -1:
            return 'left_skewed'
        elif kurtosis > 5:
            return 'heavy_tailed'
        elif kurtosis < 1:
            return 'light_tailed'
        else:
            return 'unknown'

class DataExporter:
    """
    Advanced data export utilities with multiple format support.
    """
    
    def __init__(self):
        self.supported_formats = ['csv', 'excel', 'json', 'parquet', 'hdf5', 'pickle', 'sql']
    
    def export_data(self, df: pd.DataFrame, filepath: str, format: str = 'auto', **kwargs) -> bool:
        """Export DataFrame to various formats."""
        
        if format == 'auto':
            # Infer format from file extension
            extension = filepath.split('.')[-1].lower()
            format_mapping = {
                'csv': 'csv',
                'xlsx': 'excel',
                'xls': 'excel',
                'json': 'json',
                'parquet': 'parquet',
                'h5': 'hdf5',
                'hdf': 'hdf5',
                'pkl': 'pickle',
                'pickle': 'pickle'
            }
            format = format_mapping.get(extension, 'csv')
        
        try:
            if format == 'csv':
                df.to_csv(filepath, index=False, **kwargs)
            
            elif format == 'excel':
                df.to_excel(filepath, index=False, **kwargs)
            
            elif format == 'json':
                df.to_json(filepath, orient='records', **kwargs)
            
            elif format == 'parquet':
                df.to_parquet(filepath, **kwargs)
            
            elif format == 'hdf5':
                df.to_hdf(filepath, key='data', mode='w', **kwargs)
            
            elif format == 'pickle':
                df.to_pickle(filepath, **kwargs)
            
            elif format == 'sql':
                # This would require database connection
                print("SQL export requires database connection. Use DatabaseConnector instead.")
                return False
            
            else:
                print(f"Unsupported format: {format}")
                return False
            
            print(f"✅ Data exported to {filepath} ({format} format)")
            return True
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False
    
    def export_profile_report(self, profile_data: dict, filepath: str, format: str = 'html') -> bool:
        """Export data profile as a report."""
        
        if format == 'html':
            html_content = self._generate_html_report(profile_data)
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                print(f"✅ Profile report exported to {filepath}")
                return True
            except Exception as e:
                print(f"❌ Report export failed: {e}")
                return False
        
        elif format == 'json':
            try:
                import json
                with open(filepath, 'w') as f:
                    json.dump(profile_data, f, indent=2, default=str)
                print(f"✅ Profile report exported to {filepath}")
                return True
            except Exception as e:
                print(f"❌ Report export failed: {e}")
                return False
        
        else:
            print(f"Unsupported report format: {format}")
            return False
    
    def _generate_html_report(self, profile_data: dict) -> str:
        """Generate HTML report from profile data."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Profile Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .metric { margin: 5px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Data Profile Report</h1>
        """
        
        # Dataset info section
        if 'dataset_info' in profile_data:
            html += """
            <div class="section">
                <h2>Dataset Information</h2>
            """
            info = profile_data['dataset_info']
            html += f"<div class='metric'>Shape: {info.get('shape', 'N/A')}</div>"
            html += f"<div class='metric'>Memory Usage: {info.get('memory_usage', {}).get('total_mb', 0):.2f} MB</div>"
            html += "</div>"
        
        # Add more sections as needed...
        html += """
        </body>
        </html>
        """
        
        return html