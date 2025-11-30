# 📖 dskit API Reference Guide

This document provides detailed API documentation for all dskit classes and functions.

## 🏗️ Core Classes

### dskit Main Class

```python
class dskit:
    """Main dskit class with method chaining support."""

    def __init__(self, data=None):
        """Initialize dskit instance.

        Parameters:
        -----------
        data : pd.DataFrame, optional
            Initial DataFrame to work with
        """

    @staticmethod
    def load(filepath: str, **kwargs) -> 'dskit':
        """Load data from file and return dskit instance."""

    def fix_dtypes(self) -> 'dskit':
        """Fix data types automatically."""

    def rename_columns_auto(self) -> 'dskit':
        """Clean column names automatically."""

    def fill_missing(self, strategy: str = 'auto', fill_value=None) -> 'dskit':
        """Fill missing values with specified strategy."""

    def remove_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> 'dskit':
        """Remove outliers using specified method."""
```

---

## 📁 Module API Documentation

### 🔧 Data I/O Module (`io.py`)

```python
def load(filepath: str, **kwargs) -> pd.DataFrame:
    """Universal data loader with automatic format detection.

    Parameters:
    -----------
    filepath : str
        Path to the data file
    **kwargs : dict
        Additional parameters passed to the appropriate pandas reader

    Returns:
    --------
    pd.DataFrame
        Loaded DataFrame

    Supported Formats:
    ------------------
    - CSV: .csv
    - Excel: .xlsx, .xls
    - JSON: .json
    - Parquet: .parquet

    Examples:
    ---------
    >>> df = dskit.load("data.csv")
    >>> df = dskit.load("data.xlsx", sheet_name="Sheet1")
    >>> df = dskit.load("data.json", orient="records")
    """

def read_folder(folder_path: str, file_type: str = 'csv', **kwargs) -> pd.DataFrame:
    """Read and combine multiple files from a folder.

    Parameters:
    -----------
    folder_path : str
        Path to the folder containing files
    file_type : str, default 'csv'
        Type of files to read ('csv', 'xlsx', 'json', 'parquet')
    **kwargs : dict
        Additional parameters for file reading

    Returns:
    --------
    pd.DataFrame
        Combined DataFrame from all files

    Examples:
    ---------
    >>> df = dskit.read_folder("data_folder/", file_type="csv")
    >>> df = dskit.read_folder("excel_files/", file_type="xlsx")
    """

def save(df: pd.DataFrame, filepath: str, **kwargs) -> None:
    """Save DataFrame to file with format auto-detection.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    filepath : str
        Output file path
    **kwargs : dict
        Additional parameters for saving

    Examples:
    ---------
    >>> dskit.save(df, "output.csv")
    >>> dskit.save(df, "output.parquet")
    >>> dskit.save(df, "output.xlsx", index=False)
    """
```

### 🧹 Data Cleaning Module (`cleaning.py`)

```python
def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Automatically detect and fix data types.

    Features:
    ---------
    - Converts string numbers to numeric
    - Detects and converts datetime columns
    - Optimizes integer types
    - Handles mixed-type columns

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    --------
    pd.DataFrame
        DataFrame with optimized data types
    """

def rename_columns_auto(df: pd.DataFrame) -> pd.DataFrame:
    """Automatically clean column names.

    Cleaning Operations:
    -------------------
    - Convert to lowercase
    - Replace spaces with underscores
    - Remove special characters
    - Handle duplicate names

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    --------
    pd.DataFrame
        DataFrame with cleaned column names
    """

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive missing value summary.

    Returns:
    --------
    pd.DataFrame
        Summary with columns: Count, Percentage, Data_Type
    """

def fill_missing(df: pd.DataFrame, strategy: str = 'auto', fill_value=None) -> pd.DataFrame:
    """Intelligent missing value imputation.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    strategy : str, default 'auto'
        Imputation strategy:
        - 'auto': Automatically choose best strategy
        - 'mean': Mean for numeric, mode for categorical
        - 'median': Median for numeric, mode for categorical
        - 'mode': Most frequent value
        - 'constant': Fill with fill_value
        - 'ffill': Forward fill
        - 'bfill': Backward fill
    fill_value : any, optional
        Value to use when strategy='constant'

    Returns:
    --------
    pd.DataFrame
        DataFrame with imputed missing values

    Examples:
    ---------
    >>> df = dskit.fill_missing(df, strategy='auto')
    >>> df = dskit.fill_missing(df, strategy='constant', fill_value=0)
    """

def outlier_summary(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """Generate outlier detection summary.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    method : str, default 'iqr'
        Detection method: 'iqr', 'zscore', 'isolation'
    threshold : float, default 1.5
        Threshold for outlier detection

    Returns:
    --------
    pd.DataFrame
        Summary of outliers per column
    """

def remove_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """Remove outliers using specified method.

    Methods:
    --------
    - 'iqr': Interquartile Range method
    - 'zscore': Z-score based detection (threshold = z-score limit)
    - 'isolation': Isolation Forest (threshold = contamination rate)

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    method : str, default 'iqr'
        Outlier detection method
    threshold : float, default 1.5
        Method-specific threshold

    Returns:
    --------
    pd.DataFrame
        DataFrame with outliers removed
    """

def simple_nlp_clean(df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
    """Basic text preprocessing for NLP.

    Operations:
    -----------
    - Convert to lowercase
    - Remove extra whitespace
    - Remove special characters (optional)

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    text_cols : list
        List of text column names

    Returns:
    --------
    pd.DataFrame
        DataFrame with cleaned text columns
    """
```

### 📊 Visualization Module (`visualization.py`)

```python
def plot_missingness(df: pd.DataFrame, figsize: tuple = (12, 6)) -> None:
    """Plot missing data patterns as heatmap.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    figsize : tuple, default (12, 6)
        Figure size for the plot
    """

def plot_histograms(df: pd.DataFrame, bins: int = 30, figsize: tuple = (15, 10)) -> None:
    """Plot histograms for all numeric columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    bins : int, default 30
        Number of bins for histograms
    figsize : tuple, default (15, 10)
        Figure size for the plot
    """

def plot_correlation_heatmap(df: pd.DataFrame, method: str = 'pearson', figsize: tuple = (12, 8)) -> None:
    """Plot correlation heatmap.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    method : str, default 'pearson'
        Correlation method: 'pearson', 'spearman', 'kendall'
    figsize : tuple, default (12, 8)
        Figure size for the plot
    """

def plot_boxplots(df: pd.DataFrame, figsize: tuple = (15, 10)) -> None:
    """Plot box plots for numeric columns to show outliers.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    figsize : tuple, default (15, 10)
        Figure size for the plot
    """
```

### 🔧 Preprocessing Module (`preprocessing.py`)

```python
def auto_encode(df: pd.DataFrame, target_col: str = None, max_categories: int = 10) -> pd.DataFrame:
    """Automatically encode categorical variables.

    Encoding Strategy:
    ------------------
    - Binary columns: Keep as 0/1
    - High cardinality (>max_categories): Target encoding if target provided, else drop
    - Low cardinality: One-hot encoding

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_col : str, optional
        Target column name for target encoding
    max_categories : int, default 10
        Maximum categories for one-hot encoding

    Returns:
    --------
    pd.DataFrame
        DataFrame with encoded categorical variables
    """

def auto_scale(df: pd.DataFrame, method: str = 'standard', columns: list = None) -> pd.DataFrame:
    """Automatically scale numeric features.

    Methods:
    --------
    - 'standard': StandardScaler (mean=0, std=1)
    - 'minmax': MinMaxScaler (0 to 1)
    - 'robust': RobustScaler (median and IQR)
    - 'quantile': QuantileTransformer (uniform distribution)

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    method : str, default 'standard'
        Scaling method
    columns : list, optional
        Specific columns to scale (default: all numeric)

    Returns:
    --------
    pd.DataFrame
        DataFrame with scaled features
    """

def train_test_auto(df: pd.DataFrame, target: str, test_size: float = 0.2,
                   random_state: int = 42, stratify: bool = True) -> tuple:
    """Smart train-test splitting with automatic stratification.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target : str
        Target column name
    test_size : float, default 0.2
        Proportion for test set
    random_state : int, default 42
        Random state for reproducibility
    stratify : bool, default True
        Whether to stratify split based on target

    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
```

### 🤖 Machine Learning Module (`modeling.py`)

```python
class QuickModel:
    """Quick model training with minimal setup.

    Supported Models:
    -----------------
    Classification:
        - 'random_forest', 'rf'
        - 'xgboost', 'xgb'
        - 'lightgbm', 'lgb'
        - 'catboost', 'cat'
        - 'logistic', 'lr'
        - 'svm', 'svc'
        - 'knn'
        - 'naive_bayes', 'nb'

    Regression:
        - 'random_forest_reg', 'rf_reg'
        - 'xgboost_reg', 'xgb_reg'
        - 'lightgbm_reg', 'lgb_reg'
        - 'catboost_reg', 'cat_reg'
        - 'linear', 'linear_reg'
        - 'ridge'
        - 'lasso'
        - 'svr'
    """

    def __init__(self, model_name: str, task: str = 'auto', **kwargs):
        """Initialize QuickModel.

        Parameters:
        -----------
        model_name : str
            Name of the model algorithm
        task : str, default 'auto'
            'classification', 'regression', or 'auto'
        **kwargs : dict
            Model-specific parameters
        """

    def fit(self, X, y):
        """Fit the model to training data."""

    def predict(self, X):
        """Make predictions on new data."""

    def predict_proba(self, X):
        """Predict probabilities (classification only)."""

    def score(self, X, y):
        """Calculate model score."""

def compare_models(X, y, task: str = 'auto', cv: int = 5,
                  scoring: str = None, random_state: int = 42) -> pd.DataFrame:
    """Compare multiple machine learning models.

    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    task : str, default 'auto'
        'classification', 'regression', or 'auto'
    cv : int, default 5
        Cross-validation folds
    scoring : str, optional
        Scoring metric (auto-selected if None)
    random_state : int, default 42
        Random state for reproducibility

    Returns:
    --------
    pd.DataFrame
        Comparison results with mean scores and standard deviations
    """

def evaluate_model(model, X_test, y_test, task: str = 'auto',
                  plot: bool = True) -> dict:
    """Comprehensive model evaluation.

    Returns:
    --------
    dict
        Evaluation metrics including:
        - Classification: accuracy, precision, recall, f1, auc
        - Regression: mae, mse, rmse, r2, mape
    """
```

### 🎯 AutoML Module (`auto_ml.py`)

```python
def auto_hpo(model_name: str, X, y, method: str = 'optuna',
            max_evals: int = 100, cv: int = 5, scoring: str = None,
            random_state: int = 42, **kwargs):
    """Automated hyperparameter optimization.

    Parameters:
    -----------
    model_name : str
        Name of the model to optimize
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    method : str, default 'optuna'
        Optimization method: 'optuna', 'random', 'grid', 'bayesian'
    max_evals : int, default 100
        Maximum number of evaluations
    cv : int, default 5
        Cross-validation folds
    scoring : str, optional
        Optimization metric
    random_state : int, default 42
        Random state

    Returns:
    --------
    Optimized model instance
    """

def auto_tune_model(model_name: str, X, y, method: str = 'optuna',
                   max_evals: int = 100, **kwargs):
    """End-to-end automated model tuning.

    Features:
    ---------
    - Automatic task detection
    - Smart parameter space definition
    - Cross-validation based optimization
    - Best model training and return

    Returns:
    --------
    Tuned model ready for prediction
    """
```

### 🔧 Feature Engineering Module (`feature_engineering.py`)

```python
def create_polynomial_features(df: pd.DataFrame, degree: int = 2,
                              interaction_only: bool = False,
                              include_bias: bool = False,
                              columns: list = None) -> pd.DataFrame:
    """Create polynomial and interaction features.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    degree : int, default 2
        Degree of polynomial features
    interaction_only : bool, default False
        If True, only interaction features are produced
    include_bias : bool, default False
        If True, include bias column (all ones)
    columns : list, optional
        Specific columns to transform

    Returns:
    --------
    pd.DataFrame
        DataFrame with polynomial/interaction features
    """

def create_date_features(df: pd.DataFrame, date_cols: list,
                        drop_original: bool = False) -> pd.DataFrame:
    """Extract comprehensive date/time features.

    Created Features:
    -----------------
    - Basic: year, month, day, hour, minute
    - Derived: weekday, quarter, day_of_year, week_of_year
    - Binary: is_weekend, is_month_start, is_month_end
    - Cyclical: sin/cos transformations for cyclical features

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    date_cols : list
        List of datetime column names
    drop_original : bool, default False
        Whether to drop original date columns

    Returns:
    --------
    pd.DataFrame
        DataFrame with date features
    """

def create_target_encoding(df: pd.DataFrame, cat_cols: list, target_col: str,
                          smoothing: float = 10.0, min_samples: int = 10) -> pd.DataFrame:
    """Create target encoding for categorical variables.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    cat_cols : list
        Categorical columns to encode
    target_col : str
        Target column name
    smoothing : float, default 10.0
        Smoothing parameter for regularization
    min_samples : int, default 10
        Minimum samples per category

    Returns:
    --------
    pd.DataFrame
        DataFrame with target encoded features
    """

def create_aggregation_features(df: pd.DataFrame, group_col: str,
                               agg_cols: list, agg_funcs: list = None) -> pd.DataFrame:
    """Create aggregation features grouped by categorical column.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    group_col : str
        Column to group by
    agg_cols : list
        Columns to aggregate
    agg_funcs : list, optional
        Aggregation functions (default: ['mean', 'std', 'min', 'max'])

    Returns:
    --------
    pd.DataFrame
        DataFrame with aggregation features
    """

def select_features_univariate(X, y, k: int = 10, score_func=None):
    """Univariate feature selection.

    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    k : int, default 10
        Number of features to select
    score_func : callable, optional
        Scoring function (auto-selected if None)

    Returns:
    --------
    Selected features and selector object
    """

def apply_pca(df: pd.DataFrame, n_components: int = None,
              variance_threshold: float = 0.95) -> tuple:
    """Apply Principal Component Analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    n_components : int, optional
        Number of components (auto if None)
    variance_threshold : float, default 0.95
        Minimum variance to retain

    Returns:
    --------
    tuple
        (transformed_df, pca_object, explained_variance_ratio)
    """
```

### 📝 NLP Module (`nlp_utils.py`)

```python
def advanced_text_clean(df: pd.DataFrame, text_cols: list,
                       remove_urls: bool = True, remove_emails: bool = True,
                       remove_phone: bool = True, expand_contractions: bool = True,
                       remove_punctuation: bool = False) -> pd.DataFrame:
    """Advanced text preprocessing pipeline.

    Operations:
    -----------
    - URL removal
    - Email removal
    - Phone number removal
    - Contraction expansion
    - Case normalization
    - Whitespace normalization
    - Optional punctuation removal

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    text_cols : list
        Text column names
    remove_urls : bool, default True
        Remove URLs
    remove_emails : bool, default True
        Remove email addresses
    remove_phone : bool, default True
        Remove phone numbers
    expand_contractions : bool, default True
        Expand contractions (can't -> cannot)
    remove_punctuation : bool, default False
        Remove punctuation marks

    Returns:
    --------
    pd.DataFrame
        DataFrame with cleaned text
    """

def sentiment_analysis(df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
    """Perform sentiment analysis using TextBlob.

    Creates New Columns:
    -------------------
    - {col}_sentiment: Sentiment polarity (-1 to 1)
    - {col}_subjectivity: Subjectivity score (0 to 1)
    - {col}_sentiment_label: 'positive', 'negative', 'neutral'

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    text_cols : list
        Text column names

    Returns:
    --------
    pd.DataFrame
        DataFrame with sentiment features
    """

def extract_text_features(df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
    """Extract comprehensive text statistics.

    Features Created:
    -----------------
    - Character count
    - Word count
    - Sentence count
    - Average word length
    - Punctuation count
    - Capital letter count
    - Special character count

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    text_cols : list
        Text column names

    Returns:
    --------
    pd.DataFrame
        DataFrame with text features
    """

def extract_keywords(df: pd.DataFrame, text_col: str, top_n: int = 10) -> list:
    """Extract top keywords using TF-IDF.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    text_col : str
        Text column name
    top_n : int, default 10
        Number of top keywords

    Returns:
    --------
    list
        Top keywords with TF-IDF scores
    """
```

### ⏰ Time Series Module (`time_series_utils.py`)

```python
class TimeSeriesUtils:
    """Comprehensive time series analysis and feature engineering."""

    def create_comprehensive_time_features(self, df: pd.DataFrame,
                                         date_col: str,
                                         drop_original: bool = False) -> pd.DataFrame:
        """Create 25+ time-based features.

        Features Created:
        -----------------
        Basic Time Components:
        - year, month, day, hour, minute, second
        - weekday, day_of_year, week_of_year, quarter

        Binary Indicators:
        - is_weekend, is_month_start, is_month_end
        - is_quarter_start, is_quarter_end, is_year_start, is_year_end

        Cyclical Features:
        - month_sin, month_cos, day_sin, day_cos
        - hour_sin, hour_cos, weekday_sin, weekday_cos

        Business Calendar:
        - business_day_of_month, days_since_month_start

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        date_col : str
            Date column name
        drop_original : bool, default False
            Whether to drop original date column

        Returns:
        --------
        pd.DataFrame
            DataFrame with comprehensive time features
        """

    def detect_seasonality(self, series: pd.Series, periods: list = None) -> dict:
        """Detect seasonality patterns in time series.

        Parameters:
        -----------
        series : pd.Series
            Time series data
        periods : list, optional
            Periods to test (default: [7, 30, 365])

        Returns:
        --------
        dict
            Seasonality detection results with strengths
        """

    def add_lag_features(self, df: pd.DataFrame, col: str,
                        lags: list) -> pd.DataFrame:
        """Add lag features for time series.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        col : str
            Column to create lags for
        lags : list
            List of lag periods

        Returns:
        --------
        pd.DataFrame
            DataFrame with lag features
        """

    def add_rolling_features(self, df: pd.DataFrame, col: str,
                           windows: list, functions: list = None) -> pd.DataFrame:
        """Add rolling statistics features.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        col : str
            Column to calculate rolling features
        windows : list
            Window sizes for rolling calculations
        functions : list, optional
            Functions to apply (default: ['mean', 'std', 'min', 'max'])

        Returns:
        --------
        pd.DataFrame
            DataFrame with rolling features
        """
```

---

## 🔧 Advanced Classes

### Advanced Preprocessing

```python
class AdvancedPreprocessor:
    """Advanced data preprocessing with sophisticated algorithms."""

    def advanced_imputation(self, df: pd.DataFrame, method: str = 'iterative',
                           n_neighbors: int = 5, max_iter: int = 10) -> pd.DataFrame:
        """Advanced missing value imputation.

        Methods:
        --------
        - 'iterative': IterativeImputer using BayesianRidge
        - 'knn': KNN-based imputation
        - 'mice': Multiple Imputation by Chained Equations
        """

    def detect_and_handle_outliers(self, df: pd.DataFrame, method: str = 'isolation_forest',
                                  contamination: float = 0.1) -> pd.DataFrame:
        """Advanced outlier detection and handling.

        Methods:
        --------
        - 'isolation_forest': Isolation Forest algorithm
        - 'local_outlier_factor': Local Outlier Factor
        - 'one_class_svm': One-Class SVM
        - 'elliptic_envelope': Robust covariance estimation
        """

    def smart_feature_selection(self, X, y, method: str = 'mutual_info',
                               k: int = 10, task: str = 'auto') -> tuple:
        """Intelligent feature selection.

        Methods:
        --------
        - 'mutual_info': Mutual information
        - 'chi2': Chi-squared test
        - 'f_classif': F-test for classification
        - 'f_regression': F-test for regression
        - 'rfe': Recursive Feature Elimination
        """
```

### Model Validation

```python
class ModelValidator:
    """Advanced model validation and analysis."""

    def comprehensive_cross_validation(self, model, X, y, cv_strategies: list = None) -> dict:
        """Multi-strategy cross-validation analysis."""

    def stability_analysis(self, model, X, y, n_iterations: int = 10) -> dict:
        """Analyze model stability across different data splits."""

    def learning_curve_analysis(self, model, X, y, train_sizes: list = None) -> dict:
        """Generate learning curves to analyze training efficiency."""

    def bias_variance_analysis(self, model, X, y, n_iterations: int = 100) -> dict:
        """Bias-variance decomposition analysis."""

class ModelInterpreter:
    """Model interpretation and explainability."""

    def permutation_importance(self, model, X, y, n_repeats: int = 10) -> dict:
        """Calculate permutation-based feature importance."""

    def partial_dependence_analysis(self, model, X, features: list) -> dict:
        """Analyze partial dependence of features."""

    def local_explanation(self, model, X_instance, X_train, mode: str = 'classification') -> dict:
        """LIME-based local explanation for single instance."""
```

### Data Auditing

```python
class DataAuditor:
    """Comprehensive data quality auditing."""

    def comprehensive_audit(self, df: pd.DataFrame, target_col: str = None) -> dict:
        """Complete data quality audit.

        Audit Components:
        -----------------
        - Data quality score (0-100)
        - Missing value analysis
        - Data type consistency
        - Outlier detection
        - Duplicate analysis
        - Target correlation analysis
        - Statistical summaries
        """

    def print_audit_report(self):
        """Print formatted audit report."""

class DataSampler:
    """Advanced data sampling strategies."""

    def stratified_sample(self, df: pd.DataFrame, target_col: str,
                         n_samples: int, random_state: int = 42) -> pd.DataFrame:
        """Stratified sampling maintaining target distribution."""

    def time_aware_sample(self, df: pd.DataFrame, date_col: str,
                         n_samples: int, method: str = 'recent') -> pd.DataFrame:
        """Time-aware sampling strategies."""

    def balanced_sample(self, df: pd.DataFrame, target_col: str,
                       method: str = 'undersample') -> pd.DataFrame:
        """Balanced sampling for imbalanced datasets."""
```

### Database Utilities

```python
class DatabaseConnector:
    """Universal database connectivity."""

    def connect(self, db_type: str, connection_params: dict) -> str:
        """Connect to database.

        Supported Databases:
        -------------------
        - 'sqlite': SQLite database
        - 'mysql': MySQL database
        - 'postgresql': PostgreSQL database
        - 'oracle': Oracle database
        - 'sqlserver': SQL Server database
        """

    def execute_query(self, query: str, connection_name: str = 'default') -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""

    def insert_dataframe(self, df: pd.DataFrame, table_name: str,
                        connection_name: str = 'default', if_exists: str = 'append') -> bool:
        """Insert DataFrame into database table."""

class DataProfiler:
    """Advanced data profiling and analysis."""

    def comprehensive_profile(self, df: pd.DataFrame) -> dict:
        """Generate comprehensive data profile."""

class DataExporter:
    """Multi-format data export utilities."""

    def export_data(self, df: pd.DataFrame, filepath: str, format: str) -> bool:
        """Export data in various formats."""
```

### Hyperplane Analysis

```python
class Hyperplane:
    """2D and 3D hyperplane representation and visualization."""

    def __init__(self, weights: Union[List, np.ndarray], bias: float):
        """Initialize hyperplane with weights and bias.

        Parameters:
        -----------
        weights : array-like
            Weight vector (2D or 3D)
        bias : float
            Bias term
        """

    def equation(self) -> str:
        """Get mathematical equation representation."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classify points as +1 or -1 based on hyperplane side."""

    def distance(self, point: Union[List, np.ndarray]) -> float:
        """Calculate perpendicular distance from point to hyperplane."""

    def plot_2d(self, x_range: Tuple[float, float] = (-5, 5),
                X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                show_margin: bool = False) -> None:
        """Plot 2D hyperplane with optional data points and margin."""

    def plot_3d(self, range_val: Tuple[float, float] = (-5, 5),
                X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> None:
        """Plot 3D hyperplane with optional data points."""

    def plot_decision_regions(self, X: np.ndarray, y: np.ndarray,
                             resolution: int = 100) -> None:
        """Plot color-coded decision regions for 2D data."""

class HyperplaneExtractor:
    """Extract hyperplanes from trained ML models."""

    def __init__(self, model):
        """Initialize with trained sklearn model.

        Supported Models:
        ----------------
        - LinearSVC, SVC(kernel='linear')
        - LogisticRegression
        - Perceptron
        - LinearRegression, Ridge, Lasso
        - LinearDiscriminantAnalysis
        - GaussianNB (approximate)
        - DecisionTree (approximate)
        """

    def get_hyperplane(self) -> Hyperplane:
        """Get the extracted Hyperplane object."""

    def equation(self) -> str:
        """Get hyperplane equation string."""

    def analyze_model(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Comprehensive hyperplane analysis.

        Returns:
        --------
        dict
            Analysis including distances, margins, weights, etc.
        """

    def compare_models(self, other_extractor: 'HyperplaneExtractor') -> dict:
        """Compare two hyperplane models.

        Returns:
        --------
        dict
            Comparison metrics including angle, weight differences, etc.
        """

    def plot_2d(self, X: Optional[np.ndarray] = None,
                y: Optional[np.ndarray] = None, **kwargs) -> None:
        """Plot 2D hyperplane with model-specific enhancements."""

    def plot_decision_regions(self, X: np.ndarray, y: np.ndarray) -> None:
        """Plot decision regions for the extracted hyperplane."""

    # Algorithm-Specific Plotting Methods
    def plot_svm(self, X: np.ndarray, y: np.ndarray,
                 show_support_vectors: bool = True, margin_style: str = 'dashed') -> None:
        """SVM-specific plotting with margins and support vector highlighting."""

    def plot_logistic_regression(self, X: np.ndarray, y: np.ndarray,
                                show_probabilities: bool = True,
                                probability_contours: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]) -> None:
        """Logistic regression plotting with probability contours."""

    def plot_perceptron(self, X: np.ndarray, y: np.ndarray,
                       show_misclassified: bool = True) -> None:
        """Perceptron plotting with misclassified points highlighted."""

    def plot_lda(self, X: np.ndarray, y: np.ndarray,
                 show_class_centers: bool = True, show_projections: bool = False) -> None:
        """LDA plotting with class centers and projection directions."""

    def plot_linear_regression(self, X: np.ndarray, y: np.ndarray,
                              show_residuals: bool = True, confidence_interval: bool = False) -> None:
        """Linear regression plotting with residuals (1D/2D support)."""

    def plot_algorithm_comparison(self, models_dict: dict, X: np.ndarray, y: np.ndarray) -> None:
        """Compare multiple algorithm hyperplanes in subplots."""

# Convenience Functions
def extract_hyperplane(model) -> HyperplaneExtractor:
    """Extract hyperplane from any supported ML model."""

def create_hyperplane_from_points(points: List[np.ndarray]) -> Hyperplane:
    """Create hyperplane from defining points.

    Parameters:
    -----------
    points : list of arrays
        2 points for 2D line, 3 points for 3D plane
    """

# Algorithm-Specific Plotting Functions
def plot_svm_hyperplane(model, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
    """Quick SVM hyperplane plotting with margins."""

def plot_logistic_hyperplane(model, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
    """Quick logistic regression hyperplane plotting with probability contours."""

def plot_perceptron_hyperplane(model, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
    """Quick perceptron hyperplane plotting with misclassified points."""

def plot_lda_hyperplane(model, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
    """Quick LDA hyperplane plotting with class centers."""

def plot_linear_regression_hyperplane(model, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
    """Quick linear regression hyperplane plotting with residuals."""

def compare_algorithm_hyperplanes(models_dict: dict, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
    """Compare multiple algorithm hyperplanes in one visualization."""
```

### Model Deployment

```python
class ModelPersistence:
    """Model serialization and persistence."""

    def save_model(self, model, filepath: str, format: str = 'joblib',
                  metadata: dict = None) -> bool:
        """Save model with metadata."""

    def load_model(self, filepath: str, format: str = 'joblib'):
        """Load saved model."""

class ExperimentTracker:
    """ML experiment tracking and logging."""

    def start_experiment(self, experiment_name: str, description: str = '') -> str:
        """Start new experiment."""

    def log_parameter(self, key: str, value) -> None:
        """Log experiment parameter."""

    def log_metric(self, key: str, value: float, step: int = None) -> None:
        """Log experiment metric."""

    def end_experiment(self, status: str = 'completed') -> None:
        """End current experiment."""

class ConfigManager:
    """Configuration management."""

    def load_config(self, config_path: str = 'config.json') -> dict:
        """Load configuration file."""

    def get(self, key: str, default=None):
        """Get configuration value."""

    def set(self, key: str, value) -> None:
        """Set configuration value."""

class DataVersioning:
    """Data versioning and lineage tracking."""

    def create_version(self, df: pd.DataFrame, dataset_name: str,
                      description: str = '') -> str:
        """Create new data version."""

    def load_version(self, version_id: str) -> pd.DataFrame:
        """Load specific data version."""

    def list_versions(self, dataset_name: str = None) -> pd.DataFrame:
        """List all versions."""
```

---

## 🎯 Usage Patterns

### 1. Method Chaining Pattern

```python
from dskit import dskit

# Fluent API with method chaining
result = (dskit.load("data.csv")
          .fix_dtypes()
          .rename_columns_auto()
          .fill_missing(strategy='auto')
          .remove_outliers(method='iqr')
          .auto_encode(target_col='target')
          .auto_scale(method='standard')
          .train_advanced('xgboost')
          .evaluate())

print(f"Model accuracy: {result.score}")
```

### 2. Functional Pattern

```python
import dskit

# Step-by-step functional approach
df = dskit.load("data.csv")
df = dskit.fix_dtypes(df)
df = dskit.fill_missing(df, strategy='auto')
df = dskit.remove_outliers(df, method='iqr')

# Model training
model = dskit.QuickModel('xgboost')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 3. Class-Based Pattern

```python
from dskit import AdvancedPreprocessor, ModelValidator

# Advanced preprocessing
preprocessor = AdvancedPreprocessor()
df_clean = preprocessor.advanced_imputation(df, method='iterative')
df_clean = preprocessor.detect_and_handle_outliers(df_clean, method='isolation_forest')

# Model validation
validator = ModelValidator()
cv_results = validator.comprehensive_cross_validation(model, X, y)
stability_results = validator.stability_analysis(model, X, y)
```

### 11. Hyperplane Analysis and Visualization

```python
# Extract hyperplane from trained model
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Train models
svm_model = LinearSVC(random_state=42).fit(X, y)
lr_model = LogisticRegression(random_state=42).fit(X, y)
perceptron_model = Perceptron(random_state=42).fit(X, y)
lda_model = LinearDiscriminantAnalysis().fit(X, y)

# Method 1: Direct algorithm-specific plotting
dskit.plot_svm_hyperplane(svm_model, X, y)  # SVM with margins
dskit.plot_logistic_hyperplane(lr_model, X, y)  # With probability contours
dskit.plot_perceptron_hyperplane(perceptron_model, X, y)  # Highlight misclassified
dskit.plot_lda_hyperplane(lda_model, X, y)  # With class centers

# Method 2: Extractor with algorithm-specific methods
extractor = dskit.extract_hyperplane(svm_model)
extractor.plot_svm(X, y, show_support_vectors=True, margin_style='dashed')

lr_extractor = dskit.extract_hyperplane(lr_model)
lr_extractor.plot_logistic_regression(X, y,
                                     probability_contours=[0.1, 0.3, 0.5, 0.7, 0.9],
                                     show_probabilities=True)

perceptron_extractor = dskit.extract_hyperplane(perceptron_model)
perceptron_extractor.plot_perceptron(X, y, show_misclassified=True)

lda_extractor = dskit.extract_hyperplane(lda_model)
lda_extractor.plot_lda(X, y, show_class_centers=True, show_projections=True)

# Compare multiple algorithms
models = {'SVM': svm_model, 'Logistic': lr_model, 'Perceptron': perceptron_model, 'LDA': lda_model}
dskit.compare_algorithm_hyperplanes(models, X, y)

# Basic hyperplane operations
print(f"SVM Equation: {extractor.equation()}")
distances = [extractor.hyperplane.distance(point) for point in X]

# Advanced analysis
analysis = extractor.analyze_model(X, y)
print(f"Margin: {analysis['margin']:.3f}")
print(f"Mean distance: {analysis['mean_distance']:.3f}")

# Model comparison
comparison = extractor.compare_models(lr_extractor)
print(f"Angle between hyperplanes: {comparison['angle_between_normals']:.3f}°")

# Linear regression with residuals
from sklearn.linear_model import LinearRegression
reg_model = LinearRegression().fit(X, y_continuous)
dskit.plot_linear_regression_hyperplane(reg_model, X, y_continuous, show_residuals=True)

# Create custom hyperplane
hp = dskit.Hyperplane([1, -2], 3)  # x - 2y + 3 = 0
hp.plot_2d(x_range=(-5, 5))

# Hyperplane from points
points_2d = [[0, 1], [2, 3]]
hp_from_points = dskit.create_hyperplane_from_points(points_2d)
hp_from_points.plot_2d()
```

---

## 🔧 Configuration and Settings

```python
from dskit.config import dskitConfig

# Global configuration
config = dskitConfig()
config.set_visualization_backend('plotly')  # or 'matplotlib'
config.set_random_state(42)
config.set_n_jobs(-1)
config.set_verbosity(1)

# Get current settings
settings = config.get_all_settings()
```

---

## 📊 Return Types and Data Structures

### Standard Return Types

- **DataFrames**: Most functions return `pd.DataFrame`
- **Models**: Training functions return fitted model objects
- **Dictionaries**: Analysis functions return structured results
- **Tuples**: Functions returning multiple values use tuples
- **Boolean**: Success/failure operations return boolean

### Analysis Results Structure

```python
# Model evaluation results
{
    'accuracy': 0.95,
    'precision': 0.94,
    'recall': 0.96,
    'f1_score': 0.95,
    'confusion_matrix': array([[...]]),
    'classification_report': {...}
}

# Data audit results
{
    'overall_score': 88.1,
    'data_quality': {...},
    'missing_analysis': {...},
    'outlier_analysis': {...},
    'duplicate_analysis': {...}
}
```

---

This API reference provides detailed documentation for all dskit functionality. For examples and tutorials, see the main documentation and example notebooks.
