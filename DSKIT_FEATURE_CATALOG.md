# 🔧 dskit Feature Catalog & Parameter Guide

**dskit Feature Catalog** - Complete reference of all 221 functions with descriptions and configurable parameters.

---

## 📊 **Core Data Operations**

### **Data Loading & I/O**

#### `dskit.load(filepath, **kwargs)`

**Description**: Universal data loader with automatic format detection for CSV, Excel, JSON, Parquet files.
**Syntax**:

```python
# Basic usage
kit = dskit.load("data.csv")

# Custom parameters
kit = dskit.load("data.csv", sep=';', encoding='latin-1', nrows=5000)
kit = dskit.load("data.xlsx", sheet_name='Sheet2', skiprows=1)
kit = dskit.load("data.json", na_values=['NULL', 'n/a'])
```

**Editable Parameters**:

- `sep` (str): Column separator for CSV files (default: ',')
- `encoding` (str): File encoding (default: 'utf-8')
- `sheet_name` (str/int): Excel sheet name or index (default: 0)
- `nrows` (int): Number of rows to read (default: None)
- `skiprows` (int): Number of rows to skip (default: None)
- `na_values` (list): Additional strings to recognize as NA/NaN (default: None)

#### `dskit.save(df, filepath, **kwargs)`

**Description**: Multi-format data export with automatic format detection.
**Editable Parameters**:

- `index` (bool): Write row names (default: True)
- `sep` (str): Column separator for CSV (default: ',')
- `encoding` (str): File encoding (default: 'utf-8')
- `compression` (str): Compression type ('gzip', 'bz2', 'zip', etc.)

#### `read_folder(folder_path, file_type='csv', **kwargs)`

**Description**: Batch process and combine multiple files from a directory.
**Editable Parameters**:

- `file_type` (str): Type of files to read ('csv', 'excel', 'json')
- `pattern` (str): File name pattern to match (default: None)
- `recursive` (bool): Search subdirectories (default: False)

---

## 🧹 **Data Cleaning & Preprocessing**

### **Data Type Management**

#### `fix_dtypes()`

**Description**: Automatically detect and fix data types for optimal memory usage.
**Syntax**:

```python
# Basic usage
kit.fix_dtypes()

# Custom parameters
kit.fix_dtypes(infer_datetime=False, downcast_integers=True)
kit.fix_dtypes(category_threshold=0.3)  # More aggressive categorization
```

**Editable Parameters**:

- `infer_datetime` (bool): Attempt datetime conversion (default: True)
- `downcast_integers` (bool): Use smallest integer type (default: True)
- `category_threshold` (float): Threshold for categorical conversion (default: 0.5)

#### `rename_columns_auto()`

**Description**: Clean column names by removing special characters and standardizing format.
**Editable Parameters**:

- `case` (str): Target case ('lower', 'upper', 'title', 'snake')
- `remove_special` (bool): Remove special characters (default: True)
- `max_length` (int): Maximum column name length (default: None)

### **Missing Data Handling**

#### `fill_missing(strategy='auto', fill_value=None)`

**Description**: Fill missing values using various imputation strategies.
**Syntax**:

```python
# Basic strategies
kit.fill_missing(strategy='mean')
kit.fill_missing(strategy='median')
kit.fill_missing(strategy='mode')

# Advanced strategies
kit.fill_missing(strategy='interpolate', method='polynomial')
kit.fill_missing(strategy='forward', limit=3)
kit.fill_missing(fill_value=0)  # Constant fill
```

**Editable Parameters**:

- `strategy` (str): 'mean', 'median', 'mode', 'forward', 'backward', 'interpolate', 'auto'
- `fill_value` (any): Custom fill value for constant strategy
- `limit` (int): Maximum number of consecutive NaN values to fill
- `method` (str): Interpolation method ('linear', 'polynomial', 'spline')

#### `advanced_imputation(method='knn', **kwargs)`

**Description**: Advanced missing value imputation using machine learning techniques.
**Editable Parameters**:

- `method` (str): 'knn', 'iterative', 'mice', 'missforest'
- `n_neighbors` (int): Number of neighbors for KNN (default: 5)
- `max_iter` (int): Maximum iterations for iterative methods (default: 10)
- `random_state` (int): Random seed for reproducibility

### **Outlier Detection & Removal**

#### `remove_outliers(method='iqr', threshold=1.5)`

**Description**: Detect and remove outliers using statistical methods.
**Syntax**:

```python
# Statistical methods
kit.remove_outliers(method='iqr', threshold=2.0)  # More conservative
kit.remove_outliers(method='zscore', threshold=2.5)  # Custom z-score

# Machine learning methods
kit.remove_outliers(method='isolation_forest', contamination=0.05)
kit.remove_outliers(method='local_outlier_factor', contamination=0.1)

# Specific columns
kit.remove_outliers(method='iqr', columns=['age', 'income'])
```

**Editable Parameters**:

- `method` (str): 'iqr', 'zscore', 'isolation_forest', 'local_outlier_factor'
- `threshold` (float): Sensitivity threshold (1.5 for IQR, 3.0 for Z-score)
- `contamination` (float): Proportion of outliers expected (0.1 for ML methods)
- `columns` (list): Specific columns to check (default: all numeric)

#### `detect_outliers_advanced(method='isolation_forest', **kwargs)`

**Description**: Advanced outlier detection using machine learning algorithms.
**Editable Parameters**:

- `method` (str): 'isolation_forest', 'one_class_svm', 'elliptic_envelope'
- `contamination` (float): Expected outlier proportion (default: 0.1)
- `nu` (float): Upper bound on training errors for SVM (default: 0.05)
- `gamma` (str/float): Kernel coefficient for SVM ('scale', 'auto', or float)

---

## 📈 **Exploratory Data Analysis**

### **Statistical Analysis**

#### `describe_advanced()`

**Description**: Comprehensive statistical summary beyond basic describe().
**Editable Parameters**:

- `include_all` (bool): Include all data types (default: True)
- `percentiles` (list): Custom percentiles to calculate (default: [0.25, 0.5, 0.75])
- `datetime_stats` (bool): Include datetime-specific statistics (default: True)

#### `correlation_analysis(method='pearson', threshold=0.5)`

**Description**: Analyze correlations between variables with various methods.
**Syntax**:

```python
# Basic correlation analysis
kit.correlation_analysis()

# Different correlation methods
kit.correlation_analysis(method='spearman', threshold=0.3)
kit.correlation_analysis(method='kendall', plot=True, figsize=(12, 10))
kit.correlation_analysis(method='mutual_info', threshold=0.1)

# No plotting
corr_results = kit.correlation_analysis(plot=False)
```

**Editable Parameters**:

- `method` (str): 'pearson', 'spearman', 'kendall', 'mutual_info'
- `threshold` (float): Minimum correlation threshold to report (default: 0.5)
- `plot` (bool): Generate correlation heatmap (default: True)
- `figsize` (tuple): Figure size for plot (default: (10, 8))

### **Distribution Analysis**

#### `distribution_analysis(columns=None, plot=True)`

**Description**: Analyze distributions with normality tests and visualizations.
**Editable Parameters**:

- `columns` (list): Specific columns to analyze (default: all numeric)
- `plot` (bool): Generate distribution plots (default: True)
- `test_normality` (bool): Perform normality tests (default: True)
- `bins` (int): Number of histogram bins (default: 30)

#### `qq_plot_analysis(columns=None, distribution='norm')`

**Description**: Generate Q-Q plots to assess distribution fit.
**Editable Parameters**:

- `columns` (list): Columns to analyze (default: all numeric)
- `distribution` (str): Reference distribution ('norm', 'uniform', 'expon')
- `alpha` (float): Significance level for confidence bands (default: 0.05)

---

## 🔄 **Feature Engineering**

### **Feature Creation**

#### `create_polynomial_features(degree=2, interaction_only=False)`

**Description**: Generate polynomial and interaction features.
**Syntax**:

```python
# Basic polynomial features
kit.create_polynomial_features(degree=2)

# Only interaction terms
kit.create_polynomial_features(degree=3, interaction_only=True)

# Specific columns
kit.create_polynomial_features(degree=2, columns=['age', 'income'], include_bias=False)

# Higher degree polynomials
kit.create_polynomial_features(degree=4, interaction_only=False)
```

**Editable Parameters**:

- `degree` (int): Maximum degree of polynomial features (default: 2)
- `interaction_only` (bool): Only interaction terms, no powers (default: False)
- `include_bias` (bool): Include bias column (default: True)
- `columns` (list): Specific columns to transform (default: all numeric)

#### `create_date_features(date_columns, include_cyclical=True)`

**Description**: Extract comprehensive features from datetime columns.
**Editable Parameters**:

- `date_columns` (list): Column names containing dates
- `include_cyclical` (bool): Add cyclical encodings (sin/cos) (default: True)
- `include_elapsed` (bool): Days/seconds since reference date (default: True)
- `reference_date` (datetime): Reference point for elapsed time calculations

#### `create_lag_features(columns, lags=[1, 2, 3])`

**Description**: Create lagged versions of time series features.
**Editable Parameters**:

- `columns` (list): Columns to create lags for
- `lags` (list): List of lag periods (default: [1, 2, 3])
- `fill_method` (str): How to fill initial missing values ('mean', 'zero', 'drop')

### **Feature Transformation**

#### `apply_pca(n_components=None, variance_threshold=0.95)`

**Description**: Apply Principal Component Analysis for dimensionality reduction.
**Editable Parameters**:

- `n_components` (int/float): Number of components or variance ratio to retain
- `variance_threshold` (float): Cumulative variance to preserve (default: 0.95)
- `whiten` (bool): Whiten components (default: False)
- `random_state` (int): Random seed for reproducibility

#### `scale_features(method='standard', columns=None)`

**Description**: Scale features using various normalization methods.
**Syntax**:

```python
# Standard scaling (z-score normalization)
kit.scale_features(method='standard')

# MinMax scaling to [0, 1]
kit.scale_features(method='minmax')

# MinMax scaling to custom range
kit.scale_features(method='minmax', feature_range=(-1, 1))

# Robust scaling (less sensitive to outliers)
kit.scale_features(method='robust', quantile_range=(10.0, 90.0))

# Specific columns
kit.scale_features(method='standard', columns=['age', 'income', 'score'])
```

**Editable Parameters**:

- `method` (str): 'standard', 'minmax', 'robust', 'quantile', 'power'
- `columns` (list): Specific columns to scale (default: all numeric)
- `feature_range` (tuple): Range for MinMax scaling (default: (0, 1))
- `quantile_range` (tuple): Quantile range for robust scaling (default: (25.0, 75.0))

### **Text Feature Engineering**

#### `sentiment_analysis(text_columns, method='textblob')`

**Description**: Perform sentiment analysis on text columns.
**Editable Parameters**:

- `text_columns` (list): Column names containing text
- `method` (str): 'textblob', 'vader', 'custom'
- `language` (str): Text language for analysis (default: 'en')
- `normalize` (bool): Normalize scores to [-1, 1] range (default: True)

#### `extract_text_features(text_columns, method='tfidf')`

**Description**: Extract numerical features from text data.
**Editable Parameters**:

- `text_columns` (list): Text column names
- `method` (str): 'tfidf', 'count', 'word2vec', 'doc2vec'
- `max_features` (int): Maximum number of features (default: 1000)
- `ngram_range` (tuple): N-gram range (default: (1, 2))
- `min_df` (int/float): Minimum document frequency (default: 2)
- `max_df` (float): Maximum document frequency (default: 0.95)

---

## 🎯 **Machine Learning**

### **Model Training**

#### `train_model(target, task='auto', algorithm='auto')`

**Description**: Train machine learning models with automatic algorithm selection.
**Syntax**:

```python
# Automatic model selection
kit.train_model('target_column')

# Specific algorithm
kit.train_model('price', task='regression', algorithm='xgb')
kit.train_model('churn', task='classification', algorithm='rf')

# Custom parameters
kit.train_model('target', algorithm='svm', test_size=0.3, cv_folds=10, random_state=42)

# Multiple algorithms comparison
kit.train_model('target', algorithm=['rf', 'xgb', 'svm'])
```

**Editable Parameters**:

- `target` (str): Target column name
- `task` (str): 'classification', 'regression', 'auto'
- `algorithm` (str): 'auto', 'xgb', 'rf', 'svm', 'lr', etc.
- `test_size` (float): Train/test split ratio (default: 0.2)
- `random_state` (int): Random seed for reproducibility
- `cv_folds` (int): Cross-validation folds (default: 5)

#### `auto_tune(target, method='optuna', max_evals=100)`

**Description**: Automated hyperparameter tuning using optimization libraries.
**Editable Parameters**:

- `target` (str): Target column name
- `method` (str): 'optuna', 'hyperopt', 'grid', 'random'
- `max_evals` (int): Maximum optimization iterations (default: 100)
- `timeout` (int): Maximum time in seconds (default: 3600)
- `cv_folds` (int): Cross-validation folds (default: 5)
- `scoring` (str): Optimization metric ('accuracy', 'f1', 'roc_auc', 'r2')

### **Model Evaluation**

#### `evaluate_model()`

**Description**: Comprehensive model evaluation with multiple metrics.
**Editable Parameters**:

- `metrics` (list): Specific metrics to calculate
- `plot_confusion_matrix` (bool): Generate confusion matrix (default: True)
- `plot_roc` (bool): Generate ROC curve (default: True)
- `plot_feature_importance` (bool): Plot feature importance (default: True)
- `figsize` (tuple): Figure size for plots (default: (12, 8))

#### `cross_validate_advanced(cv=5, scoring=None)`

**Description**: Advanced cross-validation with multiple metrics and visualization.
**Editable Parameters**:

- `cv` (int): Number of cross-validation folds (default: 5)
- `scoring` (list/str): Scoring metrics to use
- `return_train_score` (bool): Include training scores (default: True)
- `shuffle` (bool): Shuffle data before splitting (default: True)
- `stratify` (bool): Stratified sampling for classification (default: True)

### **Model Comparison**

#### `compare_models(target, algorithms=['xgb', 'rf', 'svm'], task='auto')`

**Description**: Compare multiple algorithms on the same dataset.
**Editable Parameters**:

- `target` (str): Target column name
- `algorithms` (list): List of algorithms to compare
- `task` (str): 'classification', 'regression', 'auto'
- `cv_folds` (int): Cross-validation folds (default: 5)
- `scoring` (str): Primary metric for comparison
- `plot_comparison` (bool): Generate comparison plots (default: True)

---

## 📐 **Hyperplane Analysis**

### **Core Hyperplane Classes**

#### `Hyperplane(weights, bias)`

**Description**: Mathematical representation of hyperplanes with visualization capabilities.
**Syntax**:

```python
from dskit.hyperplane import Hyperplane

# Create hyperplane manually
hyperplane = Hyperplane(weights=[1.5, -0.8], bias=0.3)

# Get equation
print(hyperplane.equation())  # "1.5*x1 + -0.8*x2 + 0.3 = 0"

# Classify points
predictions = hyperplane.predict(X_test)  # Returns +1 or -1

# Calculate distances
distances = hyperplane.distance(X_test)  # Perpendicular distances

# Visualize 2D
hyperplane.plot_2d(X_test, y_test, margin=True, figsize=(10, 8))

# Visualize 3D (for 3D data)
hyperplane.plot_3d(X_test_3d, y_test, alpha=0.7)
```

**Editable Parameters**:

- `weights` (array): Hyperplane normal vector weights
- `bias` (float): Hyperplane bias term
- `tolerance` (float): Numerical tolerance for calculations (default: 1e-10)

**Methods**:

- `equation()` - Get mathematical equation string
- `predict(X)` - Classify points (+1/-1)
- `distance(point)` - Calculate perpendicular distance
- `plot_2d(X, y, margin=True)` - 2D visualization
- `plot_3d(X, y)` - 3D visualization

#### `HyperplaneExtractor(model)`

**Description**: Extract hyperplane parameters from trained ML models.
**Editable Parameters**:

- `model` (sklearn model): Trained linear model
- `feature_names` (list): Names of input features (default: None)
- `class_names` (list): Names of target classes (default: None)

**Methods**:

- `extract_hyperplane(X_train)` - Extract hyperplane from model
- `analyze_model(X, y)` - Comprehensive analysis
- `compare_models(other_extractor)` - Compare two models

### **Algorithm-Specific Plotting Functions**

#### `plot_svm(model, X, y, **kwargs)`

**Description**: Specialized SVM hyperplane visualization with support vectors.
**Syntax**:

```python
from dskit.hyperplane import plot_svm
from sklearn.svm import SVC

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Basic SVM plot
plot_svm(svm_model, X_test, y_test)

# Customized SVM plot
plot_svm(svm_model, X_test, y_test,
         show_support_vectors=True,
         show_margins=True,
         alpha=0.8,
         figsize=(12, 10),
         title="Custom SVM Decision Boundary")

# Minimal SVM plot
plot_svm(svm_model, X_test, y_test,
         show_support_vectors=False,
         show_margins=False)
```

**Editable Parameters**:

- `model` (SVM): Trained SVM model
- `X` (array): Feature matrix
- `y` (array): Target vector
- `show_support_vectors` (bool): Highlight support vectors (default: True)
- `show_margins` (bool): Display margin boundaries (default: True)
- `alpha` (float): Point transparency (default: 0.6)
- `figsize` (tuple): Figure size (default: (10, 8))
- `title` (str): Plot title (default: auto-generated)

#### `plot_logistic_regression(model, X, y, **kwargs)`

**Description**: Logistic regression decision boundary with probability contours.
**Syntax**:

```python
from dskit.hyperplane import plot_logistic_regression
from sklearn.linear_model import LogisticRegression

# Train logistic regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Basic plot with probability contours
plot_logistic_regression(lr_model, X_test, y_test)

# Custom probability levels
plot_logistic_regression(lr_model, X_test, y_test,
                        probability_levels=[0.2, 0.4, 0.6, 0.8],
                        colormap='plasma')

# High-resolution plot
plot_logistic_regression(lr_model, X_test, y_test,
                        resolution=200,
                        show_probabilities=True)

# Simple decision boundary only
plot_logistic_regression(lr_model, X_test, y_test,
                        show_probabilities=False)
```

**Editable Parameters**:

- `model` (LogisticRegression): Trained model
- `X` (array): Feature matrix
- `y` (array): Target vector
- `show_probabilities` (bool): Display probability contours (default: True)
- `probability_levels` (list): Contour levels (default: [0.1, 0.3, 0.5, 0.7, 0.9])
- `resolution` (int): Grid resolution for contours (default: 100)
- `colormap` (str): Colormap name (default: 'viridis')

#### `plot_perceptron(model, X, y, **kwargs)`

**Description**: Perceptron decision boundary visualization.
**Editable Parameters**:

- `model` (Perceptron): Trained perceptron model
- `X` (array): Feature matrix
- `y` (array): Target vector
- `show_decision_function` (bool): Show decision function values (default: True)
- `show_margin` (bool): Display margin around boundary (default: False)
- `line_width` (float): Decision boundary line width (default: 2)
- `point_size` (int): Data point size (default: 50)

#### `plot_lda(model, X, y, **kwargs)`

**Description**: Linear Discriminant Analysis boundary visualization.
**Editable Parameters**:

- `model` (LDA): Trained LDA model
- `X` (array): Feature matrix
- `y` (array): Target vector
- `show_class_centers` (bool): Display class centroids (default: True)
- `show_covariance` (bool): Show covariance ellipses (default: False)
- `center_marker_size` (int): Size of class center markers (default: 200)
- `ellipse_alpha` (float): Covariance ellipse transparency (default: 0.3)

#### `plot_linear_regression(model, X, y, **kwargs)`

**Description**: Linear regression line with confidence intervals.
**Editable Parameters**:

- `model` (LinearRegression): Trained model
- `X` (array): Feature matrix (must be 1D or 2D)
- `y` (array): Target vector
- `show_confidence` (bool): Display confidence intervals (default: True)
- `confidence_level` (float): Confidence level (default: 0.95)
- `show_residuals` (bool): Display residual points (default: False)
- `line_color` (str): Regression line color (default: 'red')
- `scatter_alpha` (float): Data point transparency (default: 0.6)

#### `plot_algorithm_comparison(models, X, y, **kwargs)`

**Description**: Side-by-side comparison of multiple algorithm hyperplanes.
**Syntax**:

```python
from dskit.hyperplane import plot_algorithm_comparison
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Train multiple models
svm_model = SVC(kernel='linear').fit(X_train, y_train)
lr_model = LogisticRegression().fit(X_train, y_train)
perceptron_model = Perceptron().fit(X_train, y_train)
lda_model = LinearDiscriminantAnalysis().fit(X_train, y_train)

# Compare all models
models = [svm_model, lr_model, perceptron_model, lda_model]
plot_algorithm_comparison(models, X_test, y_test)

# Custom comparison
plot_algorithm_comparison(models, X_test, y_test,
                         model_names=['SVM', 'Logistic', 'Perceptron', 'LDA'],
                         subplot_layout=(2, 2),
                         figsize=(16, 12),
                         share_axes=False)

# Compare two models
plot_algorithm_comparison([svm_model, lr_model], X_test, y_test,
                         model_names=['SVM', 'Logistic Regression'],
                         subplot_layout=(1, 2))
```

**Editable Parameters**:

- `models` (list): List of trained models to compare
- `X` (array): Feature matrix
- `y` (array): Target vector
- `model_names` (list): Custom names for models (default: auto-generated)
- `subplot_layout` (tuple): Subplot grid layout (default: auto)
- `figsize` (tuple): Overall figure size (default: (15, 10))
- `share_axes` (bool): Share axis limits across subplots (default: True)

### **Hyperplane Utility Functions**

#### `create_hyperplane_from_points(points)`

**Description**: Create hyperplane from defining points in 2D/3D space.
**Editable Parameters**:

- `points` (array): Points defining the hyperplane
- `method` (str): Fitting method ('least_squares', 'svd')
- `normalize` (bool): Normalize weights vector (default: True)

#### `extract_hyperplane(model)`

**Description**: Convenience function for hyperplane extraction.
**Editable Parameters**:

- `model` (sklearn model): Any trained linear model
- `return_extractor` (bool): Return extractor object vs hyperplane (default: False)

---

## 📊 **Visualization**

### **Statistical Plots**

#### `plot_correlation_heatmap(method='pearson', **kwargs)`

**Description**: Generate correlation matrix heatmap with customization options.
**Syntax**:

```python
# Basic correlation heatmap
kit.plot_correlation_heatmap()

# Spearman correlation with custom styling
kit.plot_correlation_heatmap(method='spearman',
                            cmap='RdBu_r',
                            figsize=(12, 10))

# Clean lower triangle only
kit.plot_correlation_heatmap(mask_upper=True,
                            threshold=0.3,
                            annot=True)

# No annotations, different colormap
kit.plot_correlation_heatmap(annot=False,
                            cmap='viridis',
                            method='kendall')
```

**Editable Parameters**:

- `method` (str): Correlation method ('pearson', 'spearman', 'kendall')
- `annot` (bool): Annotate cells with correlation values (default: True)
- `cmap` (str): Colormap name (default: 'coolwarm')
- `figsize` (tuple): Figure size (default: (10, 8))
- `mask_upper` (bool): Mask upper triangle (default: False)
- `threshold` (float): Minimum correlation to display (default: None)

#### `plot_distribution_grid(columns=None, **kwargs)`

**Description**: Grid of distribution plots for multiple variables.
**Editable Parameters**:

- `columns` (list): Columns to plot (default: all numeric)
- `plot_type` (str): 'hist', 'kde', 'box', 'violin'
- `cols` (int): Number of subplot columns (default: 3)
- `figsize` (tuple): Figure size (default: (15, 10))
- `bins` (int): Number of histogram bins (default: 30)

### **Model Visualization**

#### `plot_learning_curves(model, X, y, **kwargs)`

**Description**: Generate learning curves to diagnose model performance.
**Editable Parameters**:

- `model` (sklearn model): Model to evaluate
- `X` (array): Feature matrix
- `y` (array): Target vector
- `train_sizes` (array): Training set sizes to evaluate (default: np.linspace(0.1, 1.0, 10))
- `cv` (int): Cross-validation folds (default: 5)
- `scoring` (str): Metric for evaluation (default: 'accuracy')
- `figsize` (tuple): Figure size (default: (10, 6))

#### `plot_feature_importance(model, feature_names=None, **kwargs)`

**Description**: Visualize feature importance from trained models.
**Editable Parameters**:

- `model` (sklearn model): Trained model with feature*importances* attribute
- `feature_names` (list): Names of features (default: auto-generated)
- `top_n` (int): Number of top features to display (default: 20)
- `horizontal` (bool): Horizontal bar chart (default: True)
- `figsize` (tuple): Figure size (default: (10, 8))

---

## 🔧 **Utility Functions**

### **Data Profiling**

#### `profile_data(output_file=None)`

**Description**: Generate comprehensive data profiling report.
**Editable Parameters**:

- `output_file` (str): File path to save HTML report (default: None)
- `minimal` (bool): Generate minimal report for large datasets (default: False)
- `title` (str): Report title (default: 'Data Profile Report')
- `dark_mode` (bool): Use dark theme (default: False)
- `show_config` (bool): Display configuration details (default: True)

#### `data_quality_report()`

**Description**: Assess data quality with detailed metrics and recommendations.
**Editable Parameters**:

- `completeness_threshold` (float): Minimum completeness ratio (default: 0.8)
- `uniqueness_threshold` (float): Minimum uniqueness ratio (default: 0.1)
- `include_recommendations` (bool): Generate improvement suggestions (default: True)
- `export_format` (str): Output format ('dict', 'json', 'html')

### **Database Connectivity**

#### `connect_database(connection_string, query=None)`

**Description**: Connect to databases and execute queries.
**Editable Parameters**:

- `connection_string` (str): Database connection string
- `query` (str): SQL query to execute (default: None)
- `chunksize` (int): Number of rows to fetch at a time (default: None)
- `parse_dates` (list): Columns to parse as dates (default: None)
- `params` (dict): Query parameters for parameterized queries

#### `export_to_database(table_name, connection_string, **kwargs)`

**Description**: Export DataFrame to database table.
**Editable Parameters**:

- `table_name` (str): Target table name
- `connection_string` (str): Database connection string
- `if_exists` (str): Action if table exists ('fail', 'replace', 'append')
- `index` (bool): Write DataFrame index as column (default: False)
- `method` (str): Insertion method ('multi', 'single', callable)

### **Configuration Management**

#### `set_global_config(**kwargs)`

**Description**: Set global configuration parameters for dskit operations.
**Editable Parameters**:

- `plot_style` (str): Default plotting style ('seaborn', 'ggplot', 'classic')
- `figure_size` (tuple): Default figure size (default: (10, 6))
- `color_palette` (str): Default color palette ('viridis', 'plasma', 'tab10')
- `random_state` (int): Global random seed (default: 42)
- `n_jobs` (int): Number of parallel jobs (default: -1)
- `memory_limit` (str): Memory usage limit ('1GB', '2GB', etc.)

#### `get_config()`

**Description**: Retrieve current global configuration settings.
**Editable Parameters**:

- `section` (str): Specific configuration section ('plotting', 'memory', 'parallel')
- `format` (str): Output format ('dict', 'json', 'yaml')

---

## 📈 **Time Series Analysis**

### **Time Series Preprocessing**

#### `prepare_time_series(date_column, value_column, **kwargs)`

**Description**: Prepare data for time series analysis.
**Editable Parameters**:

- `date_column` (str): Column containing dates
- `value_column` (str): Column containing values
- `freq` (str): Target frequency ('D', 'H', 'M', 'W')
- `fill_missing` (str): Method for missing values ('interpolate', 'forward', 'backward')
- `remove_outliers` (bool): Remove outliers automatically (default: True)

#### `decompose_time_series(value_column, model='additive', **kwargs)`

**Description**: Decompose time series into trend, seasonal, and residual components.
**Editable Parameters**:

- `value_column` (str): Column to decompose
- `model` (str): Decomposition model ('additive', 'multiplicative')
- `period` (int): Seasonal period (default: auto-detect)
- `extrapolate_trend` (bool): Extrapolate trend component (default: False)

### **Forecasting**

#### `forecast_arima(value_column, periods=30, **kwargs)`

**Description**: ARIMA forecasting with automatic parameter selection.
**Editable Parameters**:

- `value_column` (str): Column to forecast
- `periods` (int): Number of periods to forecast (default: 30)
- `seasonal` (bool): Include seasonal components (default: True)
- `auto_arima` (bool): Automatically select ARIMA parameters (default: True)
- `confidence_interval` (float): Confidence level for intervals (default: 0.95)

#### `forecast_exponential_smoothing(value_column, periods=30, **kwargs)`

**Description**: Exponential smoothing forecasting methods.
**Editable Parameters**:

- `value_column` (str): Column to forecast
- `periods` (int): Forecast horizon (default: 30)
- `trend` (str): Trend component ('add', 'mul', None)
- `seasonal` (str): Seasonal component ('add', 'mul', None)
- `seasonal_periods` (int): Length of seasonal cycle (default: auto)

---

## 🎨 **Advanced Visualization**

### **Interactive Plots**

#### `create_interactive_plot(plot_type='scatter', **kwargs)`

**Description**: Create interactive plots using Plotly.
**Editable Parameters**:

- `plot_type` (str): 'scatter', 'line', 'bar', 'histogram', 'box'
- `x_column` (str): X-axis column name
- `y_column` (str): Y-axis column name
- `color_column` (str): Column for color encoding
- `size_column` (str): Column for size encoding
- `hover_data` (list): Additional columns to show on hover
- `title` (str): Plot title
- `template` (str): Plotly template ('plotly', 'plotly_dark', 'ggplot2')

#### `create_dashboard(charts=None, layout='grid')`

**Description**: Create multi-chart dashboard with interactive widgets.
**Editable Parameters**:

- `charts` (list): List of chart specifications
- `layout` (str): Dashboard layout ('grid', 'tabs', 'sidebar')
- `theme` (str): Dashboard theme ('light', 'dark', 'custom')
- `auto_refresh` (bool): Enable auto-refresh functionality (default: False)
- `export_options` (list): Available export formats (['html', 'pdf', 'png'])

### **Statistical Visualization**

#### `plot_regression_diagnostics(model, X, y, **kwargs)`

**Description**: Comprehensive regression diagnostic plots.
**Editable Parameters**:

- `model` (sklearn model): Fitted regression model
- `X` (array): Feature matrix
- `y` (array): Target vector
- `plots` (list): Diagnostic plots to include (['residuals', 'qq', 'leverage', 'cook'])
- `figsize` (tuple): Figure size (default: (15, 10))
- `alpha` (float): Point transparency (default: 0.6)

#### `plot_classification_report(y_true, y_pred, **kwargs)`

**Description**: Visual classification report with metrics heatmap.
**Editable Parameters**:

- `y_true` (array): True labels
- `y_pred` (array): Predicted labels
- `class_names` (list): Class names for display
- `normalize` (bool): Normalize confusion matrix (default: True)
- `include_support` (bool): Include support values (default: True)
- `cmap` (str): Colormap for heatmap (default: 'Blues')

---

## 🔬 **Model Interpretation**

### **SHAP Integration**

#### `explain_with_shap(model, X_explain, **kwargs)`

**Description**: Generate SHAP explanations for model predictions.
**Editable Parameters**:

- `model` (sklearn model): Trained model to explain
- `X_explain` (array): Data to generate explanations for
- `explainer_type` (str): 'tree', 'linear', 'kernel', 'deep'
- `max_display` (int): Maximum features to display (default: 20)
- `plot_type` (str): 'waterfall', 'summary', 'dependence', 'force'
- `feature_names` (list): Names of input features

#### `global_feature_importance(model, X, method='permutation', **kwargs)`

**Description**: Calculate global feature importance using various methods.
**Editable Parameters**:

- `model` (sklearn model): Trained model
- `X` (array): Feature matrix
- `method` (str): 'permutation', 'shap', 'built_in'
- `n_repeats` (int): Number of permutation repeats (default: 10)
- `random_state` (int): Random seed for reproducibility
- `scoring` (str): Metric for importance calculation

### **Model Comparison Tools**

#### `compare_feature_importance(models, X, feature_names=None, **kwargs)`

**Description**: Compare feature importance across multiple models.
**Editable Parameters**:

- `models` (list): List of trained models
- `X` (array): Feature matrix
- `feature_names` (list): Feature names for display
- `model_names` (list): Custom model names
- `top_n` (int): Number of top features to show (default: 15)
- `plot_style` (str): 'grouped', 'stacked', 'separate'

#### `cross_model_validation(models, X, y, **kwargs)`

**Description**: Comprehensive validation across multiple models.
**Editable Parameters**:

- `models` (list): Models to validate
- `X` (array): Feature matrix
- `y` (array): Target vector
- `cv_folds` (int): Cross-validation folds (default: 5)
- `scoring` (list): Metrics to evaluate
- `plot_results` (bool): Generate comparison plots (default: True)

---

## 🚀 **Model Deployment**

### **Model Persistence**

#### `save_model(model, filepath, include_metadata=True)`

**Description**: Save trained model with optional metadata.
**Editable Parameters**:

- `model` (sklearn model): Trained model to save
- `filepath` (str): File path for saving
- `include_metadata` (bool): Save training metadata (default: True)
- `compression` (int): Compression level 0-9 (default: 3)
- `protocol` (int): Pickle protocol version (default: highest)

#### `load_model(filepath, verify_compatibility=True)`

**Description**: Load saved model with compatibility checks.
**Editable Parameters**:

- `filepath` (str): Path to saved model
- `verify_compatibility` (bool): Check scikit-learn version compatibility
- `load_metadata` (bool): Load associated metadata (default: True)

### **Model Monitoring**

#### `monitor_model_drift(model, X_baseline, X_current, **kwargs)`

**Description**: Detect data drift in production models.
**Editable Parameters**:

- `model` (sklearn model): Model to monitor
- `X_baseline` (array): Reference/training data
- `X_current` (array): Current/production data
- `drift_threshold` (float): Threshold for drift detection (default: 0.1)
- `method` (str): Drift detection method ('ks', 'chi2', 'psi')
- `report_format` (str): Output format ('dict', 'html', 'json')

#### `performance_monitoring(y_true, y_pred, baseline_metrics, **kwargs)`

**Description**: Monitor model performance over time.
**Editable Parameters**:

- `y_true` (array): True labels for current period
- `y_pred` (array): Model predictions for current period
- `baseline_metrics` (dict): Reference performance metrics
- `alert_threshold` (float): Performance degradation threshold (default: 0.05)
- `metrics` (list): Metrics to monitor
- `window_size` (int): Rolling window for trend analysis

---

## 📋 **Summary**

### **Parameter Error Handling & Debugging**

#### **Error Message Decoder**

| Error Type                             | Common Cause            | Solution                |
| -------------------------------------- | ----------------------- | ----------------------- |
| `TypeError: ... expected str, got int` | Number used as string   | Add quotes: `'value'`   |
| `ValueError: ... not in valid options` | Typo in parameter value | Check spelling, case    |
| `KeyError: 'column_name'`              | Column doesn't exist    | Verify column names     |
| `IndexError: ... out of range`         | Wrong array dimensions  | Check data shape        |
| `AttributeError: ... has no attribute` | Wrong model type        | Use correct model class |

#### **Parameter Debugging Tips**

```python
# Check your data first
print(f"Data shape: {kit.data.shape}")
print(f"Columns: {list(kit.data.columns)}")
print(f"Data types: {kit.data.dtypes}")

# Test with minimal parameters first
kit.fill_missing()  # Use defaults
# Then add parameters one by one
kit.fill_missing(strategy='median')  # Add strategy
kit.fill_missing(strategy='median', limit=3)  # Add limit

# Validate parameters before complex operations
if 'target_column' in kit.data.columns:
    kit.train_model('target_column')
else:
    print("Target column not found!")
```

#### **Parameter Testing Workflow**

1. **Start Simple**: Use default parameters first
2. **Add One Parameter**: Test each parameter individually
3. **Combine Gradually**: Build up complex parameter combinations
4. **Validate Inputs**: Check data types and ranges
5. **Test Edge Cases**: Try boundary values

---

## 📋 **Summary**

**Total Features**: 221 functions and classes
**Core Categories**: 16 major modules
**Customizable Parameters**: 500+ configurable options
**Parameter Types**: 8 major input types with validation rules
**Error Prevention**: Comprehensive input validation and debugging guides
**Supported Algorithms**: 20+ ML algorithms with hyperplane analysis
**Visualization Types**: 25+ plot types including algorithm-specific hyperplane plots
**Performance Optimization**: Memory and speed parameter guidance

## 🛠️ **Parameter Input Troubleshooting Guide**

### **Common Input Errors & Solutions**

#### **Data Type Errors**

```python
# ❌ WRONG - Missing quotes for strings
kit.fill_missing(strategy=mean)  # NameError

# ✅ CORRECT - Strings need quotes
kit.fill_missing(strategy='mean')

# ❌ WRONG - Quotes around numbers
kit.remove_outliers(threshold='2.0')  # Will be treated as string

# ✅ CORRECT - Numbers without quotes
kit.remove_outliers(threshold=2.0)
```

#### **Boolean Parameter Errors**

```python
# ❌ WRONG - Quotes around booleans
kit.fix_dtypes(infer_datetime='True')  # String, not boolean

# ✅ CORRECT - No quotes for booleans
kit.fix_dtypes(infer_datetime=True)

# ❌ WRONG - Lowercase (Python is case-sensitive)
kit.fix_dtypes(infer_datetime=true)  # NameError

# ✅ CORRECT - Proper capitalization
kit.fix_dtypes(infer_datetime=False)
```

#### **List Parameter Errors**

```python
# ❌ WRONG - Missing square brackets
kit.remove_outliers(columns='age', 'income')  # SyntaxError

# ✅ CORRECT - Proper list format
kit.remove_outliers(columns=['age', 'income'])

# ❌ WRONG - Missing quotes around column names
kit.remove_outliers(columns=[age, income])  # NameError

# ✅ CORRECT - Column names as strings
kit.remove_outliers(columns=['age', 'income'])
```

#### **Tuple Parameter Errors**

```python
# ❌ WRONG - Missing parentheses
plot_svm(model, X, y, figsize=12, 8)  # SyntaxError

# ✅ CORRECT - Proper tuple format
plot_svm(model, X, y, figsize=(12, 8))

# ❌ WRONG - Single value (not a tuple)
plot_svm(model, X, y, figsize=12)  # TypeError

# ✅ CORRECT - Two values in tuple
plot_svm(model, X, y, figsize=(12, 8))
```

### **Parameter Validation Rules**

#### **Numeric Parameter Ranges**

- **Thresholds**: Usually 0.0 to 5.0 (outlier detection)
- **Probabilities**: Always 0.0 to 1.0 (contamination, alpha)
- **Counts**: Positive integers only (nrows, cv_folds)
- **Ratios**: 0.0 to 1.0 (category_threshold, test_size)

#### **String Parameter Validation**

- **Case Sensitive**: 'Mean' ≠ 'mean' ≠ 'MEAN'
- **Exact Match**: 'isolation_forest' not 'isolation forest'
- **No Typos**: 'median' not 'medain'
- **Encoding Names**: 'utf-8' not 'UTF8'

#### **Array Parameter Requirements**

- **Shape Matching**: X and y must have same number of samples
- **Data Types**: Numeric arrays for ML functions
- **No Missing Values**: Clean data before hyperplane plotting
- **Feature Count**: 2D plotting requires exactly 2 features

### **Quick Parameter Reference**

| Parameter Type | Format       | Example              | Common Errors       |
| -------------- | ------------ | -------------------- | ------------------- |
| **String**     | `'value'`    | `strategy='mean'`    | Missing quotes      |
| **Integer**    | `123`        | `nrows=1000`         | Adding quotes       |
| **Float**      | `1.5`        | `threshold=2.5`      | Using strings       |
| **Boolean**    | `True/False` | `show_margins=False` | Lowercase, quotes   |
| **List**       | `['a', 'b']` | `columns=['age']`    | Missing brackets    |
| **Tuple**      | `(1, 2)`     | `figsize=(10, 8)`    | Missing parentheses |

This catalog provides comprehensive coverage of all dskit features with their configurable parameters. Each function includes detailed parameter descriptions, input validation rules, and troubleshooting guidance to help you customize behavior for your specific use case.

### **Advanced Parameter Combination Patterns**

#### **Safe Conservative Settings** (Recommended for beginners)

```python
# Conservative data loading
kit = dskit.dskit.load("data.csv",
                      nrows=10000,  # Limit for safety
                      encoding='utf-8')  # Standard encoding

# Gentle preprocessing
kit.fix_dtypes(infer_datetime=False,  # Skip auto-datetime
              downcast_integers=False)  # Prevent overflow

kit.fill_missing(strategy='median')  # Robust to outliers
kit.remove_outliers(method='iqr', threshold=2.5)  # Conservative removal
```

#### **Aggressive Optimization Settings** (For experienced users)

```python
# Memory-optimized loading
kit = dskit.dskit.load("large_data.csv",
                      encoding='utf-8',
                      na_values=['', 'NULL', 'n/a', 'missing'])

# Aggressive optimization
kit.fix_dtypes(infer_datetime=True,
              downcast_integers=True,
              category_threshold=0.1)  # Aggressive categorization

# Advanced imputation
kit.advanced_imputation(method='iterative', max_iter=15)
kit.remove_outliers(method='isolation_forest', contamination=0.03)
```

#### **Time Series Specific Settings**

```python
# Time series preprocessing
kit.fill_missing(strategy='interpolate',
                method='spline',
                limit=5)  # Don't fill large gaps

kit.create_lag_features(['value'],
                       lags=[1, 7, 30],  # Daily, weekly, monthly
                       fill_method='forward')
```

#### **High-Dimensional Data Settings**

```python
# Dimensionality reduction
kit.apply_pca(variance_threshold=0.99,  # Keep 99% variance
              whiten=True)  # Normalize components

# Feature selection
kit.feature_selection(method='univariate', k=50)  # Top 50 features
```

#### **Visualization Parameter Combinations**

```python
# Publication-ready plots
plot_svm(model, X, y,
         figsize=(12, 9),  # Larger size
         alpha=0.7,  # Clear points
         show_support_vectors=True,
         show_margins=True,
         title='SVM Decision Boundary Analysis')

# Comparison plots
plot_algorithm_comparison(models, X, y,
                         model_names=['Linear SVM', 'RBF SVM', 'Logistic'],
                         subplot_layout=(1, 3),  # Horizontal layout
                         figsize=(18, 6),  # Wide format
                         share_axes=True)  # Same scale
```

### **Parameter Validation Checklist**

Before running functions, verify:

✅ **Data Types Match**

- Strings have quotes: `'value'`
- Numbers don't: `123` or `1.5`
- Booleans capitalized: `True`, `False`

✅ **Containers Formatted Correctly**

- Lists: `['item1', 'item2']`
- Tuples: `(value1, value2)`
- Single items in lists: `['single_item']`

✅ **Ranges Are Valid**

- Probabilities: 0.0 ≤ value ≤ 1.0
- Counts: Positive integers only
- Thresholds: Reasonable ranges (1.5-3.0 for IQR)

✅ **Column Names Exist**

- Check DataFrame column names
- Case-sensitive matching
- No typos in column references

**Complete Usage Pattern**:

```python
import dskit
from dskit.hyperplane import plot_svm, plot_logistic_regression, plot_algorithm_comparison
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# === DATA LOADING & PREPROCESSING ===
# Load with custom parameters
kit = dskit.dskit.load("data.csv",
                      sep=';',
                      encoding='utf-8',
                      nrows=10000,
                      na_values=['NULL', 'n/a'])

# Fix data types and clean
kit.fix_dtypes(infer_datetime=True, category_threshold=0.3)
kit.rename_columns_auto(case='snake', remove_special=True)

# Handle missing data
kit.fill_missing(strategy='knn', n_neighbors=5)
kit.remove_outliers(method='isolation_forest', contamination=0.05)

# === FEATURE ENGINEERING ===
# Create polynomial features
kit.create_polynomial_features(degree=2, interaction_only=False)

# Scale features
kit.scale_features(method='robust', quantile_range=(10.0, 90.0))

# Create date features
kit.create_date_features(['date_column'], include_cyclical=True)

# === EXPLORATORY ANALYSIS ===
# Correlation analysis
kit.correlation_analysis(method='spearman', threshold=0.3, figsize=(12, 10))

# Distribution analysis
kit.distribution_analysis(columns=['age', 'income'], plot=True)

# === MACHINE LEARNING ===
# Train models
kit.train_model('target', task='classification', algorithm='svm',
               test_size=0.3, cv_folds=10, random_state=42)

# Hyperparameter tuning
kit.auto_tune('target', method='optuna', max_evals=100, timeout=3600)

# Compare multiple algorithms
kit.compare_models('target', algorithms=['svm', 'rf', 'xgb'], cv_folds=5)

# === HYPERPLANE ANALYSIS ===
# Train specific models for hyperplane visualization
svm_model = SVC(kernel='linear').fit(kit.X_train, kit.y_train)
lr_model = LogisticRegression().fit(kit.X_train, kit.y_train)

# Algorithm-specific hyperplane plots
plot_svm(svm_model, kit.X_test, kit.y_test,
         show_support_vectors=True,
         show_margins=True,
         figsize=(10, 8))

plot_logistic_regression(lr_model, kit.X_test, kit.y_test,
                        show_probabilities=True,
                        probability_levels=[0.2, 0.4, 0.6, 0.8],
                        colormap='plasma')

# Compare multiple algorithms
plot_algorithm_comparison([svm_model, lr_model],
                         kit.X_test, kit.y_test,
                         model_names=['SVM', 'Logistic Regression'],
                         figsize=(15, 6))

# === MODEL EVALUATION ===
# Comprehensive evaluation
kit.evaluate_model(plot_confusion_matrix=True,
                  plot_roc=True,
                  plot_feature_importance=True)

# Cross-validation
kit.cross_validate_advanced(cv=10, scoring=['accuracy', 'f1', 'roc_auc'])

# === REPORTING ===
# Data profiling report
kit.profile_data(output_file='data_profile.html',
                dark_mode=True,
                minimal=False)

# Data quality assessment
quality_report = kit.data_quality_report(completeness_threshold=0.8,
                                        include_recommendations=True)

# === VISUALIZATION ===
# Correlation heatmap
kit.plot_correlation_heatmap(method='pearson',
                            mask_upper=True,
                            cmap='RdBu_r')

# Distribution plots
kit.plot_distribution_grid(columns=['age', 'income', 'score'],
                          plot_type='kde',
                          cols=3)

# Learning curves
kit.plot_learning_curves(kit.model, kit.X, kit.y,
                        cv=5,
                        scoring='f1_weighted')

# === CONFIGURATION ===
# Set global preferences
dskit.set_global_config(plot_style='seaborn',
                       figure_size=(12, 8),
                       color_palette='viridis',
                       random_state=42)
```
