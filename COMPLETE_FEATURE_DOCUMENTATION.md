# 📚 dskit Complete Feature Documentation

**dskit** is a comprehensive data science library with **221 functions and classes** across **16+ specialized modules**. This document provides complete documentation of all available features including advanced hyperplane analysis capabilities.

## 🚀 Getting Started

### Installation

```bash
pip install dskit
```

### First steps in Python

```python
import dskit

# Load data and run a quick health + EDA pass
df = dskit.load("data.csv")
health_score = dskit.data_health_check(df)
dskit.quick_eda(df)
```

### First steps from the CLI

```bash
# Quick EDA
dskit eda data.csv --target target_column

# Generate an HTML profile
dskit profile data.csv --output report.html
```

## 📊 Overview Statistics

- **Total Functions/Classes**: 221
- **Core Modules**: 17
- **Algorithms Supported**: 39
- **File Formats**: 12
- **Visualization Types**: 25
- **Feature Engineering Methods**: 12

---

## 🏗️ Core Architecture

### Main Entry Point

- **`dskit`** - Main class with method chaining for fluent API

### Static Methods for Data I/O

- **`load()`** - Universal data loader (CSV, Excel, JSON, Parquet)
- **`read_folder()`** - Batch processing from directories
- **`save()`** - Multi-format data export

---

### `dskit` Core Class (`core.py`)

The `dskit` class is the high-level, fluent API that wires together all core modules:

- **Initialization:** `dskit(df=None)` – wrap an existing `DataFrame` or start empty.
- **I/O:** `dskit.load(path)`, `dskit.read_folder(path)`, `kit.save(path)` – load, batch‑load, and save data.
- **Cleaning:** `fix_dtypes()`, `rename_columns_auto()`, `fill_missing()`, `remove_outliers()`, `clean()` – chained data cleaning.
- **NLP helpers:** `advanced_text_clean()`, `extract_text_features()`, `sentiment_analysis()`, `text_stats()`, `generate_wordcloud()`.
- **Visualization:** `plot_missingness()`, `plot_histograms()`, `plot_boxplots()`, `plot_correlation_heatmap()`, `plot_pairplot()`, `visualize()`.
- **Feature engineering / preprocessing:** `auto_encode()`, `auto_scale()`, `create_polynomial_features()`, `create_date_features()`, `create_binning_features()`, `apply_pca()`, `train_test_auto()`.
- **Modeling:** `train()`, `train_advanced()`, `evaluate()`, `compare_models()`, `cross_validate()`, `auto_tune()`.
- **Explainability & EDA:** `explain()`, `basic_stats()`, `quick_eda()`, `comprehensive_eda()`, `data_health_check()`, `generate_profile_report()`.

**End‑to‑end example:**

```python
from dskit import dskit

kit = (dskit.load("data.csv")
       .clean()                        # fix_dtypes + rename_columns_auto + fill_missing
       .auto_encode()
       .auto_scale()
       .train_test_auto(target="target")
       .train(model_name="random_forest")
       .evaluate()
       .explain())
```

---

## 📁 Module-by-Module Feature Guide

### 🔧 **1. Data I/O (`io.py`)**

**Functions:**

- `load(filepath)` - Smart file loader with automatic format detection
- `read_folder(folder_path, file_type='csv')` - Batch file processing
- `save(df, filepath, **kwargs)` - Multi-format data export

**Supported Formats:**

- CSV, Excel (`.xlsx`, `.xls`), JSON, Parquet
- Automatic encoding detection
- Error handling with clear messages

**Example:**

```python
import dskit
df = dskit.load("data.csv")  # Auto-detects format
combined_df = dskit.read_folder("data_folder/")  # Combines all CSV files
dskit.save(df, "output.parquet")  # Saves as Parquet
```

---

### 🧹 **2. Data Cleaning (`cleaning.py`)**

**Core Functions:**

- `fix_dtypes(df)` - Intelligent data type detection and conversion
- `rename_columns_auto(df)` - Clean column names (lowercase, underscores)
- `replace_specials(df, chars_to_remove, replacement)` - Remove special characters
- `missing_summary(df)` - Detailed missing value analysis
- `fill_missing(df, strategy, fill_value)` - Smart missing value imputation
- `outlier_summary(df, method, threshold)` - Outlier detection analysis
- `remove_outliers(df, method, threshold)` - Outlier removal
- `simple_nlp_clean(df, text_cols)` - Basic text preprocessing

**Imputation Strategies:**

- `'auto'` - Automatically choose best strategy
- `'mean'`, `'median'`, `'mode'` - Statistical imputation
- `'ffill'`, `'bfill'` - Forward/backward fill
- `'constant'` - Fill with specific value

**Outlier Detection Methods:**

- `'iqr'` - Interquartile Range method
- `'zscore'` - Z-score based detection
- `'isolation'` - Isolation Forest algorithm

**Example:**

```python
# Complete cleaning pipeline
df = dskit.fix_dtypes(df)
df = dskit.rename_columns_auto(df)
df = dskit.fill_missing(df, strategy='auto')
df = dskit.remove_outliers(df, method='iqr')
```

---

### 📊 **3. Visualization (`visualization.py` + `advanced_visualization.py`)**

**Basic Plotting Functions:**

- `plot_missingness(df)` - Missing data heatmap
- `plot_histograms(df, bins)` - Distribution plots for numeric columns
- `plot_boxplots(df)` - Box plots for outlier visualization
- `plot_correlation_heatmap(df)` - Correlation matrix heatmap
- `plot_pairplot(df)` - Pairwise scatter plots

**Advanced Plotting Functions:**

- `plot_missing_patterns_advanced(df)` - Advanced missing data patterns
- `plot_correlation_advanced(df, method, threshold)` - Enhanced correlation analysis
- `plot_feature_importance(model, feature_names, top_n)` - Model feature importance
- `plot_outliers_advanced(df, method)` - Advanced outlier visualization
- `plot_distribution_comparison(df, columns, groups)` - Distribution comparisons
- `plot_interactive_scatter(df, x, y, color, size)` - Interactive Plotly scatter plots

**Model Performance Plots:**

- `plot_confusion_matrix(y_true, y_pred, labels)` - Confusion matrix visualization
- `plot_roc_curve(y_true, y_proba)` - ROC curve analysis
- `plot_precision_recall_curve(y_true, y_proba)` - Precision-recall curves
- `plot_target_distribution(df, target_col)` - Target variable distribution
- `plot_feature_vs_target(df, feature_col, target_col)` - Feature-target relationships

**Example:**

```python
# Basic EDA visualizations
dskit.plot_missingness(df)
dskit.plot_correlation_heatmap(df)
dskit.plot_histograms(df)

# Advanced visualizations
dskit.plot_correlation_advanced(df, method='spearman', threshold=0.7)
dskit.plot_interactive_scatter(df, 'x', 'y', color='category')
```

---

### 🔧 **4. Data Preprocessing (`preprocessing.py`)**

**Encoding Functions:**

- `auto_encode(df, target_col)` - Automatic categorical encoding
- Supports: Label Encoding, One-Hot Encoding, Target Encoding

**Scaling Functions:**

- `auto_scale(df, method, columns)` - Feature scaling
- Methods: `'standard'`, `'minmax'`, `'robust'`, `'quantile'`

**Data Splitting:**

- `train_test_auto(df, target, test_size, random_state)` - Smart train-test splitting

**Example:**

```python
# Preprocessing pipeline
df_encoded = dskit.auto_encode(df, target_col='target')
df_scaled = dskit.auto_scale(df_encoded, method='standard')
X_train, X_test, y_train, y_test = dskit.train_test_auto(df_scaled, 'target')
```

---

### 🤖 **5. Machine Learning (`modeling.py` + `advanced_modeling.py`)**

**QuickModel Class:**

```python
model = dskit.QuickModel("random_forest")
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Supported Algorithms:**

- **Classification**: Random Forest, XGBoost, LightGBM, CatBoost, SVM, KNN, Naive Bayes, Logistic Regression
- **Regression**: Random Forest, XGBoost, LightGBM, CatBoost, SVR, Linear Regression, Ridge, Lasso
- **Ensemble**: Voting Classifiers, Bagging, AdaBoost

**Model Functions:**

- `compare_models(X, y, task)` - Compare multiple algorithms
- `evaluate_model(model, X_test, y_test, task)` - Comprehensive evaluation
- `error_analysis(y_true, y_pred, X_test)` - Detailed error analysis
- `cross_validate_model(model, X, y, cv, scoring)` - Cross-validation
- `create_ensemble(models, method)` - Ensemble model creation

**Advanced Models Available:**

- `RandomForestClassifier`, `RandomForestRegressor`
- `XGBClassifier`, `XGBRegressor` (if xgboost installed)
- `LGBMClassifier`, `LGBMRegressor` (if lightgbm installed)
- `CatBoostClassifier`, `CatBoostRegressor` (if catboost installed)

---

### 🎯 **6. AutoML (`auto_ml.py`)**

**Hyperparameter Optimization:**

- `auto_hpo(model, X, y, method, max_evals)` - Automated hyperparameter tuning
- `random_search_optimization()` - Random search optimization
- `grid_search_custom()` - Custom grid search
- `bayesian_optimization_simple()` - Bayesian optimization
- `optuna_optimization()` - Optuna-based optimization
- `hyperopt_optimization()` - Hyperopt-based optimization

**Optimization Methods:**

- `'random'` - Random search
- `'grid'` - Grid search
- `'bayesian'` - Bayesian optimization
- `'optuna'` - Optuna framework
- `'hyperopt'` - Hyperopt framework

**AutoML Pipeline:**

- `auto_tune_model(model_name, X, y, method, max_evals)` - End-to-end auto-tuning

**Example:**

```python
# AutoML pipeline
best_model = dskit.auto_hpo('xgboost', X_train, y_train, method='optuna', max_evals=100)
results = dskit.compare_models(X_train, y_train, task='classification')
```

---

### 🔧 **7. Feature Engineering (`feature_engineering.py`)**

**Feature Creation Functions:**

- `create_polynomial_features(df, degree, interaction_only)` - Polynomial and interaction features
- `create_date_features(df, date_cols)` - Date/time feature extraction
- `create_binning_features(df, numeric_cols, n_bins, strategy)` - Binning and discretization
- `create_target_encoding(df, cat_cols, target_col, smoothing)` - Target encoding
- `create_aggregation_features(df, group_col, agg_cols, agg_funcs)` - Aggregation features
- `create_frequency_encoding(df, cat_cols)` - Frequency-based encoding
- `create_interaction_features(df, cols, degree)` - Feature interactions
- `create_rank_features(df, numeric_cols)` - Rank-based features
- `create_statistical_features(df, numeric_cols, window)` - Statistical features
- `create_clustering_features(df, n_clusters, algorithm)` - Clustering-based features
- `create_anomaly_features(df, method, contamination)` - Anomaly detection features

**Dimensionality Reduction:**

- `apply_pca(df, n_components, variance_threshold)` - Principal Component Analysis

**Feature Selection:**

- `select_features_univariate(X, y, k, score_func)` - Univariate feature selection
- `select_features_rfe(X, y, estimator, n_features)` - Recursive Feature Elimination

**Example:**

```python
# Feature engineering pipeline
df = dskit.create_date_features(df, ['registration_date'])
df = dskit.create_polynomial_features(df, degree=2, interaction_only=True)
df = dskit.create_target_encoding(df, ['category'], 'target', smoothing=10)
df = dskit.create_binning_features(df, ['age', 'income'], n_bins=5)
```

---

### 📝 **8. NLP & Text Processing (`nlp_utils.py`)**

**Text Cleaning:**

- `advanced_text_clean(df, text_cols, remove_urls, remove_emails, expand_contractions)` - Advanced text preprocessing
- `simple_nlp_clean(df, text_cols)` - Basic text cleaning

**Text Analysis:**

- `sentiment_analysis(df, text_cols)` - Sentiment scoring using TextBlob
- `extract_text_features(df, text_cols)` - Extract text statistics
- `basic_text_stats(df, text_cols)` - Basic text statistics
- `detect_language(text)` - Language detection
- `extract_keywords(df, text_col, top_n)` - Keyword extraction

**Text Visualization:**

- `generate_wordcloud(df, text_col, max_words)` - Word cloud generation

**Example:**

```python
# NLP pipeline
df = dskit.advanced_text_clean(df, ['review_text'], remove_urls=True)
df = dskit.sentiment_analysis(df, ['review_text'])
df = dskit.extract_text_features(df, ['review_text'])
dskit.generate_wordcloud(df, 'review_text', max_words=100)
```

---

### 📊 **9. Exploratory Data Analysis (`eda.py` + `comprehensive_eda.py`)**

**Basic EDA:**

- `basic_stats(df)` - Basic statistical summary
- `quick_eda(df)` - Quick exploratory analysis

**Comprehensive EDA:**

- `data_health_check(df)` - Data quality scoring (0-100)
- `comprehensive_eda(df, target_col, sample_size)` - Complete EDA report
- `feature_analysis_report(df, target_col)` - Feature analysis
- `generate_pandas_profile(df, output_file)` - Automated profiling report

**Data Profiling:**

- `data_profiler(df, sample_size)` - Advanced data profiling
- `detect_distribution(series)` - Distribution detection

**Example:**

```python
# EDA workflow
health_score = dskit.data_health_check(df)
dskit.comprehensive_eda(df, target_col='churn', sample_size=1000)
dskit.quick_eda(df)
```

---

### 🧠 **10. Model Explainability (`explainability.py`)**

**SHAP Integration:**

- `explain_shap(model, X, plot_type)` - SHAP-based model explanations

**Explainability Features:**

- Summary plots, force plots, waterfall plots
- Feature importance analysis
- Individual prediction explanations

**Example:**

```python
# Model explainability
dskit.explain_shap(model, X_test, plot_type='summary')
```

---

### ⏰ **11. Time Series Analysis (`time_series_utils.py`)**

**TimeSeriesUtils Class:**

- `create_comprehensive_time_features(df, date_col)` - 25+ time-based features
- `detect_seasonality(series)` - Seasonality detection
- `add_lag_features(df, col, lags)` - Lag feature creation
- `add_rolling_features(df, col, windows, functions)` - Rolling statistics
- `create_difference_features(df, col, periods)` - Differencing features
- `create_expanding_features(df, col, functions)` - Expanding window features

**Time Features Created:**

- Date components: year, month, day, weekday, quarter
- Time indicators: is_weekend, is_month_end, is_quarter_end
- Cyclical features: sin/cos transformations
- Business calendar features

**Example:**

```python
ts_utils = dskit.TimeSeriesUtils()
df = ts_utils.create_comprehensive_time_features(df, 'date')
seasonality = ts_utils.detect_seasonality(df['sales'])
df = ts_utils.add_lag_features(df, 'sales', lags=[1, 7, 30])
```

---

### 📐 **12. Hyperplane Analysis (`hyperplane.py`)**

**Hyperplane Class:**

- `equation()` - Get mathematical equation representation
- `predict(X)` - Classify points (+1/-1) based on hyperplane
- `distance(point)` - Calculate perpendicular distance from hyperplane
- `plot_2d(X, y, show_margin)` - 2D visualization with optional margin
- `plot_3d(X, y)` - 3D hyperplane visualization
- `plot_decision_regions(X, y)` - Color-coded decision regions

**HyperplaneExtractor Class:**

- `extract_hyperplane(model)` - Extract from any linear ML model
- `analyze_model(X, y)` - Comprehensive hyperplane analysis
- `compare_models(other_extractor)` - Compare two hyperplanes
- `plot_2d()`, `plot_3d()`, `plot_decision_regions()` - Advanced visualizations

**Supported ML Models:**

- **SVM**: LinearSVC, SVC(kernel='linear') with margin visualization
- **Linear Models**: LogisticRegression, Perceptron, LinearRegression
- **Discriminant Analysis**: LinearDiscriminantAnalysis
- **Naive Bayes**: GaussianNB (approximate linear boundary)
- **Decision Trees**: Approximate hyperplane from feature importance

**Algorithm-Specific Plotting Methods:**

- `plot_svm()` - SVM with margins and support vectors
- `plot_logistic_regression()` - Logistic regression with probability contours
- `plot_perceptron()` - Perceptron with misclassified points highlighted
- `plot_lda()` - LDA with class centers and projection directions
- `plot_linear_regression()` - Linear/Ridge/Lasso with residuals
- `plot_algorithm_comparison()` - Compare multiple algorithms side-by-side

**Utility Functions:**

- `create_hyperplane_from_points(points)` - Create from 2D/3D points
- `extract_hyperplane(model)` - Convenience extraction function
- `plot_svm_hyperplane()` - Quick SVM plotting
- `plot_logistic_hyperplane()` - Quick logistic regression plotting
- `plot_perceptron_hyperplane()` - Quick perceptron plotting
- `plot_lda_hyperplane()` - Quick LDA plotting
- `plot_linear_regression_hyperplane()` - Quick regression plotting
- `compare_algorithm_hyperplanes()` - Quick algorithm comparison

**Analysis Features:**

- Distance metrics (mean, min, max, std)
- Margin calculation for binary classification
- Weight magnitude and bias analysis
- Angular comparison between hyperplanes
- Class-specific distance analysis

**Example:**

```python
# Extract hyperplane from trained model
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

model = LinearSVC().fit(X, y)
extractor = dskit.extract_hyperplane(model)

# Algorithm-specific plotting (Method 1: Direct functions)
dskit.plot_svm_hyperplane(svm_model, X, y)  # SVM with margins
dskit.plot_logistic_hyperplane(lr_model, X, y)  # With probability contours
dskit.plot_perceptron_hyperplane(perceptron_model, X, y)  # Highlight misclassified
dskit.plot_lda_hyperplane(lda_model, X, y)  # With class centers

# Algorithm-specific plotting (Method 2: Extractor methods)
extractor.plot_svm(X, y, show_support_vectors=True, margin_style='dashed')
extractor.plot_logistic_regression(X, y, probability_contours=[0.1, 0.3, 0.5, 0.7, 0.9])
extractor.plot_perceptron(X, y, show_misclassified=True)
extractor.plot_lda(X, y, show_class_centers=True, show_projections=True)
extractor.plot_linear_regression(X, y, show_residuals=True)

# Compare multiple algorithms
models = {'SVM': svm_model, 'LR': lr_model, 'LDA': lda_model}
dskit.compare_algorithm_hyperplanes(models, X, y)

# Analyze hyperplane properties
analysis = extractor.analyze_model(X, y)
print(f"Equation: {extractor.equation()}")
print(f"Margin: {analysis['margin']:.3f}")

# Create custom hyperplane
hp = dskit.create_hyperplane_from_points([[0, 1], [2, 3]])
hp.plot_2d()
```

---

### 🔍 **13. Advanced Preprocessing (`advanced_preprocessing.py`)**

**AdvancedPreprocessor Class:**

- `advanced_imputation(df, method, n_neighbors)` - Advanced missing value imputation
- `detect_and_handle_outliers(df, method, contamination)` - Advanced outlier handling
- `smart_feature_selection(X, y, method, k, task)` - Intelligent feature selection
- `fit_transform_scaling(df, method, columns)` - Advanced scaling methods

**Advanced Functions:**

- `advanced_data_validation(df)` - Comprehensive data validation
- `smart_type_inference(df)` - Intelligent type optimization
- `memory_optimizer(df)` - Memory usage optimization

**Example:**

```python
preprocessor = dskit.AdvancedPreprocessor()
df = preprocessor.advanced_imputation(df, method='iterative')
df = preprocessor.detect_and_handle_outliers(df, method='isolation_forest')
```

---

### 🔬 **14. Model Validation (`model_validation.py`)**

**ModelValidator Class:**

- `comprehensive_cross_validation()` - Multi-strategy cross-validation
- `stability_analysis()` - Model stability across splits
- `learning_curve_analysis()` - Learning curve generation
- `bias_variance_analysis()` - Bias-variance decomposition

**PipelineBuilder Class:**

- `build_preprocessing_pipeline()` - Automated pipeline construction
- `build_complete_pipeline()` - End-to-end pipeline building
- `evaluate_pipelines()` - Pipeline comparison

**ModelInterpreter Class:**

- `permutation_importance()` - Permutation-based importance
- `partial_dependence_analysis()` - Partial dependence plots
- `local_explanation()` - LIME-based local explanations
- `global_surrogate_model()` - Global surrogate models

---

### 🔍 **15. Data Auditing (`data_auditing.py`)**

**DataAuditor Class:**

- `comprehensive_audit(df, target_col)` - Complete data quality audit
- `print_audit_report()` - Formatted audit report

**DataSampler Class:**

- `stratified_sample(df, target_col, n_samples)` - Stratified sampling
- `time_aware_sample(df, date_col, n_samples, method)` - Time-aware sampling
- `balanced_sample(df, target_col, method)` - Balanced sampling

**Additional Functions:**

- `create_synthetic_data(df, n_samples, method)` - Synthetic data generation

---

### 🗄️ **16. Database Utilities (`database_utils.py`)**

**DatabaseConnector Class:**

- `connect(db_type, connection_params)` - Universal database connectivity
- `execute_query(query, connection_name)` - SQL query execution
- `insert_dataframe(df, table_name)` - DataFrame to database insertion
- `list_tables(connection_name)` - Database table listing
- `get_table_info(table_name)` - Table schema information

**DataProfiler Class:**

- `comprehensive_profile(df)` - Advanced data profiling
- Statistical analysis, correlation analysis, distribution analysis

**DataExporter Class:**

- `export_data(df, filepath, format)` - Multi-format data export
- `export_profile_report(profile_data, filepath, format)` - Profile report export

**Supported Databases:**

- SQLite, MySQL, PostgreSQL, Oracle, SQL Server

---

### 💾 **17. Model Deployment (`model_deployment.py`)**

**ModelPersistence Class:**

- `save_model(model, filepath, format, metadata)` - Model serialization
- `load_model(filepath, format)` - Model deserialization
- `save_pipeline(pipeline, pipeline_name)` - Pipeline saving
- `load_pipeline(pipeline_name)` - Pipeline loading
- `list_saved_models(base_path)` - Model inventory

**ExperimentTracker Class:**

- `start_experiment(experiment_name, description)` - Experiment initialization
- `log_parameter(key, value)` - Parameter logging
- `log_metric(key, value, step)` - Metric logging
- `log_artifact(artifact_path, artifact_type)` - Artifact logging
- `end_experiment(status)` - Experiment completion
- `list_experiments()` - Experiment history

**ConfigManager Class:**

- `load_config()` - Configuration management
- `get(key, default)` - Configuration retrieval
- `set(key, value)` - Configuration setting

**DataVersioning Class:**

- `create_version(df, dataset_name, description)` - Data versioning
- `load_version(version_id)` - Version loading
- `list_versions(dataset_name)` - Version history

---

## 🎯 **Complete dskit Method Chaining API**

dskit supports fluent method chaining for streamlined workflows:

```python
from dskit import dskit

# Complete ML pipeline in method chain
kit = (dskit.load("data.csv")
       .fix_dtypes()
       .rename_columns_auto()
       .fill_missing(strategy='auto')
       .remove_outliers(method='iqr')
       .auto_encode()
       .auto_scale(method='standard')
       .train_advanced('xgboost')
       .auto_tune(method='optuna', max_evals=100)
       .evaluate()
       .explain())

# Access results
print(f"Model accuracy: {kit.model.score(kit.X_test, kit.y_test)}")
```

## 🔧 **Configuration Options**

dskit supports flexible configuration:

```python
from dskit.config import set_config, get_config

# Global configuration
set_config({
    'visualization_backend': 'plotly',  # or 'matplotlib'
    'auto_save_plots': True,
    'default_test_size': 0.2,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 1
})

# Get current config
current_config = get_config()
```

Additional configuration utilities:

- `reset_config()` – restore defaults.
- `load_config_from_file(path)` / `save_config_to_file(path)` – JSON/YAML config persistence.
- `print_config()` – print current settings.
- `config_context({...})` – temporary overrides via context manager.

Environment variables such as `dskit_VERBOSE`, `dskit_RANDOM_STATE`, `dskit_N_JOBS`, `dskit_BACKEND`, and `dskit_AUTO_SAVE` can also override defaults.

---

## 🖥️ **Command Line Interface (`cli.py`)**

dskit ships with a CLI that exposes common workflows without writing Python code.

**Base command:**

```bash
dskit <command> [options]
```

**Available commands:**

- `eda` – perform comprehensive EDA (`comprehensive_eda`).
- `profile` – generate a pandas profile HTML report (`generate_profile_report`).
- `health` – compute overall data health score (`data_health_check`).
- `clean` – clean a dataset and optionally remove outliers (`clean`, `remove_outliers`, `save`).
- `compare` – compare multiple ML models (`compare_models`).
- `config` – view, reset, or update dskit configuration.
- `info` – show dataset shape, memory usage, dtypes, basic stats and missing summary.

**Examples:**

```bash
# Quick EDA with optional target and sampling
dskit eda data.csv --target churn --sample 5000

# Generate profiling report
dskit profile data.csv --output report.html

# Data health check
dskit health data.csv

# Clean data and save, removing outliers
dskit clean data.csv --output cleaned.csv --remove-outliers

# Compare models for regression task
dskit compare housing.csv --target price --task regression

# Show and tweak configuration
dskit config --show
dskit config --set default_random_state 123
```

## 📊 **Algorithm Support Matrix**

### Classification Algorithms (18)

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- LightGBM Classifier
- CatBoost Classifier
- Support Vector Classifier
- K-Nearest Neighbors Classifier
- Naive Bayes (Gaussian, Multinomial, Bernoulli)
- Decision Tree Classifier
- Extra Trees Classifier
- AdaBoost Classifier
- Bagging Classifier
- Voting Classifier
- Neural Network (MLPClassifier)

### Regression Algorithms (15)

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor
- Support Vector Regressor
- K-Nearest Neighbors Regressor
- Decision Tree Regressor
- Extra Trees Regressor
- AdaBoost Regressor
- Neural Network (MLPRegressor)

### Ensemble Methods (6)

- Voting Classifier/Regressor
- Bagging Classifier/Regressor
- AdaBoost Classifier/Regressor
- Gradient Boosting Classifier/Regressor
- Random Forest (inherently ensemble)
- Extra Trees (inherently ensemble)

## 📈 **Supported File Formats**

### Input Formats (10)

- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)
- JSON (`.json`)
- Parquet (`.parquet`)
- TSV (`.tsv`)
- Pickle (`.pkl`, `.pickle`)
- HDF5 (`.h5`, `.hdf`)
- Feather (`.feather`)
- Stata (`.dta`)
- SAS (`.sas7bdat`)

### Output Formats (12)

- All input formats plus:
- HTML (`.html`)
- XML (`.xml`)

## 🎨 **Visualization Types**

### Static Plots (15)

- Histograms
- Box plots
- Scatter plots
- Line plots
- Bar plots
- Correlation heatmaps
- Missing data heatmaps
- Distribution plots
- Q-Q plots
- Violin plots
- Pair plots
- Feature importance plots
- Confusion matrices
- ROC curves
- Precision-recall curves

### Interactive Plots (10)

- Interactive scatter plots
- Interactive correlation matrices
- Interactive feature importance
- Interactive missing data patterns
- Interactive distribution comparisons
- 3D scatter plots
- Interactive time series plots
- Interactive box plots
- Interactive histograms
- Interactive heatmaps

## 🧪 **Testing and Quality Assurance**

dskit includes comprehensive testing:

```python
# Run all tests
pytest tests/ --cov=dskit --cov-report=html

# Run specific module tests
pytest tests/test_cleaning.py -v

# Performance benchmarks
python benchmarks/run_benchmarks.py
```

## 📝 **Usage Examples by Use Case**

### 1. Quick Data Exploration

```python
import dskit

# Load and explore
df = dskit.load("data.csv")
health_score = dskit.data_health_check(df)
dskit.quick_eda(df)
```

### 2. Data Cleaning Pipeline

```python
# Complete cleaning workflow
kit = (dskit.dskit(df)
       .fix_dtypes()
       .rename_columns_auto()
       .fill_missing(strategy='auto')
       .remove_outliers(method='iqr'))
```

### 3. Feature Engineering

```python
# Advanced feature creation
df = dskit.create_date_features(df, ['date'])
df = dskit.create_polynomial_features(df, degree=2)
df = dskit.create_target_encoding(df, ['category'], 'target')
```

### 4. Model Training and Tuning

```python
# AutoML workflow
results = dskit.compare_models(X, y, task='classification')
best_model = dskit.auto_hpo('xgboost', X, y, method='optuna', max_evals=100)
```

### 5. Text Analysis

```python
# NLP pipeline
df = dskit.advanced_text_clean(df, ['text_col'])
df = dskit.sentiment_analysis(df, ['text_col'])
df = dskit.extract_text_features(df, ['text_col'])
```

### 6. Time Series Analysis

```python
# Time series feature engineering
ts_utils = dskit.TimeSeriesUtils()
df = ts_utils.create_comprehensive_time_features(df, 'date')
seasonality = ts_utils.detect_seasonality(df['sales'])
```

### 7. Model Validation and Interpretation

```python
# Advanced model validation
validator = dskit.ModelValidator()
cv_results = validator.comprehensive_cross_validation(model, X, y)
stability = validator.stability_analysis(model, X, y)
dskit.explain_shap(model, X)
```

### 8. Database Integration

```python
# Database workflow
db = dskit.DatabaseConnector()
db.connect('sqlite', {'database': 'data.db'})
results = db.execute_query("SELECT * FROM table")
```

### 9. Experiment Tracking

```python
# MLOps workflow
tracker = dskit.ExperimentTracker()
exp_id = tracker.start_experiment("model_comparison")
tracker.log_parameter("algorithm", "xgboost")
tracker.log_metric("accuracy", 0.95)
tracker.end_experiment("completed")
```

### 10. Advanced Data Auditing

```python
# Data quality assessment
auditor = dskit.DataAuditor()
audit_results = auditor.comprehensive_audit(df, target_col='target')
auditor.print_audit_report()
```

---

## 🎯 **Summary**

dskit provides a complete data science ecosystem with:

- **221 functions and classes** across **16+ modules**
- **25+ visualization types** (static and interactive)
- **20+ machine learning algorithms**
- **30+ feature engineering methods**
- **10+ data formats** supported
- **Advanced AutoML capabilities**
- **Comprehensive model validation**
- **MLOps and deployment tools**
- **Database connectivity**
- **Text processing and NLP**
- **Time series analysis**
- **Data quality auditing**

All designed with a **fluent API** for **method chaining** and **one-line operations** to make data science **simple**, **comprehensive**, and **production-ready**! 🚀
