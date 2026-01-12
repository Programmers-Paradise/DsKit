# 📚 Complete Feature Documentation - Ak-dskit

## Table of Contents

1. [Data I/O Features](#data-io-features)
2. [Data Cleaning & Quality](#data-cleaning--quality)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Preprocessing for ML](#preprocessing-for-ml)
5. [Visualization Features](#visualization-features)
6. [Feature Engineering](#feature-engineering)
7. [Machine Learning Models](#machine-learning-models)
8. [AutoML & Tuning](#automl--tuning)
9. [NLP Features](#nlp-features)
10. [Advanced Features](#advanced-features)

---

## Data I/O Features

### Load Data from Multiple Formats

```python
from dskit import dskit

# CSV files
kit = dskit.load('data.csv')

# Excel workbooks
kit = dskit.load('data.xlsx', sheet_name='Sheet1')

# JSON files
kit = dskit.load('data.json')

# Parquet files
kit = dskit.load('data.parquet')

# From directory (batch load)
kit = dskit.load_folder('data_folder/', file_type='csv')
```

### Save Data in Multiple Formats

```python
# Save as CSV (default)
kit.save('output.csv')

# Save as Excel
kit.save('output.xlsx')

# Save as Parquet (compressed)
kit.save('output.parquet')

# Save as JSON
kit.save('output.json')

# Include or exclude columns
kit.save('output.csv', columns=['col1', 'col2'])
```

### Data Information & Overview

```python
# Get data shape, columns, types
kit.info()

# Display sample data
kit.head(10)
kit.tail(5)

# Get data types
print(kit.dtypes)

# Memory usage
print(kit.memory_usage())
```

---

## Data Cleaning & Quality

### Data Type Management

```python
# Automatically fix data types
kit.fix_dtypes()

# Convert specific columns
kit.convert_to_numeric(['col1', 'col2'])
kit.convert_to_datetime(['date_col'])
kit.convert_to_categorical(['category_col'])
```

### Missing Value Handling

```python
# Analyze missing values
kit.missing_summary()
kit.plot_missingness()

# Fill missing values - Auto strategy (intelligent)
kit.fill_missing(strategy='auto')

# Fill with specific strategies
kit.fill_missing(strategy='mean')      # Numerical columns
kit.fill_missing(strategy='median')    # Robust to outliers
kit.fill_missing(strategy='mode')      # Categorical columns
kit.fill_missing(strategy='forward_fill')  # Time series
kit.fill_missing(strategy='backward_fill') # Time series

# Fill specific column
kit.fill_missing_column('col_name', value=0)

# Drop missing values
kit.drop_missing()
kit.drop_missing_threshold(threshold=0.5)  # Drop if >50% missing
```

### Outlier Detection & Handling

```python
# Detect outliers (IQR method)
outliers = kit.detect_outliers(method='iqr')

# Detect outliers (Z-score method)
outliers = kit.detect_outliers(method='zscore')

# Remove outliers
kit.remove_outliers(method='iqr', threshold=3)

# Cap outliers at percentiles
kit.cap_outliers(lower=0.05, upper=0.95)

# Visualize outliers
kit.plot_outliers()
```

### Duplicate Management

```python
# Find duplicates
kit.find_duplicates()

# Remove duplicates
kit.remove_duplicates()

# Keep first/last occurrence
kit.remove_duplicates(keep='first')
kit.remove_duplicates(keep='last')
```

### Data Standardization

```python
# Standardize column names (snake_case, remove special chars)
kit.standardize_column_names()

# Rename columns
kit.rename_columns({'old_name': 'new_name'})

# Remove special characters from string columns
kit.clean_text_columns()

# Trim whitespace
kit.trim_whitespace()
```

### Data Quality Scoring

```python
# Get data health score (0-100)
health_score = kit.data_health_check()

# Get recommendations for improvement
recommendations = kit.get_data_quality_recommendations()

# Get quality metrics
metrics = kit.quality_metrics()
```

---

## Exploratory Data Analysis

### Quick EDA

```python
# Basic EDA with visualizations
kit.quick_eda()

# Comprehensive EDA with insights
kit.comprehensive_eda(target_col='target')

# Get EDA report as dictionary
report = kit.eda_report()
```

### Statistical Analysis

```python
# Summary statistics
kit.summary_statistics()

# Describe data
kit.describe()

# Statistical tests
kit.statistical_summary()

# Distribution analysis
kit.analyze_distributions()
```

### Correlation & Relationships

```python
# Correlation matrix
correlations = kit.correlation_matrix()

# Plot correlation heatmap
kit.plot_correlation_matrix()

# Feature relationships
kit.analyze_relationships()

# Target correlation
kit.target_correlation(target='target')
```

### Missing Data Analysis

```python
# Missing data patterns
kit.missing_patterns()

# Visualize missing data
kit.plot_missingness()

# Get missing value statistics
kit.missing_statistics()
```

### Categorical Analysis

```python
# Analyze categorical variables
kit.analyze_categorical()

# Value counts for all categorical columns
kit.categorical_summary()

# Plot categorical distributions
kit.plot_categorical_distributions()
```

### Numerical Analysis

```python
# Analyze numerical variables
kit.analyze_numerical()

# Distribution analysis
kit.distribution_analysis()

# Outlier statistics
kit.outlier_statistics()
```

---

## Preprocessing for ML

### Encoding Categorical Variables

```python
# Auto-encode all categorical columns
kit.auto_encode()

# One-hot encoding
kit.one_hot_encode()

# Label encoding
kit.label_encode()

# Target encoding (with smoothing)
kit.target_encode(target='target')

# Frequency encoding
kit.frequency_encode()

# Binary encoding
kit.binary_encode()
```

### Feature Scaling

```python
# Auto-scale all features
kit.auto_scale()

# Standard scaling (z-score)
kit.standardize()

# Min-Max scaling (0-1)
kit.normalize()

# Robust scaling (resistant to outliers)
kit.robust_scale()

# Log scaling (for skewed distributions)
kit.log_scale()
```

### Train-Test Splitting

```python
# Auto split with all preprocessing
X_train, X_test, y_train, y_test = kit.train_test_auto(target='target')

# Manual split with specific test size
X_train, X_test, y_train, y_test = kit.train_test_split(
    target='target', 
    test_size=0.2, 
    random_state=42,
    stratify=True
)

# Time series split (preserves temporal order)
X_train, X_test, y_train, y_test = kit.time_series_split(
    target='target',
    test_size=0.2
)

# Cross-validation splits
folds = kit.kfold_split(n_splits=5)
```

### Feature Selection

```python
# Auto feature selection
important_features = kit.auto_select_features(target='target')

# Correlation-based selection
features = kit.select_high_correlation_features(threshold=0.8)

# Variance-based selection
features = kit.select_high_variance_features(threshold=0.1)

# Statistical test-based selection
features = kit.statistical_feature_selection(target='target')

# Recursive feature elimination
features = kit.recursive_feature_elimination(target='target', n_features=10)
```

---

## Visualization Features

### Basic Plots

```python
# Histograms
kit.plot_histogram(column='col_name')
kit.plot_all_histograms()

# Box plots
kit.plot_boxplot(column='col_name')
kit.plot_all_boxplots()

# Scatter plots
kit.plot_scatter(x='col1', y='col2')

# Line plots
kit.plot_line(x='col1', y='col2')

# Bar plots
kit.plot_bar(x='col1', y='col2')

# Pie charts
kit.plot_pie(column='col_name')
```

### Correlation Visualization

```python
# Correlation heatmap
kit.plot_correlation_matrix()

# Pairplot (all numeric features)
kit.plot_pairplot()

# Target correlation
kit.plot_target_correlation(target='target')
```

### Distribution & Quality Plots

```python
# Distribution plots
kit.plot_distributions()

# Missing data heatmap
kit.plot_missingness()

# Outlier visualization
kit.plot_outliers()

# Data quality dashboard
kit.plot_data_quality()
```

### Advanced Visualizations

```python
# Interactive plots (Plotly)
kit.plot_interactive_histogram(column='col_name')

# 3D scatter plot
kit.plot_3d_scatter(x='col1', y='col2', z='col3')

# Violin plots (distribution + box plot)
kit.plot_violin(column='col_name', hue='category')

# KDE plots (smooth distributions)
kit.plot_kde(column='col_name')
```

---

## Feature Engineering

### Polynomial Features

```python
# Create polynomial features
kit.create_polynomial_features(degree=2)

# Polynomial features for specific columns
kit.create_polynomial_features(columns=['col1', 'col2'], degree=2)

# Interaction features (degree=2 with no powers)
kit.create_interaction_features()
```

### Temporal Features

```python
# Extract date/time features
kit.extract_datetime_features('date_column')

# Creates: year, month, day, weekday, quarter, season, day_of_year, etc.

# Cyclical encoding for temporal features
kit.encode_cyclical_features()
```

### Binning & Discretization

```python
# Equal-width binning
kit.bin_feature('age', n_bins=5)

# Equal-frequency binning (quantiles)
kit.bin_feature_quantile('age', n_bins=5)

# Custom binning
kit.bin_feature_custom('age', bins=[0, 18, 35, 50, 65, 100])
```

### Target Encoding

```python
# Mean target encoding
kit.target_encode(target='target', method='mean')

# Smoothed target encoding
kit.target_encode(target='target', method='smoothed')

# CV-fold target encoding (prevent leakage)
kit.target_encode(target='target', method='cv')
```

### Dimensionality Reduction

```python
# Principal Component Analysis
kit.apply_pca(n_components=10)

# Keep explained variance
kit.apply_pca(explained_variance=0.95)

# Truncated SVD (for sparse data)
kit.apply_truncated_svd(n_components=10)
```

### Group Features

```python
# Create aggregation features by group
kit.create_group_features(group_by='category', agg_col='value', agg_funcs=['mean', 'sum', 'std'])

# Rank within groups
kit.rank_within_group(group_by='category', rank_col='value')

# Difference from group mean
kit.group_centered_features(group_by='category', target_cols=['col1', 'col2'])
```

### Text Features

```python
# Text length and statistics
kit.create_text_features(text_column='text')

# TF-IDF features from text
kit.tfidf_features(text_column='text', max_features=100)

# Word embeddings (Word2Vec)
kit.word2vec_features(text_column='text')

# Sentiment features
kit.sentiment_features(text_column='text')
```

---

## Machine Learning Models

### Classification Models

```python
# Logistic Regression
kit.train('logistic_regression')

# Decision Tree
kit.train('decision_tree')

# Random Forest
kit.train('random_forest')

# Gradient Boosting
kit.train('gradient_boosting')

# XGBoost
kit.train('xgboost')

# LightGBM
kit.train('lightgbm')

# Support Vector Machine
kit.train('svm')

# Naive Bayes
kit.train('naive_bayes')

# K-Nearest Neighbors
kit.train('knn')

# Neural Network
kit.train('neural_network')
```

### Regression Models

```python
# Linear Regression
kit.train('linear_regression')

# Ridge Regression
kit.train('ridge')

# Lasso Regression
kit.train('lasso')

# Elastic Net
kit.train('elastic_net')

# SVR (Support Vector Regression)
kit.train('svr')

# Polynomial Regression
kit.train('polynomial_regression')

# Gradient Boosting Regressor
kit.train('gradient_boosting_regressor')

# Random Forest Regressor
kit.train('random_forest_regressor')
```

### Model Evaluation

```python
# Get comprehensive metrics
metrics = kit.evaluate()

# Classification metrics
accuracy = kit.accuracy()
precision = kit.precision()
recall = kit.recall()
f1 = kit.f1_score()
auc_roc = kit.auc_roc()
confusion = kit.confusion_matrix()

# Regression metrics
mae = kit.mean_absolute_error()
mse = kit.mean_squared_error()
rmse = kit.root_mean_squared_error()
r2 = kit.r2_score()
```

### Model Visualization

```python
# Confusion matrix
kit.plot_confusion_matrix()

# ROC Curve
kit.plot_roc_curve()

# Precision-Recall Curve
kit.plot_precision_recall_curve()

# Prediction vs Actual
kit.plot_predictions()

# Feature Importance
kit.plot_feature_importance()
```

---

## AutoML & Tuning

### Automated Model Selection

```python
# Train multiple models and compare
best_model = kit.train_multiple(['logistic_regression', 'random_forest', 'xgboost'])

# Auto select best model
best_model = kit.auto_train()

# Compare model performance
kit.compare_models()
```

### Hyperparameter Tuning

```python
# Grid search
kit.grid_search(param_grid={'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 15]})

# Random search
kit.random_search(param_distributions={'n_estimators': [10, 50, 100]}, n_iter=10)

# Bayesian optimization
kit.bayesian_optimization(n_iter=20)

# Auto tuning
kit.auto_tune()
```

### Cross-Validation

```python
# K-fold cross-validation
kit.cross_validate(cv=5)

# Stratified K-fold (for imbalanced data)
kit.cross_validate(cv=5, stratified=True)

# Time series cross-validation
kit.time_series_cross_validate(n_splits=5)

# Leave-one-out cross-validation
kit.cross_validate(cv='loo')
```

---

## NLP Features

### Text Cleaning

```python
# Clean text (remove URLs, emails, HTML)
kit.clean_text('text_column')

# Remove stopwords
kit.remove_stopwords('text_column')

# Tokenization
tokens = kit.tokenize('text_column')

# Stemming
kit.stem_text('text_column')

# Lemmatization
kit.lemmatize_text('text_column')
```

### Text Feature Extraction

```python
# TF-IDF
kit.tfidf_features('text_column')

# Bag of Words
kit.bow_features('text_column')

# Word Count features
kit.word_count_features('text_column')

# Character N-grams
kit.ngram_features('text_column', n=2)
```

### Sentiment & Topic Analysis

```python
# Sentiment analysis
sentiment = kit.sentiment_analysis('text_column')

# Topic modeling
topics = kit.topic_modeling('text_column', n_topics=5)

# Word frequency
word_freq = kit.word_frequency('text_column')

# Word cloud
kit.plot_word_cloud('text_column')
```

---

## Advanced Features

### Model Explainability

```python
# SHAP values
kit.explain_shap()

# Feature importance
kit.feature_importance()

# Partial dependence
kit.partial_dependence('feature_name')

# LIME explanation
kit.explain_lime(sample_index=0)
```

### Hyperplane Algorithm

```python
# Advanced ensemble technique
kit.train('hyperplane')

# With custom parameters
kit.train('hyperplane', custom_param=value)

# Hyperplane combination
kit.hyperplane_ensemble(['model1', 'model2', 'model3'])
```

### Model Deployment

```python
# Save trained model
kit.save_model('model.pkl')

# Load saved model
kit.load_model('model.pkl')

# Export to different formats
kit.export_model('model.onnx', format='onnx')
kit.export_model('model.joblib', format='joblib')

# Create prediction API
kit.create_api()
```

### Configuration Management

```python
# Get current configuration
config = kit.get_config()

# Set configuration
kit.set_config(random_state=42, verbose=True)

# Reset to defaults
kit.reset_config()
```

### Command Line Interface

```bash
# CLI access to all features
dskit load --file data.csv
dskit clean --file data.csv
dskit eda --file data.csv --target target_col
dskit train --file data.csv --model xgboost --target target_col
```

---

## Method Chaining

All dskit operations support method chaining for fluent, readable code:

```python
from dskit import dskit

# Complete pipeline in one readable chain
result = (dskit
    .load('data.csv')
    .fix_dtypes()
    .fill_missing(strategy='auto')
    .remove_outliers()
    .auto_encode()
    .auto_scale()
    .train('xgboost')
    .auto_tune()
    .evaluate()
    .explain_shap()
)
```

---

## Examples by Use Case

### Binary Classification (Credit Default)
```python
kit = dskit.load('credit.csv')
kit.comprehensive_eda(target_col='default')
kit.clean()
X_train, X_test, y_train, y_test = kit.train_test_auto(target='default')
kit.train('xgboost').auto_tune().evaluate().explain_shap()
```

### Regression (Price Prediction)
```python
kit = dskit.load('housing.csv')
kit.quick_eda()
kit.fill_missing(strategy='auto')
kit.auto_scale()
X_train, X_test, y_train, y_test = kit.train_test_auto(target='price')
kit.train('random_forest').evaluate()
```

### Text Classification (Sentiment)
```python
kit = dskit.load('reviews.csv')
kit.sentiment_features('review_text')
kit.tfidf_features('review_text')
X_train, X_test, y_train, y_test = kit.train_test_auto(target='sentiment')
kit.train('xgboost').evaluate()
```

---

## Complete Feature Count

- **✅ 221+ Functions** across 10 modules
- **✅ 20+ ML Algorithms** available
- **✅ 50+ Feature Engineering Strategies**
- **✅ 30+ Visualization Types**
- **✅ 15+ Data Cleaning Operations**
- **✅ 25+ EDA & Analysis Functions**
- **✅ 20+ NLP Operations**

---

*For detailed API documentation, see [API_REFERENCE.md](API_REFERENCE.md)*
