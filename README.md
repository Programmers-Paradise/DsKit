# 🚀 Ak-dskit - A Unified Wrapper Library for Data Science & ML

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/ak-dskit?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/ak-dskit)

**Ak-dskit** (import as `dskit`) is a comprehensive, community-driven, open-source Python library that wraps complex Data Science and ML operations into intuitive, user-friendly 1-line commands.

> **📝 Note**: Install using `pip install Ak-dskit`, but import in Python as `from dskit import dskit`

Instead of writing hundreds of lines for cleaning, EDA, plotting, preprocessing, modeling, evaluation, and explainability, dskit makes everything **simple**, **readable**, **reusable**, and **production-ready**.

The goal is to bring a **complete end-to-end Data Science ecosystem** in one place with wrapper-style functions and classes, supporting everything from basic data manipulation to advanced AutoML.

---

## 🎯 Project Objective

To create a Python library that lets users perform complete Data Science workflows with minimal code:

```python
from dskit import dskit

# Complete ML Pipeline in a few lines!
kit = dskit.load("data.csv")
kit.comprehensive_eda(target_col="target")  # EDA report
kit.clean()  # Clean data
kit.train_test_auto(target="target")  # Split data
kit.train_advanced("xgboost").auto_tune().evaluate().explain()  # Train, tune, evaluate, explain
```

The library remains:

- ✅ **Simple**: One-line commands for complex operations
- ✅ **Comprehensive**: 221 functions covering entire ML pipeline
- ✅ **Extensible**: Modular design for easy customization
- ✅ **Beginner-friendly**: Intuitive API with smart defaults
- ✅ **Expert-ready**: Advanced features and customization options
- ✅ **Production-ready**: Robust error handling and optimization

---

## 📦 Installation

### From PyPI (Recommended)

```bash
# Basic installation
pip install Ak-dskit

# Full installation with all optional dependencies
pip install Ak-dskit[full]

# Install specific feature sets
pip install Ak-dskit[visualization]  # Plotly support
pip install Ak-dskit[nlp]           # NLP utilities
pip install Ak-dskit[automl]        # AutoML algorithms

# Development installation
pip install Ak-dskit[dev]
```

### From Source

```bash
git clone https://github.com/Programmers-Paradise/DsKit.git
cd DsKit
pip install -e .
```

### Verify Installation

```bash
# Test the package
python test_package.py

# Check CLI
dskit --help
```

---

## 📦 Core Modules

dskit includes comprehensive modules for:

### 📁 **Data I/O**

- Multi-format loading (CSV, Excel, JSON, Parquet)
- Batch folder processing
- Smart data type detection

### 🧹 **Data Cleaning**

- Auto-detect and fix data types
- Smart missing value imputation
- Outlier detection and removal
- Column name standardization
- Text preprocessing and NLP utilities

### 📊 **Exploratory Data Analysis**

- Comprehensive EDA reports
- Data health scoring
- Interactive visualizations
- Statistical summaries
- Correlation analysis
- Missing data patterns

### 🔧 **Feature Engineering**

- Polynomial and interaction features
- Date/time feature extraction
- Binning and discretization
- Target encoding
- Dimensionality reduction (PCA)
- Text feature extraction
- Sentiment analysis

### 🤖 **Machine Learning**

- 15+ algorithms (including XGBoost, LightGBM, CatBoost)
- AutoML capabilities
- Hyperparameter optimization
- Cross-validation
- Ensemble methods
- Imbalanced data handling

### 📈 **Visualization**

- Static plots (matplotlib/seaborn)
- Interactive plots (plotly)
- Model performance charts
- Feature importance plots
- Advanced correlation heatmaps

### 🧠 **Model Explainability**

- SHAP integration
- Feature importance analysis
- Model performance metrics
- Error analysis
- Learning curves

### 📐 **Hyperplane Analysis**

- Algorithm-specific hyperplane visualization
- SVM margins and support vectors
- Logistic regression probability contours
- Perceptron misclassification highlighting
- LDA class centers and projections
- Linear regression residual analysis
- Multi-algorithm comparison tools

### 🎯 **AutoML Features**

- Automated preprocessing pipelines
- Model comparison and selection
- Hyperparameter tuning (Grid, Random, Bayesian, Optuna)
- Automated feature selection
- Pipeline optimization

---

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install Ak-dskit

# Full installation with all optional dependencies
pip install Ak-dskit[full]

# Development installation
git clone https://github.com/Programmers-Paradise/DsKit.git
cd DsKit
pip install -e .[dev,full]
```

### ✅ Verified Working Example

```python
import pandas as pd
from dskit import dskit

# 1. Load data
kit = dskit.load("your_data.csv")

# 2. Basic data exploration
print(f"Data shape: {kit.df.shape}")
health_score = kit.data_health_check()
print(f"Data health score: {health_score}/100")

# 3. Data cleaning
kit = kit.fix_dtypes().fill_missing(strategy='auto').remove_outliers()

# 4. EDA (generates comprehensive report)
kit.comprehensive_eda(target_col="your_target_column")

# 5. Feature engineering
if 'date_column' in kit.df.columns:
    kit.create_date_features(['date_column'])
if 'text_column' in kit.df.columns:
    kit.advanced_text_clean(['text_column'])
    kit.sentiment_analysis(['text_column'])

# 6. Model training
X_train, X_test, y_train, y_test = kit.train_test_auto(target="your_target_column")
kit.train(model_name="random_forest")
kit.evaluate()

# 7. Model explainability
kit.explain()  # Generates SHAP explanations
```

### Basic Usage

```python
from dskit import dskit

# Load and explore data
kit = dskit.load("your_data.csv")
health_score = kit.data_health_check()  # Get data quality score
kit.comprehensive_eda(target_col="target")  # Full EDA report

# Clean and preprocess
kit.clean()  # Auto-clean: fix types, handle missing, normalize columns
# Create features manually
kit.create_polynomial_features(degree=2)
kit.create_date_features(["date_column"])

# Train and evaluate models
kit.train_test_auto(target="your_target")
kit.compare_models("your_target")  # Compare multiple algorithms
kit.train_advanced("xgboost").auto_tune()  # Train with hyperparameter tuning
kit.evaluate().explain()  # Evaluate and generate SHAP explanations
```

### Advanced Features

```python
# Advanced text processing
kit.sentiment_analysis(["text_column"])
kit.extract_text_features(["text_column"])
kit.generate_wordcloud("text_column")

# Feature engineering
kit.create_polynomial_features(degree=3)
kit.create_date_features(["date_column"])
kit.apply_pca(variance_threshold=0.95)

# AutoML
kit.auto_tune(method="optuna", max_evals=100)
best_models = kit.compare_models("target", task="classification")

# Advanced visualizations
kit.plot_feature_importance(top_n=20)
# Learning curves and validation curves are available through model validation module
# from dskit.model_validation import ModelValidator
# validator = ModelValidator()
# validator.learning_curve_analysis(model, X, y)

# Algorithm-specific hyperplane visualization
dskit.plot_svm_hyperplane(svm_model, X, y)  # SVM with margins
dskit.plot_logistic_hyperplane(lr_model, X, y)  # Probability contours
dskit.plot_perceptron_hyperplane(perceptron_model, X, y)  # Misclassified points

# Compare multiple algorithm hyperplanes
models = {'SVM': svm, 'LR': lr, 'Perceptron': perceptron}
dskit.compare_algorithm_hyperplanes(models, X, y)
```

---

## 📚 Complete Feature Documentation

### 🧩 IMPLEMENTED FEATURES (All Tasks Complete)

Each task below is numbered and written in simple language with enough theory so that any contributor — even new ones — can understand exactly what to build.

---

## 📖 Examples & Tutorials

### Complete ML Pipeline Example

```python
import pandas as pd
from dskit import dskit

# 1. Load and explore
kit = dskit.load("customer_data.csv")
health_score = kit.data_health_check()  # Returns: 85.3/100

# 2. Comprehensive EDA
kit.comprehensive_eda(target_col="churn", sample_size=1000)
kit.generate_profile_report("eda_report.html")  # Automated EDA report

# 3. Advanced text processing (if text columns exist)
kit.advanced_text_clean(["feedback"])
kit.sentiment_analysis(["feedback"])
kit.extract_text_features(["feedback"])

# 4. Feature engineering
kit.create_date_features(["registration_date"])
kit.create_polynomial_features(degree=2, interaction_only=True)
kit.create_binning_features(["age", "income"], n_bins=5)

# 5. Preprocessing
kit.clean()  # Auto-clean pipeline
# Handle imbalanced data if needed
# from dskit.advanced_modeling import handle_imbalanced_data
# X_balanced, y_balanced = handle_imbalanced_data(X, y, method="smote")

# 6. Model training and optimization
X_train, X_test, y_train, y_test = kit.train_test_auto("churn")
comparison = kit.compare_models("churn")  # Compare 10+ algorithms
kit.train_advanced("xgboost").auto_tune(method="optuna", max_evals=50)

# 7. Evaluation and explainability
kit.evaluate().explain()  # Comprehensive evaluation + SHAP
kit.plot_feature_importance()
kit.cross_validate(cv=5)
```

### NLP Pipeline Example

```python
# Text analysis workflow
kit = dskit.load("reviews.csv")
kit.text_stats(["review_text"])  # Basic text statistics
kit.advanced_text_clean(["review_text"], remove_urls=True, expand_contractions=True)
kit.sentiment_analysis(["review_text"])  # Add sentiment scores
kit.generate_wordcloud("review_text", max_words=100)
kit.extract_keywords("review_text", top_n=20)
```

### Time Series Feature Engineering

```python
# Date/time feature extraction
kit.create_date_features(["transaction_date"])
# Creates: year, month, day, weekday, quarter, is_weekend columns

kit.create_aggregation_features("customer_id", ["amount"], ["mean", "std", "count"])
# Creates aggregated features grouped by customer
```

---

## 🎯 AutoML Capabilities

dskit includes comprehensive AutoML features:

- **Automated Preprocessing**: Smart data cleaning and feature engineering
- **Model Selection**: Automatic algorithm comparison and selection
- **Hyperparameter Optimization**: Grid, Random, Bayesian, and Optuna-based tuning
- **Feature Selection**: Univariate, RFE, and embedded methods
- **Ensemble Methods**: Voting classifiers and advanced ensembles
- **Performance Optimization**: Cross-validation and learning curve analysis

---

## 📊 Supported Algorithms

### Classification & Regression

- **Traditional**: Random Forest, Gradient Boosting, SVM, KNN, Naive Bayes
- **Advanced**: XGBoost, LightGBM, CatBoost, Neural Networks
- **Ensemble**: Voting Classifiers, Stacking, Bagging

### Preprocessing

- **Scaling**: Standard, MinMax, Robust, Quantile
- **Encoding**: Label, One-Hot, Target, Binary
- **Imputation**: Mean, Median, Mode, KNN, Iterative
- **Feature Selection**: SelectKBest, RFE, RFECV, Embedded

---

## 🔧 Configuration

dskit supports flexible configuration:

```python
# Global configuration
from dskit.config import set_config
set_config({
    'visualization_backend': 'plotly',  # or 'matplotlib'
    'auto_save_plots': True,
    'default_test_size': 0.2,
    'random_state': 42,
    'n_jobs': -1
})

# Method-specific parameters
kit.auto_tune(method="optuna", max_evals=100, timeout=3600)
kit.comprehensive_eda(sample_size=5000, include_correlations=True)
```

---

## 🔧 Troubleshooting Common Issues

### Import Errors

```python
# ❌ This might fail with import errors
from dskit import non_existent_function

# ✅ Import correctly
from dskit import dskit, load, fix_dtypes, quick_eda
```

### Method Chaining

```python
# ❌ Some methods don't return self
result = kit.missing_summary().fill_missing()  # Error!

# ✅ Correct approach
missing_info = kit.missing_summary()  # Returns DataFrame
kit = kit.fill_missing()  # Returns dskit object
```

### Data Loading

```python
# ❌ File not found
kit = dskit.load("non_existent_file.csv")

# ✅ Check file exists first
import os
if os.path.exists("data.csv"):
    kit = dskit.load("data.csv")
else:
    print("File not found!")
```

### Target Column Issues

```python
# ❌ Target column doesn't exist
kit.train_test_auto(target="non_existent_column")

# ✅ Check columns first
print("Available columns:", kit.df.columns.tolist())
if "target" in kit.df.columns:
    X_train, X_test, y_train, y_test = kit.train_test_auto(target="target")
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/Programmers-Paradise/DsKit.git
cd DsKit
pip install -e .[dev,full]
pre-commit install
```

### Running Tests

```bash
pytest tests/ --cov=dskit --cov-report=html  # If tests directory exists
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built on top of excellent libraries: pandas, scikit-learn, matplotlib, seaborn, plotly
- Inspired by the need for simplified data science workflows
- Community-driven development with contributions from data scientists worldwide

---

**Ak-dskit (dskit) - Making Data Science Simple, Comprehensive, and Accessible! 🚀**
