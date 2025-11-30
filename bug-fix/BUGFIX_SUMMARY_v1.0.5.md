# Ak-dskit Version 1.0.5 - Bug Fixes and API Updates

**Release Date:** November 30, 2025
**Previous Version:** 1.0.4

## Summary

This release fixes critical API compatibility issues discovered during comprehensive testing with the Colab demo notebook. All functions now match the expected usage patterns documented in tutorials and examples.

## Critical Fixes

### 1. **QuickModel Class** - `dskit/modeling.py`

**Issue:** `QuickModel` only accepted `model_name` parameter, but documentation showed `model_type`.

**Fix:** Added support for both `model_type` and `model_name` parameters for backwards compatibility.

```python
# Before
QuickModel(model_name='random_forest', task='classification')

# After - both work
QuickModel(model_type='rf', task='classification')  # Short form
QuickModel(model_name='random_forest', task='classification')  # Long form
```

**Changes:**
- Added `model_type` parameter to `__init__`
- Added support for short model abbreviations: 'rf', 'gb', 'lr', 'svc', 'knn', 'dt'
- Added `score()` method for easy accuracy/R2 calculation
- Added proper support for Ridge and Lasso regression models

### 2. **compare_models Function** - `dskit/modeling.py`

**Issue:** Function had incorrect signature - expected `(X, y)` but users needed `(X_train, y_train, X_test, y_test, models=[...])`.

**Fix:** Complete rewrite of function signature and logic.

```python
# Before
compare_models(X, y, task='classification')  # Wrong!

# After
compare_models(X_train, y_train, X_test, y_test, 
               models=['lr', 'rf', 'gb', 'svc'], 
               task='classification')
```

**New Features:**
- Accepts pre-split train/test data
- Allows selection of specific models via `models` parameter
- Supports model abbreviations: 'lr', 'rf', 'gb', 'svc', 'knn' (classification)
- Supports model abbreviations: 'lr', 'ridge', 'lasso', 'rf', 'gb', 'svr' (regression)
- Better error handling for unsupported models

### 3. **auto_hpo Function** - `dskit/modeling.py`

**Issue:** Function signature didn't match expected usage pattern with Optuna integration.

**Fix:** Rewrote function to accept proper parameters for hyperparameter optimization.

```python
# Before
auto_hpo(model, param_grid, X, y, method='grid', cv=3)  # Wrong!

# After
auto_hpo(X_train, y_train, X_test=None, y_test=None,
         model_type='rf', task='classification',
         n_trials=20, scoring='accuracy', cv=3)
```

**New Features:**
- Uses Optuna for optimization (with automatic fallback if not installed)
- Accepts `model_type` abbreviations ('rf', 'gb')
- Returns tuple of (best_model, best_params)
- Automatic parameter space based on model type
- Falls back to default parameters if Optuna not available

### 4. **explain_shap Function** - `dskit/explainability.py`

**Issue:** Function didn't accept `feature_names` parameter, making it difficult to use with numpy arrays.

**Fix:** Added optional `feature_names` parameter.

```python
# Before
explain_shap(model, X)  # No way to specify feature names!

# After
explain_shap(model, X_test[:100], feature_names=X_cancer.columns.tolist())
```

**New Features:**
- Accepts optional `feature_names` parameter
- Automatically converts arrays to DataFrame with proper column names
- Better error messages

## Additional Improvements

### Import Updates

Added missing imports to `dskit/modeling.py`:
```python
from sklearn.linear_model import Ridge, Lasso
```

### Model Abbreviation Support

All modeling functions now support consistent abbreviations:

**Classification:**
- 'lr' → LogisticRegression
- 'rf' → RandomForestClassifier
- 'gb' → GradientBoostingClassifier
- 'svc' → SVC
- 'knn' → KNeighborsClassifier
- 'dt' → DecisionTreeClassifier

**Regression:**
- 'lr' → LinearRegression
- 'ridge' → Ridge
- 'lasso' → Lasso
- 'rf' → RandomForestRegressor
- 'gb' → GradientBoostingRegressor
- 'svr' → SVR
- 'dt' → DecisionTreeRegressor

## Testing

All changes have been validated using the comprehensive Colab demo notebook which tests:
- ✅ Breast Cancer Dataset (Classification)
- ✅ Wine Dataset (Multi-class Classification)  
- ✅ Diabetes Dataset (Regression)
- ✅ Iris Dataset (Classification)

## Breaking Changes

⚠️ **NONE** - All changes are backwards compatible. Old code will continue to work.

## Migration Guide

No migration needed! The package is fully backwards compatible.

If you were using workarounds for the old API, you can now use the simpler, documented syntax:

```python
# Old workaround
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
# ... then manually train each model

# New - just use compare_models directly!
results = compare_models(X_train, y_train, X_test, y_test,
                        models=['lr', 'rf', 'gb'],
                        task='classification')
```

## Installation

```bash
pip install --upgrade Ak-dskit
```

## Files Modified

- `dskit/modeling.py` - Major API fixes
- `dskit/explainability.py` - Added feature_names parameter
- `setup.py` - Version bump to 1.0.5
- `pyproject.toml` - Version bump to 1.0.5
- `dskit/__init__.py` - Version bump to 1.0.5

## Contributors

- Akshay Garg (@akshagr10)

## Next Steps

Version 1.0.6 will focus on:
- Adding more comprehensive error messages
- Improving documentation with more examples
- Adding type hints
- Performance optimizations

---

**Full Changelog:** https://github.com/Programmers-Paradise/DsKit/compare/v1.0.4...v1.0.5
