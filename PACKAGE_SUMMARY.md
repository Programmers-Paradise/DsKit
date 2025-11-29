# Ak-dskit Package Summary

## Package Information

- **PyPI Package Name**: `Ak-dskit` (with hyphen)
- **Python Import Name**: `dskit` (no hyphen - Python doesn't allow hyphens in imports)
- **Version**: 1.0.2
- **Main Class**: `dskit` (lowercase)

## Installation

```bash
pip install Ak-dskit
```

## Import and Usage

```python
# Correct import
from dskit import dskit

# Create dskit object
import pandas as pd
data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
kit = dskit(data)

# Method chaining is supported
kit = kit.fix_dtypes().fill_missing().remove_outliers()
```

## Important Notes

1. **Package vs Import Name**: 
   - Install with: `pip install Ak-dskit`
   - Import with: `from dskit import dskit`
   - This is standard practice (e.g., `pip install scikit-learn` but `import sklearn`)

2. **Class Name**: 
   - The main class is `dskit` (lowercase), not `DSKit`
   - Correct: `kit = dskit(data)`
   - Incorrect: `kit = DSKit(data)`

3. **Method Chaining**:
   - All methods return `self` for chaining
   - Example: `kit.fix_dtypes().fill_missing().remove_outliers()`

4. **Available Functions**:
   - 221+ public functions and classes
   - Covers: cleaning, EDA, visualization, preprocessing, modeling, explainability
   - 39 ML algorithms supported

## Test Scripts

- `quick_test.py` - Quick package verification
- `sample_script.py` - Comprehensive demonstration
- `simple_test.py` - Installation testing

## Files Updated for Package Rename

1. `setup.py` - Package name changed to `Ak-dskit`, version to 1.0.2
2. `pyproject.toml` - Already configured with `Ak-dskit`
3. Import statements remain as `from dskit import ...` (correct)

## Publishing

The package is ready for publishing to PyPI:
- Package name: Ak-dskit
- All imports use: dskit
- No conflicts between package name and import name
