# 🧪 DsKit Testing Guide

A comprehensive guide for running and understanding the DsKit test suite.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running Tests](#running-tests)
- [Test Structure](#test-structure)
- [Writing New Tests](#writing-new-tests)
- [Troubleshooting](#troubleshooting)

---

## Overview

DsKit uses **pytest** as its testing framework. The test suite covers core functionality including:

| Module | What It Tests |
|--------|--------------|
| `io` | Data loading/saving (CSV, JSON, Excel, Parquet) |
| `cleaning` | Data cleaning, missing values, outliers |
| `preprocessing` | Encoding, scaling, train-test splitting |
| `feature_engineering` | Polynomial features, date features, PCA |

---

## Prerequisites

Before running tests, ensure you have:

- **Python 3.8+** installed
- **pip** package manager
- **Git** (for cloning the repository)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Programmers-Paradise/DsKit.git
cd DsKit
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install DsKit with Dev Dependencies

```bash
pip install -e .[dev]
```

This installs:
- **pytest** - Testing framework
- **pytest-cov** - Coverage reporting
- **black** - Code formatting
- **flake8** - Linting
- **mypy** - Type checking

---

## Running Tests

### Run All Tests

```bash
python -m pytest tests/ -v
```

### Run Specific Test File

```bash
# Test only I/O operations
python -m pytest tests/test_io.py -v

# Test only cleaning functions
python -m pytest tests/test_cleaning.py -v

# Test only preprocessing
python -m pytest tests/test_preprocessing.py -v

# Test only feature engineering
python -m pytest tests/test_feature_engineering.py -v
```

### Run Specific Test Class

```bash
python -m pytest tests/test_cleaning.py::TestFillMissing -v
```

### Run Single Test

```bash
python -m pytest tests/test_cleaning.py::TestFillMissing::test_fill_missing_auto_numeric -v
```

### Run with Coverage Report

```bash
python -m pytest tests/ -v --cov=dskit --cov-report=term-missing
```

### Run with HTML Coverage Report

```bash
python -m pytest tests/ --cov=dskit --cov-report=html
# Open htmlcov/index.html in browser
```

---

## Test Structure

```
tests/
├── __init__.py              # Package marker
├── conftest.py              # Shared fixtures
├── test_io.py               # I/O function tests
├── test_cleaning.py         # Data cleaning tests
├── test_preprocessing.py    # Preprocessing tests
└── test_feature_engineering.py  # Feature engineering tests
```

### Understanding conftest.py

The `conftest.py` file contains **fixtures** - reusable test data and resources:

| Fixture | Description |
|---------|------------|
| `sample_df` | Basic DataFrame with mixed types |
| `sample_df_with_missing` | DataFrame with missing values |
| `sample_df_with_outliers` | DataFrame with outlier values |
| `sample_df_with_text` | DataFrame with text columns |
| `sample_df_dirty_columns` | DataFrame with messy column names |
| `temp_dir` | Temporary directory for file operations |
| `temp_csv_file` | Temporary CSV file |
| `temp_json_file` | Temporary JSON file |
| `temp_excel_file` | Temporary Excel file |
| `temp_folder_with_csvs` | Folder with multiple CSV files |

---

## Writing New Tests

### Step 1: Create a Test File

Test files must be named `test_*.py` to be discovered by pytest.

### Step 2: Import Required Modules

```python
import pytest
import pandas as pd
import numpy as np

from dskit import your_module
```

### Step 3: Write Test Classes and Functions

```python
class TestYourFunction:
    """Tests for your_function()."""

    def test_basic_functionality(self, sample_df):
        """Test basic use case."""
        result = your_module.your_function(sample_df)
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_edge_case(self):
        """Test edge case handling."""
        df = pd.DataFrame({'col': []})
        result = your_module.your_function(df)
        
        assert len(result) == 0
```

### Step 4: Use Fixtures

```python
def test_with_fixtures(self, sample_df_with_missing, temp_dir):
    """Test using multiple fixtures."""
    # sample_df_with_missing is automatically injected
    # temp_dir is a temporary directory that gets cleaned up
    pass
```

### Test Naming Convention

- Test files: `test_<module_name>.py`
- Test classes: `Test<FunctionName>`
- Test functions: `test_<what_is_being_tested>`

---

## Troubleshooting

### pytest not found

```bash
# Use python -m pytest instead
python -m pytest tests/ -v
```

### Import errors

```bash
# Make sure DsKit is installed in development mode
pip install -e .
```

### Skipped tests

Some tests are skipped due to pandas version differences. This is expected - the tests are designed to be compatible across versions.

### Missing dependencies

```bash
# Install all optional dependencies
pip install -e .[full,dev]
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `python -m pytest tests/` | Run all tests |
| `python -m pytest tests/ -v` | Verbose output |
| `python -m pytest tests/ -x` | Stop on first failure |
| `python -m pytest tests/ -s` | Show print statements |
| `python -m pytest tests/ --tb=short` | Short traceback |
| `python -m pytest tests/ --tb=long` | Full traceback |
| `python -m pytest tests/ -k "missing"` | Run tests matching "missing" |

---

## Contributing New Tests

1. Fork the repository
2. Create a new branch: `git checkout -b feature/add-tests-for-xyz`
3. Add your tests following the conventions above
4. Run all tests to ensure they pass
5. Submit a Pull Request

**Thank you for contributing to DsKit! 🚀**
