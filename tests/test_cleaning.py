import pandas as pd
import pytest
from dskit import cleaning

def test_missing_summary_returns_dataframe():
    df = pd.DataFrame({"A": [1, None, 3], "B": [None, None, 2]})
    result = cleaning.missing_summary(df)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["Missing Count", "Missing %"]

def test_outlier_summary_returns_series():
    df = pd.DataFrame({"A": [1, 2, 100, 3, 4]})
    result = cleaning.outlier_summary(df)
    assert isinstance(result, pd.Series)
    assert result.name == "Outlier Count"

def test_fill_missing_auto_strategy():
    df = pd.DataFrame({"A": [1, None, 3], "B": ["x", None, "y"]})
    result = cleaning.fill_missing(df, strategy="auto")
    # Check that missing values are filled
    assert result["A"].isnull().sum() == 0
    assert result["B"].isnull().sum() == 0
