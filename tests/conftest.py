"""
Shared pytest fixtures for DsKit tests.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import shutil


@pytest.fixture
def sample_df():
    """Basic DataFrame with mixed data types."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 40, 45],
        'salary': [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        'department': ['HR', 'IT', 'IT', 'HR', 'Finance'],
        'join_date': pd.to_datetime(['2020-01-15', '2019-06-20', '2021-03-10', '2018-11-05', '2022-02-28'])
    })


@pytest.fixture
def sample_df_with_missing():
    """DataFrame with missing values for testing fill_missing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', None, 'Charlie', 'David', None],
        'age': [25, 30, None, 40, 45],
        'salary': [50000.0, None, 70000.0, None, 90000.0],
        'department': ['HR', 'IT', None, 'HR', 'Finance']
    })


@pytest.fixture
def sample_df_with_outliers():
    """DataFrame with outlier values for testing outlier functions."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'value': [10, 12, 11, 13, 12, 100, 11, 12, 13, -50],  # 100 and -50 are outliers
        'score': [85, 90, 88, 92, 87, 89, 91, 86, 500, 88]   # 500 is outlier
    })


@pytest.fixture
def sample_df_with_text():
    """DataFrame with text columns for NLP testing."""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'review': [
            'This is a GREAT product!!! I love it.',
            '  Bad quality...  terrible  experience  ',
            'OK product, nothing special #hashtag @mention'
        ],
        'category': ['Electronics', 'Clothing', 'Home']
    })


@pytest.fixture
def sample_df_dirty_columns():
    """DataFrame with messy column names."""
    return pd.DataFrame({
        'First Name': ['Alice', 'Bob'],
        'Last  Name': ['Smith', 'Jones'],
        'Age (Years)': [25, 30],
        'Salary$': [50000, 60000],
        '  Department  ': ['HR', 'IT']
    })


@pytest.fixture
def sample_df_string_numbers():
    """DataFrame with numbers stored as strings."""
    return pd.DataFrame({
        'id': ['1', '2', '3'],
        'price': ['99.99', '149.50', '75.00'],
        'quantity': ['10', '5', '20'],
        'date': ['2023-01-15', '2023-02-20', '2023-03-25']
    })


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file I/O tests."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


@pytest.fixture
def temp_csv_file(temp_dir, sample_df):
    """Create a temporary CSV file with sample data."""
    filepath = os.path.join(temp_dir, 'test_data.csv')
    sample_df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def temp_json_file(temp_dir, sample_df):
    """Create a temporary JSON file with sample data."""
    filepath = os.path.join(temp_dir, 'test_data.json')
    sample_df.to_json(filepath, orient='records')
    return filepath


@pytest.fixture
def temp_excel_file(temp_dir, sample_df):
    """Create a temporary Excel file with sample data."""
    filepath = os.path.join(temp_dir, 'test_data.xlsx')
    sample_df.to_excel(filepath, index=False)
    return filepath


@pytest.fixture
def temp_folder_with_csvs(temp_dir, sample_df):
    """Create a folder with multiple CSV files."""
    for i in range(3):
        df = sample_df.copy()
        df['batch'] = i
        filepath = os.path.join(temp_dir, f'data_{i}.csv')
        df.to_csv(filepath, index=False)
    return temp_dir
