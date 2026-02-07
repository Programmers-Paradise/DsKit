import pandas as pd
import numpy as np
import re

def fix_dtypes(df):
    """
    Auto-detects and converts column types in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame whose column types need to be inferred and converted.

    Returns
    -------
    pandas.DataFrame
        A copy of the DataFrame with columns converted to appropriate types:
        - Numeric if possible
        - Datetime if possible
        - Category if object dtype with low cardinality

    Notes
    -----
    - Numeric conversion is attempted first, followed by datetime conversion.
    - Object columns with unique values less than 50% of total rows are converted to category.
    - Category conversion is heuristic-based; users should validate conversions for critical datasets.

    Example
    -------
    >>> import pandas as pd
    >>> from dskit.cleaning import fix_dtypes
    >>> df = pd.DataFrame({"A": ["1", "2", "3"], "B": ["2020-01-01", "2020-01-02", "2020-01-03"]})
    >>> fix_dtypes(df).dtypes
    A             int64
    B    datetime64[ns]
    dtype: object
    """

    df = df.copy()
    for col in df.columns:
        # Try converting to numeric
        try:
            df[col] = pd.to_numeric(df[col])
            continue
        except (ValueError, TypeError):
            pass

        # Try converting to datetime
        try:
            df[col] = pd.to_datetime(df[col])
            continue
        except (ValueError, TypeError):
            pass
            
        # Convert object to category if low cardinality
        if df[col].dtype == 'object':
            if df[col].nunique() / len(df) < 0.5: # Heuristic
                df[col] = df[col].astype('category')
                
    return df

def rename_columns_auto(df):
    """
    Automatically cleans DataFrame column names.

    This function:
    - Converts column names to lowercase
    - Replaces spaces with underscores
    - Removes special characters

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame whose column names need cleaning.

    Returns
    -------
    pandas.DataFrame
        DataFrame with cleaned column names.

    Raises
    ------
    TypeError
        If input is not a pandas DataFrame.

    Example
    -------
    >>> import pandas as pd
    >>> from dskit.cleaning import rename_columns_auto
    >>> df = pd.DataFrame(columns=["User Name", "Total$Amount"])
    >>> rename_columns_auto(df).columns
    Index(['user_name', 'totalamount'], dtype='object')
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    df = df.copy()
    new_cols = []
    for col in df.columns:
        clean_col = str(col).strip().lower()
        clean_col = clean_col.replace(' ', '_')
        clean_col = re.sub(r'[^a-z0-9_]', '', clean_col)
        new_cols.append(clean_col)
    df.columns = new_cols
    return df


def replace_specials(df, chars_to_remove=r'[@#%$]', replacement=''):
    """
    Removes or replaces special characters from text columns in a DataFrame.

    This function applies a regular expression replacement to all
    object/string columns in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing text columns.
    chars_to_remove : str, default=r'[@#%$]'
        Regular expression pattern of characters to remove or replace.
    replacement : str, default=''
        String to replace the matched characters with.

    Returns
    -------
    pandas.DataFrame
        DataFrame with special characters removed or replaced in text columns.

    Raises
    ------
    TypeError
        If input is not a pandas DataFrame.

    Example
    -------
    >>> import pandas as pd
    >>> from dskit.cleaning import replace_specials
    >>> df = pd.DataFrame({'text': ['Hello@World!', 'Price#$100']})
    >>> replace_specials(df)
             text
    0  HelloWorld
    1   Price100
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    df = df.copy()
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str).str.replace(
            chars_to_remove, replacement, regex=True
        )
    return df


def missing_summary(df):
    """
    Generates a summary of missing values in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns: 
        - 'Missing Count': number of missing values per column 
        - 'Missing %': percentage of missing values per column 
        Only columns with missing values are included.

    Example
    -------
    >>> import pandas as pd
    >>> from dskit.cleaning import missing_summary
    >>> df = pd.DataFrame({"A": [1, None, 3], "B": [None, None, 2]})
    >>> missing_summary(df)
       Missing Count  Missing %
    B              2  66.666667
    A              1  33.333333
    """

    missing = df.isnull().sum()
    missing_percent = 100 * df.isnull().sum() / len(df)
    summary = pd.concat([missing, missing_percent], axis=1, keys=['Missing Count', 'Missing %'])
    return summary[summary['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)

def fill_missing(df, strategy='auto', fill_value=None):
    """
    Fills missing values in a DataFrame using various strategies.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    strategy : str, default='auto'
        Strategy for filling missing values:
        - 'auto': mean for numeric, mode for non-numeric
        - 'mean': fill with column mean
        - 'median': fill with column median
        - 'mode': fill with column mode
        - 'ffill': forward fill
        - 'bfill': backward fill
        - 'constant': fill with a specified constant value
    fill_value : any, optional
        Value to use when strategy='constant'.

    Returns
    -------
    pandas.DataFrame
        DataFrame with missing values filled.

    Example
    -------
    >>> import pandas as pd
    >>> from dskit.cleaning import fill_missing
    >>> df = pd.DataFrame({"A": [1, None, 3], "B": ["x", None, "y"]})
    >>> fill_missing(df, strategy="auto")
       A  B
    0  1  x
    1  2  x
    2  3  y

    Notes
    -----
    - If strategy='constant' and fill_value=None, missing values will remain unchanged.
    - For 'auto', numeric columns use mean, non-numeric columns use mode.

    """

    df = df.copy()
    
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
            
        col_strategy = strategy
        if strategy == 'auto':
            if pd.api.types.is_numeric_dtype(df[col]):
                col_strategy = 'mean'
            else:
                col_strategy = 'mode'
        
        if col_strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        elif col_strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif col_strategy == 'mode':
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
        elif col_strategy == 'ffill':
            df[col] = df[col].fillna(method='ffill')
        elif col_strategy == 'bfill':
            df[col] = df[col].fillna(method='bfill')
        elif col_strategy == 'constant' and fill_value is not None:
            df[col] = df[col].fillna(fill_value)
            
    return df

def outlier_summary(df, method='iqr', threshold=1.5):
    """
    Summarizes the number of outliers in numeric columns of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    method : str, default='iqr'
        Method for detecting outliers:
        - 'iqr': Interquartile Range method
        - 'zscore': Z-score method
    threshold : float, default=1.5
        Threshold multiplier for IQR method. For z-score, cutoff is fixed at 3.

    Returns
    -------
    pandas.Series
        Series with: 
        - Index: column names 
        - Values: outlier counts per column

    Example
    -------
    Standard case:
    >>> import pandas as pd
    >>> from dskit.cleaning import outlier_summary
    >>> df = pd.DataFrame({"A": [1, 2, 100, 3, 4]})
    >>> outlier_summary(df)
    A    1
    Name: Outlier Count, dtype: int64

    Edge case (no outliers):
    >>> import pandas as pd
    >>> from dskit.cleaning import outlier_summary
    >>> df = pd.DataFrame({"A": [1, 2, 3, 4]})  # no outliers
    >>> outlier_summary(df)
    Series([], Name: Outlier Count, dtype: int64)

    """

    summary = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (threshold * IQR)
            upper_bound = Q3 + (threshold * IQR)
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:
                continue
            z_scores = (df[col] - mean) / std
            outliers = df[np.abs(z_scores) > 3]
        
        if not outliers.empty:
            summary[col] = len(outliers)
            
    return pd.Series(summary, name='Outlier Count').sort_values(ascending=False)

def remove_outliers(df, method='iqr', threshold=1.5):
    """
    Removes rows containing outliers from numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    method : str, default='iqr'
        Method for detecting outliers:
        - 'iqr': Interquartile Range method
        - 'zscore': Z-score method
    threshold : float, default=1.5
        Threshold multiplier for IQR method. For z-score, cutoff is fixed at 3.

    Returns
    -------
    pandas.DataFrame
        DataFrame with outlier rows removed.

    Example
    -------
    >>> import pandas as pd
    >>> from dskit.cleaning import remove_outliers
    >>> df = pd.DataFrame({"A": [1, 2, 100, 3, 4]})
    >>> remove_outliers(df)
       A
    0  1
    1  2
    3  3
    4  4
    """

    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (threshold * IQR)
            upper_bound = Q3 + (threshold * IQR)
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            if std==0:
                continue
            z_scores = (df[col] - mean) / std
            df = df[np.abs(z_scores) <= 3]
            
    return df

def simple_nlp_clean(df, text_cols=None):
    """
    Performs basic text cleaning on specified columns.

    Operations
    ----------
    - Converts text to lowercase
    - Removes punctuation
    - Removes extra spaces

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    text_cols : list of str, optional
        List of column names to clean. If None, all object/string columns are cleaned.

    Returns
    -------
    pandas.DataFrame
        DataFrame with cleaned text columns.

    Example
    -------
    Standard case:
    >>> import pandas as pd
    >>> from dskit.cleaning import simple_nlp_clean
    >>> df = pd.DataFrame({"text": ["Hello, World!!", "Python   is GREAT"]})
    >>> simple_nlp_clean(df)
               text
    0    hello world
    1  python is great

    Edge case (no outliers):
    >>> df = pd.DataFrame({"text": ["   ", None]})
    >>> simple_nlp_clean(df)
        text
    0      
    1   none

    """

    df = df.copy()
    if text_cols is None:
        text_cols = df.select_dtypes(include=['object', 'string']).columns
        
    for col in text_cols:
        # Lowercase
        df[col] = df[col].astype(str).str.lower()
        # Remove punctuation
        df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
        # Remove extra spaces
        df[col] = df[col].str.replace(r'\s+', ' ', regex=True).str.strip()
        
    return df
