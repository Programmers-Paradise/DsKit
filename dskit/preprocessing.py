import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def auto_encode(df, max_unique_for_onehot=10):
    """
    Automatically encodes categorical variables.
    """
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns

    for col in datetime_cols:
        # Convert datetime to timestamp (numeric)
        df[col] = df[col].astype('int64') // 10**9
    
    for col in categorical_cols:
        if df[col].nunique() <= max_unique_for_onehot:
            # One-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)
        else:
            # Label encoding
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            
    return df

def auto_scale(df, method='standard'):
    """
    Automatically scales numeric data.
    method: 'standard' or 'minmax'
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Method must be 'standard' or 'minmax'")
        
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def train_test_auto(df, target=None, test_size=0.2, random_state=42):
    """
    Automatically splits data into training and testing sets.
    """
    if target is None:
        # Try to guess target - usually the last column
        target = df.columns[-1]
        print(f"Target not specified. Assuming '{target}' is the target.")
        
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")
        
    X = df.drop(target, axis=1)
    y = df[target]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
