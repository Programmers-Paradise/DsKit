import pandas as pd
import os
import glob

def load(filepath):
    """
    Loads a file into a pandas DataFrame.
    Supports CSV, Excel, JSON, Parquet.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file '{filepath}' was not found.")

    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext == '.csv':
            return pd.read_csv(filepath)
        elif ext in ['.xls', '.xlsx']:
            return pd.read_excel(filepath)
        elif ext == '.json':
            return pd.read_json(filepath)
        elif ext == '.parquet':
            return pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def read_folder(folder_path, file_type='csv'):
    """
    Load multiple files from a folder and return 
    a list of pandas DataFrames.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' was not found.")

    all_files = glob.glob(os.path.join(folder_path, f"*.{file_type}"))
    
    if not all_files:
        print(f"No files found with extension .{file_type} in {folder_path}")
        return None

    df_list = []
    for filename in all_files:
        df = load(filename)
        if df is not None:
            df_list.append(df)

    if df_list:
        return df_list
    else:
        return None

def save(df, filepath, **kwargs):
    """
    Saves the DataFrame to a file.
    """
    if df is None:
        print("No DataFrame to save.")
        return

    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == '.csv':
            df.to_csv(filepath, index=False, **kwargs)
        elif ext in ['.xls', '.xlsx']:
            df.to_excel(filepath, index=False, **kwargs)
        elif ext == '.json':
            df.to_json(filepath, orient='records', **kwargs)
        elif ext == '.parquet':
            df.to_parquet(filepath, index=False, **kwargs)
        else:
            print(f"Unsupported file format for saving: {ext}")
    except Exception as e:
        print(f"Error saving file: {e}")
