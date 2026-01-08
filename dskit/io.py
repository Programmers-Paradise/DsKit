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

def read_folder(folder_path:str, file_type:str='csv',dynamic:bool=False,display_ignored:bool=False):
    """
    Load and concatenate tabular files from a folder.
    Parameters
    ----------
    folder_path : str
        Path to the directory containing the files to be loaded.

    file_type : str, default='csv'
        File extension to filter files when `dynamic=False`.
        Example: 'csv', 'xlsx', 'parquet'.

    dynamic : bool, default=False
        If True, loads all files regardless of extension.
        If False, only files matching `file_type` are loaded.

    display_ignored : bool, default=False
        If True, prints the list of files that were skipped
        because they could not be loaded by the `load()` function.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' was not found.")
    if dynamic:
        all_files = glob.glob(os.path.join(folder_path, "*.*"))
        if not all_files:
            print(f"No files found!")
            return None
    else:
        all_files = glob.glob(os.path.join(folder_path, f"*.{file_type}"))
        if not all_files:
            print(f"No files found with extension .{file_type} in {folder_path}")
            return None

    df_list = []
    ignored=[]
    for filename in all_files:
        df = load(filename)
        if df is not None:
            df_list.append(df)
        else:
            ignored.append(filename)
    if display_ignored:
        print("Ignored Files : ","\n".join(ignored))
    if df_list:
        return pd.concat(df_list, ignore_index=True)
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
