"""
Data Preprocessing Module
==========================

This module contains functions for loading, cleaning, and preprocessing
the crop recommendation dataset for intelligent agriculture.

Functions:
    - load_data: Load dataset from CSV
    - check_missing_values: Check for missing values
    - handle_missing_values: Handle missing data
    - encode_categorical: Encode categorical variables (crop labels)
    - scale_features: Scale numerical features (N, P, K, temp, humidity, ph, rainfall)
    - split_data: Split data into train/test sets
    - preprocess_pipeline: Complete preprocessing pipeline
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """
    Load the heart disease dataset from a CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    df : pandas.DataFrame
        Loaded dataset
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Data loaded successfully!")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"✗ Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        return None


def check_missing_values(df):
    """
    Check for missing values in the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
        
    Returns:
    --------
    missing_info : pandas.DataFrame
        DataFrame containing missing value information
    """
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    missing_info = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': missing_count.values,
        'Missing_Percent': missing_percent.values
    })
    
    missing_info = missing_info[missing_info['Missing_Count'] > 0].sort_values(
        'Missing_Count', ascending=False
    )
    
    if len(missing_info) == 0:
        print("✓ No missing values found!")
    else:
        print(f"⚠ Found missing values in {len(missing_info)} columns:")
        print(missing_info.to_string(index=False))
    
    return missing_info


def handle_missing_values(df, strategy='mean', categorical_strategy='most_frequent'):
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    strategy : str
        Strategy for numerical columns ('mean', 'median', 'constant')
    categorical_strategy : str
        Strategy for categorical columns ('most_frequent', 'constant')
        
    Returns:
    --------
    df_cleaned : pandas.DataFrame
        DataFrame with handled missing values
    """
    df_cleaned = df.copy()
    
    # Identify numerical and categorical columns
    numerical_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
    
    # Handle numerical columns
    if len(numerical_cols) > 0:
        num_imputer = SimpleImputer(strategy=strategy)
        df_cleaned[numerical_cols] = num_imputer.fit_transform(df_cleaned[numerical_cols])
    
    # Handle categorical columns
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy=categorical_strategy)
        df_cleaned[categorical_cols] = cat_imputer.fit_transform(df_cleaned[categorical_cols])
    
    print(f"✓ Missing values handled using {strategy} strategy for numerical columns")
    print(f"  and {categorical_strategy} strategy for categorical columns")
    
    return df_cleaned


def encode_categorical(df, columns=None, method='label'):
    """
    Encode categorical variables.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    columns : list
        List of columns to encode (if None, auto-detect)
    method : str
        Encoding method ('label' or 'onehot')
        
    Returns:
    --------
    df_encoded : pandas.DataFrame
        DataFrame with encoded categorical variables
    encoders : dict
        Dictionary of encoders for each column
    """
    df_encoded = df.copy()
    encoders = {}
    
    if columns is None:
        columns = df_encoded.select_dtypes(include=['object', 'category']).columns
    
    if method == 'label':
        for col in columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                encoders[col] = le
                print(f"✓ Label encoded column: {col}")
    
    elif method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=True)
        print(f"✓ One-hot encoded {len(columns)} columns")
    
    return df_encoded, encoders


def scale_features(X, method='standard', feature_range=(0, 1)):
    """
    Scale numerical features.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Features to scale
    method : str
        Scaling method ('standard' or 'minmax')
    feature_range : tuple
        Range for MinMaxScaler (default: (0, 1))
        
    Returns:
    --------
    X_scaled : numpy.ndarray
        Scaled features
    scaler : object
        Fitted scaler object
    """
    if method == 'standard':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("✓ Features scaled using StandardScaler")
    
    elif method == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)
        X_scaled = scaler.fit_transform(X)
        print(f"✓ Features scaled using MinMaxScaler (range: {feature_range})")
    
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    return X_scaled, scaler


def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train, validation, and test sets.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Features
    y : pandas.Series or numpy.ndarray
        Target variable
    test_size : float
        Proportion of test set (default: 0.2)
    val_size : float
        Proportion of validation set from training data (default: 0.1)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_val, X_test : array-like
        Split feature sets
    y_train, y_val, y_test : array-like
        Split target sets
    """
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
    else:
        X_train, y_train = X_temp, y_temp
        X_val, y_val = None, None
    
    print(f"✓ Data split completed:")
    print(f"  Train set: {X_train.shape[0]} samples")
    if val_size > 0:
        print(f"  Validation set: {X_val.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_pipeline(df, target_column='target', test_size=0.2, 
                        val_size=0.1, scale_method='standard',
                        random_state=42):
    """
    Complete preprocessing pipeline.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_column : str
        Name of the target column
    test_size : float
        Proportion of test set
    val_size : float
        Proportion of validation set
    scale_method : str
        Scaling method ('standard' or 'minmax')
    random_state : int
        Random seed
        
    Returns:
    --------
    dict : Dictionary containing:
        - X_train, X_val, X_test: Scaled features
        - y_train, y_val, y_test: Target values
        - scaler: Fitted scaler object
        - feature_names: List of feature names
    """
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # 1. Check for missing values
    print("\n1. Checking for missing values...")
    check_missing_values(df)
    
    # 2. Handle missing values
    print("\n2. Handling missing values...")
    df_cleaned = handle_missing_values(df)
    
    # 3. Separate features and target
    print("\n3. Separating features and target...")
    if target_column not in df_cleaned.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    X = df_cleaned.drop(columns=[target_column])
    y = df_cleaned[target_column]
    feature_names = list(X.columns)
    print(f"✓ Features: {len(feature_names)} columns")
    print(f"  Target: {target_column}")
    
    # 4. Split data
    print("\n4. Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=test_size, val_size=val_size, random_state=random_state
    )
    
    # 5. Scale features
    print("\n5. Scaling features...")
    X_train_scaled, scaler = scale_features(X_train, method=scale_method)
    X_test_scaled = scaler.transform(X_test)
    
    if val_size > 0:
        X_val_scaled = scaler.transform(X_val)
    else:
        X_val_scaled = None
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    # Return results as dictionary
    results = {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_names
    }
    
    return results


def get_dataset_info(df):
    """
    Get comprehensive information about the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
        
    Returns:
    --------
    None (prints information)
    """
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    
    print(f"\n1. Shape: {df.shape}")
    print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print(f"\n2. Data Types:")
    print(df.dtypes)
    
    print(f"\n3. Memory Usage:")
    print(df.memory_usage(deep=True))
    
    print(f"\n4. Statistical Summary:")
    print(df.describe())
    
    print(f"\n5. Missing Values:")
    check_missing_values(df)
    
    print("\n" + "=" * 60)


# Example usage
if __name__ == "__main__":
    # Example of how to use this module
    print("Data Preprocessing Module")
    print("This module provides functions for data preprocessing.")
    print("\nExample usage:")
    print("  from src.data_preprocessing import load_data, preprocess_pipeline")
    print("  df = load_data('data/raw/heart.csv')")
    print("  results = preprocess_pipeline(df)")

