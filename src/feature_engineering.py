"""
Feature Engineering Module
===========================

This module contains functions for feature engineering and selection.

Functions:
    - create_interaction_features: Create interaction terms
    - create_polynomial_features: Create polynomial features
    - select_features_correlation: Select features based on correlation
    - select_features_importance: Select features based on importance
    - engineer_features: Complete feature engineering pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')


def create_interaction_features(df, feature_pairs):
    """
    Create interaction features (multiplication of feature pairs).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    feature_pairs : list of tuples
        List of feature pairs to interact
        
    Returns:
    --------
    df_with_interactions : pandas.DataFrame
        DataFrame with added interaction features
    """
    df_with_interactions = df.copy()
    
    for feat1, feat2 in feature_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            interaction_name = f"{feat1}_x_{feat2}"
            df_with_interactions[interaction_name] = df[feat1] * df[feat2]
            print(f"✓ Created interaction feature: {interaction_name}")
    
    return df_with_interactions


def create_polynomial_features(X, degree=2, include_bias=False):
    """
    Create polynomial features.
    
    Parameters:
    -----------
    X : array-like
        Input features
    degree : int
        Degree of polynomial features
    include_bias : bool
        Whether to include bias column
        
    Returns:
    --------
    X_poly : array-like
        Polynomial features
    poly : PolynomialFeatures object
        Fitted transformer
    """
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    X_poly = poly.fit_transform(X)
    
    print(f"✓ Created polynomial features (degree={degree})")
    print(f"  Original features: {X.shape[1]}")
    print(f"  Polynomial features: {X_poly.shape[1]}")
    
    return X_poly, poly


def select_features_correlation(df, target_column, threshold=0.1):
    """
    Select features based on correlation with target.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_column : str
        Name of target column
    threshold : float
        Minimum absolute correlation threshold
        
    Returns:
    --------
    selected_features : list
        List of selected feature names
    correlations : pandas.Series
        Correlations with target
    """
    # Calculate correlations
    correlations = df.corr()[target_column].abs().sort_values(ascending=False)
    correlations = correlations.drop(target_column)
    
    # Select features above threshold
    selected_features = correlations[correlations >= threshold].index.tolist()
    
    print(f"✓ Selected {len(selected_features)} features with correlation >= {threshold}")
    print(f"  Features: {selected_features}")
    
    return selected_features, correlations


def select_features_importance(X, y, k=10, score_func=f_classif):
    """
    Select top k features based on statistical tests.
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target
    k : int
        Number of features to select
    score_func : callable
        Scoring function (f_classif, mutual_info_classif, etc.)
        
    Returns:
    --------
    X_selected : array-like
        Selected features
    selector : SelectKBest object
        Fitted selector
    """
    selector = SelectKBest(score_func=score_func, k=k)
    X_selected = selector.fit_transform(X, y)
    
    print(f"✓ Selected top {k} features using {score_func.__name__}")
    print(f"  Original features: {X.shape[1]}")
    print(f"  Selected features: {X_selected.shape[1]}")
    
    return X_selected, selector


def get_feature_scores(X, y, feature_names, score_func=f_classif):
    """
    Get feature importance scores.
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target
    feature_names : list
        Feature names
    score_func : callable
        Scoring function
        
    Returns:
    --------
    scores_df : pandas.DataFrame
        DataFrame with feature scores
    """
    selector = SelectKBest(score_func=score_func, k='all')
    selector.fit(X, y)
    
    scores_df = pd.DataFrame({
        'Feature': feature_names,
        'Score': selector.scores_
    }).sort_values('Score', ascending=False)
    
    return scores_df


def create_binned_features(df, column, bins, labels=None):
    """
    Create binned categorical features from continuous variables.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    column : str
        Column to bin
    bins : int or list
        Number of bins or bin edges
    labels : list
        Labels for bins
        
    Returns:
    --------
    df_binned : pandas.DataFrame
        DataFrame with binned feature
    """
    df_binned = df.copy()
    binned_col_name = f"{column}_binned"
    
    df_binned[binned_col_name] = pd.cut(df[column], bins=bins, labels=labels)
    
    print(f"✓ Created binned feature: {binned_col_name}")
    print(f"  Original range: [{df[column].min():.2f}, {df[column].max():.2f}]")
    print(f"  Number of bins: {len(bins) if isinstance(bins, list) else bins}")
    
    return df_binned


def create_ratio_features(df, numerator_col, denominator_col, new_name=None):
    """
    Create ratio features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    numerator_col : str
        Numerator column
    denominator_col : str
        Denominator column
    new_name : str
        Name for new feature (auto-generated if None)
        
    Returns:
    --------
    df_with_ratio : pandas.DataFrame
        DataFrame with ratio feature
    """
    df_with_ratio = df.copy()
    
    if new_name is None:
        new_name = f"{numerator_col}_div_{denominator_col}"
    
    # Avoid division by zero
    df_with_ratio[new_name] = df[numerator_col] / (df[denominator_col] + 1e-10)
    
    print(f"✓ Created ratio feature: {new_name}")
    
    return df_with_ratio


def engineer_features(df, target_column='target', 
                     create_interactions=False,
                     create_polynomials=False,
                     polynomial_degree=2):
    """
    Complete feature engineering pipeline.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_column : str
        Name of target column
    create_interactions : bool
        Whether to create interaction features
    create_polynomials : bool
        Whether to create polynomial features
    polynomial_degree : int
        Degree for polynomial features
        
    Returns:
    --------
    df_engineered : pandas.DataFrame
        DataFrame with engineered features
    """
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    df_engineered = df.copy()
    
    # Create interaction features
    if create_interactions:
        print("\n1. Creating interaction features...")
        # Example: age x cholesterol, age x trestbps, etc.
        feature_pairs = [
            ('age', 'chol'),
            ('age', 'trestbps'),
            ('thalach', 'oldpeak')
        ]
        df_engineered = create_interaction_features(df_engineered, feature_pairs)
    
    # Create polynomial features would be done separately on X
    if create_polynomials:
        print("\n2. Polynomial features...")
        print("  (Note: Apply after train/test split to avoid data leakage)")
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETED!")
    print(f"Original features: {df.shape[1]}")
    print(f"Engineered features: {df_engineered.shape[1]}")
    print("=" * 60)
    
    return df_engineered


# Example usage
if __name__ == "__main__":
    print("Feature Engineering Module")
    print("This module provides functions for feature engineering.")
    print("\nExample usage:")
    print("  from src.feature_engineering import create_interaction_features")
    print("  df_new = create_interaction_features(df, [('age', 'chol')])")

