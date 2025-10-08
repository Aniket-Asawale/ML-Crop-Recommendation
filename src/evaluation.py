"""
Model Evaluation Module
========================

This module contains functions for evaluating machine learning models
and calculating various performance metrics.

Functions:
    - evaluate_classifier: Comprehensive classification evaluation
    - evaluate_regressor: Comprehensive regression evaluation
    - calculate_classification_metrics: Calculate all classification metrics
    - calculate_regression_metrics: Calculate all regression metrics
    - cross_validate_model: Perform k-fold cross-validation
    - compare_models: Compare multiple models
    - plot_confusion_matrix: Plot confusion matrix
    - plot_roc_curve: Plot ROC curve
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.model_selection import cross_val_score, cross_validate
import warnings
warnings.filterwarnings('ignore')


def calculate_classification_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive classification metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities for ROC-AUC
        
    Returns:
    --------
    metrics : dict
        Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0),
    }
    
    # Add ROC-AUC if probabilities are provided
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['roc_auc'] = None
    
    return metrics


def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate comprehensive regression metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    metrics : dict
        Dictionary containing all metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2_score': r2_score(y_true, y_pred)
    }
    
    return metrics


def evaluate_classifier(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Comprehensive evaluation of a classification model.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test labels
    model_name : str
        Name of the model for display
        
    Returns:
    --------
    results : dict
        Dictionary containing all evaluation results
    """
    print("=" * 70)
    print(f"EVALUATION: {model_name}")
    print("=" * 70)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Predicted probabilities (if available)
    try:
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
    except:
        y_train_proba = None
        y_test_proba = None
    
    # Calculate metrics
    train_metrics = calculate_classification_metrics(y_train, y_train_pred, y_train_proba)
    test_metrics = calculate_classification_metrics(y_test, y_test_pred, y_test_proba)
    
    # Print results
    print("\nTRAINING SET PERFORMANCE:")
    print("-" * 70)
    for metric, value in train_metrics.items():
        if value is not None:
            print(f"  {metric.upper():15s}: {value:.4f}")
    
    print("\nTEST SET PERFORMANCE:")
    print("-" * 70)
    for metric, value in test_metrics.items():
        if value is not None:
            print(f"  {metric.upper():15s}: {value:.4f}")
    
    # Confusion Matrix
    print("\nCONFUSION MATRIX (Test Set):")
    print("-" * 70)
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # Classification Report
    print("\nCLASSIFICATION REPORT (Test Set):")
    print("-" * 70)
    print(classification_report(y_test, y_test_pred))
    
    print("=" * 70)
    
    # Compile results
    results = {
        'model_name': model_name,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'y_train_proba': y_train_proba,
        'y_test_proba': y_test_proba,
        'confusion_matrix': cm
    }
    
    return results


def evaluate_regressor(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Comprehensive evaluation of a regression model.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test values
    model_name : str
        Name of the model for display
        
    Returns:
    --------
    results : dict
        Dictionary containing all evaluation results
    """
    print("=" * 70)
    print(f"EVALUATION: {model_name}")
    print("=" * 70)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_regression_metrics(y_train, y_train_pred)
    test_metrics = calculate_regression_metrics(y_test, y_test_pred)
    
    # Print results
    print("\nTRAINING SET PERFORMANCE:")
    print("-" * 70)
    for metric, value in train_metrics.items():
        print(f"  {metric.upper():15s}: {value:.4f}")
    
    print("\nTEST SET PERFORMANCE:")
    print("-" * 70)
    for metric, value in test_metrics.items():
        print(f"  {metric.upper():15s}: {value:.4f}")
    
    print("=" * 70)
    
    # Compile results
    results = {
        'model_name': model_name,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }
    
    return results


def cross_validate_model(model, X, y, cv=5, scoring=None, model_name="Model"):
    """
    Perform k-fold cross-validation.
    
    Parameters:
    -----------
    model : sklearn model
        Model to evaluate
    X : array-like
        Features
    y : array-like
        Target
    cv : int
        Number of folds
    scoring : str or list
        Scoring metric(s)
    model_name : str
        Name of the model
        
    Returns:
    --------
    cv_results : dict
        Cross-validation results
    """
    print("=" * 70)
    print(f"CROSS-VALIDATION: {model_name}")
    print(f"Folds: {cv}")
    print("=" * 70)
    
    if scoring is None:
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    cv_results = cross_validate(
        model, X, y, cv=cv, scoring=scoring, 
        return_train_score=True, n_jobs=-1
    )
    
    print("\nCROSS-VALIDATION RESULTS:")
    print("-" * 70)
    
    for metric in scoring:
        test_key = f'test_{metric}'
        if test_key in cv_results:
            mean_score = cv_results[test_key].mean()
            std_score = cv_results[test_key].std()
            print(f"  {metric.upper():15s}: {mean_score:.4f} (+/- {std_score:.4f})")
    
    print("=" * 70)
    
    return cv_results


def compare_models(results_list, metric='accuracy', dataset='test'):
    """
    Compare multiple models based on a specific metric.
    
    Parameters:
    -----------
    results_list : list of dict
        List of evaluation results from evaluate_classifier
    metric : str
        Metric to compare ('accuracy', 'f1_score', etc.)
    dataset : str
        Dataset to compare ('train' or 'test')
        
    Returns:
    --------
    comparison_df : pandas.DataFrame
        DataFrame with model comparisons
    """
    comparison_data = []
    
    for results in results_list:
        metrics_key = f'{dataset}_metrics'
        if metrics_key in results:
            model_data = {
                'Model': results['model_name'],
                **results[metrics_key]
            }
            comparison_data.append(model_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by the specified metric (descending)
    if metric in comparison_df.columns:
        comparison_df = comparison_df.sort_values(metric, ascending=False)
    
    print("=" * 90)
    print(f"MODEL COMPARISON ({dataset.upper()} SET)")
    print("=" * 90)
    print(comparison_df.to_string(index=False))
    print("=" * 90)
    
    return comparison_df


def calculate_clustering_metrics(X, labels):
    """
    Calculate clustering evaluation metrics.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    labels : array-like
        Cluster labels
        
    Returns:
    --------
    metrics : dict
        Dictionary containing clustering metrics
    """
    metrics = {}
    
    # Check if we have at least 2 clusters
    n_clusters = len(np.unique(labels))
    if n_clusters > 1:
        try:
            metrics['silhouette_score'] = silhouette_score(X, labels)
        except:
            metrics['silhouette_score'] = None
        
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
        except:
            metrics['davies_bouldin_score'] = None
        
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
        except:
            metrics['calinski_harabasz_score'] = None
    
    metrics['n_clusters'] = n_clusters
    
    return metrics


def print_model_summary(results):
    """
    Print a summary of model evaluation results.
    
    Parameters:
    -----------
    results : dict
        Evaluation results from evaluate_classifier or evaluate_regressor
    """
    print("\n" + "=" * 70)
    print(f"MODEL SUMMARY: {results['model_name']}")
    print("=" * 70)
    
    if 'test_metrics' in results:
        print("\nKEY METRICS (Test Set):")
        print("-" * 70)
        for metric, value in results['test_metrics'].items():
            if value is not None:
                print(f"  • {metric.upper()}: {value:.4f}")
    
    print("=" * 70 + "\n")


# Example usage
if __name__ == "__main__":
    print("Model Evaluation Module")
    print("This module provides functions for model evaluation.")
    print("\nExample usage:")
    print("  from src.evaluation import evaluate_classifier")
    print("  results = evaluate_classifier(model, X_train, X_test, y_train, y_test)")

