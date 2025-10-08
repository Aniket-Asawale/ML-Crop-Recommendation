"""
Visualization Module
====================

This module contains functions for creating various plots and visualizations
for machine learning analysis.

Functions:
    - plot_confusion_matrix: Plot confusion matrix heatmap
    - plot_roc_curve: Plot ROC curve
    - plot_feature_importance: Plot feature importance
    - plot_correlation_matrix: Plot correlation heatmap
    - plot_distribution: Plot feature distributions
    - plot_model_comparison: Compare multiple models
    - plot_learning_curve: Plot learning curves
    - plot_pca_variance: Plot PCA explained variance
    - plot_clusters: Plot clustering results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix",
                          figsize=(8, 6), cmap='Blues', save_path=None):
    """
    Plot confusion matrix as a heatmap.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : list
        Label names
    title : str
        Plot title
    figsize : tuple
        Figure size
    cmap : str
        Color map
    save_path : str
        Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve", 
                   figsize=(8, 6), save_path=None):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str
        Path to save the figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.show()


def plot_feature_importance(importances, feature_names, title="Feature Importance",
                           top_n=None, figsize=(10, 8), save_path=None):
    """
    Plot feature importance as a horizontal bar chart.
    
    Parameters:
    -----------
    importances : array-like
        Feature importance values
    feature_names : list
        Feature names
    title : str
        Plot title
    top_n : int
        Number of top features to display
    figsize : tuple
        Figure size
    save_path : str
        Path to save the figure
    """
    # Create dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    # Select top N features
    if top_n:
        importance_df = importance_df.tail(top_n)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.barh(importance_df['Feature'], importance_df['Importance'], 
             color='steelblue', edgecolor='black')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.show()


def plot_correlation_matrix(df, title="Correlation Matrix", 
                            figsize=(12, 10), cmap='coolwarm', save_path=None):
    """
    Plot correlation matrix heatmap.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with features
    title : str
        Plot title
    figsize : tuple
        Figure size
    cmap : str
        Color map
    save_path : str
        Path to save the figure
    """
    corr = df.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap, 
                center=0, square=True, linewidths=1,
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.show()


def plot_distribution(df, columns=None, figsize=(15, 10), save_path=None):
    """
    Plot distributions of multiple features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with features
    columns : list
        Columns to plot (if None, plot all)
    figsize : tuple
        Figure size
    save_path : str
        Path to save the figure
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    n_cols = 3
    n_rows = int(np.ceil(len(columns) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, col in enumerate(columns):
        axes[i].hist(df[col].dropna(), bins=30, color='steelblue', 
                    edgecolor='black', alpha=0.7)
        axes[i].set_title(col, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Value', fontsize=10)
        axes[i].set_ylabel('Frequency', fontsize=10)
        axes[i].grid(alpha=0.3)
    
    # Hide empty subplots
    for i in range(len(columns), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.show()


def plot_model_comparison(comparison_df, metric='accuracy', 
                         title="Model Comparison", figsize=(12, 6), save_path=None):
    """
    Plot comparison of multiple models.
    
    Parameters:
    -----------
    comparison_df : pandas.DataFrame
        DataFrame with model comparison results
    metric : str
        Metric to plot
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str
        Path to save the figure
    """
    if metric not in comparison_df.columns:
        print(f"✗ Metric '{metric}' not found in comparison dataframe")
        return
    
    # Sort by metric
    comparison_df_sorted = comparison_df.sort_values(metric, ascending=True)
    
    plt.figure(figsize=figsize)
    plt.barh(comparison_df_sorted['Model'], comparison_df_sorted[metric],
             color='steelblue', edgecolor='black')
    plt.xlabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(comparison_df_sorted[metric]):
        plt.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.show()


def plot_pca_variance(pca, title="PCA Explained Variance", 
                      figsize=(12, 5), save_path=None):
    """
    Plot PCA explained variance.
    
    Parameters:
    -----------
    pca : sklearn.decomposition.PCA
        Fitted PCA object
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str
        Path to save the figure
    """
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Individual variance
    ax1.bar(range(1, len(explained_variance) + 1), explained_variance,
            color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('Variance Explained by Each Component', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Cumulative variance
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
             marker='o', linestyle='-', color='darkorange', linewidth=2, markersize=8)
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title('Cumulative Explained Variance', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.show()


def plot_clusters(X, labels, centers=None, title="Clustering Results",
                 figsize=(10, 8), save_path=None):
    """
    Plot clustering results (2D projection).
    
    Parameters:
    -----------
    X : array-like
        Feature matrix (2D or will use first 2 features)
    labels : array-like
        Cluster labels
    centers : array-like
        Cluster centers (optional)
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str
        Path to save the figure
    """
    # Use first 2 features if more than 2D
    if X.shape[1] > 2:
        X_plot = X[:, :2]
        if centers is not None:
            centers_plot = centers[:, :2]
    else:
        X_plot = X
        centers_plot = centers
    
    plt.figure(figsize=figsize)
    
    # Plot points
    scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, 
                         cmap='viridis', alpha=0.6, edgecolors='black', s=50)
    
    # Plot centers if provided
    if centers_plot is not None:
        plt.scatter(centers_plot[:, 0], centers_plot[:, 1], 
                   c='red', marker='X', s=200, edgecolors='black',
                   linewidth=2, label='Centroids')
        plt.legend()
    
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.show()


def plot_regression_results(y_true, y_pred, title="Regression Results",
                           figsize=(12, 5), save_path=None):
    """
    Plot regression results (actual vs predicted).
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str
        Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.6, edgecolors='black')
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
             'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Values', fontsize=12)
    ax1.set_ylabel('Predicted Values', fontsize=12)
    ax1.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Residuals plot
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors='black')
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Values', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.show()


# Example usage
if __name__ == "__main__":
    print("Visualization Module")
    print("This module provides functions for data visualization.")
    print("\nExample usage:")
    print("  from src.visualization import plot_confusion_matrix, plot_roc_curve")
    print("  plot_confusion_matrix(y_true, y_pred)")

