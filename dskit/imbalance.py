"""
Imbalanced data utilities for DsKit.

This module provides functions for handling class imbalance in machine learning:
- get_class_weights: Calculate sklearn-style class weights
- apply_smote: Resample using SMOTE (with graceful fallback)
- threshold_tuning: Find optimal classification threshold
"""

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def get_class_weights(y):
    """
    Calculate sklearn-style class weights for imbalanced datasets.
    
    Parameters
    ----------
    y : array-like
        Target variable with class labels.
    
    Returns
    -------
    dict
        Dictionary mapping class labels to their weights.
        Weights are computed using sklearn's 'balanced' strategy.
    
    Examples
    --------
    >>> y = np.array([0, 0, 0, 0, 1])
    >>> weights = get_class_weights(y)
    >>> weights
    {0: 0.625, 1: 2.5}
    """
    y = np.asarray(y)
    classes = np.unique(y)
    
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y
    )
    
    return dict(zip(classes, weights))


def apply_smote(X, y, random_state=42):
    """
    Resample data using SMOTE for handling class imbalance.
    
    Falls back gracefully if imblearn is not installed, returning original
    data with a warning message suggesting to use class weights instead.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target variable.
    random_state : int, default=42
        Random seed for reproducibility.
    
    Returns
    -------
    X_resampled : array-like
        Resampled feature matrix (or original if SMOTE unavailable).
    y_resampled : array-like
        Resampled target variable (or original if SMOTE unavailable).
    
    Notes
    -----
    Requires imblearn package: pip install imbalanced-learn
    
    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    >>> y = np.array([0, 0, 0, 0, 1])
    >>> X_res, y_res = apply_smote(X, y)
    """
    try:
        from imblearn.over_sampling import SMOTE
        
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Print distribution info
        original_dist = pd.Series(y).value_counts().to_dict()
        resampled_dist = pd.Series(y_resampled).value_counts().to_dict()
        print(f"Original class distribution: {original_dist}")
        print(f"Resampled class distribution: {resampled_dist}")
        
        return X_resampled, y_resampled
        
    except ImportError:
        print("Warning: imbalanced-learn (imblearn) is not installed.")
        print("SMOTE resampling is not available.")
        print("Suggestion: Use get_class_weights() and pass to your model's class_weight parameter.")
        print("Install imblearn with: pip install imbalanced-learn")
        return X, y


def threshold_tuning(y_true, y_proba, metric="f1"):
    """
    Find the optimal classification threshold for a given metric.
    
    Tests thresholds from 0.1 to 0.9 and returns the threshold that
    maximizes the specified metric.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_proba : array-like of shape (n_samples,) or (n_samples, 2)
        Predicted probabilities. If 2D, uses the probability of the
        positive class (column index 1).
    metric : str, default="f1"
        Metric to optimize. Options: "f1", "precision", "recall", "accuracy".
    
    Returns
    -------
    dict
        Dictionary with:
        - 'best_threshold': float, optimal threshold value
        - 'best_score': float, best score achieved
        - 'metric': str, the metric that was optimized
        - 'all_results': list of dicts with threshold and score pairs
    
    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1, 1])
    >>> y_proba = np.array([0.2, 0.4, 0.5, 0.7, 0.9])
    >>> result = threshold_tuning(y_true, y_proba, metric="f1")
    >>> print(f"Best threshold: {result['best_threshold']}")
    """
    # Handle 2D probability arrays
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]  # Use positive class probability
    
    y_true = np.asarray(y_true)
    
    # Define metric functions
    metric_functions = {
        'f1': lambda y, p: f1_score(y, p, zero_division=0),
        'precision': lambda y, p: precision_score(y, p, zero_division=0),
        'recall': lambda y, p: recall_score(y, p, zero_division=0),
        'accuracy': accuracy_score
    }
    
    if metric not in metric_functions:
        raise ValueError(f"metric must be one of {list(metric_functions.keys())}")
    
    metric_func = metric_functions[metric]
    
    # Test thresholds
    thresholds = np.arange(0.1, 0.95, 0.05)
    all_results = []
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Skip if all predictions are the same class
        if len(np.unique(y_pred)) < 2:
            continue
            
        score = metric_func(y_true, y_pred)
        all_results.append({'threshold': round(threshold, 2), 'score': score})
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return {
        'best_threshold': round(best_threshold, 2),
        'best_score': float(best_score),
        'metric': metric,
        'all_results': all_results
    }


def detect_imbalance(
    data,
    target,
    threshold: float = 0.3,
    visualize: bool = True,
    recommend: bool = True
) -> dict:
    """
    Detect and analyze class imbalance in classification data.

    Parameters
    ----------
    data : DataFrame or array-like
        Dataset containing features and target (or ignored if `target` is an array).
    target : str or array-like
        Target column name (if data is DataFrame) or target array.
    threshold : float, default=0.3
        Imbalance threshold (e.g., 0.3 means 30:70 split triggers warning).
    visualize : bool, default=True
        Generate visualization of class distribution.
    recommend : bool, default=True
        Provide handling strategy recommendations.

    Returns
    -------
    dict
        Contains `is_imbalanced`, `severity`, `class_distribution`, `imbalance_ratio`,
        `proportions`, and `recommendations`.
    """
    # Resolve target array/series
    if isinstance(data, pd.DataFrame) and isinstance(target, str):
        y = data[target]
    else:
        # If target is provided as an array-like, use it; otherwise assume `data` is the target
        if isinstance(target, (pd.Series, np.ndarray, list)):
            y = pd.Series(target)
        else:
            y = pd.Series(data)

    y = y.dropna()

    counts = y.value_counts()
    total = counts.sum()
    props = (counts / total)

    # Basic imbalance metrics
    majority_count = int(counts.max())
    minority_count = int(counts.min())
    imbalance_ratio = float(majority_count / minority_count) if minority_count > 0 else float('inf')

    max_prop = float(props.max())
    min_prop = float(props.min())

    # Severity: difference between largest and smallest class proportions (0-1)
    severity = float(max_prop - min_prop)

    # Determine if imbalanced based on threshold
    is_imbalanced = (min_prop < threshold) or (max_prop > (1.0 - threshold))

    # Recommendations based on severity
    recommendations = []
    if recommend:
        if severity < 0.1:
            recommendations = [
                'Data appears balanced; no resampling required',
                'Consider stratified cross-validation to preserve distributions'
            ]
        elif severity < 0.3:
            recommendations = [
                'Use class weights in model training',
                'Consider threshold tuning or calibration',
                'Monitor minority-class metrics (precision/recall/f1)'
            ]
        elif severity < 0.6:
            recommendations = [
                'Use SMOTE (oversampling) for the minority class',
                'Combine oversampling with class weights',
                'Consider undersampling the majority class or ensemble methods'
            ]
        else:
            recommendations = [
                'Severe imbalance: use SMOTE or advanced oversampling techniques',
                'Collect more minority-class data if possible',
                'Combine resampling with class weights and careful validation',
                'Consider reframing as anomaly detection if appropriate'
            ]

    # Visualization
    if visualize:
        try:
            import matplotlib.pyplot as plt
            try:
                import seaborn as sns
                sns.set_style('whitegrid')
            except Exception:
                pass

            fig, ax = plt.subplots()
            # Color coding: green=balanced, orange=moderate, red=severe
            if severity < 0.3:
                bar_color = 'green'
            elif severity < 0.6:
                bar_color = 'orange'
            else:
                bar_color = 'red'

            labels = [str(x) for x in counts.index]
            values = counts.values

            # Use a single color for bars to indicate overall severity
            ax.bar(labels, values, color=[bar_color] * len(labels))
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title(f'Class distribution (severity={severity:.2f})')

            # Annotate percentages
            for i, v in enumerate(values):
                pct = props.iloc[i] * 100
                ax.text(i, v + total * 0.01, f"{pct:.1f}%", ha='center')

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Visualization skipped or failed: {e}")

    return {
        'is_imbalanced': bool(is_imbalanced),
        'severity': round(float(severity), 3),
        'class_distribution': counts.to_dict(),
        'imbalance_ratio': float(imbalance_ratio) if np.isfinite(imbalance_ratio) else imbalance_ratio,
        'proportions': {str(k): float(v) for k, v in props.items()},
        'recommendations': recommendations
    }
