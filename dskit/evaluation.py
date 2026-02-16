"""
Comprehensive ML Model Evaluation Suite

This module provides two primary functions:
- `comprehensive_classification_report` for classification evaluation
- `regression_evaluation_suite` for regression diagnostics

The implementations produce publication-quality matplotlib/seaborn
figures and return structured metric dictionaries.
"""

from typing import Dict, List, Optional, Any, Union

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_fscore_support
import os

from scipy import stats

# Expose a concise alias for array-like inputs
ArrayLike = Union[np.ndarray, pd.Series, list]


def comprehensive_classification_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_proba: Optional[Union[np.ndarray, pd.Series]] = None,
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    display: bool = True,
) -> Dict[str, Any]:
    """
    Generate a comprehensive classification evaluation report with plots.

    Parameters
    ----------
    y_true : array-like
        True labels (1D).
    y_pred : array-like
        Predicted labels (1D).
    y_proba : array-like, optional
        Predicted probabilities (NxC) or (N,) for binary. If provided,
        ROC and precision-recall curves will be computed.
    labels : list of str, optional
        Class label names in order of encoded labels.
    save_path : str, optional
        Directory or base path to save generated figures. If None, figures
        are not saved.
    display : bool, default True
        If True, show figures; otherwise close figures (keeps them in return dict).

    Returns
    -------
    dict
        A dictionary containing:
        - 'metrics': pd.DataFrame of per-class precision/recall/f1/support
        - 'confusion_matrix': np.ndarray
        - 'figures': dict of matplotlib.Figure objects
        - 'roc_auc': float or dict (for multiclass)
        - 'pr_auc': float or dict (for multiclass)

    Examples
    --------
    >>> from dskit.evaluation import comprehensive_classification_report
    >>> results = comprehensive_classification_report(y_true, y_pred, y_proba)
    >>> print(results['metrics'])
    """
    # --- Input validation and conversions ---
    if y_true is None or len(y_true) == 0:
        raise ValueError("y_true cannot be empty")

    if y_pred is None or len(y_pred) == 0:
        raise ValueError("y_pred cannot be empty")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Handle NaNs
    if np.any(pd.isna(y_true)) or np.any(pd.isna(y_pred)):
        warnings.warn("NaN values detected in y_true/y_pred, removing corresponding rows")
        mask = ~(pd.isna(y_true) | pd.isna(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if y_proba is not None:
            y_proba = np.asarray(y_proba)[mask]

    # Determine classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    # Build classification metrics table
    clf_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics_df = pd.DataFrame(clf_report).transpose()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig_cm = _plot_confusion_matrix(cm, labels=(labels if labels is not None else [str(c) for c in classes]))

    figures: Dict[str, plt.Figure] = {"confusion_matrix": fig_cm}

    # ROC and Precision-Recall computations
    roc_auc_result: Union[float, Dict[str, float], None] = None
    pr_auc_result: Union[float, Dict[str, float], None] = None

    if y_proba is not None:
        y_proba = np.asarray(y_proba)

        # Binary classification
        if n_classes == 2:
            # Support either (n_samples,) or (n_samples,2)
            if y_proba.ndim == 1:
                pos_proba = y_proba
            elif y_proba.ndim == 2 and y_proba.shape[1] == 2:
                pos_proba = y_proba[:, 1]
            else:
                # Try to select column corresponding to positive class
                pos_proba = y_proba[:, 1] if y_proba.ndim == 2 and y_proba.shape[1] >= 2 else None

            if pos_proba is None:
                warnings.warn("Unable to interpret y_proba shape for binary task; skipping ROC/PR")
            else:
                fig_roc, roc_auc = _plot_roc_curve(y_true, pos_proba, labels=None)
                fig_pr, pr_auc = _plot_precision_recall_curve(y_true, pos_proba, labels=None)
                figures.update({"roc_curve": fig_roc, "pr_curve": fig_pr})
                roc_auc_result = roc_auc
                pr_auc_result = pr_auc

        else:
            # Multiclass: expect y_proba shape (n_samples, n_classes)
            if y_proba.ndim != 2 or y_proba.shape[1] != n_classes:
                warnings.warn("y_proba shape does not match number of classes; skipping ROC/PR for multiclass")
            else:
                fig_roc, roc_auc = _plot_roc_curve(y_true, y_proba, labels=(labels if labels is not None else [str(c) for c in classes]))
                fig_pr, pr_auc = _plot_precision_recall_curve(y_true, y_proba, labels=(labels if labels is not None else [str(c) for c in classes]))
                figures.update({"roc_curve": fig_roc, "pr_curve": fig_pr})
                roc_auc_result = roc_auc
                pr_auc_result = pr_auc

    # Save figures if requested
    if save_path is not None:
        try:
            if os.path.isdir(save_path):
                base = save_path
            else:
                # treat as base filename (without extension)
                base = os.path.dirname(save_path) or "."
            for name, fig in figures.items():
                fname = os.path.join(base, f"classification_{name}.png")
                fig.savefig(fname, bbox_inches="tight")
        except Exception as exc:
            warnings.warn(f"Failed to save figures: {exc}")

    # Handle display / close
    if display:
        plt.show()
    else:
        for fig in figures.values():
            plt.close(fig)

    return {
        "metrics": metrics_df,
        "confusion_matrix": cm,
        "figures": figures,
        "roc_auc": roc_auc_result,
        "pr_auc": pr_auc_result,
    }


def regression_evaluation_suite(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    display: bool = True,
    top_n_features: int = 3,
) -> Dict[str, Any]:
    """
    Generate regression diagnostics and metrics with visualization.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    X_test : array-like or DataFrame, optional
        Test features used to produce `y_pred`. If provided, residuals vs
        features plots will be created for the top N features.
    feature_names : list of str, optional
        Names of features corresponding to columns in `X_test`.
    save_path : str, optional
        Directory or base path to save generated figures.
    display : bool, default True
        If True, show figures; otherwise close figures (figures are still
        returned in the output dictionary).
    top_n_features : int, default 3
        Number of top features (by absolute correlation with residuals)
        to plot residuals against when `X_test` is provided.

    Returns
    -------
    dict
        A dictionary containing:
        - 'metrics': pd.DataFrame of MAE, MSE, RMSE, R2, MAPE, MedAE
        - 'residuals': np.ndarray
        - 'figures': dict of matplotlib.Figure objects
        - 'normality_test': dict with Shapiro-Wilk results
        - 'outliers': list of indices flagged as outliers (>3 std dev)

    Examples
    --------
    >>> from dskit.evaluation import regression_evaluation_suite
    >>> results = regression_evaluation_suite(y_test, y_pred, X_test)
    >>> print(results['metrics'])
    """
    # --- Input validation and conversions ---
    if y_true is None or len(y_true) == 0:
        raise ValueError("y_true cannot be empty")

    if y_pred is None or len(y_pred) == 0:
        raise ValueError("y_pred cannot be empty")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Handle NaNs
    if np.any(pd.isna(y_true)) or np.any(pd.isna(y_pred)):
        warnings.warn("NaN values detected in y_true/y_pred, removing corresponding rows")
        mask = ~(pd.isna(y_true) | pd.isna(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if X_test is not None:
            X_test = np.asarray(X_test)[mask]

    # Compute residuals
    residuals = y_true - y_pred

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    mape = _calculate_mape(y_true, y_pred)

    metrics = pd.DataFrame(
        {
            "MAE": [mae],
            "MSE": [mse],
            "RMSE": [rmse],
            "R2": [r2],
            "MAPE": [mape],
            "MedAE": [medae],
        },
        index=[0],
    )

    # Generate plots
    figures = _plot_residuals(y_true, y_pred, X_test=X_test, feature_names=feature_names, top_n=top_n_features)

    # Normality test (Shapiro-Wilk) — use sample if too large
    shapiro_sample = residuals
    if residuals.size > 5000:
        rng = np.random.default_rng(42)
        shapiro_sample = rng.choice(residuals, size=5000, replace=False)

    try:
        shapiro_stat, shapiro_p = stats.shapiro(shapiro_sample)
        normality_test = {"statistic": float(shapiro_stat), "p_value": float(shapiro_p)}
    except Exception as exc:
        normality_test = {"error": str(exc)}

    # Outlier detection (>3 std dev from mean residual)
    resid_mean = float(np.mean(residuals))
    resid_std = float(np.std(residuals))
    if resid_std == 0:
        outlier_idx = []
    else:
        outlier_mask = np.abs(residuals - resid_mean) > 3 * resid_std
        outlier_idx = list(np.where(outlier_mask)[0])

    # Save figures if requested
    if save_path is not None:
        try:
            base = save_path if os.path.isdir(save_path) else (os.path.dirname(save_path) or ".")
            for name, fig in figures.items():
                fname = os.path.join(base, f"regression_{name}.png")
                fig.savefig(fname, bbox_inches="tight")
        except Exception as exc:
            warnings.warn(f"Failed to save regression figures: {exc}")

    # Display/close behavior
    if display:
        plt.show()
    else:
        for fig in figures.values():
            plt.close(fig)

    return {
        "metrics": metrics,
        "residuals": residuals,
        "figures": figures,
        "normality_test": normality_test,
        "outliers": outlier_idx,
    }


# Helper function stubs (implementations to follow)
def _plot_confusion_matrix(cm: np.ndarray, labels: Optional[List[str]] = None) -> plt.Figure:
    """Plot confusion matrix heatmap and return the Figure object."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    total = np.sum(cm)
    # Prepare annotations with counts and percentages
    annot = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = 100.0 * count / total if total > 0 else 0.0
            annot[i, j] = f"{count}\n{pct:.1f}%"

    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", ax=ax, cbar=True,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def _plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, labels: Optional[List[str]] = None) -> plt.Figure:
    """Plot ROC curve (binary or multiclass OvR) and return the Figure object."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    # If y_proba is 1-D assume binary positive probability
    if y_proba.ndim == 1 or (y_proba.ndim == 2 and y_proba.shape[1] == 1):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc_val:.3f})", color="C0")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        plt.tight_layout()
        return fig, float(roc_auc_val)

    # Multiclass OvR
    classes = np.unique(y_true)
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=classes)
    roc_auc_dict: Dict[str, float] = {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc_dict[str(labels[i]) if labels is not None else str(classes[i])] = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{labels[i] if labels is not None else classes[i]} (AUC={roc_auc_dict[str(labels[i]) if labels is not None else str(classes[i])]:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multiclass ROC Curve (OvR)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig, roc_auc_dict


def _plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray, labels: Optional[List[str]] = None) -> plt.Figure:
    """Plot precision-recall curve(s) and return the Figure object."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Binary
    if y_proba.ndim == 1 or (y_proba.ndim == 2 and y_proba.shape[1] == 1):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc_val = auc(recall, precision)
        ax.plot(recall, precision, label=f"PR (AUC={pr_auc_val:.3f})", color="C0")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="lower left")
        plt.tight_layout()
        return fig, float(pr_auc_val)

    # Multiclass
    classes = np.unique(y_true)
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=classes)
    pr_auc_dict: Dict[str, float] = {}
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        pr_auc_dict[str(labels[i]) if labels is not None else str(classes[i])] = auc(recall, precision)
        ax.plot(recall, precision, label=f"{labels[i] if labels is not None else classes[i]} (AUC={pr_auc_dict[str(labels[i]) if labels is not None else str(classes[i])]:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Multiclass)")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    return fig, pr_auc_dict


def _plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None, feature_names: Optional[List[str]] = None, top_n: int = 3) -> Dict[str, plt.Figure]:
    """Generate residual diagnostic plots and return a dict of Figures."""
    sns.set_style("whitegrid")
    figs: Dict[str, plt.Figure] = {}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    # Actual vs Predicted scatter with R^2 line and residuals coloring
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sc = ax1.scatter(y_true, y_pred, c=np.abs(residuals), cmap="viridis", alpha=0.7)
    lims = [np.min(np.concatenate([y_true, y_pred])), np.max(np.concatenate([y_true, y_pred]))]
    ax1.plot(lims, lims, linestyle="--", color="gray")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Actual vs Predicted")
    cbar = fig1.colorbar(sc, ax=ax1)
    cbar.set_label("Absolute Residual")
    # R^2 annotation
    try:
        r2 = r2_score(y_true, y_pred)
        ax1.text(0.02, 0.98, f"$R^2$ = {r2:.3f}", transform=ax1.transAxes, verticalalignment="top")
    except Exception:
        pass
    plt.tight_layout()
    figs["actual_vs_predicted"] = fig1

    # Residuals vs Predicted
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(y_pred, residuals, alpha=0.7)
    ax2.axhline(0, color="red", linestyle="--")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residuals vs Predicted")
    plt.tight_layout()
    figs["residuals_vs_predicted"] = fig2

    # Residuals histogram with normal curve overlay
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals, kde=False, stat="density", bins=30, ax=ax3, color="C0")
    # Overlay normal curve
    mu, sigma = np.mean(residuals), np.std(residuals)
    xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    if sigma > 0:
        ax3.plot(xs, stats.norm.pdf(xs, mu, sigma), color="red", linestyle="--")
    ax3.set_title("Residuals Distribution")
    ax3.set_xlabel("Residual")
    plt.tight_layout()
    figs["residuals_hist"] = fig3

    # Q-Q plot
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title("Q-Q Plot of Residuals")
    plt.tight_layout()
    figs["qq_plot"] = fig4

    # Residuals vs Features (top N by absolute correlation)
    if X_test is not None:
        X_arr = X_test
        if isinstance(X_test, pd.DataFrame):
            X_arr = X_test.values
            feature_names = feature_names or list(X_test.columns)
        else:
            X_arr = np.asarray(X_test)
            if feature_names is None and X_arr.ndim == 2:
                feature_names = [f"f{i}" for i in range(X_arr.shape[1])]

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        # Compute absolute Pearson correlation between each feature and residuals
        corrs = []
        for i in range(X_arr.shape[1]):
            col = X_arr[:, i]
            try:
                # handle constant columns
                if np.std(col) == 0:
                    corr = 0.0
                else:
                    corr = abs(np.corrcoef(col, residuals)[0, 1])
            except Exception:
                corr = 0.0
            corrs.append(corr)

        idx_sorted = np.argsort(corrs)[::-1]
        top_idx = idx_sorted[:top_n]
        for idx in top_idx:
            name = feature_names[idx] if feature_names is not None else f"f{idx}"
            figf, axf = plt.subplots(figsize=(8, 6))
            axf.scatter(X_arr[:, idx], residuals, alpha=0.7)
            axf.axhline(0, color="red", linestyle="--")
            axf.set_xlabel(name)
            axf.set_ylabel("Residuals")
            axf.set_title(f"Residuals vs Feature: {name}")
            plt.tight_layout()
            figs[f"residuals_vs_{name}"] = figf

    return figs


def _calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAPE while handling zeros in `y_true` safely."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Avoid division by zero: ignore terms where y_true == 0
    eps = np.finfo(float).eps
    denom = np.where(np.abs(y_true) < eps, np.nan, y_true)
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs((y_true - y_pred) / denom)
    # Filter out nan contributions (where y_true was zero)
    ape_nonan = ape[~np.isnan(ape)]
    if ape_nonan.size == 0:
        warnings.warn("MAPE is undefined because all true values are zero; returning NaN")
        return float("nan")
    return float(np.mean(ape_nonan) * 100.0)
