"""
Demo: Cross-Validation with Metrics
====================================
This demo showcases the cross_validate_with_metrics function in dskit.
"""

from dskit import cross_validate_with_metrics
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression


def demo_classification():
    """Demo 1: Cross-Validation for Classification"""
    print("=" * 60)
    print("DEMO 1: Cross-Validation (Classification)")
    print("=" * 60)
    
    print("\n🔧 Creating classification dataset...")
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    print("   Model: RandomForestClassifier")
    print("   CV Folds: 5")
    print("   Task: Classification (stratified)")
    
    print("\n🔧 Running cross-validation...")
    results = cross_validate_with_metrics(
        model, X, y, cv=5, task="classification", stratified=True
    )
    
    print(f"\n✓ Classification Metrics:")
    print(f"   Accuracy:  {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}")
    print(f"   F1 Score:  {results['f1_mean']:.3f} ± {results['f1_std']:.3f}")
    print(f"   Precision: {results['precision_mean']:.3f} ± {results['precision_std']:.3f}")
    print(f"   Recall:    {results['recall_mean']:.3f} ± {results['recall_std']:.3f}")
    
    if 'roc_auc_mean' in results:
        print(f"\n   ROC-AUC:   {results['roc_auc_mean']:.3f} ± {results['roc_auc_std']:.3f}")
    if 'pr_auc_mean' in results:
        print(f"   PR-AUC:    {results['pr_auc_mean']:.3f} ± {results['pr_auc_std']:.3f}")


def demo_regression():
    """Demo 2: Cross-Validation for Regression"""
    print("\n" + "=" * 60)
    print("DEMO 2: Cross-Validation (Regression)")
    print("=" * 60)
    
    print("\n🔧 Creating regression dataset...")
    X, y = make_regression(
        n_samples=200, n_features=10, n_informative=5, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    
    print("   Model: RandomForestRegressor")
    print("   CV Folds: 5")
    print("   Task: Regression")
    
    print("\n🔧 Running cross-validation...")
    results = cross_validate_with_metrics(
        model, X, y, cv=5, task="regression"
    )
    
    print(f"\n✓ Regression Metrics:")
    print(f"   R² Score: {results['r2_mean']:.3f} ± {results['r2_std']:.3f}")
    print(f"   MAE:      {results['mae_mean']:.3f}")
    print(f"   RMSE:     {results['rmse_mean']:.3f}")


def demo_multiclass():
    """Demo 3: Multiclass Classification"""
    print("\n" + "=" * 60)
    print("DEMO 3: Cross-Validation (Multiclass)")
    print("=" * 60)
    
    print("\n🔧 Creating multiclass dataset...")
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_classes=3, n_clusters_per_class=1, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    print("   Model: RandomForestClassifier")
    print("   Classes: 3")
    print("   CV Folds: 5")
    
    print("\n🔧 Running cross-validation...")
    results = cross_validate_with_metrics(
        model, X, y, cv=5, task="classification"
    )
    
    print(f"\n✓ Multiclass Metrics:")
    print(f"   Accuracy:  {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}")
    print(f"   F1 Score:  {results['f1_mean']:.3f} ± {results['f1_std']:.3f}")
    
    if 'roc_auc_mean' in results:
        print(f"   ROC-AUC (OVR): {results['roc_auc_mean']:.3f} ± {results['roc_auc_std']:.3f}")


if __name__ == "__main__":
    print("\n" + "📊" * 30)
    print("CROSS-VALIDATION DEMO".center(60))
    print("📊" * 30 + "\n")
    
    demo_classification()
    demo_regression()
    demo_multiclass()
    
    print("\n" + "✅" * 30)
    print("ALL DEMOS COMPLETED".center(60))
    print("✅" * 30 + "\n")
