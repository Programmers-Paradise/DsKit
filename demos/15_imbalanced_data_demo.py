"""
Demo: Imbalanced Data Utilities
===============================
This demo showcases imbalanced data handling functions in dskit.
"""

from dskit import get_class_weights, apply_smote, threshold_tuning
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def demo_class_weights():
    """Demo 1: Class Weights"""
    print("=" * 60)
    print("DEMO 1: Class Weights for Imbalanced Data")
    print("=" * 60)
    
    # Create imbalanced data
    y_balanced = np.array([0, 0, 1, 1])
    y_imbalanced = np.array([0]*80 + [1]*20)
    
    print("\n📊 Balanced dataset [0, 0, 1, 1]:")
    weights_balanced = get_class_weights(y_balanced)
    print(f"   Class weights: {weights_balanced}")
    
    print("\n📊 Imbalanced dataset (80% class 0, 20% class 1):")
    weights_imbalanced = get_class_weights(y_imbalanced)
    print(f"   Class weights: {weights_imbalanced}")
    print(f"   → Minority class (1) has {weights_imbalanced[1]/weights_imbalanced[0]:.1f}x higher weight")
    
    print("\n💡 Usage: Pass to model's class_weight parameter")
    print("   model = RandomForestClassifier(class_weight=weights)")


def demo_smote():
    """Demo 2: SMOTE Resampling"""
    print("\n" + "=" * 60)
    print("DEMO 2: SMOTE Resampling")
    print("=" * 60)
    
    # Create imbalanced dataset
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.array([0]*85 + [1]*15)
    
    print("\n📊 Original dataset:")
    print(f"   Shape: {X.shape}")
    print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    print("\n🔧 Applying SMOTE...")
    X_resampled, y_resampled = apply_smote(X, y, random_state=42)
    
    print(f"\n✓ Resampled dataset:")
    print(f"   Shape: {X_resampled.shape}")
    print(f"   Class distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")


def demo_threshold_tuning():
    """Demo 3: Threshold Tuning"""
    print("\n" + "=" * 60)
    print("DEMO 3: Classification Threshold Tuning")
    print("=" * 60)
    
    # Simulate predictions
    np.random.seed(42)
    y_true = np.array([0]*40 + [1]*60)
    # Simulate well-calibrated probabilities
    y_proba = np.where(y_true == 1, 
                       np.random.uniform(0.4, 0.95, len(y_true)),
                       np.random.uniform(0.05, 0.6, len(y_true)))
    
    print("\n📊 Default threshold (0.5):")
    y_pred_default = (y_proba >= 0.5).astype(int)
    print(f"   F1: {f1_score(y_true, y_pred_default):.3f}")
    print(f"   Precision: {precision_score(y_true, y_pred_default):.3f}")
    print(f"   Recall: {recall_score(y_true, y_pred_default):.3f}")
    
    print("\n🔧 Finding optimal threshold for F1...")
    result = threshold_tuning(y_true, y_proba, metric="f1")
    
    print(f"\n✓ Optimal Threshold: {result['best_threshold']}")
    print(f"   Best F1 Score: {result['best_score']:.3f}")
    
    print("\n📊 All thresholds tested:")
    for r in result['all_results'][:5]:
        print(f"   Threshold {r['threshold']}: F1 = {r['score']:.3f}")
    print("   ...")
    
    # Compare different metrics
    print("\n🔧 Comparing optimal thresholds for different metrics:")
    for metric in ['precision', 'recall', 'accuracy']:
        result = threshold_tuning(y_true, y_proba, metric=metric)
        print(f"   {metric.capitalize()}: threshold={result['best_threshold']}, score={result['best_score']:.3f}")


if __name__ == "__main__":
    print("\n" + "⚖️ " * 20)
    print("IMBALANCED DATA DEMO".center(60))
    print("⚖️ " * 20 + "\n")
    
    demo_class_weights()
    demo_smote()
    demo_threshold_tuning()
    
    print("\n" + "✅" * 30)
    print("ALL DEMOS COMPLETED".center(60))
    print("✅" * 30 + "\n")
