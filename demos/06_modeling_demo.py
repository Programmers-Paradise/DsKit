"""
Demo: Machine Learning Modeling
===============================
This demo showcases modeling functions in dskit.
"""

from dskit import (
    QuickModel, compare_models, auto_hpo,
    evaluate_model, error_analysis,
    auto_encode, auto_scale, train_test_auto
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import pandas as pd
import numpy as np


def create_classification_data():
    """Create sample classification dataset"""
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        'feature_1': np.random.normal(50, 10, n),
        'feature_2': np.random.normal(100, 20, n),
        'feature_3': np.random.exponential(5, n),
        'feature_4': np.random.uniform(0, 100, n),
        'category': np.random.choice(['A', 'B'], n)
    })

    # Create target based on features
    df['target'] = (
        ((df['feature_1'] > 50) &
         (df['feature_2'] > 100)) |
        (df['feature_3'] > 5)
    ).astype(int)

    return df


def demo_quick_model():
    """Demo 1: Quick model training"""
    print("=" * 60)
    print("DEMO 1: Quick Model Training")
    print("=" * 60)

    df = create_classification_data()
    df_encoded = auto_encode(df)
    df_scaled = auto_scale(df_encoded)
    X_train, X_test, y_train, y_test = train_test_auto(df_scaled, target='target')

    # FIX 1: Cast target to integer
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    print("\n🤖 Training Quick Random Forest model...")
    qm = QuickModel(model_type='rf', task='classification')
    qm.fit(X_train, y_train)

    print("\n✓ Model trained successfully")
    print(f"  Model type: Random Forest")
    print(f"  Training samples: {len(X_train)}")

    print("\n📊 Making predictions...")
    predictions = qm.predict(X_test)
    print(f"✓ Predicted {len(predictions)} samples")

    print("\n📈 Quick evaluation:")
    score = qm.score(X_test, y_test)
    print(f"  Accuracy: {score:.4f}")


def demo_compare_models():
    """Demo 2: Compare multiple models"""
    print("\n" + "=" * 60)
    print("DEMO 2: Model Comparison")
    print("=" * 60)

    df = create_classification_data()
    df_encoded = auto_encode(df)
    df_scaled = auto_scale(df_encoded)
    X_train, X_test, y_train, y_test = train_test_auto(df_scaled, target='target')

    # FIX 1: Cast target to integer
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    print("\n🤖 Comparing multiple models...")
    print("   Models: Random Forest, Logistic Regression, SVM")

    # FIX 2: Use short-codes ('rf', 'lr', 'svm')
    models = {
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'lr': LogisticRegression(random_state=42, max_iter=1000),
        'svm': SVC(random_state=42)
    }

    results = compare_models(
        X_train, y_train, X_test, y_test,
        models=models,
        task='classification'
    )

    print("\n✓ Model comparison completed:")
    print(results)


def demo_hyperparameter_tuning():
    """Demo 3: Automatic hyperparameter optimization"""
    print("\n" + "=" * 60)
    print("DEMO 3: Hyperparameter Optimization")
    print("=" * 60)

    df = create_classification_data()
    df_encoded = auto_encode(df)
    df_scaled = auto_scale(df_encoded)
    X_train, X_test, y_train, y_test = train_test_auto(df_scaled, target='target')

    # FIX 1: Cast target to integer
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    print("\n🔧 Tuning Random Forest hyperparameters...")
    print("   Method: Random Search")

    # FIX 3: Robustness - Wrap in try-except for missing 'optuna'
    try:
        best_model, best_params, best_score = auto_hpo(
            X_train, y_train, X_test, y_test,
            model_type='rf',
            task='classification'
        )
        print("\n✓ Hyperparameter tuning completed:")
        print(f"  Best score: {best_score:.4f}")
        print(f"  Best parameters: {best_params}")

    except Exception as e:
        print(f"\n⚠️ SKIPPING: {str(e)}")
        print("   (Install 'optuna' to run this feature: pip install optuna)")


def demo_model_evaluation():
    """Demo 4: Detailed model evaluation"""
    print("\n" + "=" * 60)
    print("DEMO 4: Model Evaluation")
    print("=" * 60)

    df = create_classification_data()
    df_encoded = auto_encode(df)
    df_scaled = auto_scale(df_encoded)
    X_train, X_test, y_train, y_test = train_test_auto(df_scaled, target='target')

    # FIX 1: Cast target to integer
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    print("\n🤖 Training model for evaluation...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("\n📊 Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test, task='classification')

    # FIX 4: Handle case where metrics is None
    if metrics:
        print("\n✓ Evaluation metrics:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
    else:
        print("\n✓ Evaluation completed (metrics printed above)")


def demo_error_analysis():
    """Demo 5: Error analysis"""
    print("\n" + "=" * 60)
    print("DEMO 5: Error Analysis")
    print("=" * 60)

    df = create_classification_data()
    df_encoded = auto_encode(df)
    df_scaled = auto_scale(df_encoded)
    X_train, X_test, y_train, y_test = train_test_auto(df_scaled, target='target')

    # FIX 1: Cast target to integer
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    print("\n🤖 Training model for error analysis...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("\n🔍 Analyzing prediction errors...")
    analysis = error_analysis(model, X_test, y_test, task='classification')

    print("\n✓ Error analysis completed")
    print(f"  Total predictions: {len(y_test)}")

    # ✅ FIX 5: analysis might be a DataFrame (cannot do `if analysis`)
    if analysis is None:
        pass

    elif isinstance(analysis, dict):
        if "error_rate" in analysis:
            print(f"  Error rate: {analysis['error_rate']:.4f}")

    elif isinstance(analysis, pd.DataFrame):
        if not analysis.empty:
            print(f"  Errors found: {len(analysis)}")
        else:
            print("  No errors found 🎉")

    else:
        # fallback for unexpected return types
        print(f"  Returned analysis type: {type(analysis)}")


if __name__ == "__main__":
    print("\n" + "🤖" * 30)
    print("MACHINE LEARNING MODELING DEMO".center(60))
    print("🤖" * 30 + "\n")

    demo_quick_model()
    demo_compare_models()
    demo_hyperparameter_tuning()
    demo_model_evaluation()
    demo_error_analysis()

    print("\n" + "✅" * 30)
    print("ALL DEMOS COMPLETED".center(60))
    print("✅" * 30 + "\n")
