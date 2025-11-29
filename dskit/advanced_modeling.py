import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import VotingClassifier, VotingRegressor, BaggingClassifier, BaggingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
try:
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    xgb = None
    lgb = None
    CatBoostClassifier = None
    CatBoostRegressor = None
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    SMOTE = None
    RandomOverSampler = None
    RandomUnderSampler = None
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedModel:
    """
    Advanced model wrapper with more algorithms and techniques.
    """
    def __init__(self, model_name, task='classification', **kwargs):
        self.model_name = model_name
        self.task = task
        self.model = self._get_model(**kwargs)
        self.is_fitted = False
        
    def _get_model(self, **kwargs):
        if self.task == 'classification':
            models = {
                'xgboost': xgb.XGBClassifier(**kwargs) if xgb else None,
                'lightgbm': lgb.LGBMClassifier(**kwargs) if lgb else None,
                'catboost': CatBoostClassifier(verbose=False, **kwargs) if CatBoostClassifier else None,
                'adaboost': AdaBoostClassifier(**kwargs)
            }
        else:
            models = {
                'xgboost': xgb.XGBRegressor(**kwargs) if xgb else None,
                'lightgbm': lgb.LGBMRegressor(**kwargs) if lgb else None,
                'catboost': CatBoostRegressor(verbose=False, **kwargs) if CatBoostRegressor else None,
                'adaboost': AdaBoostRegressor(**kwargs)
            }
        
        if self.model_name not in models:
            raise ValueError(f"Model '{self.model_name}' not supported.")
        
        model = models[self.model_name]
        if model is None:
            raise ImportError(f"Library for '{self.model_name}' not installed.")
        
        return model
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("This model does not support predict_proba")

def create_ensemble(models, task='classification', voting='soft'):
    """
    Create ensemble of multiple models.
    """
    estimators = [(f'model_{i}', model) for i, model in enumerate(models)]
    
    if task == 'classification':
        ensemble = VotingClassifier(estimators=estimators, voting=voting)
    else:
        ensemble = VotingRegressor(estimators=estimators)
    
    return ensemble

def cross_validate_model(model, X, y, cv=5, scoring=None):
    """
    Perform cross-validation on a model.
    """
    if scoring is None:
        scoring = 'accuracy' if hasattr(model, 'predict_proba') else 'r2'
    
    if hasattr(model, 'predict_proba'):  # Classification
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    else:  # Regression
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring)
    
    return {
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'scores': scores
    }

def plot_confusion_matrix(y_true, y_pred, classes=None):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_roc_curve(y_true, y_proba, classes=None):
    """
    Plot ROC curve for binary or multiclass classification.
    """
    if len(np.unique(y_true)) == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()
    else:
        # Multiclass ROC
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        from itertools import cycle
        
        y_bin = label_binarize(y_true, classes=np.unique(y_true))
        n_classes = y_bin.shape[1]
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.figure(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
        for i, color in zip(range(n_classes), colors):
            class_name = classes[i] if classes else f'Class {i}'
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Multiclass')
        plt.legend(loc="lower right")
        plt.show()

def plot_precision_recall_curve(y_true, y_proba):
    """
    Plot Precision-Recall curve.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

def detailed_classification_report(y_true, y_pred, y_proba=None, classes=None):
    """
    Generate detailed classification report with visualizations.
    """
    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=classes))
    
    # Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, classes)
    
    # ROC Curve if probabilities are available
    if y_proba is not None:
        plot_roc_curve(y_true, y_proba, classes)
        if len(np.unique(y_true)) == 2:
            plot_precision_recall_curve(y_true, y_proba)

def handle_imbalanced_data(X, y, method='smote', random_state=42):
    """
    Handle imbalanced datasets using various techniques.
    """
    if method == 'smote' and SMOTE is not None:
        sampler = SMOTE(random_state=random_state)
    elif method == 'oversample' and RandomOverSampler is not None:
        sampler = RandomOverSampler(random_state=random_state)
    elif method == 'undersample' and RandomUnderSampler is not None:
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        print(f"Method '{method}' not available or imbalanced-learn not installed.")
        return X, y
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    print(f"Original distribution: {pd.Series(y).value_counts().to_dict()}")
    print(f"Resampled distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
    
    return X_resampled, y_resampled

def learning_curve_analysis(model, X, y, cv=5):
    """
    Plot learning curves to analyze model performance vs training size.
    """
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def validation_curve_analysis(model, X, y, param_name, param_range, cv=5):
    """
    Plot validation curves to analyze model performance vs hyperparameter values.
    """
    from sklearn.model_selection import validation_curve
    
    train_scores, val_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring='accuracy', n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(param_range, val_mean, 'o-', color='red', label='Validation score')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.title(f'Validation Curve for {param_name}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()