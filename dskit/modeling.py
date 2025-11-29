import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class QuickModel:
    """
    A wrapper class for training ML models with simple commands.
    """
    def __init__(self, model_name, task='classification'):
        self.model_name = model_name
        self.task = task
        self.model = self._get_model()
        
    def _get_model(self):
        if self.task == 'classification':
            models = {
                'random_forest': RandomForestClassifier(),
                'gradient_boosting': GradientBoostingClassifier(),
                'logistic_regression': LogisticRegression(),
                'svm': SVC(probability=True),
                'knn': KNeighborsClassifier(),
                'decision_tree': DecisionTreeClassifier()
            }
        else: # regression
            models = {
                'random_forest': RandomForestRegressor(),
                'gradient_boosting': GradientBoostingRegressor(),
                'linear_regression': LinearRegression(),
                'svm': SVR(),
                'knn': KNeighborsRegressor(),
                'decision_tree': DecisionTreeRegressor()
            }
            
        if self.model_name not in models:
            raise ValueError(f"Model '{self.model_name}' not supported. Choose from {list(models.keys())}")
            
        return models[self.model_name]
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("This model does not support predict_proba")

def compare_models(X, y, task='classification'):
    """
    Trains multiple ML models and compares their performance.
    """
    if task == 'classification':
        models = {
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier()
        }
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    else:
        models = {
            'Random Forest': RandomForestRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(),
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor()
        }
        metrics = ['RMSE', 'MAE', 'R2']
        
    results = []
    
    # Split data for comparison
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if task == 'classification':
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            results.append([name, acc, prec, rec, f1])
        else:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results.append([name, rmse, mae, r2])
            
    return pd.DataFrame(results, columns=['Model'] + metrics).sort_values(by=metrics[0], ascending=False)

def auto_hpo(model, param_grid, X, y, method='grid', cv=3):
    """
    Performs automatic hyperparameter tuning.
    """
    if method == 'grid':
        search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1)
    elif method == 'random':
        search = RandomizedSearchCV(model, param_grid, cv=cv, n_jobs=-1)
    else:
        raise ValueError("Method must be 'grid' or 'random'")
        
    search.fit(X, y)
    print(f"Best Params: {search.best_params_}")
    return search.best_estimator_

def evaluate_model(model, X_test, y_test, task='classification'):
    """
    Provides evaluation metrics for a trained model.
    """
    y_pred = model.predict(X_test)
    
    if task == 'classification':
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
        print("Recall:", recall_score(y_test, y_pred, average='weighted', zero_division=0))
        print("F1 Score:", f1_score(y_test, y_pred, average='weighted', zero_division=0))
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
                if len(np.unique(y_test)) == 2:
                    print("ROC-AUC:", roc_auc_score(y_test, y_prob[:, 1]))
                else:
                    print("ROC-AUC:", roc_auc_score(y_test, y_prob, multi_class='ovr'))
        except Exception as e:
            print(f"Could not calculate ROC-AUC: {e}")
            
    else:
        print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
        print("MAE:", mean_absolute_error(y_test, y_pred))
        print("R2 Score:", r2_score(y_test, y_pred))

def error_analysis(model, X_test, y_test, task='classification'):
    """
    Analyzes wrong predictions.
    """
    y_pred = model.predict(X_test)
    
    if isinstance(X_test, pd.DataFrame):
        analysis_df = X_test.copy()
    else:
        analysis_df = pd.DataFrame(X_test)
        
    analysis_df['Actual'] = y_test
    analysis_df['Predicted'] = y_pred
    
    if task == 'classification':
        errors = analysis_df[analysis_df['Actual'] != analysis_df['Predicted']]
        print(f"Total Errors: {len(errors)} out of {len(y_test)}")
        return errors
    else:
        analysis_df['Error'] = analysis_df['Actual'] - analysis_df['Predicted']
        analysis_df['AbsError'] = analysis_df['Error'].abs()
        return analysis_df.sort_values(by='AbsError', ascending=False)
