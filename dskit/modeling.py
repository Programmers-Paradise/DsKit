import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class QuickModel:
    """
    A wrapper class for training ML models with simple commands.
    """
    def __init__(self, model_type=None, model_name=None, task='classification'):
        # Support both model_type and model_name for backwards compatibility
        self.model_name = model_type or model_name
        self.task = task
        self.model = self._get_model()
        
    def _get_model(self):
        if self.task == 'classification':
            models = {
                'rf': RandomForestClassifier(),
                'random_forest': RandomForestClassifier(),
                'gb': GradientBoostingClassifier(),
                'gradient_boosting': GradientBoostingClassifier(),
                'lr': LogisticRegression(max_iter=1000),
                'logistic_regression': LogisticRegression(max_iter=1000),
                'svm': SVC(probability=True),
                'svc': SVC(probability=True),
                'knn': KNeighborsClassifier(),
                'dt': DecisionTreeClassifier(),
                'decision_tree': DecisionTreeClassifier()
            }
        else: # regression
            models = {
                'rf': RandomForestRegressor(),
                'random_forest': RandomForestRegressor(),
                'gb': GradientBoostingRegressor(),
                'gradient_boosting': GradientBoostingRegressor(),
                'lr': LinearRegression(),
                'linear_regression': LinearRegression(),
                'ridge': Ridge(),
                'lasso': Lasso(),
                'svm': SVR(),
                'svr': SVR(),
                'knn': KNeighborsRegressor(),
                'dt': DecisionTreeRegressor(),
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
    
    def score(self, X, y):
        """Calculate accuracy score for classification or R2 for regression"""
        return self.model.score(X, y)

def compare_models(X_train, y_train, X_test, y_test, models=None, task='classification'):
    """
    Trains multiple ML models and compares their performance.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    models : list, optional
        List of model abbreviations to compare. 
        For classification: ['lr', 'rf', 'gb', 'svc', 'knn', 'dt']
        For regression: ['lr', 'ridge', 'lasso', 'rf', 'gb', 'svr']
    task : str
        'classification' or 'regression'
    """
    if models is None:
        if task == 'classification':
            models = ['rf', 'gb', 'lr', 'dt']
        else:
            models = ['rf', 'gb', 'lr', 'dt']
    
    if task == 'classification':
        model_map = {
            'lr': LogisticRegression(max_iter=1000),
            'rf': RandomForestClassifier(),
            'gb': GradientBoostingClassifier(),
            'svc': SVC(probability=True),
            'svm': SVC(probability=True),
            'knn': KNeighborsClassifier(),
            'dt': DecisionTreeClassifier()
        }
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    else:
        from sklearn.linear_model import Ridge, Lasso
        model_map = {
            'lr': LinearRegression(),
            'ridge': Ridge(),
            'lasso': Lasso(),
            'rf': RandomForestRegressor(),
            'gb': GradientBoostingRegressor(),
            'svr': SVR(),
            'svm': SVR(),
            'dt': DecisionTreeRegressor()
        }
        metrics = ['RMSE', 'MAE', 'R2']
        
    results = []
    
    for model_abbr in models:
        if model_abbr not in model_map:
            print(f"Warning: Model '{model_abbr}' not recognized. Skipping.")
            continue
            
        model = model_map[model_abbr]
        model_name = model.__class__.__name__
        
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if task == 'classification':
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                results.append([model_name, acc, prec, rec, f1])
            else:
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                results.append([model_name, rmse, mae, r2])
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue
            
    df_results = pd.DataFrame(results, columns=['Model'] + metrics)
    
    if task == 'classification':
        df_results = df_results.sort_values(by='Accuracy', ascending=False)
    else:
        df_results = df_results.sort_values(by='R2', ascending=False)
        
    return df_results

def auto_hpo(X_train, y_train, X_test=None, y_test=None, model_type='rf', task='classification', 
             n_trials=20, scoring='accuracy', cv=3):
    """
    Performs automatic hyperparameter optimization using Optuna.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_test : array-like, optional
        Test features (for final evaluation)
    y_test : array-like, optional
        Test labels (for final evaluation)
    model_type : str
        Model abbreviation ('rf', 'gb', 'lr', etc.)
    task : str
        'classification' or 'regression'
    n_trials : int
        Number of optimization trials
    scoring : str
        Scoring metric ('accuracy', 'f1', 'r2', etc.)
    cv : int
        Number of cross-validation folds
        
    Returns:
    --------
    best_model : trained model
        The best model with optimized parameters
    best_params : dict
        The best parameters found
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("Optuna not installed. Using basic hyperparameter tuning...")
        # Fallback to simple default model
        if task == 'classification':
            if model_type in ['rf', 'random_forest']:
                best_model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type in ['gb', 'gradient_boosting']:
                best_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            else:
                best_model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            if model_type in ['rf', 'random_forest']:
                best_model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type in ['gb', 'gradient_boosting']:
                best_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            else:
                best_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        best_model.fit(X_train, y_train)
        return best_model, {}
    
    def objective(trial):
        if task == 'classification':
            if model_type in ['rf', 'random_forest']:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                    'random_state': 42
                }
                model = RandomForestClassifier(**params)
            elif model_type in ['gb', 'gradient_boosting']:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'random_state': 42
                }
                model = GradientBoostingClassifier(**params)
            else:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'random_state': 42
                }
                model = RandomForestClassifier(**params)
        else:  # regression
            if model_type in ['rf', 'random_forest']:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                    'random_state': 42
                }
                model = RandomForestRegressor(**params)
            elif model_type in ['gb', 'gradient_boosting']:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'random_state': 42
                }
                model = GradientBoostingRegressor(**params)
            else:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'random_state': 42
                }
                model = RandomForestRegressor(**params)
        
        # Cross-validation
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    # Train best model
    best_params = study.best_params
    best_params['random_state'] = 42
    
    if task == 'classification':
        if model_type in ['rf', 'random_forest']:
            best_model = RandomForestClassifier(**best_params)
        elif model_type in ['gb', 'gradient_boosting']:
            best_model = GradientBoostingClassifier(**best_params)
        else:
            best_model = RandomForestClassifier(**best_params)
    else:
        if model_type in ['rf', 'random_forest']:
            best_model = RandomForestRegressor(**best_params)
        elif model_type in ['gb', 'gradient_boosting']:
            best_model = GradientBoostingRegressor(**best_params)
        else:
            best_model = RandomForestRegressor(**best_params)
    
    best_model.fit(X_train, y_train)
    return best_model, best_params

def auto_hpo_old(model, param_grid, X, y, method='grid', cv=3):
    """
    Performs automatic hyperparameter tuning (legacy function).
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
