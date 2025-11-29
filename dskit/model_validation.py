import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings

class ModelValidator:
    """
    Advanced model validation and performance analysis.
    """
    
    def __init__(self):
        self.validation_results = {}
        self.models = {}
    
    def comprehensive_cross_validation(self, model, X, y, cv_strategies=['kfold', 'stratified', 'time_series'], 
                                     scoring_metrics=None, cv_folds=5):
        """Perform comprehensive cross-validation with multiple strategies."""
        from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
        
        if scoring_metrics is None:
            if hasattr(model, 'predict_proba'):
                scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            else:
                scoring_metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
        
        results = {}
        
        # Different CV strategies
        cv_methods = {
            'kfold': KFold(n_splits=cv_folds, shuffle=True, random_state=42),
            'stratified': StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            'time_series': TimeSeriesSplit(n_splits=cv_folds)
        }
        
        for cv_name in cv_strategies:
            if cv_name not in cv_methods:
                continue
                
            cv_splitter = cv_methods[cv_name]
            results[cv_name] = {}
            
            for metric in scoring_metrics:
                try:
                    if cv_name == 'stratified' and not hasattr(model, 'predict_proba'):
                        continue  # Skip stratified for regression
                    
                    scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=metric, n_jobs=-1)
                    results[cv_name][metric] = {
                        'mean': scores.mean(),
                        'std': scores.std(),
                        'scores': scores.tolist()
                    }
                except Exception as e:
                    print(f"Error with {cv_name} CV and {metric}: {e}")
        
        return results
    
    def stability_analysis(self, model, X, y, n_runs=10, test_size=0.2):
        """Analyze model stability across different train/test splits."""
        from sklearn.model_selection import train_test_split
        
        results = []
        
        for run in range(n_runs):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=run
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if hasattr(model, 'predict_proba'):
                # Classification metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                results.append({
                    'run': run,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
            else:
                # Regression metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results.append({
                    'run': run,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                })
        
        results_df = pd.DataFrame(results)
        
        # Calculate stability metrics
        stability_report = {
            'mean_performance': results_df.drop('run', axis=1).mean().to_dict(),
            'std_performance': results_df.drop('run', axis=1).std().to_dict(),
            'coefficient_of_variation': (results_df.drop('run', axis=1).std() / results_df.drop('run', axis=1).mean()).to_dict()
        }
        
        return stability_report, results_df
    
    def learning_curve_analysis(self, model, X, y, train_sizes=None, cv=5):
        """Generate and analyze learning curves."""
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=cv, n_jobs=-1,
            scoring='accuracy' if hasattr(model, 'predict_proba') else 'r2'
        )
        
        analysis = {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores_mean': train_scores.mean(axis=1).tolist(),
            'train_scores_std': train_scores.std(axis=1).tolist(),
            'val_scores_mean': val_scores.mean(axis=1).tolist(),
            'val_scores_std': val_scores.std(axis=1).tolist(),
            'final_gap': abs(train_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-1]),
            'convergence_point': self._find_convergence_point(train_scores.mean(axis=1), val_scores.mean(axis=1))
        }
        
        return analysis
    
    def _find_convergence_point(self, train_scores, val_scores, threshold=0.02):
        """Find the point where train and validation scores converge."""
        gaps = np.abs(train_scores - val_scores)
        converged_indices = np.where(gaps < threshold)[0]
        return converged_indices[0] if len(converged_indices) > 0 else None
    
    def bias_variance_analysis(self, model, X, y, n_bootstraps=100, test_size=0.2):
        """Perform bias-variance decomposition analysis."""
        from sklearn.model_selection import train_test_split
        from sklearn.utils import resample
        
        # Generate bootstrap samples
        predictions = []
        true_values = []
        
        for i in range(n_bootstraps):
            # Create bootstrap sample
            X_boot, y_boot = resample(X, y, random_state=i)
            
            # Split into train and test
            X_train, X_test, y_train, y_test = train_test_split(
                X_boot, y_boot, test_size=test_size, random_state=i
            )
            
            # Fit model and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            predictions.append(y_pred)
            true_values.append(y_test)
        
        # Convert to numpy arrays
        all_predictions = np.array(predictions)
        all_true = np.concatenate(true_values)
        
        # Calculate bias and variance
        mean_prediction = np.mean(all_predictions, axis=0)
        
        if hasattr(model, 'predict_proba'):
            # For classification, use different approach
            bias_squared = np.mean((mean_prediction - all_true)**2)
            variance = np.mean(np.var(all_predictions, axis=0))
        else:
            # For regression
            bias_squared = np.mean((mean_prediction - all_true)**2)
            variance = np.mean(np.var(all_predictions, axis=0))
        
        noise = np.var(all_true)  # Irreducible error
        
        return {
            'bias_squared': float(bias_squared),
            'variance': float(variance),
            'noise': float(noise),
            'total_error': float(bias_squared + variance + noise)
        }

class PipelineBuilder:
    """
    Build and optimize ML pipelines automatically.
    """
    
    def __init__(self):
        self.pipelines = {}
        self.best_pipeline = None
        self.pipeline_scores = {}
    
    def build_preprocessing_pipeline(self, df, target_col=None, handle_categorical='onehot', 
                                   handle_numeric='standard', handle_missing='simple'):
        """Build comprehensive preprocessing pipeline."""
        
        # Identify column types
        if target_col and target_col in df.columns:
            feature_df = df.drop(columns=[target_col])
        else:
            feature_df = df
        
        numeric_features = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = feature_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Build transformers
        transformers = []
        
        # Numeric transformer
        if numeric_features:
            numeric_steps = []
            
            if handle_missing == 'simple':
                from sklearn.impute import SimpleImputer
                numeric_steps.append(('imputer', SimpleImputer(strategy='mean')))
            elif handle_missing == 'iterative':
                from sklearn.impute import IterativeImputer
                numeric_steps.append(('imputer', IterativeImputer(random_state=42)))
            elif handle_missing == 'knn':
                from sklearn.impute import KNNImputer
                numeric_steps.append(('imputer', KNNImputer(n_neighbors=5)))
            
            if handle_numeric == 'standard':
                numeric_steps.append(('scaler', StandardScaler()))
            elif handle_numeric == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                numeric_steps.append(('scaler', MinMaxScaler()))
            elif handle_numeric == 'robust':
                from sklearn.preprocessing import RobustScaler
                numeric_steps.append(('scaler', RobustScaler()))
            
            transformers.append(('num', Pipeline(numeric_steps), numeric_features))
        
        # Categorical transformer
        if categorical_features:
            categorical_steps = []
            
            if handle_missing in ['simple', 'iterative', 'knn']:
                from sklearn.impute import SimpleImputer
                categorical_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
            
            if handle_categorical == 'onehot':
                categorical_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
            elif handle_categorical == 'ordinal':
                from sklearn.preprocessing import OrdinalEncoder
                categorical_steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
            elif handle_categorical == 'target':
                # Note: Target encoding requires y, implement separately
                pass
            
            if categorical_steps:
                transformers.append(('cat', Pipeline(categorical_steps), categorical_features))
        
        # Combine transformers
        if transformers:
            preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
            return preprocessor
        else:
            return None
    
    def build_complete_pipeline(self, df, target_col, models_to_try=None, preprocessing_options=None):
        """Build complete ML pipelines with different combinations."""
        
        if models_to_try is None:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.linear_model import LogisticRegression, LinearRegression
            from sklearn.svm import SVC, SVR
            
            # Detect task type
            if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() < 20:
                models_to_try = {
                    'rf': RandomForestClassifier(random_state=42),
                    'lr': LogisticRegression(random_state=42, max_iter=1000),
                    'svm': SVC(random_state=42, probability=True)
                }
            else:
                models_to_try = {
                    'rf': RandomForestRegressor(random_state=42),
                    'lr': LinearRegression(),
                    'svm': SVR()
                }
        
        if preprocessing_options is None:
            preprocessing_options = [
                {'handle_categorical': 'onehot', 'handle_numeric': 'standard'},
                {'handle_categorical': 'onehot', 'handle_numeric': 'minmax'},
                {'handle_categorical': 'ordinal', 'handle_numeric': 'robust'}
            ]
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        pipelines = {}
        
        for prep_name, prep_options in enumerate(preprocessing_options):
            preprocessor = self.build_preprocessing_pipeline(df, target_col, **prep_options)
            
            for model_name, model in models_to_try.items():
                pipeline_name = f"{model_name}_prep_{prep_name}"
                
                if preprocessor:
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', model)
                    ])
                else:
                    pipeline = Pipeline([('classifier', model)])
                
                pipelines[pipeline_name] = pipeline
        
        return pipelines
    
    def evaluate_pipelines(self, pipelines, X, y, cv=5, scoring=None):
        """Evaluate multiple pipelines and return best one."""
        
        if scoring is None:
            # Auto-detect scoring metric
            if hasattr(list(pipelines.values())[0], 'predict_proba') or 'Classifier' in str(type(list(pipelines.values())[0])):
                scoring = 'accuracy'
            else:
                scoring = 'r2'
        
        results = {}
        
        for name, pipeline in pipelines.items():
            try:
                scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                results[name] = {
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'scores': scores.tolist()
                }
                print(f"{name}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                results[name] = {'error': str(e)}
        
        # Find best pipeline
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_pipeline_name = max(valid_results.keys(), key=lambda x: valid_results[x]['mean_score'])
            self.best_pipeline = pipelines[best_pipeline_name]
            self.pipeline_scores = results
            
            print(f"\nBest pipeline: {best_pipeline_name}")
            print(f"Best score: {valid_results[best_pipeline_name]['mean_score']:.4f}")
        
        return results

class ModelInterpreter:
    """
    Advanced model interpretation and explanation utilities.
    """
    
    def __init__(self):
        self.interpretation_results = {}
    
    def permutation_importance(self, model, X, y, scoring=None, n_repeats=10):
        """Calculate permutation importance for features."""
        from sklearn.inspection import permutation_importance
        
        if scoring is None:
            scoring = 'accuracy' if hasattr(model, 'predict_proba') else 'r2'
        
        result = permutation_importance(
            model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=42, n_jobs=-1
        )
        
        # Create importance dataframe
        feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        return importance_df
    
    def partial_dependence_analysis(self, model, X, features=None, n_features=5):
        """Analyze partial dependence for top features."""
        try:
            from sklearn.inspection import partial_dependence, PartialDependenceDisplay
        except ImportError:
            print("Partial dependence requires scikit-learn >= 0.22")
            return None
        
        if features is None:
            # Use permutation importance to select top features
            importance_df = self.permutation_importance(model, X, X.iloc[:, 0])  # Dummy y for feature selection
            features = importance_df.head(n_features)['feature'].tolist()
        
        # Calculate partial dependence
        pd_results = {}
        feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        
        for feature in features:
            if feature in feature_names:
                feature_idx = list(feature_names).index(feature)
                pd_result = partial_dependence(model, X, [feature_idx])
                pd_results[feature] = {
                    'values': pd_result['values'][0].tolist(),
                    'grid_values': pd_result['grid_values'][0].tolist()
                }
        
        return pd_results
    
    def local_explanation(self, model, X, instance_idx=0, n_features=10):
        """Generate local explanation for a specific instance."""
        try:
            import lime
            import lime.lime_tabular
        except ImportError:
            print("LIME not available. Install with: pip install lime")
            return None
        
        # Create LIME explainer
        feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X.values,
            feature_names=feature_names,
            class_names=['class_0', 'class_1'] if hasattr(model, 'predict_proba') else None,
            mode='classification' if hasattr(model, 'predict_proba') else 'regression'
        )
        
        # Generate explanation for specific instance
        explanation = explainer.explain_instance(
            X.iloc[instance_idx].values,
            model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
            num_features=n_features
        )
        
        # Extract explanation data
        explanation_data = {
            'instance_idx': instance_idx,
            'prediction': model.predict([X.iloc[instance_idx].values])[0],
            'feature_contributions': explanation.as_list()
        }
        
        if hasattr(model, 'predict_proba'):
            explanation_data['prediction_proba'] = model.predict_proba([X.iloc[instance_idx].values])[0].tolist()
        
        return explanation_data
    
    def global_surrogate_model(self, model, X, y=None):
        """Create a global surrogate model for interpretation."""
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        
        # Generate predictions from the original model
        if y is None:
            y_surrogate = model.predict(X)
        else:
            y_surrogate = model.predict(X)
        
        # Train surrogate model
        if hasattr(model, 'predict_proba'):
            surrogate = DecisionTreeClassifier(max_depth=5, random_state=42)
        else:
            surrogate = DecisionTreeRegressor(max_depth=5, random_state=42)
        
        surrogate.fit(X, y_surrogate)
        
        # Calculate fidelity (how well surrogate approximates original)
        surrogate_pred = surrogate.predict(X)
        
        if hasattr(model, 'predict_proba'):
            fidelity = accuracy_score(y_surrogate, surrogate_pred)
        else:
            fidelity = r2_score(y_surrogate, surrogate_pred)
        
        return {
            'surrogate_model': surrogate,
            'fidelity': fidelity,
            'feature_importance': dict(zip(
                X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])],
                surrogate.feature_importances_
            ))
        }