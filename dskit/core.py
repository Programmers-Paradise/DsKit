import pandas as pd
from . import io, cleaning, visualization, preprocessing, modeling, explainability, eda
from . import feature_engineering, nlp_utils, advanced_visualization, advanced_modeling, auto_ml, comprehensive_eda
from typing import Literal,Optional,Annotated
class dskit:
    def __init__(self, df=None):
        self.df = df
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    # --- IO ---
    @staticmethod
    def load(filepath):
        return dskit(io.load(filepath))

    @staticmethod
    def read_folder(folder_path:str, file_type:Literal['csv','xls','xlsx','json','parquet']='csv',dynamic:bool=False,display_ignored:bool=False):
        return dskit(io.read_folder(folder_path, file_type,dynamic,display_ignored))
    
    def save(self, filepath, **kwargs):
        io.save(self.df, filepath, **kwargs)
        return self

    # --- Cleaning ---
    def fix_dtypes(self):
        self.df = cleaning.fix_dtypes(self.df)
        return self

    def rename_columns_auto(self):
        self.df = cleaning.rename_columns_auto(self.df)
        return self

    def replace_specials(self, chars_to_remove=r'[@#%$]', replacement=''):
        self.df = cleaning.replace_specials(self.df, chars_to_remove, replacement)
        return self

    def missing_summary(self):
        return cleaning.missing_summary(self.df)

    def fill_missing(self, strategy='auto', fill_value=None):
        self.df = cleaning.fill_missing(self.df, strategy, fill_value)
        return self

    def outlier_summary(self, method='iqr', threshold=1.5):
        return cleaning.outlier_summary(self.df, method, threshold)

    def remove_outliers(self, method='iqr', threshold=1.5):
        self.df = cleaning.remove_outliers(self.df, method, threshold)
        return self
    
    def simple_nlp_clean(self, text_cols=None):
        self.df = cleaning.simple_nlp_clean(self.df, text_cols)
        return self
    
    def advanced_text_clean(self, text_cols=None, remove_urls=True, remove_emails=True, remove_numbers=False, expand_contractions=True):
        self.df = nlp_utils.advanced_text_clean(self.df, text_cols, remove_urls, remove_emails, remove_numbers, expand_contractions)
        return self
    
    def extract_text_features(self, text_cols=None):
        self.df = nlp_utils.extract_text_features(self.df, text_cols)
        return self
    
    def sentiment_analysis(self, text_cols=None):
        self.df = nlp_utils.sentiment_analysis(self.df, text_cols)
        return self
    
    def text_stats(self, text_cols=None):
        return nlp_utils.basic_text_stats(self.df, text_cols)
    
    def generate_wordcloud(self, text_col, max_words=100):
        nlp_utils.generate_wordcloud(self.df, text_col, max_words)
        return self
    def generate_vocabulary(self,text_col:str,case:Literal['lower','upper']=None):
        return nlp_utils.generate_vocabulary(self.df,text_col,case)
    def apply_nltk(
            self,
            text_column:Annotated[str, "Column name containing raw text"],
            output_column:Annotated[str, "Output column name for processed text"] = "cleaned_nltk",
            apply_case:Annotated[
                Optional[Literal['lower','upper','sentence','title']],
                "Case transformation to apply"
            ] = 'lower',
            allow_download:Annotated[bool,"Automatically download required NLTK resources if missing"]=False,
            remove_stopwords:Annotated[bool,"Remove stopwords using NLTK stopword corpus"]=True,
            keep_words:Annotated[
                list[str],
                "Words to retain even if stopword removal is enabled"
            ] = ["not", "no", "off"],
            remove_words:Annotated[
                list[str],
                "Words to explicitly remove from the text"
                ]=[],
            use_tokenizer:Annotated[bool,"Use NLTK tokenizer instead of simple whitespace split"]=True,
            language:Annotated[str,"Language for stopword removal"]='english',
            canonicalization:Annotated[
                Optional[Literal['stemming', 'lemmatization']],
                "Canonicalization strategy"
                ]='stemming'
        )->pd.DataFrame:
        self.df = nlp_utils.apply_nltk(
            df=self.df,
            text_column=text_column,
            output_column=output_column,
            apply_case=apply_case,
            allow_download=allow_download,
            remove_stopwords=remove_stopwords,
            keep_words=keep_words,
            remove_words=remove_words,
            use_tokenizer=use_tokenizer,
            language=language,
            canonicalization=canonicalization
        )
        return self
    def clean(self):
        """
        Chains multiple cleaning steps: fix_dtypes -> rename_columns -> fill_missing
        """
        return self.fix_dtypes().rename_columns_auto().fill_missing()

    # --- Visualization ---
    def plot_missingness(self):
        visualization.plot_missingness(self.df)
        return self

    def plot_histograms(self, bins=30):
        visualization.plot_histograms(self.df, bins)
        return self

    def plot_boxplots(self):
        visualization.plot_boxplots(self.df)
        return self

    def plot_correlation_heatmap(self):
        visualization.plot_correlation_heatmap(self.df)
        return self

    def plot_pairplot(self, hue=None):
        visualization.plot_pairplot(self.df, hue)
        return self
    
    def visualize(self):
        """
        Chains common visualizations.
        """
        self.plot_histograms()
        self.plot_correlation_heatmap()
        return self
    
    def plot_feature_importance(self, top_n=20):
        if self.model is None:
            print("Model not trained. Call train() first.")
            return self
        feature_names = self.X_train.columns if hasattr(self.X_train, 'columns') else None
        advanced_visualization.plot_feature_importance(self.model.model, feature_names, top_n)
        return self
    
    def plot_target_distribution(self, target_col, task='classification'):
        advanced_visualization.plot_target_distribution(self.df, target_col, task)
        return self
    
    def plot_missing_patterns_advanced(self):
        advanced_visualization.plot_missing_patterns_advanced(self.df)
        return self
    
    def plot_outliers_advanced(self, method='iqr'):
        advanced_visualization.plot_outliers_advanced(self.df, method)
        return self

    # --- Preprocessing ---
    def auto_encode(self):
        self.df = preprocessing.auto_encode(self.df)
        return self

    def auto_scale(self, method='standard'):
        self.df = preprocessing.auto_scale(self.df, method)
        return self
    
    def create_polynomial_features(self, degree=2, interaction_only=False, include_bias=False):
        self.df = feature_engineering.create_polynomial_features(self.df, degree, interaction_only, include_bias)
        return self
    
    def create_date_features(self, date_cols=None):
        self.df = feature_engineering.create_date_features(self.df, date_cols)
        return self
    
    def create_binning_features(self, numeric_cols=None, n_bins=5, strategy='quantile'):
        self.df = feature_engineering.create_binning_features(self.df, numeric_cols, n_bins, strategy)
        return self
    
    def apply_pca(self, n_components=None, variance_threshold=0.95):
        self.df, self.pca_model = feature_engineering.apply_pca(self.df, n_components, variance_threshold)
        return self

    def train_test_auto(self, target=None, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = preprocessing.train_test_auto(
            self.df, target, test_size, random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    # --- Modeling ---
    def train(self, model_name='random_forest', task='classification'):
        if self.X_train is None:
            print("Data not split. Call train_test_auto() first.")
            return self
            
        self.model = modeling.QuickModel(model_name, task)
        self.model.fit(self.X_train, self.y_train)
        return self

    def evaluate(self, task='classification'):
        if self.model is None:
            print("Model not trained. Call train() first.")
            return self
        modeling.evaluate_model(self.model, self.X_test, self.y_test, task)
        return self
    
    def compare_models(self, target, task='classification'):
        X = self.df.drop(target, axis=1)
        y = self.df[target]
        return modeling.compare_models(X, y, task)
    
    def train_advanced(self, model_name='xgboost', task='classification', **kwargs):
        if self.X_train is None:
            print("Data not split. Call train_test_auto() first.")
            return self
        
        self.model = advanced_modeling.AdvancedModel(model_name, task, **kwargs)
        self.model.fit(self.X_train, self.y_train)
        return self
    
    def cross_validate(self, cv=5, scoring=None):
        if self.model is None:
            print("Model not trained. Call train() first.")
            return self
        return advanced_modeling.cross_validate_model(self.model.model, self.X_train, self.y_train, cv, scoring)
    
    def auto_tune(self, method='random', max_evals=50, task='classification', model_name=None):
        if self.X_train is None:
            print("Data not split. Call train_test_auto() first.")
            return self
        
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        model_class = RandomForestClassifier if task == 'classification' else RandomForestRegressor
        
        best_model, best_params, best_score = auto_ml.auto_tune_model(
            model_class, self.X_train, self.y_train, method, max_evals, task, model_name
        )
        
        # Wrap in our model structure
        self.model = type('TunedModel', (), {'model': best_model, 'fit': lambda x, y: best_model, 'predict': best_model.predict})()
        return self

    # --- Explainability ---
    def explain(self):
        if self.model is None:
            print("Model not trained. Call train() first.")
            return self
        # Explain using test data or a sample of it
        explainability.explain_shap(self.model.model, self.X_test)
        return self

    # --- EDA ---
    def basic_stats(self):
        return eda.basic_stats(self.df)

    def quick_eda(self):
        eda.quick_eda(self.df)
        return self
    
    def comprehensive_eda(self, target_col=None, sample_size=None):
        comprehensive_eda.comprehensive_eda(self.df, target_col, sample_size)
        return self
    
    def data_health_check(self):
        return comprehensive_eda.data_health_check(self.df)
    
    def generate_profile_report(self, output_file='profile_report.html'):
        return comprehensive_eda.generate_pandas_profile(self.df, output_file)
