import pandas as pd
import numpy as np
from typing import Literal, Optional, Iterable, Callable
from collections import OrderedDict
import re
import warnings
import importlib
import string
import matplotlib.pyplot as plt
try:
    from textblob import TextBlob
    from wordcloud import WordCloud
except ImportError:
    TextBlob = None
    WordCloud = None


def basic_text_stats(df, text_cols=None):
    """
    Generate basic statistics for text columns.
    """
    if text_cols is None:
        text_cols = df.select_dtypes(include=['object', 'string']).columns
    
    stats = {}
    for col in text_cols:
        if col not in df.columns:
            continue
            
        text_series = df[col].astype(str)
        stats[col] = {
            'total_texts': len(text_series),
            'avg_length': text_series.str.len().mean(),
            'max_length': text_series.str.len().max(),
            'min_length': text_series.str.len().min(),
            'avg_words': text_series.str.split().str.len().mean(),
            'unique_texts': text_series.nunique()
        }
    
    return pd.DataFrame(stats).T

def advanced_text_clean(df, text_cols=None, remove_urls=True, remove_emails=True, 
                       remove_numbers=False, expand_contractions=True):
    """
    Advanced text cleaning with more options.
    """
    df = df.copy()
    
    if text_cols is None:
        text_cols = df.select_dtypes(include=['object', 'string']).columns
    
    for col in text_cols:
        if col not in df.columns:
            continue
            
        text_series = df[col].astype(str)
        
        # Remove URLs
        if remove_urls:
            text_series = text_series.str.replace(r'http\S+|www\S+|https\S+', '', regex=True)
        
        # Remove email addresses
        if remove_emails:
            text_series = text_series.str.replace(r'\S+@\S+', '', regex=True)
        
        # Remove numbers
        if remove_numbers:
            text_series = text_series.str.replace(r'\d+', '', regex=True)
        
        # Basic contractions expansion
        if expand_contractions:
            contractions = {
                "won't": "will not", "can't": "cannot", "n't": " not",
                "'re": " are", "'ve": " have", "'ll": " will",
                "'d": " would", "'m": " am"
            }
            for contraction, expansion in contractions.items():
                text_series = text_series.str.replace(contraction, expansion, case=False)
        
        # Remove extra whitespace
        text_series = text_series.str.replace(r'\s+', ' ', regex=True).str.strip()
        
        df[col] = text_series
    
    return df

def extract_text_features(df, text_cols=None):
    """
    Extract features from text columns.
    """
    df = df.copy()
    
    if text_cols is None:
        text_cols = df.select_dtypes(include=['object', 'string']).columns
    
    for col in text_cols:
        if col not in df.columns:
            continue
            
        text_series = df[col].astype(str)
        
        # Length features
        df[f'{col}_length'] = text_series.str.len()
        df[f'{col}_word_count'] = text_series.str.split().str.len()
        
        # Character features
        df[f'{col}_uppercase_count'] = text_series.str.count(r'[A-Z]')
        df[f'{col}_lowercase_count'] = text_series.str.count(r'[a-z]')
        df[f'{col}_digit_count'] = text_series.str.count(r'\d')
        df[f'{col}_special_char_count'] = text_series.str.count(r'[^\w\s]')
        
        # Punctuation features
        df[f'{col}_exclamation_count'] = text_series.str.count('!')
        df[f'{col}_question_count'] = text_series.str.count('\?')
        
    return df

def sentiment_analysis(df, text_cols=None):
    """
    Perform sentiment analysis using TextBlob.
    """
    if TextBlob is None:
        print("TextBlob not installed. Please install it using 'pip install textblob'")
        return df
    
    df = df.copy()
    
    if text_cols is None:
        text_cols = df.select_dtypes(include=['object', 'string']).columns
    
    for col in text_cols:
        if col not in df.columns:
            continue
            
        sentiments = []
        polarities = []
        subjectivities = []
        
        for text in df[col].astype(str):
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Classify sentiment
                if polarity > 0.1:
                    sentiment = 'positive'
                elif polarity < -0.1:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                sentiments.append(sentiment)
                polarities.append(polarity)
                subjectivities.append(subjectivity)
                
            except Exception:
                sentiments.append('neutral')
                polarities.append(0.0)
                subjectivities.append(0.0)
        
        df[f'{col}_sentiment'] = sentiments
        df[f'{col}_polarity'] = polarities
        df[f'{col}_subjectivity'] = subjectivities
    
    return df

def generate_wordcloud(df, text_col, max_words=100):
    """
    Generate a word cloud from text column.
    """
    if WordCloud is None:
        print("WordCloud not installed. Please install it using 'pip install wordcloud'")
        return
    
    if text_col not in df.columns:
        print(f"Column '{text_col}' not found.")
        return
    
    # Combine all text
    all_text = ' '.join(df[text_col].astype(str).values)
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, max_words=max_words, 
                         background_color='white').generate(all_text)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {text_col}')
    plt.tight_layout()
    plt.show()

def extract_keywords(df, text_col, top_n=20):
    """
    Extract most common words/phrases from text column.
    """
    if text_col not in df.columns:
        print(f"Column '{text_col}' not found.")
        return pd.DataFrame()
    
    # Combine all text and split into words
    all_text = ' '.join(df[text_col].astype(str).str.lower().values)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)  # Words with 3+ characters
    
    # Count word frequencies
    word_counts = pd.Series(words).value_counts().head(top_n)
    
    return word_counts.reset_index().rename(columns={'index': 'word', 0: 'count'})

def detect_language(df, text_col):
    """
    Detect language of text using simple heuristics.
    """
    if TextBlob is None:
        print("TextBlob not installed. Please install it using 'pip install textblob'")
        return df
    
    df = df.copy()
    
    if text_col not in df.columns:
        print(f"Column '{text_col}' not found.")
        return df
    
    languages = []
    for text in df[text_col].astype(str):
        try:
            blob = TextBlob(text)
            lang = blob.detect_language()
            languages.append(lang)
        except Exception:
            languages.append('unknown')
    
    df[f'{text_col}_language'] = languages
    return df


def generate_vocabulary(df:pd.DataFrame,text_col:str,case:Literal['lower','upper']=None):
    """
    returns a list of vocabulary made from a column of dataframe
    
    :param df: dataframe
    :type df: pd.DataFrame
    :param text_col: name of the text column
    :type text_col: str
    :param case: case of text. If not provided then words remains unchanged
    :type case: Literal['lower', 'upper']
    """
    if text_col not in df.columns:
        print(f"Column '{text_col}' not found.")
        return []
    vocabulary = set()
    for text in df[text_col].astype(str):
        if case=='lower':
            text=text.lower()
        elif case=='upper':
            text==text.upper()
        text = text.split()
        for t in text:
            vocabulary.add(t)
    return list(vocabulary)



# nltk application
_init_nltk_cache=OrderedDict()
_MAX_CACHE_SIZE = 2
def _get_from_cache(key):
    try:
        value = _init_nltk_cache.pop(key)
        _init_nltk_cache[key] = value  # move to end
        return value
    except KeyError:
        return None

def _set_cache(key, value):
    if key in _init_nltk_cache:
        _init_nltk_cache.pop(key)
    elif len(_init_nltk_cache) >= _MAX_CACHE_SIZE:
        _init_nltk_cache.popitem(last=False)  # evict LRU
    _init_nltk_cache[key] = value


class NLTKUnavailable(Exception):
    pass


def _init_nltk(download_list:Optional[Iterable[str]]=None,allow_download: bool = False, language: str = "english",keep_words:Optional[Iterable[str]]=None,remove_words:Optional[Iterable[str]]=None)->dict[str,object]:
    """
    Lazy initialize and return a dict with objects: tokenizer, stopwords_set, stemmer/lemmatizer.
    Raises NLTKUnavailable if nltk not present and allow_download is False.
    """
    env=None
    download_tuple = tuple(sorted([d for d in (download_list or []) if d]))
    keep_tuple = tuple(sorted([k for k in (keep_words or []) if k]))
    remove_tuple = tuple(sorted([r for r in (keep_words or []) if r]))
    key = (download_tuple, bool(allow_download), language, keep_tuple, remove_tuple)
    env=_get_from_cache(key)
    if env is not None:
        return env

    try:
        nltk = importlib.import_module("nltk")
    except ImportError:
        raise NLTKUnavailable("nltk is not installed")

    # helper to check resource and optionally download
    def _ensure(resource_name: str, download_name: Optional[str] = None):
        try:
            nltk.data.find(resource_name)
        except LookupError:
            if allow_download:
                download_target = download_name or resource_name
                nltk.download(download_target, quiet=True)
            else:
                raise LookupError(f"NLTK resource '{resource_name}' not found. Set allow_download=True or install resources manually.")

    tokenizer=None
    stopwords_set = None
    stemmer = None
    lemmatizer = None
    # punkt is used by word_tokenize
    if 'tokenizer' in download_list:
        _ensure("tokenizers/punkt", "punkt")
        tokenizer = nltk.word_tokenize
        

    
    if 'stopwords' in download_list:
        _ensure("corpora/stopwords", "stopwords")
        from nltk.corpus import stopwords as _nltk_stopwords
        stopwords_set = set(_nltk_stopwords.words(language))
        if isinstance(keep_words,(list,tuple)) and len(keep_words)>0:
            for kword in keep_words:
                stopwords_set.discard(kword)
        if isinstance(remove_words,(list,tuple)) and len(remove_words)>0:
            for rword in remove_words:
                stopwords_set.add(rword)
        

    
    # Initialize both, user picks which to use
    if 'stemming' in download_list:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
    if 'lemmatization' in download_list:
        _ensure("corpora/wordnet", "wordnet")
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
    env = {
        "nltk": nltk,
        "tokenizer": tokenizer,
        "stopwords": stopwords_set,
        "stemmer": stemmer,
        "lemmatizer": lemmatizer,
    }
    _set_cache(key,env)
    return env
def apply_nltk(
        df:pd.DataFrame,
        text_column:str,
        output_column:str="claeaned_nltk",
        apply_case:Optional[Literal['lower','upper','sentence','title']]=None,
        allow_download:bool=False,
        remove_stopwords:bool=False,
        keep_words:list=["not","no","off"],
        remove_words:list=[],
        use_tokenizer:bool=False,
        language:str='english',
        canonicalization:Optional[Literal['stemming', 'lemmatization']]=None
        )->pd.DataFrame:
    """
    Apply advanced text preprocessing using optional NLTK-based features.

    This function performs configurable text normalization on a specified
    DataFrame column, including case transformation, tokenization,
    stopword removal, and canonicalization (stemming or lemmatization).
    NLTK is treated as an optional dependency and is only initialized
    when explicitly required by the chosen options.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the text data.

    text_column : str
        Name of the column in `df` that contains text to be processed.

    output_column : str, default="claeaned_nltk"
        Name of the output column where the processed text will be stored.

    apply_case : {"lower", "upper", "sentence", "title"}, optional
        Case transformation to apply to the text:

    allow_download : bool, default=False
        If True, automatically downloads required NLTK resources
        (e.g., punkt, stopwords, wordnet) when missing.
        If False, missing resources will raise an error.

    remove_stopwords : bool, default=False
        If True, removes stopwords using the specified language.

    keep_words : list of str, default=["not", "no", "off"]
        Words that should be retained even if stopword removal is enabled.
        Useful for preserving negations.

    remove_words : list of str, default=[]
        Explicit list of words to remove from the text regardless
        of stopword settings.

    use_tokenizer : bool, default=False
        If True, uses NLTK's tokenizer (requires `punkt`).
        If False, falls back to a lightweight whitespace-based tokenizer.

    language : str, default="english"
        Language used for stopword removal.

    canonicalization : {"stemming", "lemmatization"}, optional
        Word normalization strategy:
        - "stemming"       : applies Porter stemming
        - "lemmatization"  : applies WordNet lemmatization

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with an additional column
        containing the processed text.

    Raises
    ------
    KeyError
        If `text_column` does not exist in the DataFrame.

    ImportError
        If NLTK is required but not installed.

    LookupError
        If required NLTK resources are missing and `allow_download=False`.

    Notes
    -----
    - NLTK initialization and resource loading are cached internally
      to avoid repeated overhead across multiple calls.

    """
    if text_column not in df.columns:
        raise IndexError(f"Column '{text_column}' not found.") 
    df=df.copy()
    text_field = df[text_column].astype('str')
    tokenizer: Callable[[str], list[str]] = None
    stopwords_set: Optional[set[str]] = set()
    stemmer = None
    lemmatizer = None
    download_list = []
    if canonicalization:
        download_list.append(canonicalization)
    if remove_stopwords:
        download_list.append('stopwords')
    if use_tokenizer:
        download_list.append('tokenizer')
    try:
        env = _init_nltk(download_list=download_list,allow_download=allow_download, language=language,keep_words=keep_words,remove_words=remove_words)
        tokenizer = env["tokenizer"]
        stopwords_set = env["stopwords"]
        stemmer = env["stemmer"]
        lemmatizer = env["lemmatizer"]
    except (NLTKUnavailable, LookupError) as e:
        warnings.warn(f"NLTK unavailable or missing corpora: Falling back to lightweight tokenizer. Set allow_download=True to auto-download resources. {e}", UserWarning,stacklevel=2)


    def _apply(text):
        text=text.lower()
        if use_tokenizer:
            text = tokenizer(text)
        else:
            text = text.split()
        
        if canonicalization == 'stemming':
            text = [stemmer.stem(word) for word in text if word not in stopwords_set]
        elif canonicalization == 'lemmatization':
            text = [lemmatizer.lemmatize(word) for word in text if word not in stopwords_set]
        else:
            warnings.warn("No canonicalization used",UserWarning,stacklevel=2)
            if remove_stopwords:
                text = [word for word in text if word not in stopwords_set]
        
        text =' '.join(text)
        if apply_case=="upper":
            text=text.upper()
        elif apply_case=="sentence":
            text=text.capitalize()
        elif apply_case=="title":
            text=text.title()
        return text 
    text_field = text_field.apply(_apply)
    df[output_column]=text_field
    return df  
    