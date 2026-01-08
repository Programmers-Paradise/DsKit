import pandas as pd
import numpy as np
from typing import Literal
try:
    import nltk
    from nltk import WordNetLemmatizer
    from nltk.corpus import stopwords
    from textblob import TextBlob
    from wordcloud import WordCloud
except ImportError:
    nltk = None
    TextBlob = None
    WordCloud = None
import re
import string
import matplotlib.pyplot as plt

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
                       remove_numbers=False, expand_contractions=True, auto_smart_cleaning:bool=False,language:str='english'):
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
        if auto_smart_cleaning: # applies nltk for removing common words and uses word lemmatization
            try:
                nltk.download('stopwords') # download stopwords
                nltk.download('punkt_tab') # download punkt
                nltk.download('wordnet') # download wordnet
                all_stopwords = stopwords.words(language) # load stopwords
                for remove_stop in ['no', 'not', 'off']: # removing from stopwords
                    all_stopwords.remove(remove_stop)
                lemmatizer = WordNetLemmatizer() #lamitizer
                
                # Function to lemmatize and remove stopwords
                def lemmatize_text(text):
                    from nltk.tokenize import word_tokenize
                    words = word_tokenize(text.lower())  # Lowercase for better lemmatization
                    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in all_stopwords]
                    return ' '.join(lemmatized_words)
                
                # Apply lemmatization to the text series
                text_series = text_series.apply(lemmatize_text)
                
            except Exception as e:
                print(f"Advanced cleaning failed: {e}")
                pass
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

