import pytest
import pandas as pd
from dskit import nlp_utils


@pytest.fixture
def sample_text_df():
    """Fixture providing sample text data for testing."""
    return pd.DataFrame({
        "text": [
            "Hello world! This is NLP.",
            "Pytest makes testing easy.",
            "Natural Language Processing is powerful."
        ]
    })


@pytest.fixture
def sample_text_df_with_urls():
    """Fixture providing sample text with URLs and emails."""
    return pd.DataFrame({
        "text": [
            "Visit https://example.com for more info. Contact us at test@example.com",
            "Check out www.github.com and email support@github.com",
            "Email me at john@test.co.uk or visit http://mysite.com"
        ]
    })


@pytest.fixture
def sample_multi_col_df():
    """Fixture providing text data with multiple text columns."""
    return pd.DataFrame({
        "text": [
            "Hello world this is a test",
            "Another sample text here",
            "Third text example"
        ],
        "description": [
            "First description with some words",
            "Second description text",
            "Third description example"
        ]
    })


# ==================== TESTS FOR 5 BASIC FUNCTIONS ====================

def test_basic_text_stats(sample_text_df):
    """Test that basic_text_stats returns a DataFrame with expected columns."""
    result = nlp_utils.basic_text_stats(sample_text_df, text_cols=["text"])
    
    # Verify return type
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    
    # Verify expected columns exist
    assert "total_texts" in result.columns, "Should have 'total_texts' column"
    assert "avg_length" in result.columns, "Should have 'avg_length' column"
    assert "max_length" in result.columns, "Should have 'max_length' column"
    assert "min_length" in result.columns, "Should have 'min_length' column"
    assert "avg_words" in result.columns, "Should have 'avg_words' column"
    assert "unique_texts" in result.columns, "Should have 'unique_texts' column"
    
    # Verify data correctness
    assert result.loc["text", "total_texts"] == 3, "Should count 3 texts"
    assert result.loc["text", "avg_words"] > 0, "Average words should be positive"
    assert result.loc["text", "unique_texts"] == 3, "All texts are unique"


def test_basic_text_stats_multiple_columns(sample_multi_col_df):
    """Test basic_text_stats with multiple text columns."""
    result = nlp_utils.basic_text_stats(sample_multi_col_df, text_cols=["text", "description"])
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2, "Should have stats for 2 columns"
    assert "text" in result.index
    assert "description" in result.index
    assert result.loc["text", "total_texts"] == 3
    assert result.loc["description", "total_texts"] == 3


def test_advanced_text_clean(sample_text_df_with_urls):
    """Test that advanced_text_clean removes URLs and emails."""
    cleaned = nlp_utils.advanced_text_clean(
        sample_text_df_with_urls,
        text_cols=["text"],
        remove_urls=True,
        remove_emails=True,
        expand_contractions=True
    )
    
    # Verify return type
    assert isinstance(cleaned, pd.DataFrame), "Result should be a DataFrame"
    
    # Verify column exists
    assert "text" in cleaned.columns, "Text column should exist"
    
    # Verify URLs and emails are removed
    assert "https://" not in cleaned["text"].iloc[0], "URLs should be removed"
    assert "@" not in cleaned["text"].iloc[0], "Email addresses should be removed"
    
    # Verify text length is reduced
    assert len(cleaned["text"].iloc[0]) < len(sample_text_df_with_urls["text"].iloc[0]), \
        "Cleaned text should be shorter"


def test_advanced_text_clean_number_removal():
    """Test advanced_text_clean with number removal."""
    df = pd.DataFrame({"text": ["Hello 123 world 456", "Test 789"]})
    
    # Without number removal
    result_with_nums = nlp_utils.advanced_text_clean(df.copy(), text_cols=["text"], remove_numbers=False)
    assert "123" in result_with_nums["text"].iloc[0], "Numbers should remain when remove_numbers=False"
    
    # With number removal
    result_no_nums = nlp_utils.advanced_text_clean(df.copy(), text_cols=["text"], remove_numbers=True)
    assert "123" not in result_no_nums["text"].iloc[0], "Numbers should be removed when remove_numbers=True"


def test_extract_text_features(sample_text_df):
    """Test that extract_text_features creates expected feature columns."""
    features = nlp_utils.extract_text_features(sample_text_df, text_cols=["text"])
    
    # Verify return type
    assert isinstance(features, pd.DataFrame), "Result should be a DataFrame"
    
    # Verify feature columns were created
    assert "text_length" in features.columns, "Should have text_length feature"
    assert "text_word_count" in features.columns, "Should have text_word_count feature"
    assert "text_uppercase_count" in features.columns, "Should have text_uppercase_count feature"
    assert "text_lowercase_count" in features.columns, "Should have text_lowercase_count feature"
    assert "text_digit_count" in features.columns, "Should have text_digit_count feature"
    assert "text_special_char_count" in features.columns, "Should have text_special_char_count feature"
    assert "text_exclamation_count" in features.columns, "Should have text_exclamation_count feature"
    assert "text_question_count" in features.columns, "Should have text_question_count feature"
    
    # Verify original columns are preserved
    assert "text" in features.columns, "Original text column should be preserved"
    
    # Verify data correctness
    assert len(features) == 3, "Should have same number of rows"
    assert features["text_word_count"].iloc[0] > 0, "First text should have words"
    assert features["text_exclamation_count"].iloc[0] == 1, "First text has 1 exclamation"


def test_generate_vocabulary(sample_text_df):
    """Test that generate_vocabulary returns a list of unique words."""
    vocab = nlp_utils.generate_vocabulary(sample_text_df, text_col="text")
    
    # Verify return type
    assert isinstance(vocab, list), "Result should be a list"
    
    # Verify non-empty result
    assert len(vocab) > 0, "Vocabulary should not be empty"
    
    # Verify contains expected words
    vocab_lower = [w.lower() for w in vocab]
    assert any("hello" in w.lower() for w in vocab), "Should contain 'hello'"
    assert any("world" in w.lower() for w in vocab), "Should contain 'world'"
    
    # Verify all items are strings
    assert all(isinstance(word, str) for word in vocab), "All vocabulary items should be strings"


def test_generate_vocabulary_with_case_lower():
    """Test generate_vocabulary with lowercase case option."""
    df = pd.DataFrame({"text": ["Hello World Test"]})
    vocab = nlp_utils.generate_vocabulary(df, text_col="text", case="lower")
    
    assert isinstance(vocab, list), "Result should be a list"
    assert len(vocab) > 0, "Vocabulary should not be empty"
    # All words should be lowercase (or at least check that processing occurred)
    assert any(word in vocab for word in ["hello", "world", "test"]), \
        "Should contain expected words"


def test_generate_vocabulary_multiple_texts():
    """Test generate_vocabulary with multiple text entries."""
    df = pd.DataFrame({
        "text": [
            "apple banana cherry",
            "banana date elderberry",
            "fig grape apple"
        ]
    })
    vocab = nlp_utils.generate_vocabulary(df, text_col="text")
    
    assert isinstance(vocab, list), "Result should be a list"
    assert len(vocab) >= 7, "Should have at least 7 unique words (apple, banana, etc.)"
    
    # Check specific words are in vocabulary
    vocab_lower = [w.lower() for w in vocab]
    assert any("apple" in w for w in vocab_lower), "Should contain 'apple'"
    assert any("banana" in w for w in vocab_lower), "Should contain 'banana'"


def test_extract_keywords(sample_text_df):
    """Test that extract_keywords returns a DataFrame with word frequencies."""
    keywords = nlp_utils.extract_keywords(sample_text_df, text_col="text", top_n=5)
    
    # Verify return type
    assert isinstance(keywords, pd.DataFrame), "Result should be a DataFrame"
    
    # Verify expected columns
    assert "word" in keywords.columns, "Should have 'word' column"
    assert "count" in keywords.columns, "Should have 'count' column"
    
    # Verify correct number of results
    assert len(keywords) <= 5, "Should return at most top_n results"
    assert len(keywords) > 0, "Should return at least one keyword"
    
    # Verify results are sorted by frequency (descending)
    assert keywords["count"].iloc[0] >= keywords["count"].iloc[-1], \
        "Keywords should be sorted by frequency (highest first)"


def test_extract_keywords_specific_count(sample_text_df):
    """Test extract_keywords with different top_n values."""
    keywords_3 = nlp_utils.extract_keywords(sample_text_df, text_col="text", top_n=3)
    keywords_10 = nlp_utils.extract_keywords(sample_text_df, text_col="text", top_n=10)
    
    assert len(keywords_3) <= 3, "Should return at most 3 keywords"
    assert len(keywords_10) <= 10, "Should return at most 10 keywords"
    assert len(keywords_3) <= len(keywords_10), "More keywords requested should give more results"


def test_extract_keywords_word_filtering():
    """Test that extract_keywords filters out very short words (3+ chars)."""
    df = pd.DataFrame({
        "text": [
            "I am very happy about this special journey",
            "The quick brown fox jumps over lazy dog",
            "Amazing and wonderful experience"
        ]
    })
    keywords = nlp_utils.extract_keywords(df, text_col="text", top_n=20)
    
    # All words should be 3+ characters long
    assert all(len(word) >= 3 for word in keywords["word"]), \
        "All keywords should be at least 3 characters long"


# ==================== ADDITIONAL VALIDATION TESTS ====================

def test_nlp_utils_integration():
    """Integration test combining multiple NLP operations."""
    df = pd.DataFrame({
        "text": [
            "This is a great product! I love it.",
            "This product is terrible and broken.",
            "Average product, nothing special."
        ]
    })
    
    # Test basic stats
    stats = nlp_utils.basic_text_stats(df, text_cols=["text"])
    assert stats.loc["text", "total_texts"] == 3
    
    # Test cleaning
    cleaned = nlp_utils.advanced_text_clean(df, text_cols=["text"])
    assert len(cleaned) == 3
    
    # Test features
    features = nlp_utils.extract_text_features(df, text_cols=["text"])
    assert len(features) == 3
    
    # Test vocabulary
    vocab = nlp_utils.generate_vocabulary(df, text_col="text")
    assert len(vocab) > 0
    
    # Test keywords
    keywords = nlp_utils.extract_keywords(df, text_col="text", top_n=5)
    assert len(keywords) > 0


def test_empty_dataframe_handling():
    """Test that functions handle empty DataFrames gracefully."""
    empty_df = pd.DataFrame({"text": []})
    
    # basic_text_stats with empty df
    stats = nlp_utils.basic_text_stats(empty_df, text_cols=["text"])
    assert isinstance(stats, pd.DataFrame)
    
    # advanced_text_clean with empty df
    cleaned = nlp_utils.advanced_text_clean(empty_df, text_cols=["text"])
    assert len(cleaned) == 0
    
    # extract_text_features with empty df
    features = nlp_utils.extract_text_features(empty_df, text_cols=["text"])
    assert len(features) == 0
    
    # generate_vocabulary with empty df
    vocab = nlp_utils.generate_vocabulary(empty_df, text_col="text")
    assert vocab == []
    
    # extract_keywords with empty df
    keywords = nlp_utils.extract_keywords(empty_df, text_col="text", top_n=5)
    assert isinstance(keywords, pd.DataFrame)


def test_special_characters_handling():
    """Test that functions handle special characters correctly."""
    df = pd.DataFrame({
        "text": [
            "Hello!!! How are you???",
            "Special chars: @#$%^&*()",
            "Mixed content: abc123!@#"
        ]
    })
    
    features = nlp_utils.extract_text_features(df, text_cols=["text"])
    assert "text_exclamation_count" in features.columns
    assert features["text_exclamation_count"].iloc[0] == 3
    assert features["text_question_count"].iloc[0] == 3
