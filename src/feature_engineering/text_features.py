import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

class TextFeatureExtractor:
    """Extract TF-IDF features from movie plot summaries and descriptions"""
    
    def __init__(self, max_features=5000, min_df=2, max_df=0.8):
        self.logger = logging.getLogger(__name__)
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),  # Include bigrams
            preprocessor=self.preprocess_text
        )
        
        # Download required NLTK data
        self._download_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
    
    def preprocess_text(self, text):
        """Preprocess text for TF-IDF"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        stop_words = set(stopwords.words('english'))
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def fit_transform(self, movies_df):
        """Fit TF-IDF vectorizer and transform movie descriptions"""
        self.logger.info("Extracting TF-IDF features from movie overviews...")
        
        # Prepare text data
        text_data = movies_df['overview'].fillna('')
        
        # Fit and transform
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_data)
        
        # Get feature names
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        self.logger.info(f"Created TF-IDF matrix: {tfidf_matrix.shape}")
        self.logger.info(f"Top features: {list(feature_names[:10])}")
        
        return tfidf_matrix, feature_names
    
    def transform(self, movies_df):
        """Transform new movie descriptions using fitted vectorizer"""
        text_data = movies_df['overview'].fillna('')
        return self.tfidf_vectorizer.transform(text_data)
    
    def save_vectorizer(self, path):
        """Save fitted vectorizer"""
        with open(path, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        self.logger.info(f"TF-IDF vectorizer saved to {path}")
    
    def load_vectorizer(self, path):
        """Load fitted vectorizer"""
        with open(path, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        self.logger.info(f"TF-IDF vectorizer loaded from {path}")
