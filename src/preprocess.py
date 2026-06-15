import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    """
    Text Preprocessing Pipeline for Sentiment Analysis
    
    Author: Jajitha
    Course: Deep Learning for NLP using Python
    """
    
    def __init__(self, vocab_size=10000, max_length=200):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and normalize raw text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words
            and len(token) > 2
        ]
        
        return tokens
    
    def preprocess(self, texts):
        """Full preprocessing pipeline"""
        cleaned = [self.clean_text(text) for text in texts]
        tokenized = [
            ' '.join(self.tokenize_and_lemmatize(text)) 
            for text in cleaned
        ]
        return tokenized
    
    def fit_tokenizer(self, texts):
        """Fit tokenizer on training data"""
        self.tokenizer.fit_on_texts(texts)
        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
    
    def texts_to_sequences(self, texts):
        """Convert texts to padded sequences"""
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding='post',
            truncating='post'
        )
        return padded
    
    def prepare_data(self, texts, fit=False):
        """Complete data preparation pipeline"""
        preprocessed = self.preprocess(texts)
        
        if fit:
            self.fit_tokenizer(preprocessed)
        
        sequences = self.texts_to_sequences(preprocessed)
        return sequences


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "This movie was absolutely amazing! I loved every minute of it.",
        "Terrible film. Complete waste of time and money.",
        "It was okay, nothing special but not bad either."
    ]
    
    preprocessor = TextPreprocessor(
        vocab_size=10000,
        max_length=200
    )
    
    processed = preprocessor.prepare_data(sample_texts, fit=True)
    print(f"Processed shape: {processed.shape}")
    print("Preprocessing pipeline working successfully!")
