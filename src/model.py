import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, 
    Dropout, Bidirectional
)
from tensorflow.keras.optimizers import Adam

class SentimentAnalyzer:
    """
    Deep Learning model for Sentiment Analysis
    using Bidirectional LSTM architecture.
    
    Author: Jajitha
    Course: Deep Learning for NLP using Python
    """
    
    def __init__(
        self, 
        vocab_size=10000, 
        embedding_dim=128,
        max_length=200
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.model = self._build_model()
    
    def _build_model(self):
        """Build Bidirectional LSTM model"""
        model = Sequential([
            # Embedding layer
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_length,
                name='embedding_layer'
            ),
            
            # Bidirectional LSTM layers
            Bidirectional(
                LSTM(128, return_sequences=True),
                name='bilstm_1'
            ),
            Dropout(0.3, name='dropout_1'),
            
            Bidirectional(
                LSTM(64, return_sequences=False),
                name='bilstm_2'
            ),
            Dropout(0.3, name='dropout_2'),
            
            # Dense layers
            Dense(64, activation='relu', name='dense_1'),
            Dropout(0.2, name='dropout_3'),
            Dense(3, activation='softmax', name='output')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """Train the sentiment analysis model"""
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )
        return history
    
    def predict(self, text_sequence):
        """
        Predict sentiment of input text
        Returns: positive, negative, or neutral
        """
        labels = ['Negative', 'Neutral', 'Positive']
        prediction = self.model.predict(text_sequence)
        sentiment_idx = np.argmax(prediction, axis=1)
        confidence = np.max(prediction, axis=1)
        
        return {
            'sentiment': labels[sentiment_idx[0]],
            'confidence': float(confidence[0]),
            'probabilities': {
                'negative': float(prediction[0][0]),
                'neutral': float(prediction[0][1]),
                'positive': float(prediction[0][2])
            }
        }
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return {
            'test_loss': loss,
            'test_accuracy': accuracy
        }
    
    def save_model(self, filepath='models/sentiment_model.h5'):
        """Save trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/sentiment_model.h5'):
        """Load pre-trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    analyzer = SentimentAnalyzer(
        vocab_size=10000,
        embedding_dim=128,
        max_length=200
    )
    
    # Display model architecture
    analyzer.model.summary()
    print("\nSentiment Analyzer initialized successfully!")
    print("Ready for training on IMDB dataset.")
