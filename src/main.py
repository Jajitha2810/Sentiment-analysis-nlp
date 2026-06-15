import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
from preprocess import TextPreprocessor
from model import SentimentAnalyzer
from visualize import SentimentVisualizer

def load_imdb_data(vocab_size=10000):
    """Load and prepare IMDB dataset"""
    print("Loading IMDB dataset...")
    (X_train, y_train), (X_test, y_test) = imdb.load_data(
        num_words=vocab_size
    )
    
    # Convert binary to 3-class sentiment
    # 0-3 = Negative, 4-6 = Neutral, 7-10 = Positive
    def convert_rating(rating):
        if rating <= 3:
            return 0  # Negative
        elif rating <= 6:
            return 1  # Neutral
        else:
            return 2  # Positive
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    return (X_train, y_train), (X_test, y_test)


def main():
    """Main training and evaluation pipeline"""
    print("="*60)
    print("SENTIMENT ANALYSIS - DEEP LEARNING FOR NLP")
    print("Author: Jajitha")
    print("="*60)
    
    # Configuration
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 128
    MAX_LENGTH = 200
    EPOCHS = 10
    BATCH_SIZE = 32
    
    # Load data
    (X_train, y_train), (X_test, y_test) = load_imdb_data(VOCAB_SIZE)
    
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)
    
    # Initialize components
    analyzer = SentimentAnalyzer(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        max_length=MAX_LENGTH
    )
    visualizer = SentimentVisualizer()
    
    # Display model architecture
    print("\nModel Architecture:")
    analyzer.model.summary()
    
    # Train model
    print("\nTraining model...")
    history = analyzer.train(
        X_train, y_train_cat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    results = analyzer.evaluate(X_test, y_test_cat)
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    
    # Generate predictions
    y_pred = np.argmax(
        analyzer.model.predict(X_test), 
        axis=1
    )
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualizer.plot_training_history(history)
    visualizer.plot_confusion_matrix(y_test, y_pred)
    visualizer.plot_sentiment_distribution(y_pred)
    visualizer.print_classification_report(y_test, y_pred)
    
    # Save model
    analyzer.save_model('models/sentiment_model.h5')
    
    # Test with custom input
    print("\n" + "="*60)
    print("CUSTOM PREDICTION EXAMPLE")
    print("="*60)
    
    sample_reviews = [
        "This movie was absolutely brilliant! Outstanding performance!",
        "Worst film I have ever seen. Complete disaster.",
        "It was decent enough, had some good and bad moments."
    ]
    
    preprocessor = TextPreprocessor(VOCAB_SIZE, MAX_LENGTH)
    processed = preprocessor.prepare_data(sample_reviews, fit=True)
    
    print("\nSample Predictions:")
    for review, sequence in zip(sample_reviews, processed):
        sequence = sequence.reshape(1, -1)
        result = analyzer.predict(sequence)
        print(f"\nReview: {review[:50]}...")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()
