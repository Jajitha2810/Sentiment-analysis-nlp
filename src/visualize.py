import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

class SentimentVisualizer:
    """
    Visualization tools for Sentiment Analysis Results
    
    Author: Jajitha
    Course: Deep Learning for NLP using Python
    """
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        self.labels = ['Negative', 'Neutral', 'Positive']
    
    def plot_training_history(self, history):
        """Plot model training accuracy and loss"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        axes[0].plot(
            history.history['accuracy'],
            color=self.colors[2],
            linewidth=2,
            label='Training Accuracy'
        )
        axes[0].plot(
            history.history['val_accuracy'],
            color=self.colors[0],
            linewidth=2,
            linestyle='--',
            label='Validation Accuracy'
        )
        axes[0].set_title(
            'Model Accuracy Over Epochs',
            fontsize=14,
            fontweight='bold'
        )
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(
            history.history['loss'],
            color=self.colors[2],
            linewidth=2,
            label='Training Loss'
        )
        axes[1].plot(
            history.history['val_loss'],
            color=self.colors[0],
            linewidth=2,
            linestyle='--',
            label='Validation Loss'
        )
        axes[1].set_title(
            'Model Loss Over Epochs',
            fontsize=14,
            fontweight='bold'
        )
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=150)
        plt.show()
        print("Training history plot saved!")
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix heatmap"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.labels,
            yticklabels=self.labels
        )
        plt.title(
            'Confusion Matrix',
            fontsize=14,
            fontweight='bold'
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png', dpi=150)
        plt.show()
        print("Confusion matrix saved!")
    
    def plot_sentiment_distribution(self, predictions):
        """Plot distribution of predicted sentiments"""
        unique, counts = np.unique(predictions, return_counts=True)
        percentages = counts / counts.sum() * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar chart
        bars = axes[0].bar(
            self.labels,
            percentages,
            color=self.colors,
            edgecolor='white',
            linewidth=1.5
        )
        axes[0].set_title(
            'Sentiment Distribution',
            fontsize=14,
            fontweight='bold'
        )
        axes[0].set_ylabel('Percentage (%)')
        
        for bar, pct in zip(bars, percentages):
            axes[0].text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 0.5,
                f'{pct:.1f}%',
                ha='center',
                fontweight='bold'
            )
        
        # Pie chart
        axes[1].pie(
            percentages,
            labels=self.labels,
            colors=self.colors,
            autopct='%1.1f%%',
            startangle=90
        )
        axes[1].set_title(
            'Sentiment Breakdown',
            fontsize=14,
            fontweight='bold'
        )
        
        plt.tight_layout()
        plt.savefig('results/sentiment_distribution.png', dpi=150)
        plt.show()
        print("Sentiment distribution plot saved!")
    
    def print_classification_report(self, y_true, y_pred):
        """Print detailed classification metrics"""
        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(
            y_true, y_pred,
            target_names=self.labels
        ))


if __name__ == "__main__":
    visualizer = SentimentVisualizer()
    
    # Example with dummy data
    sample_predictions = np.random.choice([0, 1, 2], size=100)
    sample_true = np.random.choice([0, 1, 2], size=100)
    
    print("Visualizer initialized successfully!")
    print("Ready to plot sentiment analysis results!")
