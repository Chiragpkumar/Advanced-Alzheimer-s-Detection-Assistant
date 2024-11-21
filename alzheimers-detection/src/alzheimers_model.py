import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
import mlflow
import mlflow.keras
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
from typing import Tuple, List, Dict, Any
import logging
import os
import ssl
import certifi

# Import custom model utilities
from .model_utils import load_efficientnet_weights

# Configure SSL context with certifi
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl.get_default_https_context = lambda: ssl_context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlzheimerDetectionModel:
    def __init__(self, input_shape: Tuple[int, ...], num_classes: int = 2):
        """
        Initialize Alzheimer's Detection Model for 2D image classification
        
        Args:
            input_shape: Shape of input image
            num_classes: Number of classification categories
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        try:
            self.model = self._build_model()
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            # Fallback to a simple model if EfficientNetB0 fails
            self.model = self._build_fallback_model()
        
        self.history = None
    
    def _build_model(self) -> keras.Model:
        """
        Build a transfer learning-based CNN for Alzheimer's detection
        
        Returns:
            Compiled Keras model
        """
        # Use custom weight loading utility
        base_model = load_efficientnet_weights(self.input_shape)
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create and compile the model
        model = keras.Model(inputs=base_model.input, outputs=output)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                     keras.metrics.Precision(), 
                     keras.metrics.Recall()]
        )
        
        return model
    
    def _build_fallback_model(self) -> keras.Model:
        """
        Build a simple CNN model as a fallback if transfer learning fails
        
        Returns:
            Compiled Keras model with basic architecture
        """
        logger.warning("Using fallback model architecture")
        
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                     keras.metrics.Precision(), 
                     keras.metrics.Recall()]
        )
        
        return model
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray, 
              X_val: np.ndarray, 
              y_val: np.ndarray,
              epochs: int = 50, 
              batch_size: int = 32) -> keras.callbacks.History:
        """
        Train the Alzheimer's detection model
        
        Args:
            X_train: Training image data
            y_train: Training labels
            X_val: Validation image data
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Training batch size
        
        Returns:
            Training history
        """
        # Convert labels to categorical
        y_train_cat = keras.utils.to_categorical(y_train, num_classes=self.num_classes)
        y_val_cat = keras.utils.to_categorical(y_val, num_classes=self.num_classes)
        
        # Early stopping and model checkpoint
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            'best_alzheimer_model.h5', 
            monitor='val_accuracy', 
            save_best_only=True
        )
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint]
        )
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Input images
        
        Returns:
            Prediction probabilities
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test image data
            y_test: Test labels
        
        Returns:
            Evaluation metrics
        """
        # Convert labels to categorical
        y_test_cat = keras.utils.to_categorical(y_test, num_classes=self.num_classes)
        
        # Evaluate model
        results = self.model.evaluate(X_test, y_test_cat)
        
        return {
            'loss': results[0],
            'accuracy': results[1],
            'precision': results[2],
            'recall': results[3]
        }
    
    def plot_model_performance(self, X_test, y_test, output_dir='results'):
        """
        Generate and save model performance visualization plots
        
        Args:
            X_test: Test image data
            y_test: Test labels
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Confusion Matrix
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test[:, 1], y_pred[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()

def main():
    """Example usage with advanced features"""
    # Initialize model with advanced architecture
    input_shape = (256, 256, 3)
    model = AlzheimerDetectionModel(input_shape, num_classes=2)
    
    # Generate sample data
    X_train = np.random.rand(100, 256, 256, 3)
    y_train = np.random.randint(2, size=(100, 1))
    X_val = np.random.rand(20, 256, 256, 3)
    y_val = np.random.randint(2, size=(20, 1))
    
    # Train with advanced features
    history = model.train(X_train, y_train, X_val, y_val)
    
    # Generate visualizations
    # model.plot_training_metrics()
    # model.plot_confusion_matrix(y_val, model.predict(X_val))
    # model.plot_roc_curve(y_val, model.predict(X_val))
    
    # Uncertainty estimation
    # mean_pred, std_pred = model.predict_with_uncertainty(X_val[:5])
    # logger.info(f"Prediction mean: {mean_pred}")
    # logger.info(f"Prediction uncertainty: {std_pred}")

if __name__ == "__main__":
    main()
