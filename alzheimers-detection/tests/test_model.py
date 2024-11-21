import pytest
import numpy as np
import tensorflow as tf
from src.alzheimers_model import AlzheimerDetectionModel
from src.data_preprocessing import AlzheimerDataPreprocessor

class TestAlzheimerDetectionModel:
    @pytest.fixture
    def model(self):
        """
        Create a model instance for testing
        """
        input_shape = (128, 128, 128, 1)
        return AlzheimerDetectionModel(input_shape)
    
    @pytest.fixture
    def sample_data(self):
        """
        Generate sample MRI data for testing
        """
        # Simulated 3D MRI scan data
        X_train = np.random.rand(100, 128, 128, 128, 1)
        y_train = tf.keras.utils.to_categorical(
            np.random.randint(2, size=(100, 1)), 
            num_classes=2
        )
        
        X_val = np.random.rand(20, 128, 128, 128, 1)
        y_val = tf.keras.utils.to_categorical(
            np.random.randint(2, size=(20, 1)), 
            num_classes=2
        )
        
        return X_train, y_train, X_val, y_val
    
    def test_model_creation(self, model):
        """
        Test model creation and basic properties
        """
        assert model is not None, "Model should be created successfully"
        assert model.model is not None, "Keras model should be initialized"
        assert len(model.model.layers) > 0, "Model should have layers"
    
    def test_model_training(self, model, sample_data):
        """
        Test model training process
        """
        X_train, y_train, X_val, y_val = sample_data
        
        # Train the model
        history = model.train(X_train, y_train, X_val, y_val, epochs=2)
        
        # Check training history
        assert 'accuracy' in history.history, "Training should track accuracy"
        assert 'val_accuracy' in history.history, "Validation accuracy should be tracked"
        
        # Check model performance
        assert len(history.history['accuracy']) > 0, "Training should occur"
    
    def test_model_prediction(self, model, sample_data):
        """
        Test model prediction capabilities
        """
        X_train, y_train, X_val, y_val = sample_data
        
        # Train the model first
        model.train(X_train, y_train, X_val, y_val, epochs=2)
        
        # Test prediction
        prediction = model.predict(X_val[:5])
        
        assert prediction.shape[0] == 5, "Prediction should match input size"
        assert prediction.shape[1] == 2, "Prediction should have 2 classes"
        assert np.all(prediction >= 0) and np.all(prediction <= 1), "Predictions should be probabilities"
    
    def test_model_evaluation(self, model, sample_data):
        """
        Test model evaluation metrics
        """
        X_train, y_train, X_val, y_val = sample_data
        
        # Train the model
        model.train(X_train, y_train, X_val, y_val, epochs=2)
        
        # Evaluate the model
        test_loss, test_accuracy, test_auc = model.evaluate(X_val, y_val)
        
        assert test_loss is not None, "Test loss should be calculated"
        assert 0 <= test_accuracy <= 1, "Accuracy should be between 0 and 1"
        assert 0 <= test_auc <= 1, "AUC should be between 0 and 1"

class TestDataPreprocessor:
    @pytest.fixture
    def preprocessor(self):
        """
        Create a preprocessor instance for testing
        """
        return AlzheimerDataPreprocessor('/test/data/path')
    
    def test_data_augmentation(self, preprocessor):
        """
        Test MRI data augmentation
        """
        # Simulated MRI data
        mock_mri = np.random.rand(128, 128, 128)
        
        augmented_images = preprocessor.augment_mri_data(mock_mri)
        
        assert len(augmented_images) == 5, "Should generate 5 augmented images"
        assert all(img.shape == mock_mri.shape for img in augmented_images), "Augmented images should maintain original shape"
    
    def test_cognitive_data_preprocessing(self, preprocessor):
        """
        Test cognitive data preprocessing
        """
        import pandas as pd
        
        # Create mock cognitive assessment data
        mock_data = pd.DataFrame({
            'memory_score': [80, 75, 90],
            'orientation_score': [85, 70, 95],
            'attention_score': [90, 80, 85],
            'language_score': [75, 85, 95]
        })
        
        preprocessed_data = preprocessor.preprocess_cognitive_data(mock_data)
        
        assert preprocessed_data.shape == (3, 4), "Preprocessed data should maintain original number of samples"
        assert np.isclose(preprocessed_data.mean(), 0, atol=1e-7), "Scaled data should have near-zero mean"
        assert np.isclose(preprocessed_data.std(), 1, atol=1e-7), "Scaled data should have unit variance"

def main():
    """
    Run tests
    """
    pytest.main([__file__])

if __name__ == "__main__":
    main()
