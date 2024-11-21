import os
import numpy as np
import cv2
import albumentations as A
from typing import List, Union
from PIL import Image

class AlzheimerImagePreprocessor:
    def __init__(self, input_shape=(224, 224, 3), data_dir=None):
        """
        Initialize image preprocessor for Alzheimer's detection
        
        Args:
            input_shape: Target input shape for neural network
            data_dir: Optional directory for data loading
        """
        self.input_shape = input_shape
        self.data_dir = data_dir
        self.augmentation_pipeline = self._create_augmentation_pipeline()
    
    def _create_augmentation_pipeline(self):
        """
        Create a robust image augmentation pipeline
        
        Returns:
            Albumentations composition of augmentations
        """
        return A.Compose([
            # Geometric transformations
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            # Spatial-level transforms
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=15, 
                p=0.5
            ),
            
            # Intensity transformations
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.3
            ),
            
            # Color space transformations
            A.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2, 
                hue=0.2, 
                p=0.3
            ),
            
            # Noise augmentations
            A.GaussNoise(var_limit=(0.01, 0.1), p=0.3),
        ])
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess a single image
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Preprocessed image as numpy array
        """
        # Read image using OpenCV
        image = cv2.imread(image_path)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target input shape
        image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def load_images_from_directory(self, directory: str, label: int) -> List[dict]:
        """
        Load images from a directory with their corresponding label
        
        Args:
            directory: Path to image directory
            label: Class label for images in this directory
        
        Returns:
            List of dictionaries containing image and label
        """
        images = []
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(directory, filename)
                try:
                    image = self.load_image(image_path)
                    images.append({
                        'image': image,
                        'label': label
                    })
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        return images
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to a single image
        
        Args:
            image: Input image numpy array
        
        Returns:
            Augmented image
        """
        # Apply augmentation pipeline
        augmented = self.augmentation_pipeline(image=image)['image']
        
        return augmented
    
    def generate_augmented_batch(self, images: List[np.ndarray], num_augmentations: int = 5) -> List[np.ndarray]:
        """
        Generate multiple augmented versions of input images
        
        Args:
            images: List of input images
            num_augmentations: Number of augmentations per image
        
        Returns:
            List of augmented images
        """
        augmented_batch = []
        for image in images:
            # Generate multiple augmentations
            image_augmentations = [self.augment_image(image) for _ in range(num_augmentations)]
            augmented_batch.extend(image_augmentations)
        
        return augmented_batch
    
    def prepare_dataset(self, 
                        alzheimers_dir: str, 
                        normal_dir: str, 
                        test_split: float = 0.2, 
                        augment: bool = True) -> tuple:
        """
        Prepare complete dataset with train/test split and optional augmentation
        
        Args:
            alzheimers_dir: Directory with Alzheimer's images
            normal_dir: Directory with normal brain images
            test_split: Proportion of data to use for testing
            augment: Whether to apply data augmentation
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        # Load images
        alzheimers_images = self.load_images_from_directory(alzheimers_dir, label=1)
        normal_images = self.load_images_from_directory(normal_dir, label=0)
        
        # Combine and shuffle datasets
        all_images = alzheimers_images + normal_images
        np.random.shuffle(all_images)
        
        # Separate images and labels
        X = np.array([item['image'] for item in all_images])
        y = np.array([item['label'] for item in all_images])
        
        # Optional augmentation
        if augment:
            augmented_images = self.generate_augmented_batch(X)
            X = np.concatenate([X, np.array(augmented_images)])
            y = np.concatenate([y, np.repeat(y, len(augmented_images) // len(y))])
        
        # Split into train and test sets
        split_idx = int(len(X) * (1 - test_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, y_train, X_test, y_test

# Example usage
if __name__ == "__main__":
    preprocessor = AlzheimerImagePreprocessor()
    X_train, y_train, X_test, y_test = preprocessor.prepare_dataset(
        alzheimers_dir='/path/to/alzheimers/images',
        normal_dir='/path/to/normal/images'
    )
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
