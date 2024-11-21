import numpy as np
import pandas as pd
import nibabel as nib
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, List, Dict, Any, Optional
import logging
from scipy import ndimage
import albumentations as A
from pathlib import Path
import json
import h5py

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add error handling for elasticdeform
try:
    import elasticdeform
except ImportError:
    logger.warning("Elasticdeform library not available. Some augmentation techniques will be limited.")
    elasticdeform = None

class AlzheimerDataPreprocessor:
    def __init__(self, data_path: str, config: Optional[Dict] = None):
        """
        Initialize the data preprocessor with advanced configuration
        
        Args:
            data_path: Path to the medical imaging or assessment data
            config: Configuration dictionary for preprocessing parameters
        """
        self.data_path = Path(data_path)
        self.config = config or self._default_config()
        self.scaler = self._get_scaler()
        self._setup_augmentation()
    
    def _default_config(self) -> Dict:
        """Default configuration for preprocessing"""
        return {
            'scaler_type': 'robust',
            'slice_thickness': 1.0,
            'target_shape': (128, 128, 128),
            'normalization': 'z_score',
            'augmentation': {
                'rotation_range': 15,
                'zoom_range': 0.1,
                'shear_range': 0.1,
                'brightness_range': (0.9, 1.1),
                'elastic_deformation': True
            },
            'artifact_removal': {
                'denoise': True,
                'bias_field_correction': True,
                'skull_stripping': True
            }
        }
    
    def _get_scaler(self) -> Any:
        """Initialize appropriate scaler"""
        if self.config['scaler_type'] == 'robust':
            return RobustScaler()
        return StandardScaler()
    
    def _setup_augmentation(self) -> None:
        """Setup augmentation pipeline"""
        self.aug_pipeline = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1.0, shift_limit=0.5, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.3),
        ])
    
    def load_mri_data(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess MRI data with advanced techniques
        
        Args:
            file_path: Path to NIfTI image file
        
        Returns:
            Preprocessed MRI image data
        """
        try:
            # Load NIfTI image
            nifti_img = nib.load(file_path)
            img_data = nifti_img.get_fdata()
            
            # Apply preprocessing pipeline
            if self.config['artifact_removal']['denoise']:
                img_data = self._denoise_image(img_data)
            
            if self.config['artifact_removal']['bias_field_correction']:
                img_data = self._correct_bias_field(img_data)
            
            if self.config['artifact_removal']['skull_stripping']:
                img_data = self._strip_skull(img_data)
            
            # Normalize
            img_data = self._normalize_image(img_data)
            
            # Resize to target shape
            img_data = self._resize_volume(img_data, self.config['target_shape'])
            
            return img_data
            
        except Exception as e:
            logger.error(f"Error loading MRI data: {e}")
            return None
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Apply advanced denoising"""
        return ndimage.gaussian_filter(image, sigma=1)
    
    def _correct_bias_field(self, image: np.ndarray) -> np.ndarray:
        """
        Correct bias field in MRI
        Note: This is a simplified version. For production, consider using N4ITK
        """
        # Implement bias field correction
        # This is a placeholder for actual implementation
        return image
    
    def _strip_skull(self, image: np.ndarray) -> np.ndarray:
        """
        Remove skull from brain MRI
        Note: This is a simplified version. For production, consider using BET
        """
        # Implement skull stripping
        # This is a placeholder for actual implementation
        return image
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Advanced image normalization"""
        if self.config['normalization'] == 'z_score':
            return (image - image.mean()) / (image.std() + 1e-8)
        elif self.config['normalization'] == 'min_max':
            return (image - image.min()) / (image.max() - image.min() + 1e-8)
        return image
    
    def _resize_volume(self, image: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """Resize 3D volume while maintaining aspect ratio"""
        current_depth = image.shape[0]
        current_height = image.shape[1]
        current_width = image.shape[2]
        
        depth = target_shape[0]
        height = target_shape[1]
        width = target_shape[2]
        
        # Compute depth factor
        depth_factor = depth / current_depth
        width_factor = width / current_width
        height_factor = height / current_height
        
        # Resize across z-axis
        resized_volume = ndimage.zoom(image, (depth_factor, height_factor, width_factor))
        
        return resized_volume
    
    def preprocess_cognitive_data(self, cognitive_df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess cognitive assessment data with advanced techniques
        
        Args:
            cognitive_df: Cognitive assessment dataframe
        
        Returns:
            Preprocessed cognitive features
        """
        # Handle missing values with advanced imputation
        cognitive_df = self._handle_missing_values(cognitive_df)
        
        # Feature engineering
        cognitive_df = self._engineer_features(cognitive_df)
        
        # Select relevant features
        features = [
            'memory_score', 
            'orientation_score', 
            'attention_score', 
            'language_score',
            'executive_function_score',
            'visuospatial_score'
        ]
        
        # Scale features
        scaled_features = self.scaler.fit_transform(cognitive_df[features])
        
        return scaled_features
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced missing value handling"""
        # Implement sophisticated imputation methods
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns
        
        # For numeric columns, use interpolation
        df[numeric_columns] = df[numeric_columns].interpolate(method='cubic')
        
        # For categorical columns, use mode
        for col in categorical_columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering"""
        # Example: Create composite scores
        if all(col in df.columns for col in ['memory_score', 'attention_score']):
            df['cognitive_composite'] = df['memory_score'] * 0.6 + df['attention_score'] * 0.4
        
        # Add more sophisticated feature engineering as needed
        return df
    
    def augment_mri_data(self, mri_image: np.ndarray) -> List[np.ndarray]:
        """
        Advanced data augmentation for MRI images
        
        Args:
            mri_image: Original MRI image
        
        Returns:
            List of augmented MRI images
        """
        return self._augment_mri(mri_image)
    
    def _augment_mri(self, mri_image: np.ndarray) -> List[np.ndarray]:
        """
        Apply advanced data augmentation techniques to MRI image
        
        Args:
            mri_image: Input 3D MRI image
        
        Returns:
            List of augmented images
        """
        augmented_images = [
            mri_image,  # Original image
            np.rot90(mri_image),  # 90-degree rotation
            np.rot90(mri_image, 2),  # 180-degree rotation
            np.fliplr(mri_image),
            np.flipud(mri_image)
        ]
        
        # Elastic deformation
        if self.config['augmentation']['elastic_deformation'] and elasticdeform is not None:
            try:
                deformed_image = elasticdeform.deform_random_grid(
                    mri_image,
                    sigma=2,
                    points=3
                )
                augmented_images.append(deformed_image)
            except Exception as e:
                logger.warning(f"Elastic deformation failed: {e}")
        
        return augmented_images
    
    def save_preprocessed_data(self, data: np.ndarray, metadata: Dict, 
                             output_path: str) -> None:
        """
        Save preprocessed data with metadata
        
        Args:
            data: Preprocessed image data
            metadata: Associated metadata
            output_path: Path to save the data
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            # Save data
            f.create_dataset('data', data=data, compression='gzip')
            
            # Save metadata
            metadata_group = f.create_group('metadata')
            for key, value in metadata.items():
                metadata_group.attrs[key] = value
    
    def load_preprocessed_data(self, file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Load preprocessed data and metadata
        
        Args:
            file_path: Path to preprocessed data file
        
        Returns:
            Tuple of (data, metadata)
        """
        with h5py.File(file_path, 'r') as f:
            data = f['data'][:]
            metadata = dict(f['metadata'].attrs)
        
        return data, metadata

def main():
    """Example usage with advanced features"""
    # Initialize preprocessor with custom config
    config = {
        'scaler_type': 'robust',
        'target_shape': (128, 128, 128),
        'normalization': 'z_score',
        'augmentation': {
            'elastic_deformation': True
        },
        'artifact_removal': {
            'denoise': True,
            'bias_field_correction': True,
            'skull_stripping': True
        }
    }
    
    preprocessor = AlzheimerDataPreprocessor('/path/to/data', config)
    
    # Example: Process MRI data
    mri_data = preprocessor.load_mri_data('/path/to/mri.nii.gz')
    if mri_data is not None:
        # Augment data
        augmented_images = preprocessor.augment_mri_data(mri_data)
        
        # Save preprocessed data
        metadata = {
            'preprocessing_date': '2023-01-01',
            'original_shape': mri_data.shape,
            'normalization': 'z_score'
        }
        preprocessor.save_preprocessed_data(
            mri_data,
            metadata,
            'preprocessed_data.h5'
        )
    
    # Example: Process cognitive data
    cognitive_data = pd.DataFrame({
        'memory_score': [80, 75, 90],
        'orientation_score': [85, 70, 95],
        'attention_score': [90, 80, 85],
        'language_score': [75, 85, 95]
    })
    
    preprocessed_cognitive = preprocessor.preprocess_cognitive_data(cognitive_data)
    logger.info(f"Preprocessed cognitive data shape: {preprocessed_cognitive.shape}")

if __name__ == "__main__":
    main()
