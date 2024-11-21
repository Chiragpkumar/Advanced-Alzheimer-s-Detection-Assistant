import numpy as np
import albumentations as A

class MedicalImageAugmentation:
    def __init__(self, input_shape=(128, 128, 128, 1)):
        """
        Initialize medical image augmentation techniques
        
        Args:
            input_shape: Shape of input medical images
        """
        self.input_shape = input_shape
        self.augmentation_pipeline = self._create_augmentation_pipeline()
    
    def _create_augmentation_pipeline(self):
        """
        Create a robust medical image augmentation pipeline
        
        Returns:
            Albumentations composition of augmentations
        """
        return A.Compose([
            # 3D-compatible augmentations
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            # Intensity transformations
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.3
            ),
            
            # Noise augmentations
            A.GaussNoise(var_limit=(0.01, 0.1), p=0.3),
        ])
    
    def augment(self, image):
        """
        Apply augmentations to medical image
        
        Args:
            image: Input medical image numpy array
        
        Returns:
            Augmented image
        """
        # Ensure image is 3D
        if len(image.shape) == 4:
            augmented_slices = []
            for slice_idx in range(image.shape[0]):
                # Extract 2D slice and convert to uint8
                slice_2d = image[slice_idx, :, :, 0]
                slice_2d = (slice_2d * 255).astype(np.uint8)
                
                # Apply augmentation
                augmented_slice = self.augmentation_pipeline(image=slice_2d)['image']
                
                # Normalize back to original scale
                augmented_slice = augmented_slice.astype(np.float32) / 255.0
                augmented_slices.append(augmented_slice)
            
            # Reconstruct 3D volume
            augmented_image = np.stack(augmented_slices, axis=0)
            augmented_image = np.expand_dims(augmented_image, axis=-1)
            
            return augmented_image
        
        return image
    
    def generate_augmented_batch(self, images, num_augmentations=5):
        """
        Generate multiple augmented versions of input images
        
        Args:
            images: Batch of input medical images
            num_augmentations: Number of augmentations per image
        
        Returns:
            Augmented image batch
        """
        augmented_batch = []
        for image in images:
            image_augmentations = [self.augment(image) for _ in range(num_augmentations)]
            augmented_batch.extend(image_augmentations)
        
        return np.array(augmented_batch)
