import os
import requests
import ssl
import urllib3
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0

# Disable SSL warnings and verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

def find_local_weights(filename='efficientnetb0_notop.h5'):
    """
    Search for weights file in potential local directories
    
    Args:
        filename: Name of the weights file
    
    Returns:
        str: Full path to weights file if found, None otherwise
    """
    # Potential directories to search
    search_dirs = [
        os.path.join(os.path.dirname(__file__), '..', 'data'),  # Project data folder
        os.path.join(os.path.dirname(__file__), 'data'),        # Local data folder
        os.path.expanduser('~/.keras/models'),                 # Keras default cache
        os.getcwd()                                            # Current working directory
    ]
    
    for directory in search_dirs:
        weights_path = os.path.join(directory, filename)
        if os.path.exists(weights_path):
            print(f"Found weights file at: {weights_path}")
            return weights_path
    
    print(f"Weights file {filename} not found in searched directories")
    return None

def download_efficientnet_weights(cache_dir=None):
    """
    Manually download EfficientNetB0 weights with SSL bypass
    
    Args:
        cache_dir: Directory to save weights. Defaults to Keras default cache.
    
    Returns:
        str: Path to downloaded weights file
    """
    # First, check for local weights
    local_weights = find_local_weights()
    if local_weights:
        return local_weights
    
    if cache_dir is None:
        cache_dir = os.path.expanduser('~/.keras/models')
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # EfficientNetB0 weights URL
    weights_url = "https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5"
    weights_filename = "efficientnetb0_notop.h5"
    weights_path = os.path.join(cache_dir, weights_filename)
    
    # Check if weights already exist
    if os.path.exists(weights_path):
        return weights_path
    
    try:
        # Download weights with SSL verification disabled
        print(f"Downloading EfficientNetB0 weights to {weights_path}")
        response = requests.get(
            weights_url, 
            stream=True, 
            verify=False  # Disable SSL verification
        )
        response.raise_for_status()
        
        with open(weights_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return weights_path
    
    except Exception as e:
        print(f"Failed to download weights: {e}")
        return None

def load_efficientnet_weights(input_shape):
    """
    Load EfficientNetB0 weights with multiple fallback strategies
    
    Args:
        input_shape: Shape of input image
    
    Returns:
        Loaded EfficientNetB0 model
    """
    # Try loading from local weights first
    local_weights = find_local_weights()
    
    try:
        if local_weights:
            # Try loading with local weights file
            model = EfficientNetB0(
                weights=local_weights, 
                include_top=False, 
                input_shape=input_shape
            )
            print(f"Loaded weights from local file: {local_weights}")
            return model
    except Exception as e:
        print(f"Failed to load local weights: {e}")
    
    # List of weight loading strategies
    weight_strategies = [
        'imagenet',  # Official imagenet weights
        None,        # Random initialization
    ]
    
    for strategy in weight_strategies:
        try:
            model = EfficientNetB0(
                weights=strategy, 
                include_top=False, 
                input_shape=input_shape
            )
            print(f"Loaded weights using strategy: {strategy}")
            return model
        except Exception as e:
            print(f"Failed to load weights with strategy {strategy}: {e}")
    
    # Absolute fallback to random initialization
    return EfficientNetB0(
        weights=None, 
        include_top=False, 
        input_shape=input_shape
    )
