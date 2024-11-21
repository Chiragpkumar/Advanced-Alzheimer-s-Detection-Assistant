import pytest
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from frontend.app import AlzheimerDetectionApp

def test_app_initialization():
    """Test that the Alzheimer's Detection App can be initialized"""
    app = AlzheimerDetectionApp()
    assert app is not None, "App initialization failed"

def test_model_initialization():
    """Test that the model can be initialized"""
    app = AlzheimerDetectionApp()
    assert app.model is not None, "Model initialization failed"

def test_critical_imports():
    """Verify critical imports are available"""
    try:
        import streamlit as st
        import numpy as np
        import tensorflow as tf
        import plotly.graph_objects as go
        import nibabel as nib
        import cv2
    except ImportError as e:
        pytest.fail(f"Critical import failed: {e}")

def test_dependencies_installed():
    """Check if all required dependencies are installed"""
    required_packages = [
        'streamlit', 'numpy', 'tensorflow', 'plotly', 
        'nibabel', 'opencv-python', 'pandas', 'seaborn'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            pytest.fail(f"Required package {package} is not installed")

# Add more specific tests as needed
def test_preprocessing_import():
    """Verify data preprocessing module can be imported"""
    try:
        from src.data_preprocessing import AlzheimerDataPreprocessor
        assert AlzheimerDataPreprocessor is not None
    except ImportError:
        pytest.fail("Failed to import AlzheimerDataPreprocessor")
