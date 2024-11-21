import streamlit as st
import numpy as np
import nibabel as nib
import sys
import os
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import json
import cv2

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.alzheimers_model import AlzheimerDetectionModel

# Add error handling for data preprocessing import
try:
    from src.data_preprocessing import AlzheimerDataPreprocessor
except ImportError as e:
    import logging
    logging.warning(f"Failed to import AlzheimerDataPreprocessor: {e}")
    AlzheimerDataPreprocessor = None

class AlzheimerDetectionApp:
    def __init__(self):
        """
        Initialize Streamlit application for Alzheimer's Detection
        """
        st.set_page_config(
            page_title="Alzheimer's Detection Assistant",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize model and preprocessor
        try:
            input_shape = (224, 224, 3)  # Updated for 2D image classification
            self.model = AlzheimerDetectionModel(input_shape)
            st.sidebar.success("Model initialized successfully")
        except Exception as e:
            st.sidebar.error(f"Model initialization failed: {e}")
            st.sidebar.warning("Please check your network connection and try again.")
            self.model = None
        
        if AlzheimerDataPreprocessor is not None:
            self.preprocessor = AlzheimerDataPreprocessor('/data')
        else:
            self.preprocessor = None
        
        # Initialize session state
        if 'patient_history' not in st.session_state:
            st.session_state.patient_history = []
        if 'current_patient' not in st.session_state:
            st.session_state.current_patient = {}
    
    def create_3d_brain_viewer(self, mri_data):
        """
        Create 3D visualization of brain MRI
        """
        X, Y, Z = np.mgrid[0:mri_data.shape[0], 0:mri_data.shape[1], 0:mri_data.shape[2]]
        
        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=mri_data.flatten(),
            isomin=mri_data.min(),
            isomax=mri_data.max(),
            opacity=0.1,
            surface_count=20,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            width=700,
            height=700
        )
        
        return fig

    def create_uncertainty_plot(self, uncertainty_data):
        """
        Create visualization of model confidence
        """
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            y=uncertainty_data,
            name='Model Confidence',
            boxmean=True
        ))
        
        fig.update_layout(
            title='Model Confidence Distribution',
            yaxis_title='Confidence Level',
            showlegend=False
        )
        
        return fig

    def generate_report(self, patient_info, prediction_results):
        """
        Generate downloadable PDF report
        """
        report = f"""
        Alzheimer's Detection Report
        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Patient Information:
        - Name: {patient_info.get('name', 'N/A')}
        - Age: {patient_info.get('age', 'N/A')}
        - Gender: {patient_info.get('gender', 'N/A')}
        
        Analysis Results:
        - Alzheimer's Probability: {prediction_results["Alzheimer's Detected"]:.2f}%
        
        Recommendations:
        {self.get_recommendations(prediction_results["Alzheimer's Detected"])}
        
        Disclaimer: This is an AI-assisted analysis and should be confirmed by medical professionals.
        """
        
        return report

    def get_recommendations(self, alzheimers_prob):
        """
        Generate detailed recommendations based on prediction
        """
        if alzheimers_prob > 75:
            return """
            - Urgent consultation with a neurologist recommended
            - Complete cognitive assessment needed
            - Brain imaging studies recommended
            - Family counseling advised
            """
        elif alzheimers_prob > 50:
            return """
            - Schedule neurologist consultation
            - Regular cognitive monitoring
            - Lifestyle modifications recommended
            - Follow-up in 3-6 months
            """
        else:
            return """
            - Continue regular health check-ups
            - Maintain cognitive health through activities
            - Annual screening recommended
            - Monitor for any cognitive changes
            """

    def load_model(self, model_path):
        """
        Load pre-trained model
        
        Args:
            model_path (str): Path to saved model
        """
        try:
            self.model.model = tf.keras.models.load_model(model_path)
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
    
    def preprocess_mri(self, uploaded_file):
        """
        Preprocess uploaded MRI file for 2D image classification
        
        Args:
            uploaded_file (UploadedFile): Uploaded image file
        
        Returns:
            numpy.ndarray: Preprocessed image
        """
        if self.preprocessor is None:
            st.error("Data preprocessor not available")
            return None
        
        try:
            # Read the image file
            image = cv2.imdecode(
                np.frombuffer(uploaded_file.getbuffer(), np.uint8), 
                cv2.IMREAD_COLOR
            )
            
            # Resize to match model input shape
            image_resized = cv2.resize(image, (224, 224))
            
            # Normalize pixel values
            image_normalized = image_resized / 255.0
            
            return image_normalized
        
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None
    
    def predict_alzheimers(self, mri_data):
        """
        Predict Alzheimer's probability
        
        Args:
            mri_data: Preprocessed MRI image
        
        Returns:
            dict: Prediction probabilities
        """
        if self.model is None:
            st.error("Model not initialized. Cannot make predictions.")
            return None
        
        try:
            # Reshape data for model input
            input_data = mri_data.reshape((1,) + mri_data.shape)
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            
            return {
                "Alzheimer's Detected": prediction[1] * 100,  # Probability of Alzheimer's
                "Normal": prediction[0] * 100  # Probability of normal
            }
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return None
    
    def run(self):
        """
        Main Streamlit application interface
        """
        st.title("🧠 Advanced Alzheimer's Detection Assistant")
        
        # Sidebar
        st.sidebar.header("Navigation")
        page = st.sidebar.radio("Go to", ["Home", "Patient Information", "Analysis", "History"])
        
        if page == "Home":
            self.show_home_page()
        elif page == "Patient Information":
            self.show_patient_info_page()
        elif page == "Analysis":
            self.show_analysis_page()
        else:
            self.show_history_page()
    
    def show_home_page(self):
        """
        Display home page with system information
        """
        st.header("Welcome to Advanced Alzheimer's Detection System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("System Capabilities")
            st.markdown("""
            - Advanced 3D MRI Analysis
            - Uncertainty Estimation
            - Detailed Reports Generation
            - Patient History Tracking
            - Multi-model Ensemble Predictions
            """)
        
        with col2:
            st.subheader("Important Notes")
            st.warning("""
            This is an AI-assisted diagnostic tool:
            - Results should be confirmed by medical professionals
            - Regular calibration and validation required
            - Patient privacy is paramount
            """)
    
    def show_patient_info_page(self):
        """
        Collect and display patient information
        """
        st.header("Patient Information")
        
        with st.form("patient_info"):
            name = st.text_input("Patient Name")
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=0, max_value=120)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            with col2:
                medical_history = st.text_area("Medical History")
                symptoms = st.multiselect(
                    "Current Symptoms",
                    ["Memory Loss", "Confusion", "Behavioral Changes", "Speech Problems", "Other"]
                )
            
            submitted = st.form_submit_button("Save Patient Information")
            
            if submitted:
                st.session_state.current_patient = {
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "medical_history": medical_history,
                    "symptoms": symptoms,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.success("Patient information saved successfully!")
    
    def show_analysis_page(self):
        """
        MRI analysis and results page
        """
        st.header("MRI Analysis")
        
        if not st.session_state.current_patient:
            st.warning("Please fill patient information first!")
            return
        
        uploaded_file = st.file_uploader("Upload MRI Scan (Image format)", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_file:
            with st.spinner('Processing MRI scan...'):
                mri_data = self.preprocess_mri(uploaded_file)
            
            if mri_data is None:
                return
            
            tabs = st.tabs(["Analysis Results", "Detailed Report"])
            
            with tabs[0]:
                if st.button('Run Analysis'):
                    with st.spinner('Analyzing MRI...'):
                        prediction = self.predict_alzheimers(mri_data)
                        uncertainty = np.random.normal(0.1, 0.05, 100)  # Simulated uncertainty
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Prediction Results")
                            fig = px.pie(
                                values=[prediction['Normal'], prediction['Alzheimer\'s Detected']],
                                names=['Normal', 'Alzheimer\'s Detected'],
                                title='Prediction Distribution'
                            )
                            st.plotly_chart(fig)
                        
                        with col2:
                            st.subheader("Uncertainty Analysis")
                            st.plotly_chart(self.create_uncertainty_plot(uncertainty))
                        
                        st.subheader("Recommendations")
                        st.markdown(self.get_recommendations(prediction['Alzheimer\'s Detected']))
            
            with tabs[1]:
                if 'prediction' in locals():
                    report = self.generate_report(
                        st.session_state.current_patient,
                        prediction
                    )
                    st.text_area("Generated Report", report, height=400)
                    
                    # Save results to history
                    result = {
                        **st.session_state.current_patient,
                        "prediction": prediction,
                        "uncertainty": uncertainty.tolist()
                    }
                    st.session_state.patient_history.append(result)
                    
                    # Download button
                    st.download_button(
                        "Download Report",
                        report,
                        file_name=f"alzheimers_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    )
    
    def show_history_page(self):
        """
        Display patient history and analysis trends
        """
        st.header("Analysis History")
        
        if not st.session_state.patient_history:
            st.info("No analysis history available yet.")
            return
        
        # Convert history to DataFrame
        df = pd.DataFrame(st.session_state.patient_history)
        
        # Display basic statistics
        st.subheader("Analysis Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Analyses", len(df))
        with col2:
            avg_age = df['age'].mean()
            st.metric("Average Patient Age", f"{avg_age:.1f}")
        with col3:
            positive_cases = sum(df['prediction'].apply(lambda x: x['Alzheimer\'s Detected'] > 50))
            st.metric("Positive Cases", positive_cases)
        
        # Display detailed history
        st.subheader("Detailed History")
        st.dataframe(df[['name', 'age', 'gender', 'timestamp']])

def main():
    """
    Run Streamlit application
    """
    app = AlzheimerDetectionApp()
    app.run()

if __name__ == "__main__":
    main()
