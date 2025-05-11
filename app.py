# medical_anomaly_detector/app.py
import streamlit as st
from PIL import Image
import io
import os
import numpy as np # For np.nan checks
from typing import Dict # MODIFICATION: Import Dict for type hinting

# Assuming pipeline.py is in the same directory or accessible via PYTHONPATH
from pipeline import run_analysis_pipeline, GraphState # Import run_pipeline and GraphState for type hinting

st.set_page_config(layout="wide", page_title="Medical Anomaly Detector")
st.title("⚕️ Medical Image Anomaly Detector")
st.markdown("Upload a Lung or Breast image to check for anomalies using VQ-VAE and Mahalanobis distance.")

# --- Dummy Asset Creation (for first run if assets are missing) ---
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
DUMMY_FILES_TO_CHECK = [
    os.path.join(ASSETS_DIR, "lung_vqvae_model.pth"),
    os.path.join(ASSETS_DIR, "breast_vqvae_model.pth"),
    os.path.join(ASSETS_DIR, "lung_mahalanobis_scorer_params.npz"),
    os.path.join(ASSETS_DIR, "breast_mahalanobis_scorer_params.npz"),
    os.path.join(ASSETS_DIR, "lung_thresholds.json"),
    os.path.join(ASSETS_DIR, "breast_thresholds.json")
]
os.makedirs(ASSETS_DIR, exist_ok=True)
for dummy_file_path in DUMMY_FILES_TO_CHECK:
    if not os.path.exists(dummy_file_path):
        try:
            with open(dummy_file_path, 'w') as f:
                if dummy_file_path.endswith(".json"):
                    f.write('{"hybrid_score": 0.5, "image_mse": 0.01, "max_patch_mse": 0.05, "mahalanobis_distance": 50.0}')
                else:
                    f.write("This is a DUMMY placeholder file. Replace with your actual trained asset.")
            print(f"Created dummy file: {dummy_file_path}")
        except IOError as e:
            print(f"Could not create dummy file {dummy_file_path}: {e}")
            try:
                st.error(f"Could not create dummy asset file: {os.path.basename(dummy_file_path)}. Please check permissions for '{ASSETS_DIR}'.")
            except Exception:
                pass


# --- Sidebar for Upload and Controls ---
st.sidebar.header("Image Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
user_selected_type = st.sidebar.selectbox(
    "Select Image Type:",
    ("Lung", "Breast"), 
    help="Select the type of medical image you are uploading."
)

if 'pipeline_state_result' not in st.session_state:
    st.session_state.pipeline_state_result = None

if st.sidebar.button("⚡ Analyze Image", type="primary", use_container_width=True):
    if uploaded_file is not None:
        st.session_state.pipeline_state_result = None 
        image_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name
        
        with st.spinner(f"Analyzing {filename} as {user_selected_type}... This may take a moment."):
            try:
                final_graph_state: GraphState = run_analysis_pipeline(image_bytes, filename, user_selected_type)
                st.session_state.pipeline_state_result = final_graph_state
                st.success("Analysis complete!")
            except Exception as e:
                st.error(f"An error occurred during pipeline execution: {e}")
                st.exception(e) 
                st.session_state.pipeline_state_result = {"error_message": str(e), "image_filename": filename, "user_selected_type": user_selected_type}
    else:
        st.sidebar.warning("Please upload an image first.")

# --- Main Area for Displaying Results ---
if st.session_state.pipeline_state_result:
    state_result: Dict = st.session_state.pipeline_state_result # Using imported Dict
    filename_display = state_result.get("image_filename", "Uploaded Image")
    
    st.header(f"Results for: {filename_display}")

    if state_result.get("error_message"):
        st.error(f"Processing Error: {state_result['error_message']}")
    else:
        original_pil_image = state_result.get("original_pil") 
        heatmap_pil_image = state_result.get("heatmap_pil")
        is_anomaly = state_result.get("is_anomaly", False)
        
        col1, col2 = st.columns(2)
        with col1:
            if original_pil_image:
                st.image(original_pil_image, caption="Original Image", use_column_width=True)
            else:
                if uploaded_file: 
                     st.image(uploaded_file, caption="Original Image (from upload)", use_column_width=True)
                else:
                     st.warning("Original image preview not available from pipeline state.")
        
        with col2:
            st.subheader("Analysis Summary")
            st.write(f"**Determined Image Type:** {state_result.get('determined_type', 'N/A').capitalize()}")
            
            status_color = "red" if is_anomaly else "green"
            status_text = "🚨 ANOMALY DETECTED 🚨" if is_anomaly else "✅ NORMAL ✅"
            st.markdown(f"**Overall Status: <span style='color:{status_color}; font-weight:bold;'>{status_text}</span>**", unsafe_allow_html=True)
            
            hybrid_score_val = state_result.get('hybrid_score')
            threshold_val = state_result.get('actual_threshold_used')

            if hybrid_score_val is not None and not np.isnan(hybrid_score_val) and \
               threshold_val is not None and not np.isnan(threshold_val):
                st.metric(label="Hybrid Anomaly Score", 
                          value=f"{hybrid_score_val:.4f}",
                          delta=f"Threshold Used: {threshold_val:.4f}",
                          delta_color="inverse" if is_anomaly else "normal")
            elif hybrid_score_val is not None and not np.isnan(hybrid_score_val):
                 st.metric(label="Hybrid Anomaly Score", value=f"{hybrid_score_val:.4f}")
            else:
                st.write("Hybrid Score: Not available or NaN")

            with st.expander("Detailed Scores", expanded=False):
                mse = state_result.get('mse_score', np.nan)
                patch_mse = state_result.get('patch_mse_score', np.nan)
                maha = state_result.get('mahalanobis_score', np.nan)

                # Corrected f-string formatting
                formatted_mse = f"{mse:.6f}" if not np.isnan(mse) else "N/A"
                formatted_patch_mse = f"{patch_mse:.6f}" if not np.isnan(patch_mse) else "N/A"
                formatted_maha = f"{maha:.6f}" if not np.isnan(maha) else "N/A"
                
                st.markdown(f"- **Image MSE:** `{formatted_mse}`")
                st.markdown(f"- **Max Patch MSE:** `{formatted_patch_mse}`")
                st.markdown(f"- **Mahalanobis Distance:** `{formatted_maha}`")

            if is_anomaly and heatmap_pil_image:
                st.image(heatmap_pil_image, caption="Anomaly Heatmap", use_column_width=True)
            elif is_anomaly:
                st.info("Anomaly detected, but heatmap is not available for display (check pipeline's generate_heatmap_node).")
else:
    st.info("⬆️ Upload an image and click 'Analyze Image' to begin.")

st.sidebar.markdown("---")
st.sidebar.info("This application uses a VQ-VAE based pipeline to detect anomalies in medical images. Ensure 'assets/' folder contains your trained models, scorer parameters, and threshold files for actual functionality.")
