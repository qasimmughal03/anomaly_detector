# medical_anomaly_detector/app.py
import streamlit as st
from PIL import Image
import io
import os
import numpy as np
from typing import Dict, Optional # Added Optional

# Import pipeline function and constants
from pipeline import run_analysis_pipeline, GraphState
from vqvae_models import LUNG_MODEL_TYPE_ID, BREAST_MODEL_TYPE_ID
# Import utils only if needed directly in app.py (e.g., for tensor_to_pil)
# from utils import tensor_to_pil # Moved import lower

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Medical Anomaly Detector")
st.title("⚕️ Medical Image Anomaly Detector (2-Component Hybrid)")
st.markdown("""
Upload a Lung or Breast image ('png', 'jpg', 'jpeg', 'bmp', 'tif').
Select 'Auto-Detect' (requires Google API Key) or manually specify the type.
The system provides:
- Anomaly status based on Image MSE, Max Patch MSE, and Mahalanobis Distance individually (vs. thresholds).
- A final anomaly decision based on a **Hybrid Score** (weighted combination of scaled Max Patch MSE and scaled Mahalanobis Distance).
- An anomaly heatmap overlay if the final decision is 'Anomaly'.
""")

# --- Asset Directory and Dummy File Creation ---
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# Placeholder content for JSON files (USERS MUST REPLACE WITH ACTUAL GENERATED FILES)
# Use slightly different placeholder values for breast for demonstration
placeholder_runtime_config_lung = """
{
    "model_type_trained": "lung",
    "params": {
        "in_channels": 1, "h_dim": 128, "res_h_dim": 64, "n_res_layers": 2,
        "n_embeddings": 128, "embedding_dim": 32, "beta": 0.25, "out_channels": 1
    },
    "img_size": 256,
    "patch_size_ratio": 16,
    "clahe_params": {"apply_clahe": false, "clip_limit": 2.0, "tile_grid_size": [8, 8]},
    "scaling_parameters_from_training": {
        "mse_min_train": 0.0001, "mse_max_train": 0.01,
        "patch_mse_min_train": 0.001, "patch_mse_max_train": 0.05,
        "mahalanobis_min_train": 1.0, "mahalanobis_max_train": 50.0
    },
    "hybrid_score_weights_training": {
        "patch_mse": 0.6,
        "mahalanobis": 0.4
    }
}"""
placeholder_thresholds_lung = """
{
    "image_mse": 0.005,
    "max_patch_mse": 0.02,
    "mahalanobis_distance": 10.0,
    "hybrid_score": 0.45
}"""

# Dummy Breast Config (adjust params if needed for a real breast model)
placeholder_runtime_config_breast = """
{
    "model_type_trained": "breast",
    "params": {
        "in_channels": 1, "h_dim": 128, "res_h_dim": 64, "n_res_layers": 2,
        "n_embeddings": 128, "embedding_dim": 32, "beta": 0.25, "out_channels": 1
    },
    "img_size": 256,
    "patch_size_ratio": 16,
     "clahe_params": {"apply_clahe": false, "clip_limit": 2.0, "tile_grid_size": [8, 8]},
    "scaling_parameters_from_training": {
        "mse_min_train": 0.0002, "mse_max_train": 0.015,
        "patch_mse_min_train": 0.002, "patch_mse_max_train": 0.06,
        "mahalanobis_min_train": 2.0, "mahalanobis_max_train": 60.0
    },
    "hybrid_score_weights_training": {
        "patch_mse": 0.55,
        "mahalanobis": 0.45
    }
}"""
placeholder_thresholds_breast = """
{
    "image_mse": 0.006,
    "max_patch_mse": 0.025,
    "mahalanobis_distance": 12.0,
    "hybrid_score": 0.50
}"""


# Dictionary mapping TYPE_ID to its dummy files content
dummy_files_map = {
    LUNG_MODEL_TYPE_ID: {
        f"{LUNG_MODEL_TYPE_ID}_model_runtime_config.json": placeholder_runtime_config_lung,
        f"{LUNG_MODEL_TYPE_ID}_thresholds.json": placeholder_thresholds_lung,
    },
    BREAST_MODEL_TYPE_ID: {
        f"{BREAST_MODEL_TYPE_ID}_model_runtime_config.json": placeholder_runtime_config_breast,
        f"{BREAST_MODEL_TYPE_ID}_thresholds.json": placeholder_thresholds_breast,
    }
}

# Create dummy JSON files if they don't exist
for type_id, files_dict in dummy_files_map.items():
    for fname, content in files_dict.items():
        fpath = os.path.join(ASSETS_DIR, fname)
        if not os.path.exists(fpath):
            try:
                with open(fpath, 'w') as f: f.write(content)
                print(f"Created dummy asset for demonstration: {fpath}")
            except IOError as e:
                # Use st.error only if Streamlit context is available, otherwise print
                try: st.error(f"Fatal Error: Could not create required dummy asset {fname} in {ASSETS_DIR}. Check permissions. Error: {e}")
                except: print(f"Fatal Error: Could not create required dummy asset {fname} in {ASSETS_DIR}. Check permissions. Error: {e}")

# --- Crucial Warning about Real Assets ---
st.warning("""
**IMPORTANT:** This application requires pre-trained model files (`<type>_vqvae_model.pth`)
and calculated Mahalanobis scorer parameters (`<type>_mahalanobis_scorer_params.npz`)
to be present in the `assets/` directory for **each** supported image type (`lung`, `breast`).
The provided dummy JSON files are placeholders only. **Analysis will fail without the real `.pth` and `.npz` files.**
""")

# --- Sidebar ---
st.sidebar.header("Image Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp", "tif"])

# Check API key status dynamically
# We access the variable checked in utils.py at startup
try:
    from utils import GEMINI_CONFIGURED as gemini_api_key_available
    if gemini_api_key_available:
         st.sidebar.success("✅ Google API Key detected.")
    else:
         st.sidebar.warning("⚠️ Google API Key not detected/configured. 'Auto-Detect' disabled.")
except ImportError:
     gemini_api_key_available = False
     st.sidebar.error("Error importing utils status.")


# Setup type selection options
type_options = [LUNG_MODEL_TYPE_ID.capitalize(), BREAST_MODEL_TYPE_ID.capitalize()]
selectbox_options = []
if gemini_api_key_available:
     selectbox_options.append("Auto-Detect")
selectbox_options.extend(type_options)

default_index = 0 if gemini_api_key_available else 0 # Default to Auto-Detect if available, else first type

user_selected_type: Optional[str] = None
if selectbox_options:
    user_selected_type = st.sidebar.selectbox(
        "Select Image Type:",
        selectbox_options,
        index=default_index,
        help="Choose 'Auto-Detect' to use AI classification (requires API Key) or manually select the type."
    )
else:
    st.sidebar.error("Cannot configure type selection.")


# Initialize session state
if 'pipeline_state_result' not in st.session_state:
    st.session_state.pipeline_state_result = None

if st.sidebar.button("🚀 Analyze Image", type="primary", use_container_width=True, disabled=(not uploaded_file or not user_selected_type)):
    if uploaded_file and user_selected_type:
        st.session_state.pipeline_state_result = None # Reset previous results
        image_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name
        st.info(f"Starting analysis for '{filename}' with mode: '{user_selected_type}'...")
        with st.spinner(f"Analyzing {filename}..."):
            try:
                # Run the pipeline (which now has more internal checks)
                final_state: GraphState = run_analysis_pipeline(image_bytes, filename, user_selected_type)
                st.session_state.pipeline_state_result = final_state # Store result regardless of errors

                # Display immediate feedback based on final state error
                if final_state.get("error_message"):
                     st.error(f"Analysis completed with errors. See details below and check console logs.")
                else:
                     st.success("Analysis pipeline completed successfully!")

            except Exception as e:
                # Catch errors outside the pipeline itself (e.g., LangGraph invocation issues)
                st.error(f"CRITICAL ERROR during pipeline execution: {e}")
                st.exception(e) # Show traceback in Streamlit
                # Store a basic error state
                st.session_state.pipeline_state_result = {"error_message": f"Critical pipeline execution error: {e}", "image_filename": filename}

    elif not uploaded_file:
        st.sidebar.warning("Please upload an image file.")
    elif not user_selected_type:
         st.sidebar.error("Please select an image type (or Auto-Detect).")


# --- Main Area for Results ---
if st.session_state.pipeline_state_result:
    res: Dict = st.session_state.pipeline_state_result
    fname_disp = res.get("image_filename", "Uploaded Image")
    st.header(f"Results for: {fname_disp}")

    # Display Determination Info
    det_type = res.get("determined_type")
    sel_type = res.get("user_selected_type", "N/A") # Get the originally selected type
    if sel_type == "Auto-Detect":
         if det_type and not res.get("error_message"): # Check for error too
              st.info(f"LLM Auto-Detection classified image as: **{det_type.capitalize()}**")
         elif det_type: # Type determined but error occurred later
              st.warning(f"LLM Auto-Detection classified image as: **{det_type.capitalize()}** (but errors occurred later)")
         else: # Error during detection itself
              st.error(f"LLM Auto-Detection failed. Check error message below.")
    elif det_type:
        st.info(f"User selected type: **{det_type.capitalize()}**")


    # --- Display Errors Prominently ---
    if res.get("error_message"):
        st.error(f"**Pipeline Error(s):** {res['error_message']}")
        # Stop displaying results if core components are missing
        if res.get("original_pil") is None or res.get("model_config") is None:
             st.warning("Cannot display further results due to critical early errors.")
             # Clear the rest of the display area if needed, or just return
             st.stop()


    # --- Display Images ---
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        orig_pil = res.get("original_pil")
        if orig_pil:
            st.image(orig_pil, caption="Original Image", use_column_width=True)
        else:
            st.warning("Original image data N/A.")

    with img_col2:
        recon_tensor = res.get("reconstructed_tensor")
        if recon_tensor is not None:
            try:
                 from utils import tensor_to_pil # Import here to avoid top-level issues if utils fails
                 recon_pil = tensor_to_pil(recon_tensor)
                 st.image(recon_pil, caption="Reconstructed Image", use_column_width=True)
            except Exception as e:
                 st.warning(f"Could not display reconstructed image: {e}")
        elif not res.get("error_message"): # Only show N/A if no other errors reported
             st.info("Reconstructed image N/A (or VQVAE step failed).")
        # If error occurred, message is already shown


    # --- Display Anomaly Analysis ---
    st.markdown("---"); st.subheader("Anomaly Analysis")

    is_ano_final = res.get("is_anomaly") # Final decision True/False/None
    hybrid_score = res.get("hybrid_score_pipeline") # Scaled hybrid score value
    th_hybrid = res.get("threshold_hybrid_pipeline") # Threshold for the hybrid score

    # Display Final Status based on Hybrid Score
    if is_ano_final is not None: # Check if decision was made (True or False)
        s_color = "red" if is_ano_final else "green"
        s_text = "🚨 ANOMALY (Hybrid)" if is_ano_final else "✅ NORMAL (Hybrid)"
        st.markdown(f"**Final Status: <span style='color:{s_color};'>{s_text}</span>**", unsafe_allow_html=True)

        # Display the metric driving the final status
        if hybrid_score is not None and not np.isnan(hybrid_score) and th_hybrid is not None and not np.isnan(th_hybrid):
            delta_val = hybrid_score - th_hybrid
            st.metric(
                "Pipeline Hybrid Score (Scaled PatchMSE+Maha)",
                f"{hybrid_score:.4f}",
                f"Δ vs Th ({th_hybrid:.4f}) = {delta_val:+.4f}",
                delta_color="inverse" if is_ano_final else "normal" # Red if > th, Green if <= th
            )
        elif hybrid_score is not None and not np.isnan(hybrid_score):
             st.write(f"**Hybrid Score:** {hybrid_score:.4f} (Threshold: N/A or invalid)")
        else:
            st.write("Hybrid Score: N/A")
    elif not res.get("error_message"): # No error, but no decision? Should not happen with current logic.
         st.warning("Final anomaly status (Hybrid) could not be determined, but no specific error reported.")
    # If error message exists, it's already displayed


    # --- Display Individual Metrics ---
    st.markdown("---")
    st.markdown("#### Individual Metric Status (Raw Scores vs. Thresholds):")

    def display_metric(cols, name, score_key, threshold_key, is_anomaly_key, fmt="{:.6f}"):
         score = res.get(score_key)
         th = res.get(threshold_key)
         is_ano = res.get(is_anomaly_key)
         with cols:
            st.markdown(f"**{name}**")
            score_disp = fmt.format(score) if score is not None and not np.isnan(score) else 'N/A'
            thresh_disp = fmt.format(th) if th is not None and not np.isnan(th) else 'N/A'
            st.write(f"Score: `{score_disp}`")
            st.write(f"Th: `{thresh_disp}`")
            if is_ano is not None:
                status_color = "red" if is_ano else "green"
                status_text = "Anomaly" if is_ano else "Normal"
                st.markdown(f"Status: <span style='color:{status_color};'>{status_text}</span>", unsafe_allow_html=True)
            else:
                 # Don't show N/A if score/thresh were just missing
                 if score is not None and not np.isnan(score) and th is not None and not np.isnan(th):
                    st.write("Status: `Error during comparison`")
                 else:
                    st.write("Status: `N/A (Missing Score/Th)`")


    m_cols = st.columns(3)
    display_metric(m_cols[0], "Image MSE", "mse_score", "threshold_mse", "is_anomaly_mse")
    display_metric(m_cols[1], "Max Patch MSE", "patch_mse_score", "threshold_patch_mse", "is_anomaly_patch_mse")
    display_metric(m_cols[2], "Mahalanobis Dist.", "mahalanobis_score", "threshold_mahalanobis", "is_anomaly_mahalanobis", fmt="{:.4f}")

    # --- Display Scaled Scores (Optional) ---
    with st.expander("View Scaled Scores (inputs to Pipeline Hybrid)", expanded=False):
        sc_cols = st.columns(3)
        mse_sc = res.get('mse_score_scaled', np.nan)
        pmse_sc = res.get('patch_mse_score_scaled', np.nan)
        maha_sc = res.get('mahalanobis_score_scaled', np.nan)
        with sc_cols[0]: st.metric("Scaled Image MSE", f"{mse_sc:.4f}" if not np.isnan(mse_sc) else "N/A")
        with sc_cols[1]: st.metric("Scaled Max Patch MSE", f"{pmse_sc:.4f}" if not np.isnan(pmse_sc) else "N/A")
        with sc_cols[2]: st.metric("Scaled Mahalanobis", f"{maha_sc:.4f}" if not np.isnan(maha_sc) else "N/A")

    # --- Display Heatmap ---
    heatmap_pil = res.get("heatmap_pil")
    if is_ano_final and heatmap_pil: # Only show if final decision is anomaly AND heatmap exists
        st.subheader("Anomaly Heatmap Overlay")
        st.image(heatmap_pil, caption="Heatmap on Original Image", use_column_width=True)
    elif is_ano_final and not heatmap_pil and not res.get("error_message"): # Anomaly, but no heatmap and no error?
         st.info("Image classified as Anomaly (Hybrid), but heatmap could not be generated (check logs).")
    # If error occurred, message already shown. If normal, no heatmap expected.


# --- Footer Info ---
elif uploaded_file is None and not st.session_state.get('pipeline_state_result'):
    st.info("⬆️ Upload an image and click 'Analyze Image'.")

st.sidebar.markdown("---")
st.sidebar.info("""
**Pipeline Logic:** Final decision uses a 2-Component Hybrid score (Scaled Max Patch MSE + Scaled Mahalanobis). Individual raw metrics are also compared to their thresholds.
**Requirements:** Ensure `assets/` contains **real** `<type>_vqvae_model.pth` and `<type>_mahalanobis_scorer_params.npz` files. Also needs correctly formatted `<type>_model_runtime_config.json` and `<type>_thresholds.json`. Auto-Detect requires a valid `GOOGLE_API_KEY` environment variable.
""")