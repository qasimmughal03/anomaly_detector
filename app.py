# medical_anomaly_detector/app.py
import streamlit as st
from PIL import Image
import io
import os
import numpy as np
from typing import Dict, Optional, List

import torch # Added List

# Import pipeline function and constants
from pipeline import run_analysis_pipeline, GraphState # GraphState might be useful for typing hints
from vqvae_models import LUNG_MODEL_TYPE_ID, BREAST_MODEL_TYPE_ID
# Import IL utils for the callback
from il_utils import add_to_buffer, get_buffer_path
# Import utils only if needed directly (tensor_to_pil)
# from utils import tensor_to_pil # Moved lower

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Medical Anomaly Detector (Incremental)")
st.title("⚕️ Medical Image Anomaly Detector (Incremental Learning Enabled)")
st.markdown("""
Upload a Lung or Breast image. The system uses a VQ-VAE (M0) and incrementally trained classifiers (C1, C2...)
to detect anomalies. Provide feedback when requested to help the system learn!
""")

# --- Asset Directory and Dummy File Creation ---
# ... (Keep the ASSETS_DIR, os.makedirs, placeholder content, and dummy file creation logic as before) ...
# ... (It's important that the dummy JSONs exist for both types) ...
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)
# ...(Copy the dummy file creation logic from previous steps)...

# --- Crucial Warning about Real Assets ---
# ... (Keep the st.warning about needing real .pth/.npz files) ...
st.warning("""
**IMPORTANT:** This application requires pre-trained model files (`<type>_vqvae_model.pth`)
and calculated Mahalanobis scorer parameters (`<type>_mahalanobis_scorer_params.npz`)
to be present in the `assets/` directory for **each** supported image type (`lung`, `breast`).
The provided dummy JSON files are placeholders only. **Analysis will fail without the real `.pth` and `.npz` files.**
""")
# --- Add Warning about IL State ---
st.info("""
**Incremental Learning:** This version saves learning state (buffers, classifiers) in the `il_state/` directory. Ensure the initial `buffer_normal_replay_*.npz` files are present there.
""")


# --- Ground Truth Labeling Callback Function ---
# Define this function within app.py so it can use st elements easily
def handle_gt_label(label: str, image_key: str):
    """Callback to store features based on user's GT label."""
    print(f"Handling GT label '{label}' for image key '{image_key}'")
    # Retrieve data stored in session state by the results display logic
    features = st.session_state.get(f"features_for_{image_key}")
    img_type = st.session_state.get(f"img_type_for_{image_key}")

    if features is not None and img_type is not None:
        target_buffer_name = None
        if label == "normal_fp":
            target_buffer_name = "false_positives"
        elif label == "cancer":
             target_buffer_name = "cancer_anomalies"
        elif label == "benign":
             target_buffer_name = "benign_anomalies"
        # Add more labels/buffers here if needed

        if target_buffer_name:
            target_buffer_path = get_buffer_path(target_buffer_name, img_type)
            try:
                # Ensure features is a standard numpy array before saving
                if isinstance(features, torch.Tensor): # Should be numpy already from pipeline
                     features_np = features.cpu().detach().numpy()
                else:
                     features_np = np.array(features) # Ensure it's numpy

                add_to_buffer(target_buffer_path, features_np, label) # Use the function from il_utils
                st.success(f"Feedback received: Image features stored as '{label}' for type '{img_type}'.")
                # Clear the temporary features and flag from session state after saving
                if f"features_for_{image_key}" in st.session_state:
                    del st.session_state[f"features_for_{image_key}"]
                if f"img_type_for_{image_key}" in st.session_state:
                    del st.session_state[f"img_type_for_{image_key}"]
                # Clear the result flag to potentially hide buttons after click? Or rely on rerun.
                # If pipeline_state_result held the flag, nullifying it might be needed
                # st.session_state.pipeline_state_result['gt_label_needed'] = False # Example if state is mutable dict

            except Exception as e:
                st.error(f"Error saving features to buffer '{target_buffer_path}': {e}")
        else:
            st.error(f"Could not determine target buffer for label '{label}'.")
    else:
        st.error("Could not retrieve features or image type from session state to save label. Please try analyzing again.")

# --- Sidebar ---
# ... (Keep sidebar logic for upload, API key check, type selection, and analyze button as refined before) ...
st.sidebar.header("Image Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp", "tif"])
# ... (API key check and display) ...
try:
    # Assuming utils is importable and GEMINI_CONFIGURED reflects reality
    # If utils causes issues, remove this dynamic check
    from utils import GEMINI_CONFIGURED as gemini_api_key_available
    if gemini_api_key_available: st.sidebar.success("✅ Google API Key detected.")
    else: st.sidebar.warning("⚠️ Google API Key not detected/configured. 'Auto-Detect' disabled.")
except ImportError:
     gemini_api_key_available = bool(os.environ.get('GOOGLE_API_KEY')) # Fallback check
     st.sidebar.caption("API Key status check might be limited.")


type_options = [LUNG_MODEL_TYPE_ID.capitalize(), BREAST_MODEL_TYPE_ID.capitalize()]
selectbox_options = []
if gemini_api_key_available: selectbox_options.append("Auto-Detect")
selectbox_options.extend(type_options)
default_index = 0 if gemini_api_key_available else 0
user_selected_type: Optional[str] = None
if selectbox_options:
    user_selected_type = st.sidebar.selectbox("Select Image Type:", selectbox_options, index=default_index)


if 'pipeline_state_result' not in st.session_state: st.session_state.pipeline_state_result = None

# Clean up temporary feature storage if analysis is re-run
if st.sidebar.button("🚀 Analyze Image", type="primary", use_container_width=True, disabled=(not uploaded_file or not user_selected_type)):
    # Clear previous feature cache before running
    keys_to_delete = [k for k in st.session_state if k.startswith("features_for_") or k.startswith("img_type_for_")]
    for key in keys_to_delete:
        del st.session_state[key]

    # ... (Rest of the button logic: run pipeline, handle errors, store result in st.session_state.pipeline_state_result) ...
    if uploaded_file and user_selected_type:
        st.session_state.pipeline_state_result = None
        image_bytes = uploaded_file.getvalue(); filename = uploaded_file.name
        st.info(f"Starting analysis for '{filename}' with mode: '{user_selected_type}'...")
        with st.spinner(f"Analyzing {filename}... (May include training check)"):
            try:
                # Make sure run_analysis_pipeline returns the full GraphState dict
                final_state_dict: Dict = run_analysis_pipeline(image_bytes, filename, user_selected_type)
                st.session_state.pipeline_state_result = final_state_dict

                if final_state_dict.get("error_message"):
                     st.error(f"Analysis completed with errors. See details below.")
                else:
                     st.success("Analysis pipeline completed successfully!")
            except Exception as e:
                st.error(f"CRITICAL ERROR during pipeline execution: {e}"); st.exception(e)
                st.session_state.pipeline_state_result = {"error_message": f"Critical pipeline execution error: {e}", "image_filename": filename}
    # ... (warnings for missing file/type) ...


# --- Main Area for Results (Heavily Modified) ---
if st.session_state.pipeline_state_result:
    res: Dict = st.session_state.pipeline_state_result # Result is a dictionary (GraphState)
    fname_disp = res.get("image_filename", "Uploaded Image")
    st.header(f"Results for: {fname_disp}")

    # --- Extract IL and other relevant states ---
    det_type = res.get("determined_type")
    sel_type = res.get("user_selected_type")
    final_classification = res.get("final_classification")
    gt_label_needed = res.get("gt_label_needed", False)
    known_classes = res.get("known_classes_for_classifier") # List or None
    buffer_counts = res.get("buffer_counts") # Dict or None
    error_message = res.get("error_message")
    orig_pil = res.get("original_pil")
    recon_tensor = res.get("reconstructed_tensor")
    heatmap_pil = res.get("heatmap_pil")

    # --- Display Type Determination ---
    if sel_type == "Auto-Detect":
         if det_type and not error_message: st.info(f"LLM Auto-Detection classified image as: **{det_type.capitalize()}**")
         elif det_type: st.warning(f"LLM Auto-Detection classified image as: **{det_type.capitalize()}** (but errors occurred)")
         else: st.error(f"LLM Auto-Detection failed. Check error message.")
    elif det_type:
        st.info(f"User selected type: **{det_type.capitalize()}**")

    # --- Display Errors First ---
    if error_message:
        st.error(f"**Pipeline Error(s):** {error_message}")
        if orig_pil is None: # Critical early error
             st.warning("Cannot display further results due to critical early errors.")
             st.stop() # Stop rendering further for this run

    # --- Display Images (Original / Reconstructed) ---
    if orig_pil:
         img_col1, img_col2 = st.columns(2)
         with img_col1:
             st.image(orig_pil, caption="Original Image", use_column_width=True)
         with img_col2:
             if recon_tensor is not None:
                 try:
                      from utils import tensor_to_pil # Import here
                      recon_pil = tensor_to_pil(recon_tensor)
                      st.image(recon_pil, caption="Reconstructed Image", use_column_width=True)
                 except Exception as e: st.warning(f"Could not display recon image: {e}")
             elif not error_message: st.info("Reconstructed image N/A.")
    else:
         st.warning("Original image could not be loaded.")


    st.markdown("---")
    # --- Display Incremental Learning State ---
    st.subheader("Incremental Learning Status")
    col1, col2 = st.columns(2)
    with col1:
        if known_classes is not None: # Should be [] if none known
             classifier_desc = "M0 Only (No Classifier)" if not known_classes else f"Current Classifier (Knows Normal + {', '.join(sorted(known_classes))})"
             st.metric("Active Classifier Knows", f"{1 + len(known_classes)} Classes", classifier_desc)
        else:
             st.info("Classifier status unknown.")
    with col2:
        if buffer_counts is not None:
             st.write("**Buffer Sizes:**")
             # Filter which buffers to show - maybe just anomaly counts?
             display_buffers = {k: v for k, v in buffer_counts.items() if 'anomalies' in k}
             st.json(display_buffers) # Simple JSON display
        else:
             st.info("Buffer counts unavailable.")

    # --- Display Final Classification & Anomaly Scores ---
    st.markdown("---")
    st.subheader("Analysis Result")

    # Display the main classification output
    if final_classification:
         st.metric("Final Classification", final_classification)
         # Add color/icon based on classification?
         if final_classification == "Normal": st.success("Overall status: Normal")
         elif "Error" in final_classification: st.error(f"Overall status: Error ({final_classification})")
         else: st.warning(f"Overall status: Anomaly/Uncertain ({final_classification})")
    elif not error_message:
         st.warning("Final classification could not be determined.")

    # Display VQ-VAE Anomaly Score (still useful context)
    hybrid_score = res.get("hybrid_score_pipeline")
    th_hybrid = res.get("threshold_hybrid_pipeline")
    if hybrid_score is not None and not np.isnan(hybrid_score) and th_hybrid is not None and not np.isnan(th_hybrid):
        is_m0_anom = hybrid_score > th_hybrid
        st.metric(
            "VQ-VAE (M0) Anomaly Score",
            f"{hybrid_score:.4f}",
            f"vs M0 Th ({th_hybrid:.4f}) = {'Above (Anomalous by M0)' if is_m0_anom else 'Below (Normal-like by M0)'}",
            delta_color="inverse" if is_m0_anom else "normal"
            )
    elif hybrid_score is not None and not np.isnan(hybrid_score):
        st.write(f"VQ-VAE (M0) Score: `{hybrid_score:.4f}` (Threshold N/A)")
    else:
         st.write("VQ-VAE (M0) Score: `N/A`")

    # --- Ground Truth Input Section ---
    if gt_label_needed and orig_pil: # Only show if flag is set AND image available
        st.markdown("---")
        st.subheader("🔬 Expert Feedback Required!")
        st.warning(f"The system classified this as **'{final_classification}'** and requires your input.")

        # Use filename as a relatively stable key for session state
        image_key = fname_disp.replace(" ", "_").replace(".", "_") # Make key safer

        # Store features/type needed by the callback BEFORE rendering buttons
        extracted_features = res.get("extracted_features")
        if extracted_features is not None and det_type is not None:
            st.session_state[f"features_for_{image_key}"] = extracted_features
            st.session_state[f"img_type_for_{image_key}"] = det_type
            print(f"Stored features for GT callback under key: features_for_{image_key}") # Debug print

            # Display image again for context if needed, or assume user sees above image
            # st.image(orig_pil, caption="Image Requiring Label", use_column_width=True)

            st.write("Please provide the correct label for this image:")
            btn_cols = st.columns(3)
            with btn_cols[0]:
                if st.button("Mark as False Positive (Normal)", key=f"fp_btn_{image_key}"):
                    handle_gt_label("normal_fp", image_key)
                    st.rerun() # Rerun script to reflect state change & hide buttons
            with btn_cols[1]:
                if st.button("Mark as Cancer", key=f"cancer_btn_{image_key}"):
                    handle_gt_label("cancer", image_key)
                    st.rerun()
            with btn_cols[2]:
                 if st.button("Mark as Benign", key=f"benign_btn_{image_key}"):
                    handle_gt_label("benign", image_key)
                    st.rerun()
            # Add more buttons for other potential ground truth labels if needed
        else:
             st.error("Cannot request label: Features or image type were not found in the result state.")

    # --- Display Heatmap ---
    # Decide which classifications trigger heatmap
    anomaly_labels_for_heatmap = [
        "Cancer", "Benign",
        "Potential_Anomaly_M0",
        "Potential_Novel_Anomaly", # Cover C1/C2 novel cases broadly
        "Potential_Unknown_Anomaly_C2",
        "Potential_FP_Known", # Optional: show heatmap even for potential FPs
    ]
    # Check if final_classification contains any part of these strings
    show_heatmap_flag = False
    if final_classification:
         for label_part in anomaly_labels_for_heatmap:
              if label_part in final_classification:
                   show_heatmap_flag = True
                   break

    if show_heatmap_flag and heatmap_pil:
        st.markdown("---")
        st.subheader("Anomaly Heatmap Overlay")
        st.image(heatmap_pil, caption="Heatmap on Original Image", use_column_width=True)
    elif show_heatmap_flag and not error_message: # Anomaly classification, but no heatmap generated
         st.info("Anomaly classification, but heatmap is unavailable (check pipeline logs).")


    # --- Optional: Display Raw M0 Scores ---
    with st.expander("View Raw VQ-VAE (M0) Metric Scores", expanded=False):
         # ... (Keep the display_metric logic and calls from before if desired) ...
        def display_metric_raw(cols, name, score_key, threshold_key, is_anomaly_key, fmt="{:.6f}"):
             score = res.get(score_key); th = res.get(threshold_key); is_ano = res.get(is_anomaly_key)
             with cols:
                st.markdown(f"**{name}**"); score_disp = fmt.format(score) if score is not None and not np.isnan(score) else 'N/A'
                thresh_disp = fmt.format(th) if th is not None and not np.isnan(th) else 'N/A'
                st.write(f"Score: `{score_disp}`"); st.write(f"Th: `{thresh_disp}`")
                if is_ano is not None: st.markdown(f"Status: {'<span style=\"color:red;\">Anomaly</span>' if is_ano else '<span style=\"color:green;\">Normal</span>'}", unsafe_allow_html=True)
                else: st.write("Status: `N/A`")

        m_cols = st.columns(3)
        display_metric_raw(m_cols[0], "Image MSE", "mse_score", "threshold_mse", "is_anomaly_mse")
        display_metric_raw(m_cols[1], "Max Patch MSE", "patch_mse_score", "threshold_patch_mse", "is_anomaly_patch_mse")
        display_metric_raw(m_cols[2], "Mahalanobis Dist.", "mahalanobis_score", "threshold_mahalanobis", "is_anomaly_mahalanobis", fmt="{:.4f}")


# --- Footer Info ---
elif uploaded_file is None and not st.session_state.get('pipeline_state_result'):
    st.info("⬆️ Upload an image and click 'Analyze Image'.")

st.sidebar.markdown("---")
st.sidebar.info("""
**Pipeline:** VQ-VAE (M0) + Incrementally Trained Classifier (MLP).
**IL State:** Stored in `il_state/` (buffers, classifier, known classes).
**Feedback:** Provide ground truth labels when prompted to enable learning.
**Requirements:** Ensure `assets/` contains real `.pth`/`.npz` files and `il_state/` contains initial `buffer_normal_replay_*.npz`. Check API Key setup.
""")