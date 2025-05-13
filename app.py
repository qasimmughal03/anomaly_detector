# medical_anomaly_detector/app.py
# medical_anomaly_detector/app.py
import streamlit as st
from PIL import Image
import io
import os
import numpy as np
from typing import Dict, Optional, List # Added List

# Import pipeline function and constants
from pipeline import run_analysis_pipeline, GraphState
from vqvae_models import LUNG_MODEL_TYPE_ID, BREAST_MODEL_TYPE_ID
# Import IL utils for the callback
from il_utils import add_to_buffer, get_buffer_path # Assuming add_to_buffer expects numpy array
# Import utils only if needed directly (tensor_to_pil)
# from utils import tensor_to_pil

# --- Page Configuration & Warnings (Keep as before) ---
# ... (st.set_page_config, st.title, st.markdown, ASSETS_DIR, dummy file creation, warnings) ...
st.set_page_config(layout="wide", page_title="Medical Anomaly Detector (Incremental)")
st.title("⚕️ Medical Image Anomaly Detector (Incremental Learning Enabled)")
st.markdown("""
Upload a Lung or Breast image. The system uses a VQ-VAE (M0) and incrementally trained classifiers (C1, C2...)
to detect anomalies. Provide feedback when requested, or correct a classification, to help the system learn!
""")
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)
# ...(Copy the dummy file creation logic from previous steps)...
st.warning("""
**IMPORTANT:** This application requires pre-trained model files (`<type>_vqvae_model.pth`)
and calculated Mahalanobis scorer parameters (`<type>_mahalanobis_scorer_params.npz`)
to be present in the `assets/` directory for **each** supported image type (`lung`, `breast`).
The provided dummy JSON files are placeholders only. **Analysis will fail without the real `.pth` and `.npz` files.**
""")
st.info("""
**Incremental Learning:** This version saves learning state (buffers, classifiers) in the `il_state/` directory. Ensure the initial `buffer_normal_replay_*.npz` files are present there.
""")


# --- Ground Truth Labeling Callback Function (Ensure it expects NumPy array) ---
def handle_gt_label(label: str, image_key: str):
    """Callback to store features based on user's GT label."""
    print(f"Handling GT label '{label}' for image key '{image_key}'")
    features_np = st.session_state.get(f"features_for_{image_key}") # Should be np.ndarray
    img_type = st.session_state.get(f"img_type_for_{image_key}")

    if features_np is not None and isinstance(features_np, np.ndarray) and img_type is not None:
        target_buffer_name = None
        # Map user-friendly labels to buffer names and internal labels if needed
        # For now, assume label passed is what's stored.
        if label == "normal_fp": # "Mark as False Positive (Normal)"
            target_buffer_name = "false_positives"
            # The label stored in the buffer for false_positives might be 'normal_fp' or just 'normal'
            # Let's use 'normal_fp' to distinguish it in the buffer if needed later.
        elif label == "cancer":
             target_buffer_name = "cancer_anomalies"
        elif label == "benign":
             target_buffer_name = "benign_anomalies"

        if target_buffer_name:
            target_buffer_path = get_buffer_path(target_buffer_name, img_type)
            try:
                add_to_buffer(target_buffer_path, features_np, label) # Pass np array and label string
                st.success(f"Feedback received: Image features stored as '{label}' for type '{img_type}'.")
                # Clear specific session state keys after processing
                keys_to_delete = [f"features_for_{image_key}", f"img_type_for_{image_key}"]
                for key in keys_to_delete:
                    if key in st.session_state:
                        del st.session_state[key]
                # Remove the entire result to prevent re-submission on simple refresh
                # or to hide the GT buttons if desired, though rerun handles this.
                # if 'pipeline_state_result' in st.session_state:
                #     st.session_state.pipeline_state_result['gt_label_needed'] = False # Mark as handled
            except Exception as e:
                st.error(f"Error saving features to buffer '{target_buffer_path}': {e}")
        else:
            st.error(f"Could not determine target buffer for label '{label}'.")
    else:
        st.error("Could not retrieve features (or features not np.ndarray) or image type from session state. Please try analyzing again.")


# --- Sidebar & Analyze Button (Keep as refined before) ---
# ... (Make sure the analyze button clears old "features_for_" session state keys) ...
st.sidebar.header("Image Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp", "tif"])
try:
    from utils import GEMINI_CONFIGURED as gemini_api_key_available
    if gemini_api_key_available: st.sidebar.success("✅ Google API Key detected.")
    else: st.sidebar.warning("⚠️ Google API Key not detected/configured. 'Auto-Detect' disabled.")
except ImportError:
     gemini_api_key_available = bool(os.environ.get('GOOGLE_API_KEY'))
     st.sidebar.caption("API Key status check may be limited.")

type_options = [LUNG_MODEL_TYPE_ID.capitalize(), BREAST_MODEL_TYPE_ID.capitalize()]
selectbox_options = []
if gemini_api_key_available: selectbox_options.append("Auto-Detect")
selectbox_options.extend(type_options)
default_index = 0 if gemini_api_key_available else 0
user_selected_type: Optional[str] = None
if selectbox_options:
    user_selected_type = st.sidebar.selectbox("Select Image Type:", selectbox_options, index=default_index)

if 'pipeline_state_result' not in st.session_state: st.session_state.pipeline_state_result = None

if st.sidebar.button("🚀 Analyze Image", type="primary", use_container_width=True, disabled=(not uploaded_file or not user_selected_type)):
    keys_to_delete = [k for k in st.session_state if k.startswith("features_for_") or k.startswith("img_type_for_")]
    for key in keys_to_delete: del st.session_state[key]
    if uploaded_file and user_selected_type:
        st.session_state.pipeline_state_result = None
        image_bytes = uploaded_file.getvalue(); filename = uploaded_file.name
        st.info(f"Starting analysis for '{filename}' with mode: '{user_selected_type}'...")
        with st.spinner(f"Analyzing {filename}... (May include training check)"):
            try:
                final_state_dict: Dict = run_analysis_pipeline(image_bytes, filename, user_selected_type)
                st.session_state.pipeline_state_result = final_state_dict
                if final_state_dict.get("error_message"): st.error(f"Analysis completed with errors.") # Details shown below
                else: st.success("Analysis pipeline completed successfully!")
            except Exception as e:
                st.error(f"CRITICAL ERROR during pipeline execution: {e}"); st.exception(e)
                st.session_state.pipeline_state_result = {"error_message": f"Critical pipeline execution error: {e}", "image_filename": filename}


# --- Main Area for Results (Modified GT Section) ---
if st.session_state.pipeline_state_result:
    res: Dict = st.session_state.pipeline_state_result
    fname_disp = res.get("image_filename", "Uploaded Image")
    st.header(f"Results for: {fname_disp}")

    # Extract states (as before)
    det_type = res.get("determined_type")
    sel_type = res.get("user_selected_type")
    final_classification = res.get("final_classification")
    gt_label_needed_from_pipeline = res.get("gt_label_needed", False) # Flag from pipeline
    known_classes = res.get("known_classes_for_classifier")
    buffer_counts = res.get("buffer_counts")
    error_message = res.get("error_message")
    orig_pil = res.get("original_pil")
    recon_tensor = res.get("reconstructed_tensor")
    heatmap_pil = res.get("heatmap_pil")
    extracted_features = res.get("extracted_features") # This should be np.ndarray

    # --- Display Type Determination, Errors, Images (as before) ---
    # ... (Keep this section as refined previously) ...
    if sel_type == "Auto-Detect":
         if det_type and not error_message: st.info(f"LLM Auto-Detection classified image as: **{det_type.capitalize()}**")
         # ... other conditions ...
    elif det_type: st.info(f"User selected type: **{det_type.capitalize()}**")
    if error_message: st.error(f"**Pipeline Error(s):** {error_message}") # ...
    if orig_pil: # ... display images ...
         img_col1, img_col2 = st.columns(2); # ...
         with img_col1: st.image(orig_pil, caption="Original Image", use_column_width=True)
         with img_col2:
             if recon_tensor is not None:
                 try: from utils import tensor_to_pil; recon_pil = tensor_to_pil(recon_tensor); st.image(recon_pil, caption="Reconstructed Image", use_column_width=True)
                 except Exception as e: st.warning(f"Could not display recon: {e}")
             elif not error_message: st.info("Recon image N/A.")
    else: st.warning("Original image N/A.")


    st.markdown("---")
    # --- Display Incremental Learning Status (as before) ---
    st.subheader("Incremental Learning Status")
    # ... (Display active classifier and buffer counts as refined before) ...
    col1_status, col2_status = st.columns(2)
    with col1_status:
        if known_classes is not None:
             classifier_desc = "M0 Only (No Classifier)" if not known_classes else f"Knows Normal + {', '.join(sorted(known_classes))}"
             st.metric("Active Classifier", classifier_desc, f"{1 + len(known_classes)} Classes")
        else: st.info("Classifier status unknown.")
    with col2_status:
        if buffer_counts is not None: st.write("**Buffer Sizes:**"); st.json({k: v for k, v in buffer_counts.items() if 'anomalies' in k or 'false_positives' in k or 'replay' in k})
        else: st.info("Buffer counts N/A.")


    # --- Display Final Classification & M0 Score (as before) ---
    st.markdown("---")
    st.subheader("Analysis Result")
    # ... (Display final_classification metric and M0 score metric as refined before) ...
    if final_classification:
         st.metric("Final Classification", final_classification)
         if final_classification == "Normal": st.success("Overall status: Normal")
         elif "Error" in final_classification: st.error(f"Overall status: Error ({final_classification})")
         else: st.warning(f"Overall status: Potential Anomaly/Uncertain ({final_classification})")
    elif not error_message: st.warning("Final classification N/A.")

    hybrid_score = res.get("hybrid_score_pipeline"); th_hybrid = res.get("threshold_hybrid_pipeline")
    if hybrid_score is not None and not np.isnan(hybrid_score) and th_hybrid is not None and not np.isnan(th_hybrid):
        is_m0_anom = hybrid_score > th_hybrid
        st.metric("VQ-VAE (M0) Anomaly Score", f"{hybrid_score:.4f}", f"vs M0 Th ({th_hybrid:.4f}) = {'Above (Anomalous by M0)' if is_m0_anom else 'Below (Normal-like by M0)'}", delta_color="inverse" if is_m0_anom else "normal")
    # ...

    # --- MODIFIED/UNIFIED Ground Truth Input Section ---
    if extracted_features is not None and det_type is not None and orig_pil is not None and not error_message: # Only show if features are available and no critical error
        st.markdown("---")
        # Prepare image key for session state (must be consistent)
        image_key = fname_disp.replace(" ", "_").replace(".", "_").replace("(", "").replace(")", "")

        # Store features in session state *before* buttons are rendered IF NOT ALREADY HANDLED
        # Check if already handled for this key, to prevent overwriting if user clicks multiple times
        # However, Streamlit reruns, so if buttons not clicked, this will re-store.
        # If a button was clicked, handle_gt_label clears it, so this section won't show buttons on immediate rerun.
        if f"features_for_{image_key}" not in st.session_state: # Only store if not already there from a previous button click on this image
             st.session_state[f"features_for_{image_key}"] = extracted_features
             st.session_state[f"img_type_for_{image_key}"] = det_type
             print(f"Stored features/type for GT callback under key: {image_key}")


        # Check if features are still available in session state for button display
        # (they will be unless a GT button was just clicked and handle_gt_label cleared them)
        if f"features_for_{image_key}" in st.session_state:
            if gt_label_needed_from_pipeline:
                st.subheader("🔬 Expert Feedback Required!")
                st.warning(f"The system classified this as **'{final_classification}'** and requires your input. Please provide the correct label:")
            else:
                st.subheader("🔍 Optionally Provide or Correct Label")
                st.info(f"The system classified this as **'{final_classification}'**. If this is incorrect, or if you want to reinforce learning, please provide the true label:")

            # Display image again in this section for clarity
            st.image(orig_pil, caption="Image for Labeling", use_column_width=True, width=300)

            btn_cols_gt = st.columns(3)
            # Use distinct keys for these buttons, e.g., by prefixing
            # The label passed to handle_gt_label is what determines the buffer.
            with btn_cols_gt[0]:
                if st.button("Mark as False Positive (Normal)", key=f"gt_fp_btn_{image_key}"):
                    handle_gt_label("normal_fp", image_key) # Label to be stored
                    st.rerun()
            with btn_cols_gt[1]:
                if st.button("Mark as Cancer", key=f"gt_cancer_btn_{image_key}"):
                    handle_gt_label("cancer", image_key)
                    st.rerun()
            with btn_cols_gt[2]:
                 if st.button("Mark as Benign", key=f"gt_benign_btn_{image_key}"):
                    handle_gt_label("benign", image_key)
                    st.rerun()
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