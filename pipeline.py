# medical_anomaly_detector/pipeline.py
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import io
import json
from typing import TypedDict, List, Optional, Dict, Any, Tuple # Added Tuple

from langgraph.graph import StateGraph, END

# Ensure model IDs are imported correctly
from vqvae_models import VQVAE, MahalanobisScorer, LUNG_MODEL_TYPE_ID, BREAST_MODEL_TYPE_ID
from utils import (
    preprocess_image_from_bytes, compute_image_mse, compute_max_patch_mse,
    compute_pixelwise_error_map, generate_heatmap_pil,
    classify_image_type_with_llm, # Keep this import
    min_max_scale_with_params
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Pipeline using device: {DEVICE}")

# --- Asset Paths ---
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
# Define paths dynamically later based on determined type

# --- LangGraph State Definition ---
# ... (GraphState remains the same) ...
class GraphState(TypedDict):
    image_bytes: Optional[bytes]
    image_filename: Optional[str]
    user_selected_type: str
    determined_type: Optional[str]
    model_config: Optional[Dict[str, Any]]
    original_pil: Optional[Image.Image]
    processed_tensor: Optional[torch.Tensor]
    reconstructed_tensor: Optional[torch.Tensor]
    latent_z_q: Optional[torch.Tensor]
    pixel_error_map_np: Optional[np.ndarray]
    mse_score: Optional[float]
    patch_mse_score: Optional[float]
    mahalanobis_score: Optional[float]
    mse_score_scaled: Optional[float]
    patch_mse_score_scaled: Optional[float]
    mahalanobis_score_scaled: Optional[float]
    hybrid_score_pipeline: Optional[float]
    is_anomaly_mse: Optional[bool]
    threshold_mse: Optional[float]
    is_anomaly_patch_mse: Optional[bool]
    threshold_patch_mse: Optional[float]
    is_anomaly_mahalanobis: Optional[bool]
    threshold_mahalanobis: Optional[float]
    is_anomaly: Optional[bool]
    threshold_hybrid_pipeline: Optional[float]
    heatmap_pil: Optional[Image.Image]
    error_message: Optional[str]


# --- Helper Functions for Loading Assets ---
# ... (load_json_file, load_vqvae_model, load_mahalanobis_scorer remain the same) ...
# Minor refinement: add file description to error messages
def load_json_file(file_path: str, file_description: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(file_path):
        print(f"   ERROR ({file_description} Loader): File not found: {file_path}")
        # Return None, the calling function should handle this
        return None
    try:
        with open(file_path, 'r') as f: data = json.load(f)
        print(f"   ({file_description} Loader): Data loaded from {file_path}")
        return data
    except Exception as e:
        print(f"   ERROR ({file_description} Loader): Could not load/parse JSON from {file_path}: {e}")
        return None

def load_vqvae_model(model_path: str, model_constructor_params: dict) -> Optional[VQVAE]:
    print(f"   Attempting to load VQ-VAE model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"   ERROR (load_vqvae_model): Model file not found: {model_path}")
        return None
    # ... rest of the function is okay ...
    try:
        required_keys = ["in_channels", "h_dim", "res_h_dim", "n_res_layers",
                         "n_embeddings", "embedding_dim", "beta", "out_channels"]
        missing_keys = [key for key in required_keys if key not in model_constructor_params]
        if missing_keys:
            print(f"   ERROR (load_vqvae_model): Missing keys in model_constructor_params for {model_path}: {missing_keys}")
            return None
        model = VQVAE(**model_constructor_params).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"   VQ-VAE Model loaded successfully: {model_path}")
        return model
    except Exception as e:
        print(f"   ERROR loading VQ-VAE model state dict or instantiating from {model_path}: {e}")
        return None


def load_mahalanobis_scorer(scorer_params_path: str) -> Optional[MahalanobisScorer]:
    print(f"   Attempting to load Mahalanobis scorer params from: {scorer_params_path}")
    if not os.path.exists(scorer_params_path):
        print(f"   ERROR (load_mahalanobis_scorer): Scorer params file not found: {scorer_params_path}")
        return None
    # ... rest of the function is okay ...
    try:
        params = np.load(scorer_params_path)
        scorer = MahalanobisScorer()
        if 'mean_embedding' not in params or 'inv_covariance' not in params:
            print(f"   ERROR (load_mahalanobis_scorer): Required keys ('mean_embedding', 'inv_covariance') missing in {scorer_params_path}.")
            return None
        scorer.fit_params(params['mean_embedding'], params['inv_covariance'])
        if not scorer.fitted:
            # fit_params should print its own error if params are None
            print(f"   Warning (load_mahalanobis_scorer): Scorer initialization failed with params from {scorer_params_path}.")
            return None # Explicitly return None if fitting failed
        print(f"   Mahalanobis scorer loaded successfully: {scorer_params_path}")
        return scorer
    except Exception as e:
        print(f"   ERROR loading or processing Mahalanobis scorer params from {scorer_params_path}: {e}")
        return None


# --- LangGraph Node Functions (prepare_initial_data_node Revised) ---
def prepare_initial_data_node(state: GraphState) -> Dict[str, Any]:
    print("\n--- Node: prepare_initial_data_node ---")
    image_bytes = state.get("image_bytes")
    user_type_selection = state.get("user_selected_type")
    image_filename = state.get("image_filename", "Unknown Filename") # Get filename for logging
    error_msg = None
    determined_type_val = None
    pipeline_model_config = {}

    if not image_bytes:
        error_msg = "CRITICAL ERROR: No image data provided to the pipeline."
        print(f"   {error_msg}")
        return {"error_message": error_msg}

    # 1. Determine Image Type (LLM or User Selection)
    if user_type_selection == "Auto-Detect":
        print(f"   Attempting Auto-Detection for {image_filename}...")
        llm_type, llm_error = classify_image_type_with_llm(image_bytes) # Get error message too
        if llm_error:
            # Prepend specific context to the LLM error
            error_msg = f"LLM Auto-Detect Failed: {llm_error}. Please select type manually."
        elif llm_type in [LUNG_MODEL_TYPE_ID, BREAST_MODEL_TYPE_ID]:
            determined_type_val = llm_type
            print(f"   Auto-Detect determined type: {determined_type_val}")
        else:
            # This case should ideally be covered by llm_error now
            error_msg = f"LLM Auto-Detect returned unexpected type '{llm_type}'. Select manually."

    elif user_type_selection.lower() in [LUNG_MODEL_TYPE_ID, BREAST_MODEL_TYPE_ID]:
        determined_type_val = user_type_selection.lower()
        print(f"   User selected type: {determined_type_val}")
    else:
        error_msg = f"Invalid user type selection: '{user_type_selection}'."

    if error_msg:
        print(f"   Error (Type Determination): {error_msg}")
        return {"error_message": error_msg, "determined_type": None} # Return determined_type as None

    # If type determination failed somehow (shouldn't happen if error_msg isn't set, but check anyway)
    if not determined_type_val:
         error_msg = "Internal Error: Image type could not be determined."
         print(f"   {error_msg}")
         return {"error_message": error_msg, "determined_type": None}

    # 2. Define and Check Required Asset Paths based on determined_type_val
    print(f"   Checking required assets for type: {determined_type_val}...")
    cfg_file = os.path.join(ASSETS_DIR, f"{determined_type_val}_model_runtime_config.json")
    thr_file = os.path.join(ASSETS_DIR, f"{determined_type_val}_thresholds.json")
    model_file = os.path.join(ASSETS_DIR, f"{determined_type_val}_vqvae_model.pth")
    scorer_file = os.path.join(ASSETS_DIR, f"{determined_type_val}_mahalanobis_scorer_params.npz")

    required_files = {
        "Runtime Config": cfg_file,
        "Thresholds": thr_file,
        "VQ-VAE Model (.pth)": model_file,
        "Mahalanobis Scorer (.npz)": scorer_file
    }
    missing_files = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append(f"{name} (Expected at: {path})")

    if missing_files:
        error_msg = f"CRITICAL ERROR: Missing required asset file(s) for type '{determined_type_val}': {'; '.join(missing_files)}. Cannot proceed."
        print(f"   {error_msg}")
        # Return determined_type so the UI knows what was attempted
        return {"error_message": error_msg, "determined_type": determined_type_val}

    print("   All required asset files appear to exist.")

    # 3. Load Config and Thresholds (safe now, as files exist)
    loaded_cfg = load_json_file(cfg_file, f"{determined_type_val} Runtime Config")
    if loaded_cfg:
        pipeline_model_config.update(loaded_cfg)
    else:
        # load_json_file prints specific errors
        error_msg = f"Failed to load or parse runtime config: {cfg_file}."

    if not error_msg:
        loaded_thr = load_json_file(thr_file, f"{determined_type_val} Thresholds")
        if loaded_thr:
            req_thr_keys = ["image_mse", "max_patch_mse", "mahalanobis_distance", "hybrid_score"]
            if not all(k in loaded_thr for k in req_thr_keys):
                error_msg = f"Missing required threshold keys in {thr_file}. Expected: {req_thr_keys}. Found: {list(loaded_thr.keys())}"
            else:
                pipeline_model_config["loaded_thresholds"] = loaded_thr
        else:
             # load_json_file prints specific errors
            error_msg = f"Failed to load or parse thresholds: {thr_file}."

    # 4. Check Essential Keys in Loaded Config
    if not error_msg:
        essential_runtime_keys = ["params", "img_size", "patch_size_ratio",
                                  "scaling_parameters_from_training",
                                  "hybrid_score_weights_training"]
        missing_config_keys = [k for k in essential_runtime_keys if k not in pipeline_model_config]
        if missing_config_keys:
            error_msg = f"Essential keys missing from loaded runtime config ({cfg_file}): {missing_config_keys}."

    if error_msg:
        print(f"   Error (Config/Threshold Loading): {error_msg}")
        return {"error_message": error_msg, "determined_type": determined_type_val, "model_config": None}

    # 5. Load and Preprocess Image
    print("   Loading and preprocessing image...")
    original_pil_val = None
    processed_tensor_val = None
    try:
        original_pil_val = Image.open(io.BytesIO(image_bytes)).convert('L')
        img_sz = pipeline_model_config["img_size"] # Known to exist now
        clahe_cfg = pipeline_model_config.get("clahe_params") # Optional
        processed_tensor_val = preprocess_image_from_bytes(image_bytes, img_sz, clahe_cfg)
        if processed_tensor_val is None:
            # preprocess_image_from_bytes prints errors
            error_msg = "Image preprocessing failed."
    except Exception as e:
        error_msg = f"Failed to load or preprocess image ({image_filename}): {e}"

    if error_msg:
        print(f"   Error (Image Processing): {error_msg}")
        return {"error_message": error_msg, "determined_type": determined_type_val,
                "model_config": pipeline_model_config, "original_pil": original_pil_val} # Return PIL if loaded

    # 6. Success Case
    print("   Initial data preparation successful.")
    return {
        "determined_type": determined_type_val,
        "model_config": pipeline_model_config,
        "original_pil": original_pil_val,
        "processed_tensor": processed_tensor_val.to(DEVICE) if processed_tensor_val is not None else None,
        "error_message": None # Explicitly set error to None on success
    }


# --- vqvae_inference_node (Revised to use determined paths) ---
def vqvae_inference_node(state: GraphState) -> Dict[str, Any]:
    print("\n--- Node: vqvae_inference_node ---")
    cfg = state.get("model_config")
    tensor = state.get("processed_tensor")
    dtype = state.get("determined_type")
    err_msg = state.get("error_message", "") # Get existing errors

    # Check if previous node already set an error
    if err_msg:
        print(f"   Skipping VQVAE inference due to previous error: {err_msg}")
        # Return the state mostly unchanged, ensuring the error persists
        return {"error_message": err_msg}

    # Check required inputs for this node
    if not cfg or tensor is None or not dtype:
        current_err = "Config, processed_tensor, or determined_type missing for VQVAE inference."
        print(f"   Error: {current_err}")
        # Combine with previous errors if any
        full_err = (err_msg + "; " if err_msg else "") + current_err
        return {"error_message": full_err}

    # Construct model path (file existence already checked in prepare_initial_data_node)
    model_p = os.path.join(ASSETS_DIR, f"{dtype}_vqvae_model.pth")
    arch_p = cfg.get("params")

    if not arch_p:
        current_err = f"VQVAE architecture 'params' missing in loaded config for type '{dtype}'."
        print(f"   Error: {current_err}")
        full_err = (err_msg + "; " if err_msg else "") + current_err
        return {"error_message": full_err}

    # Load model (loader function handles internal errors/logging)
    model = load_vqvae_model(model_p, arch_p)
    if not model:
        # Error message already printed by load_vqvae_model
        current_err = f"Failed to load VQVAE model from {model_p}."
        full_err = (err_msg + "; " if err_msg else "") + current_err
        return {"error_message": full_err}

    # Run inference
    try:
        print("   Running VQ-VAE model inference...")
        with torch.no_grad():
            _, recon, _, zq, _, _ = model(tensor)
        print("   Calculating pixel error map...")
        px_err_map = compute_pixelwise_error_map(tensor, recon)
        print("   VQ-VAE inference successful.")
        return {
            "reconstructed_tensor": recon,
            "latent_z_q": zq,
            "pixel_error_map_np": px_err_map,
            "error_message": err_msg if err_msg else None # Preserve any non-critical previous messages
         }
    except Exception as e:
        current_err = f"Error during VQ-VAE model forward pass or pixel map calculation: {e}"
        print(f"   ERROR: {current_err}")
        full_err = (err_msg + "; " if err_msg else "") + current_err
        # Return NaNs or None for outputs if inference fails
        return {
            "reconstructed_tensor": None,
            "latent_z_q": None,
            "pixel_error_map_np": None,
            "error_message": full_err
        }


# --- calculate_raw_scores_node (Revised to use determined paths) ---
def calculate_raw_scores_node(state: GraphState) -> Dict[str, Any]:
    print("\n--- Node: calculate_raw_scores_node ---")
    cfg=state.get("model_config")
    pt=state.get("processed_tensor")
    rt=state.get("reconstructed_tensor")
    zq=state.get("latent_z_q")
    img_t=state.get("determined_type")
    err_msg = state.get("error_message", "") # Get existing errors

    # Check if previous node already set an error that prevents scoring
    if err_msg and (pt is None or rt is None or zq is None):
        print(f"   Skipping raw score calculation due to previous error: {err_msg}")
        return {"error_message": err_msg, "mse_score":np.nan, "patch_mse_score":np.nan, "mahalanobis_score":np.nan}

    # Initialize scores
    mse, p_mse, maha = np.nan, np.nan, np.nan
    current_errs = [] # Collect errors from this node

    # Check for necessary inputs specifically for this node's calculations
    if pt is None or rt is None:
        current_errs.append("Processed or reconstructed tensor missing for MSE/PatchMSE.")
    if zq is None:
         current_errs.append("Latent tensor 'zq' missing for Mahalanobis score.")
    if not cfg:
         current_errs.append("Model config missing (needed for patch size).")
    if not img_t:
         current_errs.append("Determined image type missing (needed for scorer path).")

    # Calculate Image MSE
    if pt is not None and rt is not None:
        try:
            mse = compute_image_mse(pt, rt)
            print(f"   Raw Image MSE: {mse:.6f}")
        except Exception as e:
            current_errs.append(f"Image MSE calculation error: {e}")
    else:
         current_errs.append("Skipped Image MSE (missing tensors).")


    # Calculate Patch MSE
    if pt is not None and rt is not None and cfg:
        try:
            img_size_val = cfg.get("img_size")
            patch_ratio_val = cfg.get("patch_size_ratio")
            if img_size_val is not None and patch_ratio_val is not None and patch_ratio_val > 0:
                patch_s = img_size_val // patch_ratio_val
                if patch_s > 0:
                    p_mse = compute_max_patch_mse(pt, rt, patch_s)
                    print(f"   Raw Max Patch MSE (patch size {patch_s}): {p_mse:.6f}")
                else:
                     current_errs.append("Patch size calculated as 0, cannot compute Patch MSE.")
            else:
                current_errs.append("img_size or patch_size_ratio missing/invalid in config for Patch MSE.")
        except Exception as e:
            current_errs.append(f"Patch MSE calculation error: {e}")
    else:
         current_errs.append("Skipped Patch MSE (missing tensors or config).")


    # Calculate Mahalanobis Score
    if zq is not None and img_t:
        try:
            # Construct scorer path (file existence already checked)
            s_path = os.path.join(ASSETS_DIR, f"{img_t}_mahalanobis_scorer_params.npz")
            scorer = load_mahalanobis_scorer(s_path) # Loader handles errors
            if scorer and scorer.fitted:
                # Pool the latent representation
                pooled = F.adaptive_avg_pool2d(zq, (1, 1)).squeeze(-1).squeeze(-1)
                if pooled.ndim == 1: pooled = pooled.unsqueeze(0) # Ensure batch dim
                maha_arr = scorer.score(pooled.cpu().numpy()) # score method handles its internal errors
                if maha_arr is not None and maha_arr.size > 0 and not np.isnan(maha_arr[0]):
                    maha = maha_arr[0]
                    print(f"   Raw Mahalanobis Score: {maha:.4f}")
                elif maha_arr is not None and maha_arr.size > 0 and np.isnan(maha_arr[0]):
                     current_errs.append("Mahalanobis scorer returned NaN.")
                else:
                     current_errs.append("Mahalanobis scorer returned empty/None result.")
            else:
                # Error message printed by loader or fit_params
                current_errs.append(f"Mahalanobis scorer not loaded or not fitted for type '{img_t}'.")
        except Exception as e:
            current_errs.append(f"Mahalanobis score calculation error: {e}")
    else:
        current_errs.append("Skipped Mahalanobis (missing latent tensor 'zq' or image type).")

    # Combine error messages
    node_err_msg = "; ".join(current_errs) if current_errs else None
    full_err = (err_msg + "; " if err_msg else "") + (node_err_msg if node_err_msg else "")
    if node_err_msg: print(f"   Warnings/Errors in score calculation: {node_err_msg}")

    return {
        "mse_score": mse,
        "patch_mse_score": p_mse,
        "mahalanobis_score": maha,
        "error_message": full_err if full_err else None
    }


# --- evaluate_scaled_hybrid_node ---
# Keep this mostly as is, but add checks for NaN inputs and improve logging
def evaluate_scaled_hybrid_node(state: GraphState) -> Dict[str, Any]:
    print("\n--- Node: evaluate_scaled_hybrid_node (2-Component Hybrid Focus) ---")
    mse_r = state.get("mse_score") # Raw MSE
    p_mse_r = state.get("patch_mse_score") # Raw Patch MSE
    maha_r = state.get("mahalanobis_score") # Raw Mahalanobis
    cfg = state.get("model_config")
    err_msg = state.get("error_message", "") # Get existing errors
    current_errs = [] # Collect errors from this node

    # Default values
    mse_s, p_mse_s, maha_s, hybrid_pipe = np.nan, np.nan, np.nan, np.nan
    is_ano_mse, is_ano_pmse, is_ano_maha = None, None, None
    th_mse, th_pmse, th_maha, th_hybrid = np.nan, np.nan, np.nan, np.nan

    # Check if previous node already set an error that prevents evaluation
    if err_msg and (cfg is None or np.isnan(p_mse_r) or np.isnan(maha_r)):
         print(f"   Skipping evaluation due to previous error or missing inputs: {err_msg}")
         # Return NaNs and preserve error
         return {"error_message": err_msg, "mse_score_scaled": mse_s, "patch_mse_score_scaled": p_mse_s,
                 "mahalanobis_score_scaled": maha_s, "hybrid_score_pipeline": hybrid_pipe,
                 "is_anomaly_mse": is_ano_mse, "threshold_mse": th_mse,
                 "is_anomaly_patch_mse": is_ano_pmse, "threshold_patch_mse": th_pmse,
                 "is_anomaly_mahalanobis": is_ano_maha, "threshold_mahalanobis": th_maha}

    if not cfg:
        current_errs.append("Model config missing for scaling/hybrid evaluation.")
    else:
        scale_p = cfg.get("scaling_parameters_from_training")
        hybrid_w = cfg.get("hybrid_score_weights_training")
        thresholds = cfg.get("loaded_thresholds")

        if not scale_p: current_errs.append("Scaling parameters ('scaling_parameters_from_training') missing in config.")
        if not hybrid_w: current_errs.append("Hybrid weights ('hybrid_score_weights_training') missing in config.")
        if not thresholds: current_errs.append("Thresholds ('loaded_thresholds') missing in config.")

    # Proceed only if essential configs are present
    if not current_errs and cfg and scale_p and hybrid_w and thresholds:
        # --- Scaling ---
        print("   Calculating scaled scores...")
        try:
            # Check keys exist before accessing
            required_scale_keys = ["mse_min_train", "mse_max_train", "patch_mse_min_train", "patch_mse_max_train", "mahalanobis_min_train", "mahalanobis_max_train"]
            missing_scale_keys = [k for k in required_scale_keys if k not in scale_p]
            if missing_scale_keys:
                 current_errs.append(f"Missing keys in scaling_parameters_from_training: {missing_scale_keys}")
            else:
                if mse_r is not None and not np.isnan(mse_r):
                    mse_s = min_max_scale_with_params(np.array([mse_r]), scale_p["mse_min_train"], scale_p["mse_max_train"])[0]
                else: print("   Skipped scaling Image MSE (raw score is NaN or None).")

                if p_mse_r is not None and not np.isnan(p_mse_r):
                    p_mse_s = min_max_scale_with_params(np.array([p_mse_r]), scale_p["patch_mse_min_train"], scale_p["patch_mse_max_train"])[0]
                else: print("   Skipped scaling Patch MSE (raw score is NaN or None).")

                if maha_r is not None and not np.isnan(maha_r):
                    maha_s = min_max_scale_with_params(np.array([maha_r]), scale_p["mahalanobis_min_train"], scale_p["mahalanobis_max_train"])[0]
                else: print("   Skipped scaling Mahalanobis (raw score is NaN or None).")
            print(f"   Scaled Scores: ImgMSE={mse_s:.4f} (for display), PatchMSE={p_mse_s:.4f}, Maha={maha_s:.4f}")
        except Exception as e:
            current_errs.append(f"Error during score scaling: {e}")

        # --- Individual Thresholding (using RAW scores) ---
        print("   Evaluating individual metrics against thresholds...")
        try:
            # Check keys exist before accessing
            required_thresh_keys = ["image_mse", "max_patch_mse", "mahalanobis_distance", "hybrid_score"]
            missing_thresh_keys = [k for k in required_thresh_keys if k not in thresholds]
            if missing_thresh_keys:
                 current_errs.append(f"Missing keys in loaded_thresholds: {missing_thresh_keys}")
            else:
                th_mse = thresholds.get("image_mse")
                th_pmse = thresholds.get("max_patch_mse")
                th_maha = thresholds.get("mahalanobis_distance")
                th_hybrid = thresholds.get("hybrid_score") # Store hybrid threshold here

                if mse_r is not None and not np.isnan(mse_r) and th_mse is not None and not np.isnan(th_mse):
                     is_ano_mse = mse_r >= th_mse
                elif mse_r is None or np.isnan(mse_r): print("   Skipped Image MSE thresholding (raw score NaN/None).")
                elif th_mse is None or np.isnan(th_mse): print("   Skipped Image MSE thresholding (threshold NaN/None).")

                if p_mse_r is not None and not np.isnan(p_mse_r) and th_pmse is not None and not np.isnan(th_pmse):
                     is_ano_pmse = p_mse_r >= th_pmse
                elif p_mse_r is None or np.isnan(p_mse_r): print("   Skipped Patch MSE thresholding (raw score NaN/None).")
                elif th_pmse is None or np.isnan(th_pmse): print("   Skipped Patch MSE thresholding (threshold NaN/None).")


                if maha_r is not None and not np.isnan(maha_r) and th_maha is not None and not np.isnan(th_maha):
                     is_ano_maha = maha_r >= th_maha
                elif maha_r is None or np.isnan(maha_r): print("   Skipped Mahalanobis thresholding (raw score NaN/None).")
                elif th_maha is None or np.isnan(th_maha): print("   Skipped Mahalanobis thresholding (threshold NaN/None).")

            print(f"   Individual Anomaly Status: MSE={is_ano_mse} (Th:{th_mse:.4f}), PatchMSE={is_ano_pmse} (Th:{th_pmse:.4f}), Maha={is_ano_maha} (Th:{th_maha:.4f})")

        except Exception as e:
            current_errs.append(f"Error during individual thresholding: {e}")

        # --- Hybrid Score Calculation (using SCALED scores) ---
        print("   Calculating pipeline hybrid score...")
        # Check keys exist before accessing
        required_hybrid_w_keys = ["patch_mse", "mahalanobis"]
        missing_hybrid_w_keys = [k for k in required_hybrid_w_keys if k not in hybrid_w]
        if missing_hybrid_w_keys:
             current_errs.append(f"Missing keys in hybrid_score_weights_training: {missing_hybrid_w_keys}")
        elif not np.any(np.isnan([p_mse_s, maha_s])): # Check if *inputs* are valid
            try:
                 hybrid_pipe = (hybrid_w["patch_mse"] * p_mse_s +
                                hybrid_w["mahalanobis"] * maha_s)
                 print(f"   Pipeline Hybrid Score (Scaled PatchMSE * {hybrid_w['patch_mse']:.2f} + Scaled Maha * {hybrid_w['mahalanobis']:.2f}) = {hybrid_pipe:.4f}")
            except Exception as e:
                current_errs.append(f"Error calculating hybrid score value: {e}")
                hybrid_pipe = np.nan # Ensure it's NaN on error
        else:
            current_errs.append("Skipped hybrid score calculation (Scaled PatchMSE or Scaled Mahalanobis is NaN).")
            hybrid_pipe = np.nan # Ensure it's NaN

    # Combine error messages
    node_err_msg = "; ".join(current_errs) if current_errs else None
    full_err = (err_msg + "; " if err_msg else "") + (node_err_msg if node_err_msg else "")
    if node_err_msg: print(f"   Warnings/Errors in evaluation: {node_err_msg}")

    # Return all calculated values, including NaNs if steps failed
    return {
        "mse_score_scaled": mse_s, "patch_mse_score_scaled": p_mse_s, "mahalanobis_score_scaled": maha_s,
        "hybrid_score_pipeline": hybrid_pipe,
        "is_anomaly_mse": is_ano_mse, "threshold_mse": th_mse,
        "is_anomaly_patch_mse": is_ano_pmse, "threshold_patch_mse": th_pmse,
        "is_anomaly_mahalanobis": is_ano_maha, "threshold_mahalanobis": th_maha,
        # Store the hybrid threshold here if available, ready for the next node
        "threshold_hybrid_pipeline": th_hybrid if 'th_hybrid' in locals() and th_hybrid is not None else np.nan,
        "error_message": full_err if full_err else None
    }


# --- apply_final_threshold_node (Revised to use threshold from previous node) ---
def apply_final_threshold_node(state: GraphState) -> Dict[str, Any]:
    print("\n--- Node: apply_final_threshold_node (2-Component Hybrid) ---")
    hybrid_val = state.get("hybrid_score_pipeline")
    # Get the hybrid threshold calculated and stored by the previous node
    th_hybrid = state.get("threshold_hybrid_pipeline")
    err_msg = state.get("error_message", "") # Get existing errors
    is_ano_final = None # Default to None

    # Check if previous steps already failed
    if err_msg and (hybrid_val is None or np.isnan(hybrid_val) or th_hybrid is None or np.isnan(th_hybrid)):
        print(f"   Skipping final thresholding due to previous error or missing values: {err_msg}")
        return {"error_message": err_msg, "is_anomaly": None} # Keep anomaly as None

    current_errs = []
    if hybrid_val is None or np.isnan(hybrid_val):
         current_errs.append("Hybrid score is missing or NaN.")
    if th_hybrid is None or np.isnan(th_hybrid):
         current_errs.append("Hybrid score threshold is missing or NaN (check thresholds JSON and previous node).")

    if not current_errs:
        try:
            is_ano_final = hybrid_val >= th_hybrid
            print(f"   Comparing Hybrid Score {hybrid_val:.4f} against Threshold {th_hybrid:.4f}. Final Anomaly = {is_ano_final}")
        except Exception as e:
             current_errs.append(f"Error during final threshold comparison: {e}")
             is_ano_final = None # Set anomaly to None on error
    else:
         is_ano_final = None # Set anomaly to None if inputs were missing


    # Combine error messages
    node_err_msg = "; ".join(current_errs) if current_errs else None
    full_err = (err_msg + "; " if err_msg else "") + (node_err_msg if node_err_msg else "")
    if node_err_msg: print(f"   Warnings/Errors in final thresholding: {node_err_msg}")

    return {
        "is_anomaly": is_ano_final,
        # Threshold is already in the state, no need to return again unless needed elsewhere
        # "threshold_hybrid_pipeline": th_hybrid,
        "error_message": full_err if full_err else None
    }


# --- generate_heatmap_node ---
# Keep as is, but ensure it checks for errors
def generate_heatmap_node(state: GraphState) -> Dict[str, Any]:
    print("\n--- Node: generate_heatmap_node ---")
    orig_pil=state.get("original_pil")
    px_err_map=state.get("pixel_error_map_np")
    err_msg = state.get("error_message", "") # Get existing errors
    is_anomaly = state.get("is_anomaly") # Check if anomaly was detected

    # Check if previous node already set an error or if not anomaly
    if err_msg:
        print(f"   Skipping heatmap generation due to previous error: {err_msg}")
        return {"error_message": err_msg, "heatmap_pil": None}
    if not is_anomaly:
         print(f"   Skipping heatmap generation as image was not flagged as an anomaly.")
         # No error, just no heatmap needed
         return {"error_message": err_msg if err_msg else None, "heatmap_pil": None}

    current_errs = []
    heatmap_pil_val = None
    if orig_pil and px_err_map is not None:
        try:
            print("   Generating heatmap...")
            heatmap_pil_val = generate_heatmap_pil(orig_pil, px_err_map)
            print("   Heatmap generated successfully.")
        except Exception as e:
            current_errs.append(f"Error during heatmap generation: {e}")
            heatmap_pil_val = None # Ensure None on error
    elif not orig_pil:
         current_errs.append("Original PIL image missing for heatmap.")
    elif px_err_map is None:
         current_errs.append("Pixel error map missing for heatmap.")


    # Combine error messages
    node_err_msg = "; ".join(current_errs) if current_errs else None
    full_err = (err_msg + "; " if err_msg else "") + (node_err_msg if node_err_msg else "")
    if node_err_msg: print(f"   Warnings/Errors in heatmap generation: {node_err_msg}")

    return {"heatmap_pil": heatmap_pil_val, "error_message": full_err if full_err else None}

# --- error_node ---
# Keep as is
def error_node(state: GraphState) -> Dict[str, Any]:
    err = state.get('error_message', "Unspecified error reached error_node.")
    print(f"\n--- Node: error_node --- Error encountered in pipeline: {err}")
    # Potentially add more logging or handling here
    return {} # Return empty dict, no state changes needed

# --- Routing Functions (Revised) ---
def route_after_preparation(state: GraphState) -> str:
    # Always check for error message first
    if state.get("error_message"):
        print("Routing: prepare_data -> error_handler")
        return "error_handler"
    else:
        print("Routing: prepare_data -> vqvae_inference")
        return "vqvae_inference"

def route_after_vqvae(state: GraphState) -> str:
     # Check if VQVAE failed critically (no recon/latent)
    if state.get("error_message") and (state.get("reconstructed_tensor") is None or state.get("latent_z_q") is None):
         print("Routing: vqvae_inference -> error_handler (Critical VQVAE Error)")
         return "error_handler"
    else:
         # Proceed even if there were minor errors, scoring node handles missing data
         print("Routing: vqvae_inference -> calculate_raw_scores")
         return "calculate_raw_scores"

def route_after_scoring(state: GraphState) -> str:
    # Check if scoring failed critically (no usable scores for hybrid)
    if state.get("error_message") and (state.get("patch_mse_score") is None or np.isnan(state.get("patch_mse_score")) or state.get("mahalanobis_score") is None or np.isnan(state.get("mahalanobis_score"))):
         print("Routing: calculate_raw_scores -> error_handler (Critical Scoring Error)")
         return "error_handler"
    else:
         print("Routing: calculate_raw_scores -> evaluate_scaled_hybrid")
         return "evaluate_scaled_hybrid"

def route_after_evaluation(state: GraphState) -> str:
     # Check if evaluation failed critically (no hybrid score or threshold)
    if state.get("error_message") and (state.get("hybrid_score_pipeline") is None or np.isnan(state.get("hybrid_score_pipeline")) or state.get("threshold_hybrid_pipeline") is None or np.isnan(state.get("threshold_hybrid_pipeline"))):
        print("Routing: evaluate_scaled_hybrid -> error_handler (Critical Evaluation Error)")
        return "error_handler"
    else:
        print("Routing: evaluate_scaled_hybrid -> apply_final_threshold")
        return "apply_final_threshold"

# This route determines if heatmap is needed *or* if errors occurred
def route_after_final_threshold(state: GraphState) -> str:
    # Always check for errors first
    if state.get("error_message"):
        print("Routing: apply_final_threshold -> error_handler")
        return "error_handler"
    # If no error, check if anomaly was detected
    if state.get("is_anomaly"): # is_anomaly should be True/False now if no error
        print("Routing: apply_final_threshold -> generate_heatmap")
        return "generate_heatmap"
    else:
        # Normal image or is_anomaly is None (shouldn't happen without error), end workflow
        print("Routing: apply_final_threshold -> END (Normal or Undetermined)")
        return END


# --- Workflow Definition (Updated with new routes) ---
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("prepare_data", prepare_initial_data_node)
workflow.add_node("vqvae_inference", vqvae_inference_node)
workflow.add_node("calculate_raw_scores", calculate_raw_scores_node)
workflow.add_node("evaluate_scaled_hybrid", evaluate_scaled_hybrid_node)
workflow.add_node("apply_final_threshold", apply_final_threshold_node)
workflow.add_node("generate_heatmap", generate_heatmap_node)
workflow.add_node("error_handler", error_node) # Terminal node for errors

# Set entry point
workflow.set_entry_point("prepare_data")

# Define edges and conditional routing
workflow.add_conditional_edges(
    "prepare_data",
    route_after_preparation,
    {"vqvae_inference": "vqvae_inference", "error_handler": "error_handler"}
)
workflow.add_conditional_edges(
    "vqvae_inference",
    route_after_vqvae,
     {"calculate_raw_scores": "calculate_raw_scores", "error_handler": "error_handler"}
)
workflow.add_conditional_edges(
    "calculate_raw_scores",
    route_after_scoring,
     {"evaluate_scaled_hybrid": "evaluate_scaled_hybrid", "error_handler": "error_handler"}
)
workflow.add_conditional_edges(
    "evaluate_scaled_hybrid",
    route_after_evaluation,
     {"apply_final_threshold": "apply_final_threshold", "error_handler": "error_handler"}
)
workflow.add_conditional_edges(
    "apply_final_threshold",
    route_after_final_threshold,
    {"generate_heatmap": "generate_heatmap", END: END, "error_handler": "error_handler"}
)

# Final connections to END
workflow.add_edge("generate_heatmap", END)
workflow.add_edge("error_handler", END) # Errors also terminate the graph

# Compile the graph
app_graph = workflow.compile()

# --- run_analysis_pipeline Function ---
# Keep as is
def run_analysis_pipeline(image_bytes_input: bytes, image_filename_input: str, user_selected_type_input: str) -> GraphState:
    initial_state = GraphState(
        image_bytes=image_bytes_input, image_filename=image_filename_input, user_selected_type=user_selected_type_input,
        determined_type=None, model_config=None, original_pil=None, processed_tensor=None,
        reconstructed_tensor=None, latent_z_q=None, pixel_error_map_np=None,
        mse_score=np.nan, patch_mse_score=np.nan, mahalanobis_score=np.nan, # Initialize scores to NaN
        mse_score_scaled=np.nan, patch_mse_score_scaled=np.nan, mahalanobis_score_scaled=np.nan,
        hybrid_score_pipeline=np.nan,
        is_anomaly_mse=None, threshold_mse=np.nan,
        is_anomaly_patch_mse=None, threshold_patch_mse=np.nan,
        is_anomaly_mahalanobis=None, threshold_mahalanobis=np.nan,
        is_anomaly=None, threshold_hybrid_pipeline=np.nan,
        heatmap_pil=None, error_message=None
    )
    print(f"\nInvoking pipeline for: {image_filename_input}, User Type Selection: {user_selected_type_input}")
    # Increase recursion limit slightly if complex error paths might occur
    final_state = app_graph.invoke(initial_state, {"recursion_limit": 30})
    print("-" * 50)
    print(f"Pipeline finished for {image_filename_input}.")
    final_error = final_state.get('error_message')
    if final_error:
         print(f"Pipeline completed with error(s): {final_error}")
    else:
         final_status = "Anomaly" if final_state.get('is_anomaly') else "Normal"
         print(f"Pipeline completed successfully. Final Status: {final_status}")
    print("-" * 50)
    return final_state