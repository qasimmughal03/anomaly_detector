# medical_anomaly_detector/pipeline.py
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import io
import json # Import json module
from typing import TypedDict, List, Optional, Dict, Any # Ensure Optional is imported

from langgraph.graph import StateGraph, END

# Assuming vqvae_models.py and utils.py are in the same directory
# and correctly define the necessary classes and functions.
from vqvae_models import VQVAE, MahalanobisScorer, LUNG_MODEL_CONFIG, BREAST_MODEL_CONFIG
from utils import (
    preprocess_image_from_bytes, compute_image_mse, compute_max_patch_mse,
    compute_pixelwise_error_map, generate_heatmap_pil, tensor_to_pil, pil_to_bytes
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Pipeline using device: {DEVICE}")

# --- Asset Paths ---
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
LUNG_MODEL_FILE = os.path.join(ASSETS_DIR, "lung_vqvae_model.pth")
BREAST_MODEL_FILE = os.path.join(ASSETS_DIR, "breast_vqvae_model.pth")
LUNG_SCORER_FILE = os.path.join(ASSETS_DIR, "lung_mahalanobis_scorer_params.npz")
BREAST_SCORER_FILE = os.path.join(ASSETS_DIR, "breast_mahalanobis_scorer_params.npz")
LUNG_THRESHOLDS_FILE = os.path.join(ASSETS_DIR, "lung_thresholds.json")
BREAST_THRESHOLDS_FILE = os.path.join(ASSETS_DIR, "breast_thresholds.json")

# --- LangGraph State Definition ---
class GraphState(TypedDict):
    image_bytes: Optional[bytes]
    image_filename: Optional[str]
    user_selected_type: str

    # Intermediate states
    determined_type: Optional[str]
    model_config: Optional[Dict[str, Any]] # This will include loaded thresholds
    original_pil: Optional[Image.Image]
    processed_tensor: Optional[torch.Tensor] # Batch of 1, (1, C, H, W)
    reconstructed_tensor: Optional[torch.Tensor] # Batch of 1
    latent_z_q: Optional[torch.Tensor]
    pixel_error_map_np: Optional[np.ndarray]

    # Scores
    mse_score: Optional[float]
    patch_mse_score: Optional[float]
    mahalanobis_score: Optional[float]
    hybrid_score: Optional[float]

    # Decision & Output
    is_anomaly: Optional[bool]
    actual_threshold_used: Optional[float] # Stores the threshold value that was applied
    heatmap_pil: Optional[Image.Image] # For display in Streamlit

    error_message: Optional[str]


# --- Helper Functions for Loading Assets ---
def load_thresholds(threshold_file_path: str) -> Optional[Dict[str, float]]:
    if not os.path.exists(threshold_file_path):
        print(f"  ERROR (load_thresholds): Threshold file not found: {threshold_file_path}")
        return None
    try:
        with open(threshold_file_path, 'r') as f:
            thresholds = json.load(f)
        print(f"  (load_thresholds): Thresholds loaded successfully from {threshold_file_path}")
        return thresholds
    except Exception as e:
        print(f"  ERROR (load_thresholds): Could not load or parse JSON from {threshold_file_path}: {e}")
        return None

def load_vqvae_model(model_path: str, model_core_params: dict) -> Optional[VQVAE]:
    print(f"  Attempting to load VQ-VAE model...")
    print(f"  Model path: {model_path}")
    print(f"  Using core_params: {model_core_params}")
    if not os.path.exists(model_path):
        print(f"  ERROR (load_vqvae_model): Model file not found at path: {model_path}")
        return None
    try:
        required_keys = ["in_channels", "h_dim", "res_h_dim", "n_res_layers", "n_embeddings", "embedding_dim", "beta", "out_channels"]
        missing_keys = [key for key in required_keys if key not in model_core_params]
        if missing_keys:
            print(f"  ERROR (load_vqvae_model): Missing keys in model_core_params for VQVAE instantiation: {missing_keys}")
            return None

        model = VQVAE(**model_core_params).to(DEVICE)
        print(f"  (load_vqvae_model): VQVAE model instantiated with provided params.")
        
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"  (load_vqvae_model): Model state_dict loaded successfully from {model_path} and set to eval mode.")
        return model
    except RuntimeError as e:
        print(f"  RUNTIME ERROR (load_vqvae_model): Error loading state_dict for VQVAE model from {model_path}. This often indicates a mismatch between saved model architecture and definition. Details: {e}")
        print(f"  Please ensure the parameters in vqvae_models.py (specifically within LUNG_MODEL_CONFIG['params'] or BREAST_MODEL_CONFIG['params']) exactly match the architecture of the saved model at {model_path}.")
        return None
    except Exception as e:
        print(f"  GENERAL ERROR (load_vqvae_model): Error loading VQVAE model from {model_path}. Details: {e}")
        return None

def load_mahalanobis_scorer(scorer_params_path: str) -> Optional[MahalanobisScorer]:
    if not os.path.exists(scorer_params_path):
        print(f"  Warning (load_mahalanobis_scorer): Scorer params file not found: {scorer_params_path}")
        return None
    try:
        params = np.load(scorer_params_path)
        scorer = MahalanobisScorer()
        # Ensure keys exist before accessing
        if 'mean_embedding' not in params or 'inv_covariance' not in params:
            print(f"  ERROR (load_mahalanobis_scorer): mean_embedding or inv_covariance not found in {scorer_params_path}.")
            return None
        scorer.fit_params(params['mean_embedding'], params['inv_covariance'])
        if not scorer.fitted: 
            print(f"  Warning (load_mahalanobis_scorer): Scorer params loaded from {scorer_params_path} but scorer did not successfully fit.")
            return None 
        print(f"  (load_mahalanobis_scorer): Mahalanobis scorer loaded and fitted from {scorer_params_path}")
        return scorer
    except Exception as e:
        print(f"  ERROR (load_mahalanobis_scorer): Could not load or fit Mahalanobis scorer from {scorer_params_path}: {e}")
        return None

# --- LangGraph Node Functions ---
def prepare_initial_data_node(state: GraphState) -> Dict[str, Any]:
    print("\n--- Node: prepare_initial_data_node ---")
    image_bytes = state.get("image_bytes")
    user_type = state.get("user_selected_type") # This comes from Streamlit UI
    error_msg = None
    determined_type_val = None
    model_config_val = None
    original_pil_val = None
    processed_tensor_val = None

    print(f"  User selected type: {user_type}")

    if not image_bytes:
        error_msg = "No image data provided."
    else:
        threshold_file_path = None
        if user_type == "Lung":
            determined_type_val = "lung"
            model_config_val = LUNG_MODEL_CONFIG.copy() # Use a copy to add dynamic thresholds
            threshold_file_path = LUNG_THRESHOLDS_FILE
        elif user_type == "Breast":
            determined_type_val = "breast"
            model_config_val = BREAST_MODEL_CONFIG.copy()
            threshold_file_path = BREAST_THRESHOLDS_FILE
        # "Auto-Detect" was removed from Streamlit options for simplicity, can be added back later
        else:
            error_msg = f"Invalid user selected type: '{user_type}'. Please select 'Lung' or 'Breast'."
        
        print(f"  Determined type (tentative): {determined_type_val}")

        if not error_msg and model_config_val and threshold_file_path:
            loaded_thresholds_val = load_thresholds(threshold_file_path)
            if loaded_thresholds_val:
                model_config_val["loaded_thresholds"] = loaded_thresholds_val
                print(f"  Successfully loaded thresholds into model_config: {list(loaded_thresholds_val.keys())}")
            else:
                error_msg = f"Critical Error: Failed to load thresholds for {determined_type_val} from {threshold_file_path}. Ensure the JSON file exists and is correctly formatted in 'assets/' directory."
        
        print(f"  Model config assigned: {model_config_val is not None}")
        if model_config_val:
            print(f"  Model config keys after attempting threshold load: {list(model_config_val.keys())}")

        if not error_msg and model_config_val:
            try:
                original_pil_val = Image.open(io.BytesIO(image_bytes)).convert('L')
                # Ensure 'img_size' key exists before accessing
                if "img_size" not in model_config_val:
                    error_msg = f"'img_size' not found in model_config for type '{determined_type_val}'."
                else:
                    img_size = model_config_val["img_size"]
                    processed_tensor_val = preprocess_image_from_bytes(image_bytes, img_size)
                    if processed_tensor_val is None:
                        error_msg = "Image preprocessing failed."
            except Exception as e:
                error_msg = f"Failed to load/preprocess image: {str(e)}"
        elif not error_msg and not model_config_val and determined_type_val: # Only error if determined_type was set but config failed
             error_msg = f"Model configuration base not found for determined type '{determined_type_val}'."
            
    if error_msg:
        print(f"  Error in prepare_initial_data_node: {error_msg}")
        return {"error_message": error_msg, "model_config": None} # Pass along None for model_config
    
    print(f"  Exiting prepare_initial_data_node. Determined type: {determined_type_val}, Model config set: {model_config_val is not None}")
    return {
        "determined_type": determined_type_val,
        "model_config": model_config_val,
        "original_pil": original_pil_val,
        "processed_tensor": processed_tensor_val.to(DEVICE) if processed_tensor_val is not None else None,
    }

def vqvae_inference_node(state: GraphState) -> Dict[str, Any]:
    print("\n--- Node: vqvae_inference_node ---")
    determined_type = state.get("determined_type")
    model_config = state.get("model_config")
    processed_tensor = state.get("processed_tensor")
    print(f"  Type: {determined_type}, Model config set: {model_config is not None}, Processed tensor set: {processed_tensor is not None}")

    if not model_config or processed_tensor is None:
        err = "Model config or processed tensor missing for inference (from previous step)."
        print(f"  Error: {err}")
        return {"error_message": err}

    model_path = LUNG_MODEL_FILE if determined_type == "lung" else BREAST_MODEL_FILE
    
    model = load_vqvae_model(model_path, model_config["params"])
    if model is None:
        err = f"Failed to load VQ-VAE model from {model_path} for type {determined_type}."
        print(f"  Error: {err}")
        return {"error_message": err}

    with torch.no_grad():
        _vq_loss, recon_tensor, _perplexity, z_q, _z_e_pre_quant, _min_indices = model(processed_tensor)
    
    pixel_err_map = compute_pixelwise_error_map(processed_tensor, recon_tensor)
    print(f"  Inference successful. Recon tensor shape: {recon_tensor.shape}, z_q shape: {z_q.shape}")
    return {
        "reconstructed_tensor": recon_tensor,
        "latent_z_q": z_q,
        "pixel_error_map_np": pixel_err_map
    }

def calculate_scores_node(state: GraphState) -> Dict[str, Any]:
    print("\n--- Node: calculate_scores_node ---")
    img_type = state.get("determined_type")
    model_config = state.get("model_config")
    processed_tensor = state.get("processed_tensor")
    reconstructed_tensor = state.get("reconstructed_tensor")
    latent_z_q = state.get("latent_z_q")
    print(f"  Type: {img_type}, Model config set: {model_config is not None}")
    print(f"  Inputs - Processed tensor: {processed_tensor is not None}, Recon tensor: {reconstructed_tensor is not None}, z_q: {latent_z_q is not None}")

    if model_config is None or \
       processed_tensor is None or \
       reconstructed_tensor is None or \
       latent_z_q is None:
        error_parts = []
        if model_config is None: error_parts.append("model_config")
        if processed_tensor is None: error_parts.append("processed_tensor")
        if reconstructed_tensor is None: error_parts.append("reconstructed_tensor")
        if latent_z_q is None: error_parts.append("latent_z_q")
        err = f"Missing data for score calculation: {', '.join(error_parts)} missing."
        print(f"  Error: {err}")
        return {"error_message": err, "mse_score": np.nan, "patch_mse_score": np.nan, "mahalanobis_score": np.nan}

    mse = compute_image_mse(processed_tensor, reconstructed_tensor)
    print(f"  Calculated MSE: {mse:.6f}")

    if "img_size" not in model_config or "patch_size_ratio" not in model_config:
        err = "'img_size' or 'patch_size_ratio' missing in model_config for patch MSE."
        print(f"  Error: {err}")
        # Return already computed mse, others will be NaN
        return {"error_message": err, "mse_score": mse, "patch_mse_score": np.nan, "mahalanobis_score": np.nan}
    patch_size = model_config["img_size"] // model_config["patch_size_ratio"]
    max_patch_mse = compute_max_patch_mse(processed_tensor, reconstructed_tensor, patch_size)
    print(f"  Calculated Max Patch MSE (patch size {patch_size}): {max_patch_mse:.6f}")

    scorer_path = LUNG_SCORER_FILE if img_type == "lung" else BREAST_SCORER_FILE
    scorer = load_mahalanobis_scorer(scorer_path)
    
    maha_score = np.nan
    if scorer and scorer.fitted:
        try:
            pooled_z_q = F.adaptive_avg_pool2d(latent_z_q, (1, 1)).squeeze(-1).squeeze(-1)
            if pooled_z_q.ndim == 1: pooled_z_q = pooled_z_q.unsqueeze(0)
            maha_score_array = scorer.score(pooled_z_q.cpu().numpy())
            if maha_score_array is not None and maha_score_array.size > 0:
                maha_score = maha_score_array[0]
            else:
                print("  Warning: Mahalanobis scorer returned empty or None result.")
        except Exception as e:
            print(f"  Error during Mahalanobis scoring: {e}")
    else:
        print(f"  Warning: Mahalanobis scorer for {img_type} not loaded/fitted or params missing. Mahalanobis score will be NaN.")
    print(f"  Calculated Mahalanobis Score: {maha_score}")
            
    return {"mse_score": mse, "patch_mse_score": max_patch_mse, "mahalanobis_score": maha_score}

def calculate_hybrid_score_node(state: GraphState) -> Dict[str, Any]:
    print("\n--- Node: calculate_hybrid_score_node ---")
    mse = state.get("mse_score")
    maha = state.get("mahalanobis_score")
    model_config = state.get("model_config")
    print(f"  Type: {state.get('determined_type')}, Model config set: {model_config is not None}")
    print(f"  Received MSE: {mse}, Mahalanobis: {maha}")

    if model_config is None or mse is None or np.isnan(mse) or maha is None or np.isnan(maha):
        warning_msg = "MSE, Mahalanobis score, or model_config is missing/NaN. Hybrid score will be NaN."
        print(f"  Warning: {warning_msg}")
        # Propagate existing error or set new one; ensure hybrid_score is NaN
        return {"hybrid_score": np.nan, "error_message": state.get("error_message") or warning_msg} 

    if "norm_constants" not in model_config:
        err = "'norm_constants' missing in model_config for hybrid score calculation."
        print(f"  Error: {err}")
        return {"hybrid_score": np.nan, "error_message": err}
        
    norm_const = model_config["norm_constants"]
    # Ensure keys exist in norm_const
    required_norm_keys = ["mse_min", "mse_max", "maha_min", "maha_max"]
    if not all(key in norm_const for key in required_norm_keys):
        err = f"One or more normalization constants ({', '.join(required_norm_keys)}) missing in model_config['norm_constants']."
        print(f"  Error: {err}")
        return {"hybrid_score": np.nan, "error_message": err}

    norm_mse = (mse - norm_const["mse_min"]) / (norm_const["mse_max"] - norm_const["mse_min"] + 1e-9)
    norm_maha = (maha - norm_const["maha_min"]) / (norm_const["maha_max"] - norm_const["maha_min"] + 1e-9)
    norm_mse = np.clip(norm_mse, 0, 1) # Clip to [0,1] as per original logic
    norm_maha = np.clip(norm_maha, 0, 1)
    
    hybrid = 0.5 * norm_mse + 0.5 * norm_maha # Example weights
    print(f"  Normalized MSE: {norm_mse:.4f}, Normalized Mahalanobis: {norm_maha:.4f}, Calculated Hybrid Score: {hybrid:.4f}")
    return {"hybrid_score": hybrid}

def apply_threshold_node(state: GraphState) -> Dict[str, Any]:
    print("\n--- Node: apply_threshold_node ---")
    hybrid_score = state.get("hybrid_score")
    model_config = state.get("model_config")
    determined_type = state.get("determined_type")
    print(f"  Type: {determined_type}, Model config set: {model_config is not None}")
    print(f"  Received Hybrid Score: {hybrid_score}")

    if model_config is None or hybrid_score is None or np.isnan(hybrid_score):
        err = "Hybrid score or model config missing, cannot apply threshold."
        print(f"  Error: {err}")
        return {"is_anomaly": False, "actual_threshold_used": np.nan, 
                "error_message": state.get("error_message") or err}

    loaded_thresholds = model_config.get("loaded_thresholds")
    if not loaded_thresholds or "hybrid_score" not in loaded_thresholds:
        err = f"'loaded_thresholds' dictionary or 'hybrid_score' key missing in model_config for {determined_type}."
        available_keys_str = f"(Available threshold keys: {list(loaded_thresholds.keys()) if loaded_thresholds else 'None'})"
        print(f"  Error: {err} {available_keys_str}")
        return {"is_anomaly": False, "actual_threshold_used": np.nan, "error_message": err}

    threshold_to_apply = loaded_thresholds["hybrid_score"]
    is_anomaly = hybrid_score >= threshold_to_apply # Anomaly if score is greater or equal
    print(f"  Applied threshold for hybrid_score: {threshold_to_apply:.4f}. Is anomaly: {is_anomaly}")
    return {"is_anomaly": is_anomaly, "actual_threshold_used": threshold_to_apply}

def generate_heatmap_node(state: GraphState) -> Dict[str, Any]:
    print("\n--- Node: generate_heatmap_node ---")
    original_pil = state.get("original_pil")
    pixel_error_map_np = state.get("pixel_error_map_np")
    print(f"  Original PIL set: {original_pil is not None}, Pixel error map set: {pixel_error_map_np is not None}")
    
    if original_pil and pixel_error_map_np is not None:
        try:
            heatmap_pil = generate_heatmap_pil(original_pil, pixel_error_map_np)
            print(f"  Heatmap generated successfully.")
            return {"heatmap_pil": heatmap_pil}
        except Exception as e:
            err = f"Heatmap generation failed: {e}"
            print(f"  Error: {err}")
            return {"heatmap_pil": None, "error_message": state.get("error_message") or err}
    else:
        warn_msg = "Original image or pixel error map missing for heatmap generation."
        print(f"  Warning: {warn_msg}")
        return {"heatmap_pil": None, "error_message": state.get("error_message") or warn_msg}


def error_node(state: GraphState) -> Dict[str, Any]:
    # This node is mostly a sink if errors are handled by returning {"error_message": ...} in prior nodes.
    # The final state will contain the error message.
    error_msg = state.get('error_message', "An unspecified error occurred in the pipeline.")
    print(f"\n--- Node: error_node ---")
    print(f"  Image: {state.get('image_filename', 'N/A')}, Error: {error_msg}")
    # No specific state change needed here as the error_message is already in the state.
    return {}


# --- Conditional Edges ---
def route_after_preparation(state: GraphState) -> str:
    if state.get("error_message"):
        print("Routing after preparation: to error_handler")
        return "error_handler"
    print("Routing after preparation: to vqvae_inference")
    return "vqvae_inference"

def route_after_threshold(state: GraphState) -> str:
    if state.get("error_message"): # If apply_threshold itself had an error
        print("Routing after threshold: to error_handler (due to error in apply_threshold)")
        return "error_handler"
    if state.get("is_anomaly"):
        print("Routing after threshold: to generate_heatmap (anomaly)")
        return "generate_heatmap"
    print("Routing after threshold: to END (normal)")
    return END 

# --- Build LangGraph ---
workflow = StateGraph(GraphState)

workflow.add_node("prepare_data", prepare_initial_data_node)
workflow.add_node("vqvae_inference", vqvae_inference_node)
workflow.add_node("calculate_scores", calculate_scores_node)
workflow.add_node("calculate_hybrid", calculate_hybrid_score_node)
workflow.add_node("apply_threshold", apply_threshold_node)
workflow.add_node("generate_heatmap", generate_heatmap_node)
workflow.add_node("error_handler", error_node) # Central error logging/handling node

# Entry Point
workflow.set_entry_point("prepare_data")

# Edges
workflow.add_conditional_edges(
    "prepare_data", 
    route_after_preparation, 
    {"vqvae_inference": "vqvae_inference", "error_handler": "error_handler"}
)
# If vqvae_inference, calculate_scores, or calculate_hybrid sets an error_message,
# the subsequent nodes should ideally handle it or the graph could route to error_handler.
# For now, errors are propagated and checked in conditional routers or final display.
# A more robust graph would have explicit error paths from each fallible node.
workflow.add_edge("vqvae_inference", "calculate_scores")
workflow.add_edge("calculate_scores", "calculate_hybrid")
workflow.add_edge("calculate_hybrid", "apply_threshold")

workflow.add_conditional_edges(
    "apply_threshold",
    route_after_threshold,
    {"generate_heatmap": "generate_heatmap", END: END, "error_handler": "error_handler"}
)
workflow.add_edge("generate_heatmap", END)
workflow.add_edge("error_handler", END)

# Compile the graph
app_graph = workflow.compile()

# Function to invoke the graph for Streamlit
def run_analysis_pipeline(image_bytes_input: bytes, image_filename_input: str, user_selected_type_input: str) -> GraphState:
    initial_state = GraphState(
        image_bytes=image_bytes_input,
        image_filename=image_filename_input,
        user_selected_type=user_selected_type_input,
        determined_type=None, 
        model_config=None, 
        original_pil=None, 
        processed_tensor=None,
        reconstructed_tensor=None, 
        latent_z_q=None, 
        pixel_error_map_np=None,
        mse_score=None, 
        patch_mse_score=None, 
        mahalanobis_score=None, 
        hybrid_score=None,
        is_anomaly=None, 
        actual_threshold_used=None, # Corrected key
        heatmap_pil=None, 
        error_message=None
    )
    
    print(f"\nInvoking pipeline for: {image_filename_input}, Type: {user_selected_type_input}")
    # The invoke method returns the final state of the graph.
    final_state = app_graph.invoke(initial_state, {"recursion_limit": 25}) # Increased recursion limit
    print(f"Pipeline finished for {image_filename_input}. Final state error: {final_state.get('error_message')}")
    return final_state