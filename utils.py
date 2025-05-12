# medical_anomaly_detector/utils.py
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import os
import google.generativeai as genai
from typing import Optional, Tuple, Dict

# --- Gemini API Key Configuration ---
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
GEMINI_CONFIGURED = False
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        GEMINI_CONFIGURED = True
        print("Gemini API Key configured successfully.")
    except Exception as e:
        print(f"ERROR configuring Gemini API: {e}. Auto-Detect will fail.")
else:
    print("WARNING: GOOGLE_API_KEY environment variable not found. Auto-Detect feature will be unavailable.")

# --- Custom CLAHE Transform ---
# ... (keep CLAHETransformPIL as it is) ...
class CLAHETransformPIL(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        if pil_img.mode != 'L':
            pil_img = pil_img.convert('L')
        img_np = np.array(pil_img)
        img_clahe_np = self.clahe.apply(img_np)
        return Image.fromarray(img_clahe_np)

# --- Image Preprocessing ---
# ... (keep get_transform and preprocess_image_from_bytes as they are) ...
def get_transform(img_size: int, clahe_config: Optional[Dict] = None):
    transform_list = []
    if clahe_config and clahe_config.get("apply_clahe", False):
        print(f"Applying CLAHE with params: {clahe_config.get('clip_limit', 2.0)}, {clahe_config.get('tile_grid_size', (8,8))}")
        transform_list.append(CLAHETransformPIL(
            clip_limit=clahe_config.get("clip_limit", 2.0),
            tile_grid_size=tuple(clahe_config.get("tile_grid_size", (8,8)))
        ))
    transform_list.extend([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5]) # If your model was trained with this
    ])
    return transforms.Compose(transform_list)

def preprocess_image_from_bytes(image_bytes: bytes, img_size: int, clahe_config: Optional[Dict] = None) -> Optional[torch.Tensor]:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        transform = get_transform(img_size, clahe_config=clahe_config)
        return transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# --- LLM Image Type Classification (Revised) ---
def classify_image_type_with_llm(image_bytes: bytes) -> Tuple[str, Optional[str]]: # Return type, error message
    """
    Classifies an image as 'lung', 'breast', or 'unknown' using Gemini.
    Returns a tuple: (classification_result, error_message)
    """
    from vqvae_models import LUNG_MODEL_TYPE_ID, BREAST_MODEL_TYPE_ID # Import here

    if not GEMINI_CONFIGURED:
        error_msg = "LLM Classification Error: Gemini API not configured (check GOOGLE_API_KEY)."
        print(f"   {error_msg}")
        return "unknown", error_msg

    try:
        print("   Attempting LLM image classification...")
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Ensure model is compatible with vision - use a recent reliable one
        model = genai.GenerativeModel('gemini-1.5-flash') # Or 'gemini-pro-vision' if flash has issues

        prompt = (
            f"Analyze the medical image. Is it primarily an image of a human {LUNG_MODEL_TYPE_ID} or a human {BREAST_MODEL_TYPE_ID}? "
            f"Respond with only ONE single word: '{LUNG_MODEL_TYPE_ID}', '{BREAST_MODEL_TYPE_ID}', or 'unknown'."
        )

        # Generate content
        response = model.generate_content([prompt, pil_image])

        # Debugging: Print raw response
        try:
            raw_text = response.text
            print(f"   LLM Raw Response Text: '{raw_text}'")
        except Exception as e:
            # Handle potential issue if response structure is unexpected (e.g., blocked content)
            print(f"   ERROR accessing LLM response text: {e}")
            print(f"   LLM Full Response object: {response}")
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 error_msg = f"LLM Error: Blocked - {response.prompt_feedback.block_reason}"
                 return "unknown", error_msg
            return "unknown", f"LLM Error: Could not parse response text ({e})"

        llm_output = raw_text.strip().lower()

        # More specific check for the exact word
        if llm_output == LUNG_MODEL_TYPE_ID:
            print(f"   LLM classified as: {LUNG_MODEL_TYPE_ID}")
            return LUNG_MODEL_TYPE_ID, None
        elif llm_output == BREAST_MODEL_TYPE_ID:
            print(f"   LLM classified as: {BREAST_MODEL_TYPE_ID}")
            return BREAST_MODEL_TYPE_ID, None
        else:
            warn_msg = f"LLM classification unclear or unknown. Received: '{llm_output}'. Expected '{LUNG_MODEL_TYPE_ID}' or '{BREAST_MODEL_TYPE_ID}'."
            print(f"   {warn_msg}")
            # Treat unclear as an error for the pipeline
            return "unknown", warn_msg

    except Exception as e:
        error_msg = f"Error during LLM API call or processing: {e}"
        print(f"   {error_msg}")
        return "unknown", error_msg

# --- Score Calculation Helpers ---
# ... (keep compute_image_mse, compute_max_patch_mse, compute_pixelwise_error_map as they are) ...
def compute_image_mse(original_tensor: torch.Tensor, reconstructed_tensor: torch.Tensor) -> float:
    if original_tensor.device != reconstructed_tensor.device:
        reconstructed_tensor = reconstructed_tensor.to(original_tensor.device)
    return F.mse_loss(reconstructed_tensor, original_tensor).item()

def compute_max_patch_mse(original_tensor: torch.Tensor, reconstructed_tensor: torch.Tensor, patch_size: int) -> float:
    if original_tensor.device != reconstructed_tensor.device:
        reconstructed_tensor = reconstructed_tensor.to(original_tensor.device)
    if original_tensor.ndim == 4: original_tensor = original_tensor.squeeze(0)
    if reconstructed_tensor.ndim == 4: reconstructed_tensor = reconstructed_tensor.squeeze(0)
    _c, h, w = original_tensor.shape
    actual_patch_size = max(1, min(patch_size, h, w))
    if actual_patch_size == 0 :
        return F.mse_loss(reconstructed_tensor, original_tensor).item()
    patch_errors = []
    stride = actual_patch_size
    for r_idx in range(0, h - actual_patch_size + 1, stride):
        for c_idx in range(0, w - actual_patch_size + 1, stride):
            patch_orig = original_tensor[:, r_idx:r_idx+actual_patch_size, c_idx:c_idx+actual_patch_size]
            patch_recon = reconstructed_tensor[:, r_idx:r_idx+actual_patch_size, c_idx:c_idx+actual_patch_size]
            patch_errors.append(F.mse_loss(patch_recon, patch_orig).item())
    if not patch_errors: return F.mse_loss(reconstructed_tensor, original_tensor).item()
    return np.max(patch_errors)

def compute_pixelwise_error_map(original_tensor: torch.Tensor, reconstructed_tensor: torch.Tensor) -> np.ndarray:
    if original_tensor.device != reconstructed_tensor.device:
        reconstructed_tensor = reconstructed_tensor.to(original_tensor.device)
    original_np = original_tensor.cpu().detach().squeeze().numpy()
    reconstructed_np = reconstructed_tensor.cpu().detach().squeeze().numpy()
    if original_np.ndim != 2: raise ValueError(f"Original tensor for pixel_error_map not 2D. Shape: {original_np.shape}")
    if reconstructed_np.ndim != 2: raise ValueError(f"Reconstructed tensor for pixel_error_map not 2D. Shape: {reconstructed_np.shape}")
    return (original_np - reconstructed_np)**2


# --- Scaling Helper ---
# ... (keep min_max_scale_with_params as it is) ...
def min_max_scale_with_params(arr: np.ndarray, min_train_val: float, max_train_val: float) -> np.ndarray:
    if not isinstance(arr, np.ndarray): arr = np.array(arr)
    if arr.size == 0: return np.array([])
    if np.isnan(min_train_val) or np.isnan(max_train_val):
        return np.full_like(arr, np.nan, dtype=float)
    arr_clipped = np.clip(arr, min_train_val, max_train_val)
    if abs(max_train_val - min_train_val) < 1e-9:
        return np.zeros_like(arr_clipped, dtype=float)
    return (arr_clipped - min_train_val) / (max_train_val - min_train_val + 1e-9)


# --- Visualization Helpers ---
# ... (keep tensor_to_pil, generate_heatmap_pil, pil_to_bytes as they are) ...
def tensor_to_pil(tensor_image: torch.Tensor) -> Image.Image:
    if tensor_image.ndim == 4: tensor_image = tensor_image.squeeze(0)
    tensor_image = tensor_image.cpu().detach()
    if tensor_image.ndim == 3 and tensor_image.shape[0] == 1:
        tensor_image = tensor_image.squeeze(0)
    return transforms.ToPILImage()(tensor_image)

def generate_heatmap_pil(original_pil: Image.Image, error_map_np: np.ndarray,
                         colormap_name='hot', alpha=0.5) -> Image.Image:
    if original_pil.mode != 'L': original_pil_gray = original_pil.convert('L')
    else: original_pil_gray = original_pil
    original_cv_gray = np.array(original_pil_gray)
    original_cv_rgb = cv2.cvtColor(original_cv_gray, cv2.COLOR_GRAY2RGB)
    norm_error_map = error_map_np.astype(np.float32)
    min_val, max_val = np.nanmin(norm_error_map), np.nanmax(norm_error_map)
    if abs(max_val - min_val) > 1e-9: norm_error_map = (norm_error_map - min_val) / (max_val - min_val + 1e-9)
    else: norm_error_map = np.zeros_like(norm_error_map)
    norm_error_map = np.nan_to_num(norm_error_map)
    cmap = plt.get_cmap(colormap_name)
    heatmap_rgba = cmap(norm_error_map)
    heatmap_rgb_float = heatmap_rgba[..., :3]
    heatmap_rgb_uint8 = (heatmap_rgb_float * 255).astype(np.uint8)
    if heatmap_rgb_uint8.shape[:2] != original_cv_rgb.shape[:2]:
        heatmap_rgb_uint8 = cv2.resize(heatmap_rgb_uint8, (original_cv_rgb.shape[1], original_cv_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlayed_image_cv = cv2.addWeighted(original_cv_rgb, 1 - alpha, heatmap_rgb_uint8, alpha, 0)
    return Image.fromarray(overlayed_image_cv)

def pil_to_bytes(pil_image: Image.Image, format='PNG') -> bytes:
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()