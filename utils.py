# medical_anomaly_detector/utils.py
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io

# --- Image Preprocessing ---
def get_transform(img_size: int): # Add type hint for clarity
    # Assuming grayscale, no CLAHE for simplicity here.
    # If CLAHE is needed, the CLAHETransform class should be defined here or imported.
    transform_list = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # Add any normalization if your models were trained with it (e.g., transforms.Normalize)
    ]
    return transforms.Compose(transform_list)

def preprocess_image_from_bytes(image_bytes: bytes, img_size: int) -> torch.Tensor | None:
    """Loads image from bytes, converts to grayscale, and applies transform."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('L') # Ensure grayscale
        transform = get_transform(img_size)
        return transform(image).unsqueeze(0) # Add batch dimension
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# --- Score Calculation Helpers ---
def compute_image_mse(original_tensor: torch.Tensor, reconstructed_tensor: torch.Tensor) -> float:
    return F.mse_loss(reconstructed_tensor, original_tensor).item()

def compute_max_patch_mse(original_tensor: torch.Tensor, reconstructed_tensor: torch.Tensor, patch_size: int) -> float:
    if original_tensor.ndim == 4: original_tensor = original_tensor.squeeze(0)
    if reconstructed_tensor.ndim == 4: reconstructed_tensor = reconstructed_tensor.squeeze(0)
    
    _c, h, w = original_tensor.shape # Expects (C, H, W)
    actual_patch_size = max(1, min(patch_size, h, w))
    if actual_patch_size == 0 : actual_patch_size = min(h,w)

    patch_errors = []
    stride = actual_patch_size
    for r_idx in range(0, h - actual_patch_size + 1, stride):
        for c_idx in range(0, w - actual_patch_size + 1, stride):
            patch_orig = original_tensor[:, r_idx:r_idx+actual_patch_size, c_idx:c_idx+actual_patch_size]
            patch_recon = reconstructed_tensor[:, r_idx:r_idx+actual_patch_size, c_idx:c_idx+actual_patch_size]
            patch_errors.append(F.mse_loss(patch_recon, patch_orig).item())
    
    if not patch_errors:
        return F.mse_loss(reconstructed_tensor, original_tensor).item() # Overall MSE if no patches
    return np.max(patch_errors) if patch_errors else 0.0

def compute_pixelwise_error_map(original_tensor: torch.Tensor, reconstructed_tensor: torch.Tensor) -> np.ndarray:
    if original_tensor.ndim == 4: original_tensor = original_tensor.squeeze(0)
    if reconstructed_tensor.ndim == 4: reconstructed_tensor = reconstructed_tensor.squeeze(0)
    
    original_np = original_tensor.cpu().detach().squeeze().numpy() # Squeeze channel if 1
    reconstructed_np = reconstructed_tensor.cpu().detach().squeeze().numpy() # Squeeze channel if 1
    
    # Ensure 2D
    if original_np.ndim == 3 and original_np.shape[0] == 1: original_np = original_np[0]
    if reconstructed_np.ndim == 3 and reconstructed_np.shape[0] == 1: reconstructed_np = reconstructed_np[0]

    if original_np.ndim != 2 or reconstructed_np.ndim != 2:
         raise ValueError(f"Pixelwise error map inputs must resolve to 2D after squeeze. Got shapes {original_np.shape}, {reconstructed_np.shape}")
    return (original_np - reconstructed_np)**2

# --- Visualization Helpers ---
def tensor_to_pil(tensor_image: torch.Tensor) -> Image.Image:
    if tensor_image.ndim == 4: tensor_image = tensor_image[0]
    tensor_image = tensor_image.cpu().detach()
    if tensor_image.shape[0] == 1: tensor_image = tensor_image.squeeze(0)
    return transforms.ToPILImage()(tensor_image)

def generate_heatmap_pil(original_pil: Image.Image, error_map_np: np.ndarray, colormap_name='hot') -> Image.Image:
    original_cv_gray = np.array(original_pil.convert('L'))
    original_cv_bgr = cv2.cvtColor(original_cv_gray, cv2.COLOR_GRAY2BGR)

    # Normalize error map for visualization
    norm_error_map = error_map_np.astype(np.float32) # Ensure float for division
    min_val, max_val = np.min(norm_error_map), np.max(norm_error_map)
    if max_val - min_val > 1e-6: # Avoid division by zero
        norm_error_map = (norm_error_map - min_val) / (max_val - min_val)
    else:
        norm_error_map = np.zeros_like(norm_error_map)
        
    norm_error_map_uint8 = (norm_error_map * 255).astype(np.uint8)

    # Get matplotlib colormap
    cmap = plt.get_cmap(colormap_name)
    heatmap_rgba = cmap(norm_error_map) # This is (H, W, 4) RGBA
    heatmap_rgb = (heatmap_rgba[..., :3] * 255).astype(np.uint8) # Convert to RGB uint8
    heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
    
    if heatmap_bgr.shape[:2] != original_cv_bgr.shape[:2]:
        heatmap_bgr = cv2.resize(heatmap_bgr, (original_cv_bgr.shape[1], original_cv_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    overlayed_image = cv2.addWeighted(original_cv_bgr, 0.5, heatmap_bgr, 0.5, 0)
    return Image.fromarray(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))


def pil_to_bytes(pil_image: Image.Image, format='PNG') -> bytes:
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()