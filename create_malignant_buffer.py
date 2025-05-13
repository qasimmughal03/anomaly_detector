import os
import argparse
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError # Import UnidentifiedImageError
from typing import List, Tuple, Dict, Any
import glob # For finding image files

# --- Project Imports (assuming this script is in the root of medical_anomaly_detector) ---
# Add project root to path to allow imports if script is run from elsewhere,
# but best to run from project root.
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from pipeline import load_vqvae_model, load_json_file, DEVICE # DEVICE is set in pipeline.py
    from utils import preprocess_image_from_bytes
    from il_utils import get_buffer_path, save_buffer # Using your existing save_buffer
    from vqvae_models import VQVAE # Needed for type hinting and model loading
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure this script is in the 'medical_anomaly_detector' project root,")
    print("and all required files (pipeline.py, utils.py, il_utils.py, vqvae_models.py) are present.")
    exit()

def create_anomaly_buffer(
    image_dir: str,
    image_type: str, # "lung" or "breast"
    anomaly_label: str, # e.g., "cancer"
    model_path: str,
    config_path: str,
    output_buffer_name: str, # e.g., "cancer_anomalies"
    max_images: int = 50,
    device: torch.device = DEVICE
) -> None:
    """
    Creates an anomaly buffer by processing images, extracting features,
    and saving them with a given label.
    """
    print(f"Starting creation of anomaly buffer for '{anomaly_label}' from '{image_type}' images.")
    print(f"Image source directory: {image_dir}")
    print(f"Max images to process: {max_images}")

    # 1. Load Model Runtime Config
    print(f"Loading model config from: {config_path}")
    model_config = load_json_file(config_path, f"{image_type} Runtime Config for Buffer Creation")
    if not model_config:
        print("Failed to load model config. Exiting.")
        return
    try:
        img_size = model_config["img_size"]
        clahe_params = model_config.get("clahe_params") # Optional
        vqvae_arch_params = model_config["params"]
    except KeyError as e:
        print(f"Error: Missing key {e} in model configuration. Exiting.")
        return

    # 2. Load VQ-VAE Model
    print(f"Loading VQ-VAE model from: {model_path}")
    model: VQVAE = load_vqvae_model(model_path, vqvae_arch_params)
    if not model:
        print("Failed to load VQ-VAE model. Exiting.")
        return
    model.to(device)
    model.eval()

    # 3. Prepare for Buffer
    buffer_data: List[Tuple[np.ndarray, str]] = []
    output_buffer_path = get_buffer_path(output_buffer_name, image_type)
    print(f"Output buffer will be saved to: {output_buffer_path}")

    # 4. Find and Process Images
    # Get a list of common image file extensions
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    
    image_files.sort() # Sort to get a consistent "first N"
    print(f"Found {len(image_files)} image files in directory.")

    processed_count = 0
    for img_path in image_files:
        if processed_count >= max_images:
            print(f"Reached max image limit of {max_images}.")
            break

        print(f"\nProcessing image ({processed_count + 1}/{max_images}): {os.path.basename(img_path)}")
        try:
            with open(img_path, 'rb') as f:
                img_bytes = f.read()

            # Preprocess image
            processed_tensor = preprocess_image_from_bytes(img_bytes, img_size, clahe_params)
            if processed_tensor is None:
                print(f"  Skipping image due to preprocessing error: {os.path.basename(img_path)}")
                continue
            processed_tensor = processed_tensor.to(device)

            # Extract features (M0 encoder + pre_quant_conv + pooling)
            with torch.no_grad():
                # Assuming VQVAE.forward returns: vq_loss, x_recon, perplexity, z_q, z_e_pre_quant, ...
                _, _, _, _, z_e_pre_quant, _ = model(processed_tensor)

            if z_e_pre_quant is not None:
                pooled_features_tensor = torch.nn.functional.adaptive_avg_pool2d(z_e_pre_quant, (1, 1)).squeeze()
                feature_vector_np = pooled_features_tensor.cpu().numpy() # Already detached in pooling
                
                # Append (feature, label) tuple
                buffer_data.append((feature_vector_np, anomaly_label))
                print(f"  Extracted features (shape {feature_vector_np.shape}) and added to buffer with label '{anomaly_label}'.")
                processed_count += 1
            else:
                print(f"  Skipping image, z_e_pre_quant was None: {os.path.basename(img_path)}")

        except UnidentifiedImageError:
            print(f"  Skipping non-image or corrupt file: {os.path.basename(img_path)}")
        except Exception as e:
            print(f"  Error processing image {os.path.basename(img_path)}: {e}")
            import traceback
            traceback.print_exc()


    # 5. Save Buffer
    if buffer_data:
        print(f"\nSaving {len(buffer_data)} processed samples to buffer...")
        save_buffer(output_buffer_path, buffer_data) # Use the save_buffer from il_utils
        print(f"Buffer saved successfully to {output_buffer_path}")
    else:
        print("No data processed, buffer not saved.")

    print("Buffer creation process finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an anomaly buffer from images.")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing the images.")
    parser.add_argument("--img_type", type=str, required=True, choices=["lung", "breast"], help="Type of images ('lung' or 'breast').")
    parser.add_argument("--label", type=str, required=True, help="Label for these anomalies (e.g., 'cancer', 'benign').")
    parser.add_argument("--model_name", type=str, help="Name of the model (e.g., 'lung_vqvae_model.pth'). If not given, constructs from img_type.")
    parser.add_argument("--config_name", type=str, help="Name of the config file (e.g., 'lung_model_runtime_config.json'). If not given, constructs from img_type.")
    parser.add_argument("--buffer_name", type=str, help="Base name for the output buffer (e.g., 'cancer_anomalies'). If not given, constructs from label.")
    parser.add_argument("--max_images", type=int, default=50, help="Maximum number of images to process.")
    parser.add_argument("--assets_dir", type=str, default="assets", help="Path to the assets directory.")
    
    args = parser.parse_args()

    # Construct full paths for model and config if specific names not given
    model_filename = args.model_name if args.model_name else f"{args.img_type}_vqvae_model.pth"
    model_path = os.path.join(args.assets_dir, model_filename)

    config_filename = args.config_name if args.config_name else f"{args.img_type}_model_runtime_config.json"
    config_path = os.path.join(args.assets_dir, config_filename)
    
    output_buffer_name = args.buffer_name if args.buffer_name else f"{args.label}_anomalies"

    if not os.path.isdir(args.img_dir):
        print(f"Error: Image directory not found: {args.img_dir}")
        exit()
    if not os.path.isfile(model_path):
        print(f"Error: Model file not found: {model_path}")
        exit()
    if not os.path.isfile(config_path):
        print(f"Error: Config file not found: {config_path}")
        exit()

    create_anomaly_buffer(
        image_dir=args.img_dir,
        image_type=args.img_type,
        anomaly_label=args.label,
        model_path=model_path,
        config_path=config_path,
        output_buffer_name=output_buffer_name,
        max_images=args.max_images,
        # DEVICE is imported from pipeline.py, assumed to be set correctly there
    )