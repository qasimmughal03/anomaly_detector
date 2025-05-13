# il_utils.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional
import json # Needed for known classes state

# --- Directory Setup ---
IL_STATE_DIR = os.path.join(os.path.dirname(__file__), "il_state")
os.makedirs(IL_STATE_DIR, exist_ok=True)

# --- Path Definitions ---
def get_buffer_path(buffer_name: str, image_type: str):
    """Gets the path for a specific buffer file (e.g., cancer_anomalies_lung.npz)."""
    base_name = f"buffer_{buffer_name}_{image_type}"
    if not base_name.endswith('.npz'):
        base_name += '.npz'
    return os.path.join(IL_STATE_DIR, base_name)

def get_current_classifier_path(image_type: str):
    """Gets the path for the single, current classifier."""
    return os.path.join(IL_STATE_DIR, f"classifier_current_{image_type}.pt")

def get_known_classes_path(image_type: str):
    """Gets the path for the JSON file storing known anomaly classes."""
    return os.path.join(IL_STATE_DIR, f"known_classes_{image_type}.json")

# --- Known Classes State Management ---
def load_known_classes(image_type: str) -> list:
    """Loads the list of known anomaly classes from the state file."""
    path = get_known_classes_path(image_type)
    if not os.path.exists(path):
        return [] # No known anomaly classes yet
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            # Ensure it returns a list, even if file is somehow invalid
            return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {path}. Returning empty list.")
        return []
    except Exception as e:
        print(f"Error loading known classes from {path}: {e}")
        return []

def save_known_classes(image_type: str, known_classes: list):
    """Saves the list of known anomaly classes to the state file."""
    path = get_known_classes_path(image_type)
    # Ensure consistent order by sorting before saving
    known_classes_sorted = sorted(list(set(known_classes))) # Ensure unique and sorted
    try:
        with open(path, 'w') as f:
            json.dump(known_classes_sorted, f, indent=4)
        print(f"Saved known classes {known_classes_sorted} to {os.path.basename(path)}")
    except Exception as e:
        print(f"Error saving known classes to {path}: {e}")

# --- Buffer Loading/Saving/Adding (Keep as implemented in previous step) ---
def load_buffer(buffer_path: str) -> list:
    """Loads features and labels (as tuples) OR just features from an npz buffer file."""
    print(f"\n--- il_utils:load_buffer (REVISED) for {os.path.basename(buffer_path)} ---")
    if not os.path.exists(buffer_path):
        print(f"  File not found. Returning empty list.")
        return []
    try:
        data = np.load(buffer_path, allow_pickle=True)
        if 'features' not in data:
            print(f"  'features' key not found in file. Returning empty list.")
            return []

        features_data = data['features']
        print(f"  Loaded 'features' key. Type: {type(features_data)}, Shape: {features_data.shape if hasattr(features_data, 'shape') else 'N/A'}, Dtype: {features_data.dtype if hasattr(features_data, 'dtype') else 'N/A'}")

        if features_data.size == 0:
            # Handles case where an empty object array was saved e.g. np.array([], dtype=object)
            # Also handles 0-dim array containing an empty list if that ever occurs.
            if features_data.ndim == 0 and isinstance(features_data.item(), list):
                print(f"  Loaded 0-dim array containing an empty list. Returning empty list.")
                return features_data.item() # Should be []
            print(f"  Loaded array is empty (size 0). Returning empty list.")
            return []

        # If it's an object array, it should contain our tuples or individual np.ndarrays (for normal_replay)
        if features_data.dtype == object:
            print(f"  Loaded object array. Converting to list.")
            # .tolist() converts an object array of tuples back to a list of tuples.
            # If it's an object array of np.ndarrays (like normal_replay), it becomes list of np.ndarrays.
            return features_data.tolist()
        elif isinstance(features_data, np.ndarray):
            # This case is for normal_replay if it wasn't saved as dtype=object,
            # or if anomaly buffers were somehow saved as plain N-D arrays (which is the problem we're fixing)
            # If we hit this for an anomaly buffer after the fix, something is still wrong with saving.
            print(f"  Loaded non-object ndarray. Converting to list of rows/items.")
            return list(features_data) # For normal_replay (list of feature arrays)
        else:
             print(f"  Warning: Unexpected data type {type(features_data)} after loading. Returning empty list.")
             return []
    except Exception as e:
        print(f"  Error loading buffer {buffer_path}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for load errors
        return []

# ... (Keep SimpleMLP, load_current_classifier, save_current_classifier, train_classifier_internal, train_incremental_classifier)
# ... (Keep add_to_buffer with its diagnostic prints)
# il_utils.py

# il_utils.py

# ... (other imports and functions like get_buffer_path, load_buffer) ...

def add_to_buffer(buffer_path: str, feature_vector: np.ndarray, label: str):
     """Adds a single feature vector and its label to a buffer."""
     print(f"\n--- il_utils:add_to_buffer ---") # DIAGNOSTIC
     print(f"  Attempting to add to: {os.path.basename(buffer_path)}") # DIAGNOSTIC
     print(f"  Received feature_vector type: {type(feature_vector)}") # DIAGNOSTIC
     if isinstance(feature_vector, np.ndarray): # DIAGNOSTIC
         print(f"  Received feature_vector shape: {feature_vector.shape}") # DIAGNOSTIC
         print(f"  Received feature_vector dtype: {feature_vector.dtype}") # DIAGNOSTIC
     print(f"  Received label type: {type(label)}") # DIAGNOSTIC
     print(f"  Received label value: '{label}'") # DIAGNOSTIC

     buffer_data = load_buffer(buffer_path) # This loads existing items
     # Ensure feature_vector is numpy, not torch tensor, if coming from pipeline
     if isinstance(feature_vector, torch.Tensor):
         feature_vector = feature_vector.cpu().detach().numpy()

     item_to_append = (feature_vector, label)
     print(f"  Item to append to buffer_data: type={type(item_to_append)}, feature_shape={item_to_append[0].shape if hasattr(item_to_append[0], 'shape') else 'N/A'}, label='{item_to_append[1]}'")

     buffer_data.append(item_to_append)
     save_buffer(buffer_path, buffer_data) # Call save_buffer
     print(f"  Successfully called save_buffer. New total items in buffer_data list (before saving): {len(buffer_data)}")
     print(f"--- End il_utils:add_to_buffer ---\n")

def save_buffer(buffer_path: str, buffer_data: list): # list of (feature_array, label_string) tuples
    """Saves a list of (feature, label) tuples to an npz buffer file more robustly."""
    print(f"\n--- il_utils:save_buffer (REVISED) ---")
    print(f"  Saving to: {os.path.basename(buffer_path)}")
    print(f"  buffer_data (list input) contains {len(buffer_data)} items.")
    if buffer_data:
        first_item_in_list = buffer_data[0]
        print(f"  Type of first item in input buffer_data list: {type(first_item_in_list)}")
        if isinstance(first_item_in_list, tuple) and len(first_item_in_list) == 2:
            print(f"    Input First item's feature type: {type(first_item_in_list[0])}, shape: {first_item_in_list[0].shape if hasattr(first_item_in_list[0], 'shape') else 'N/A'}")
            print(f"    Input First item's label type: {type(first_item_in_list[1])}, value: '{first_item_in_list[1]}'")

    try:
        if buffer_data:
            # Explicitly create an object array of the correct size
            array_to_save = np.empty(len(buffer_data), dtype=object)
            # Fill the object array with the items from buffer_data (which should be tuples)
            array_to_save[:] = buffer_data
        else:
            # Save an empty object array if buffer_data is empty
            array_to_save = np.array([], dtype=object)

        print(f"  Shape of array_to_save (after explicit creation/filling): {array_to_save.shape}")
        if array_to_save.size > 0:
            print(f"  Type of first element in array_to_save: {type(array_to_save[0])}")
            if isinstance(array_to_save[0], tuple) and hasattr(array_to_save[0][0], 'shape'):
                 print(f"    Shape of feature in first element of array_to_save: {array_to_save[0][0].shape}")

        np.savez(buffer_path, features=array_to_save)
        print(f"  Buffer saved successfully to {os.path.basename(buffer_path)}")
    except Exception as e:
        print(f"  Error during np.savez in save_buffer: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for save errors
    print(f"--- End il_utils:save_buffer (REVISED) ---\n")

# --- Classifier Definition (SimpleMLP - Keep as is) ---
class SimpleMLP(nn.Module):
    # ... (implementation from previous step) ...
     def __init__(self, input_dim, num_classes):
         super().__init__()
         self.layer_1 = nn.Linear(input_dim, 128)
         self.relu1 = nn.ReLU()
         self.dropout1 = nn.Dropout(0.3)
         self.layer_2 = nn.Linear(128, 64)
         self.relu2 = nn.ReLU()
         self.dropout2 = nn.Dropout(0.3)
         self.output_layer = nn.Linear(64, num_classes)
     def forward(self, x):
         x = self.dropout1(self.relu1(self.layer_1(x)))
         x = self.dropout2(self.relu2(self.layer_2(x)))
         x = self.output_layer(x)
         return x

# --- Classifier Loading/Saving (Adapt slightly for single 'current' classifier) ---
def load_current_classifier(image_type: str, input_dim: int, num_classes: int, device):
    """Loads the single 'current' classifier for the image type."""
    classifier_path = get_current_classifier_path(image_type)
    if not os.path.exists(classifier_path):
        print(f"No current classifier found at {os.path.basename(classifier_path)}")
        return None # No classifier trained yet
    try:
        model = SimpleMLP(input_dim, num_classes).to(device)
        model.load_state_dict(torch.load(classifier_path, map_location=device))
        model.eval()
        print(f"Loaded current classifier {os.path.basename(classifier_path)} ({num_classes} classes)")
        return model
    except Exception as e:
        # Catch potential errors if saved state doesn't match expected num_classes/input_dim
        print(f"Error loading current classifier {classifier_path}: {e}. State might be incompatible.")
        return None

def save_current_classifier(image_type: str, model: nn.Module):
    """Saves the model as the single 'current' classifier."""
    classifier_path = get_current_classifier_path(image_type)
    try:
        torch.save(model.state_dict(), classifier_path)
        print(f"Saved current classifier to {os.path.basename(classifier_path)}")
    except Exception as e:
        print(f"Error saving current classifier {classifier_path}: {e}")


# --- Internal Trainer Helper (Keep as is) ---
def train_classifier_internal(model: nn.Module, dataloader: DataLoader, device, epochs: int = 10, lr: float = 0.001):
    # ... (implementation from previous step) ...
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_samples = len(dataloader.dataset)
    print(f"  Starting training for {epochs} epochs on {num_samples} samples...")
    # ... (training loop) ...
    for epoch in range(epochs):
        running_loss = 0.0; correct_predictions = 0; total_predictions = 0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            running_loss += loss.item() * inputs.size(0); _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0); correct_predictions += (predicted == labels).sum().item()
        epoch_loss = running_loss / num_samples; epoch_acc = correct_predictions / total_predictions
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    model.eval()
    print("  Training finished.")
    return model

# il_utils.py

# ... (imports and other functions above) ...

# --- REMOVE OLD C1 / C2 TRAINING FUNCTIONS ---
# Remove train_classifier_c1(...)
# Remove train_classifier_c2(...)
# ---

# --- Dynamic Incremental Training Function ---
# il_utils.py

# ... (imports, other functions) ...

def train_incremental_classifier(
    img_type: str,
    new_anomaly_class: str,
    currently_known_anomaly_classes: list,
    device,
    epochs: int = 15,
    lr: float = 0.001,
    batch_size: int = 8
) -> Optional[nn.Module]:
    # ... (print statements, class/label map setup as before) ...
    print(f"\n--- Training Incremental Classifier ({img_type}) ---")
    print(f"   New anomaly class to learn: '{new_anomaly_class}'")
    print(f"   Previously known anomaly classes for replay: {currently_known_anomaly_classes}")
    all_anomaly_classes = sorted(list(set([new_anomaly_class] + currently_known_anomaly_classes)))
    all_classes_for_training = ["Normal"] + all_anomaly_classes
    num_classes = len(all_classes_for_training)
    label_map = {name: i for i, name in enumerate(all_classes_for_training)}
    print(f"   Training for {num_classes} classes: {all_classes_for_training}")
    print(f"   Label Map: {label_map}")

    # 2. Load Data for all required classes (REVISED LOGIC)
    all_features = []
    all_labels = []
    input_dim = -1

    # --- Load Normal Replay (Features Only) ---
    normal_buffer_path = get_buffer_path("normal_replay", img_type)
    # Assuming load_buffer now correctly returns a list of np.ndarrays for this file
    # based on check_buffer.py output and potential load_buffer refinement
    normal_features_list = load_buffer(normal_buffer_path)
    if not normal_features_list: # Check if the list itself is empty
        print(f"  Error: Cannot train - Normal replay buffer returned empty list for {img_type}.")
        return None
    # The items ARE the features
    normal_features = [f for f in normal_features_list if isinstance(f, np.ndarray)] # Filter just in case
    if not normal_features:
        print(f"  Error: No valid numpy arrays found in normal replay buffer for {img_type}.")
        return None

    all_features.extend(normal_features)
    all_labels.extend([label_map["Normal"]] * len(normal_features))
    input_dim = normal_features[0].shape[0] # Get dim from first valid feature
    print(f"  Loaded {len(normal_features)} Normal samples (Input Dim: {input_dim}).")
    if input_dim <= 0:
         print(f"  Error: Could not determine valid input dimension ({input_dim}).")
         return None
    # --- End Load Normal Replay ---

    # --- Load New Anomaly Class Data (Tuples: feature, label) ---
    new_anomaly_buffer_path = get_buffer_path(f"{new_anomaly_class}_anomalies", img_type)
    new_anomaly_data_tuples = load_buffer(new_anomaly_buffer_path)
    # Extract feature [0] and check label [1] from tuples
    new_anomaly_features = [item[0] for item in new_anomaly_data_tuples if isinstance(item, (tuple, list)) and len(item)>1 and item[1] == new_anomaly_class and isinstance(item[0], np.ndarray)]
    if not new_anomaly_features:
        print(f"  Error: Cannot train - Buffer for new class '{new_anomaly_class}' is empty or has invalid data for {img_type}.")
        return None
    all_features.extend(new_anomaly_features)
    all_labels.extend([label_map[new_anomaly_class]] * len(new_anomaly_features))
    print(f"  Loaded {len(new_anomaly_features)} '{new_anomaly_class}' samples.")
    # --- End Load New Anomaly ---

    # --- Load Replay Data for Previously Known Anomaly Classes (Tuples: feature, label) ---
    for known_class in currently_known_anomaly_classes:
        replay_buffer_path = get_buffer_path(f"{known_class}_anomalies", img_type)
        replay_data_tuples = load_buffer(replay_buffer_path)
        # Extract feature [0] and check label [1] from tuples
        replay_features = [item[0] for item in replay_data_tuples if isinstance(item, (tuple, list)) and len(item)>1 and item[1] == known_class and isinstance(item[0], np.ndarray)]
        if not replay_features:
            print(f"  Warning: Replay buffer for known class '{known_class}' is empty or invalid for {img_type}. Model might forget.")
        else:
            all_features.extend(replay_features)
            all_labels.extend([label_map[known_class]] * len(replay_features))
            print(f"  Loaded {len(replay_features)} '{known_class}' samples for replay.")
    # --- End Load Replay ---

    # 3. Prepare Data for PyTorch (Check consistency)
    if not all_features:
        print("  Error: No features collected for training.")
        return None
    try:
        # Check dimensions before creating tensor
        first_shape = all_features[0].shape
        if first_shape[0] != input_dim: # Basic check
             print(f"  Error: Feature dimension mismatch detected! Expected dim {input_dim}, found {first_shape[0]}. Check buffers.")
             return None
        # Ensure all are numpy arrays before stacking
        features_np = np.array([np.asarray(f) for f in all_features])
        # Check shape consistency after conversion
        if features_np.shape[1] != input_dim:
             print(f"  Error: Feature dimension mismatch after stacking! Expected dim {input_dim}, got {features_np.shape[1]}. Check buffers.")
             return None

        features_tensor = torch.tensor(features_np, dtype=torch.float32)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    except Exception as e:
        print(f"  Error preparing tensors: {e}. Check feature vector consistency.")
        return None

    # 4. Create DataLoader
    # ... (rest of the function: DataLoader, Instantiate Model, Train Model) ...
    dataset = TensorDataset(features_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"  Total samples for training: {len(dataset)}")
    model = SimpleMLP(input_dim, num_classes).to(device)
    print(f"  Instantiated SimpleMLP with input_dim={input_dim}, num_classes={num_classes}")
    trained_model = train_classifier_internal(model, dataloader, device, epochs, lr)

    return trained_model

# --- End Dynamic Training Function ---