# il_utils.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import json
from sklearn.model_selection import train_test_split # <-- ADD THIS
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Optional, List, Dict, Any, Tuple # <-- ADD THESE
from sklearn.utils.class_weight import compute_class_weight # <-- ADD THIS IMPORT

from sklearn.svm import SVC  # <-- For SVM
from xgboost import XGBClassifier  # <-- For XGBoost (ensure installed: pip install xgboost)
from sklearn.ensemble import AdaBoostClassifier # <-- For AdaBoost
from sklearn.tree import DecisionTreeClassifier # <-- Common base for AdaBoost

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
    return os.path.join(IL_STATE_DIR, f"classifier_current_{image_type}")

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
         self.dropout1 = nn.Dropout(0.2)
         self.layer_2 = nn.Linear(128, 64)
         self.relu2 = nn.ReLU()
         self.dropout2 = nn.Dropout(0.2)
         self.output_layer = nn.Linear(64, num_classes)
     def forward(self, x):
         x = self.dropout1(self.relu1(self.layer_1(x)))
         x = self.dropout2(self.relu2(self.layer_2(x)))
         x = self.output_layer(x)
         return x

# --- Classifier Loading/Saving (Adapt slightly for single 'current' classifier) ---
# il_utils.py

# il_utils.py
def get_current_classifier_path_base(image_type: str) -> str:
    """Gets the base path (no extension) for the single, current classifier."""
    return os.path.join(IL_STATE_DIR, f"classifier_current_{image_type}")

    """Loads the 'current' classifier, trying PyTorch MLP then Joblib."""
    classifier_path_base = get_current_classifier_path_base(image_type)
    classifier_pt_path = classifier_path_base + ".pt"
    classifier_joblib_path = classifier_path_base + ".joblib"
    
    loaded_model = None
    model_type_loaded = None # This will be "mlp" or e.g., "sklearn_xgb"

    # Try loading PyTorch model first
    if os.path.exists(classifier_pt_path):
        print(f"Attempting to load PyTorch MLP classifier from {os.path.basename(classifier_pt_path)}...")
        try:
            # input_dim and num_classes are needed to instantiate the MLP before loading state_dict
            input_dim = model_config["params"]["embedding_dim"]
            num_classes = 1 + len(known_classes) # Assumes known_classes is just anomaly names
            
            model = SimpleMLP(input_dim, num_classes).to(device)
            model.load_state_dict(torch.load(classifier_pt_path, map_location=device))
            model.eval()
            loaded_model = model
            model_type_loaded = "mlp"
            print(f"Loaded PyTorch MLP classifier ({num_classes} classes).")
        except Exception as e:
            print(f"Error loading PyTorch MLP {classifier_pt_path}: {e}. Will try joblib if MLP was not already loaded.")
    
    # If MLP not loaded (or didn't exist), try loading joblib model
    if loaded_model is None and os.path.exists(classifier_joblib_path):
        print(f"Attempting to load scikit-learn/XGBoost/AdaBoost classifier from {os.path.basename(classifier_joblib_path)}...")
        try:
            model = joblib.load(classifier_joblib_path)
            loaded_model = model
            # You can make this type more specific if needed, but a generic one works for predict_proba
            model_type_loaded = "sklearn_model" # Generic type for joblib loaded models
            print(f"Loaded scikit-learn/XGBoost/AdaBoost classifier.")
        except Exception as e:
            print(f"Error loading scikit-learn/XGBoost/AdaBoost classifier {classifier_joblib_path}: {e}")
    
    if loaded_model is None:
        print(f"No 'current' classifier file found for {image_type} (checked .pt and .joblib).")

    return loaded_model, model_type_loaded
# il_utils.py
import joblib # Add this import at the top

def load_current_classifier(image_type: str, model_config: Dict, known_classes: List[str], device):
    """Loads the 'current' classifier, trying PyTorch MLP then Joblib, with detailed debugging."""
    print(f"\n--- il_utils:load_current_classifier for {image_type} (DIAGNOSTIC RUN) ---")
    classifier_path_base = get_current_classifier_path_base(image_type)
    classifier_pt_path = classifier_path_base + ".pt"
    classifier_joblib_path = classifier_path_base + ".joblib"
    
    print(f"  Base path for classifier: {classifier_path_base}")
    print(f"  Checking for PyTorch model at: {classifier_pt_path}")
    print(f"  Checking for Joblib model at: {classifier_joblib_path}")

    loaded_model = None
    model_type_loaded = None

    # Try loading PyTorch model first
    pt_exists = os.path.exists(classifier_pt_path)
    print(f"  PyTorch model file exists ({os.path.basename(classifier_pt_path)}): {pt_exists}")
    if pt_exists:
        print(f"  Attempting to load PyTorch MLP classifier from {os.path.basename(classifier_pt_path)}...")
        try:
            input_dim = model_config["params"]["embedding_dim"]
            num_classes = 1 + len(known_classes)
            
            model = SimpleMLP(input_dim, num_classes).to(device)
            model.load_state_dict(torch.load(classifier_pt_path, map_location=device))
            model.eval()
            loaded_model = model
            model_type_loaded = "mlp"
            print(f"  SUCCESS: Loaded PyTorch MLP classifier ({num_classes} classes).")
        except Exception as e:
            print(f"  ERROR loading PyTorch MLP {os.path.basename(classifier_pt_path)}: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for this error
            # loaded_model remains None, model_type_loaded remains None
    
    # If MLP not loaded (or didn't exist), try loading joblib model
    if loaded_model is None:
        joblib_exists = os.path.exists(classifier_joblib_path)
        print(f"  Joblib model file exists ({os.path.basename(classifier_joblib_path)}): {joblib_exists}")
        if joblib_exists:
            print(f"  Attempting to load scikit-learn/XGBoost/AdaBoost classifier from {os.path.basename(classifier_joblib_path)}...")
            try:
                model = joblib.load(classifier_joblib_path)
                loaded_model = model
                # Determine a more specific type if possible, or keep generic
                if hasattr(model, 'get_ xgb_params'): # Heuristic for XGBoost
                    model_type_loaded = "xgboost"
                elif isinstance(model, SVC):
                     model_type_loaded = "svm"
                elif isinstance(model, AdaBoostClassifier):
                     model_type_loaded = "adaboost"
                else:
                     model_type_loaded = "sklearn_model" # Generic
                print(f"  SUCCESS: Loaded Joblib model. Detected type: {model_type_loaded}.")
            except Exception as e:
                print(f"  ERROR loading Joblib model {os.path.basename(classifier_joblib_path)}: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for this error
                # loaded_model remains None, model_type_loaded remains None
    
    if loaded_model is None:
        print(f"  RESULT: No 'current' classifier was successfully loaded for {image_type} (checked .pt and .joblib).")
    else:
        print(f"  RESULT: Successfully loaded classifier of type '{model_type_loaded}'.")
    
    print(f"--- End il_utils:load_current_classifier ---")
    return loaded_model, model_type_loaded

def save_current_classifier(image_type: str, model: Any): # model can now be other types
    """Saves the model as the single 'current' classifier."""
    classifier_path = get_current_classifier_path(image_type) # This function now needs to return a .joblib path
    try:
        if isinstance(model, nn.Module): # Check if it's a PyTorch model
            torch.save(model.state_dict(), classifier_path + ".pt") # Save PyTorch model as before
            print(f"Saved current PyTorch classifier to {os.path.basename(classifier_path)}.pt")
        else: # Assume scikit-learn compatible (XGBoost, SVM, AdaBoost)
            joblib.dump(model, classifier_path + ".joblib")
            print(f"Saved current scikit-learn/XGBoost classifier to {os.path.basename(classifier_path)}.joblib")
    except Exception as e:
        print(f"Error saving current classifier to {classifier_path}: {e}")



# il_utils.py

# ... (other functions like SimpleMLP, load/save_classifier, etc.) ...

def train_classifier_internal(
    model: nn.Module,
    train_loader: DataLoader,
    device,
    epochs: int = 50, # Increased default epochs, can be overridden
    lr: float = 0.001,
    class_weights: Optional[torch.Tensor] = None # Accepts class_weights
) -> nn.Module:
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss(weight=class_weights) # Use weights if provided
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_samples_train = len(train_loader.dataset)
    if num_samples_train == 0:
        print("  Error: Training dataset is empty in train_classifier_internal. Cannot train.")
        return model # Return untrained model

    print(f"  Starting internal training on {num_samples_train} samples for {epochs} epochs (LR={lr})...")
    if class_weights is not None:
        print(f"  Using class weights: {class_weights.cpu().numpy()}")

    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        
        if total_predictions > 0:
            epoch_loss = running_loss / total_predictions
            epoch_acc = correct_predictions / total_predictions
            print(f"  Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}")
        else:
            print(f"  Epoch {epoch+1}/{epochs}, No data processed in this epoch.")
            
    print("  Internal training loop finished.")
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

# il_utils.py

# ... (imports, SimpleMLP, train_classifier_internal, etc.) ...

# def train_incremental_classifier(
#     img_type: str,
#     new_anomaly_class: str,
#     currently_known_anomaly_classes: list,
#     device,
#     epochs: int = 100, # You can adjust this default
#     lr: float = 0.001,
#     batch_size: int = 8,
#     test_size: float = 0.2 # Proportion for validation split
# ) -> Optional[nn.Module]:
#     """
#     Trains/Retrains the classifier to include a new anomaly class,
#     using replay, calculates class weights, and evaluates on a validation split.
#     """
#     print(f"\n--- Training Incremental Classifier ({img_type}) ---")
#     print(f"   New anomaly class to learn/focus on: '{new_anomaly_class}'")
#     print(f"   Previously known other anomaly classes for replay: {currently_known_anomaly_classes}")

#     # 1. Determine all classes and create label map
#     all_anomaly_classes = sorted(list(set([new_anomaly_class] + currently_known_anomaly_classes)))
#     all_classes_for_training = ["Normal"] + all_anomaly_classes
#     num_classes = len(all_classes_for_training)
#     label_map = {name: i for i, name in enumerate(all_classes_for_training)}
#     class_names_for_report = [name for name, idx in sorted(label_map.items(), key=lambda item: item[1])]
#     print(f"   Training for {num_classes} classes: {all_classes_for_training}")
#     print(f"   Label Map: {label_map}")

#     if num_classes < 2:
#         print("  Error: At least two classes (e.g., Normal and one anomaly) are required for classification training.")
#         return None

#     # 2. Load Data for all required classes
#     all_features_list: List[np.ndarray] = []
#     all_labels_list: List[int] = []
#     input_dim: int = -1

#     # Load Normal Replay
#     normal_buffer_path = get_buffer_path("normal_replay", img_type)
#     normal_features_loaded = load_buffer(normal_buffer_path)
#     if not normal_features_loaded:
#         print(f"  Error: Cannot train - Normal replay buffer is empty or failed to load for {img_type}.")
#         return None
#     current_normal_features = [f for f in normal_features_loaded if isinstance(f, np.ndarray)]
#     if not current_normal_features:
#         print(f"  Error: No valid numpy arrays found in normal replay buffer for {img_type}.")
#         return None
#     all_features_list.extend(current_normal_features)
#     all_labels_list.extend([label_map["Normal"]] * len(current_normal_features))
#     input_dim = current_normal_features[0].shape[0]
#     print(f"  Loaded {len(current_normal_features)} Normal samples (Input Dim: {input_dim}).")
#     if input_dim <= 0:
#          print(f"  Error: Could not determine valid input dimension ({input_dim}) from normal features.")
#          return None

#     # Load New/Focus Anomaly Class Data
#     focus_anomaly_buffer_path = get_buffer_path(f"{new_anomaly_class}_anomalies", img_type)
#     focus_anomaly_data_tuples = load_buffer(focus_anomaly_buffer_path)
#     current_focus_anomaly_features = [item[0] for item in focus_anomaly_data_tuples if isinstance(item, (tuple, list)) and len(item)>1 and item[1] == new_anomaly_class and isinstance(item[0], np.ndarray)]
#     if not current_focus_anomaly_features:
#         print(f"  Error: Buffer for focus class '{new_anomaly_class}' is empty or has invalid data for {img_type}. Cannot train.")
#         return None
#     all_features_list.extend(current_focus_anomaly_features)
#     all_labels_list.extend([label_map[new_anomaly_class]] * len(current_focus_anomaly_features))
#     print(f"  Loaded {len(current_focus_anomaly_features)} '{new_anomaly_class}' samples.")

#     # Load Replay Data for Previously Known (but not current focus) Anomaly Classes
#     for known_class in currently_known_anomaly_classes:
#         if known_class == new_anomaly_class: continue # Already loaded as focus class
#         replay_buffer_path = get_buffer_path(f"{known_class}_anomalies", img_type)
#         replay_data_tuples = load_buffer(replay_buffer_path)
#         current_replay_features = [item[0] for item in replay_data_tuples if isinstance(item, (tuple, list)) and len(item)>1 and item[1] == known_class and isinstance(item[0], np.ndarray)]
#         if not current_replay_features:
#             print(f"  Warning: Replay buffer for known class '{known_class}' is empty or invalid for {img_type}. Model might forget/perform poorly for this class.")
#         else:
#             all_features_list.extend(current_replay_features)
#             all_labels_list.extend([label_map[known_class]] * len(current_replay_features))
#             print(f"  Loaded {len(current_replay_features)} '{known_class}' samples for replay.")

#     # 3. Prepare Data for PyTorch & Split
#     if not all_features_list:
#         print("  Error: No features collected for training after loading all buffers.")
#         return None
#     try:
#         features_np = np.array([np.asarray(f) for f in all_features_list])
#         labels_np = np.array(all_labels_list)

#         # Check feature consistency
#         for i, f_vec in enumerate(features_np):
#             if f_vec.shape[0] != input_dim:
#                 print(f"  FATAL ERROR: Feature dimension mismatch at index {i}! Expected {input_dim}, got {f_vec.shape}. Check data loading for class associated with label {labels_np[i]}.")
#                 return None
        
#         X_train_np, X_val_np, y_train_np, y_val_np = features_np, np.array([]), labels_np, np.array([]) # Defaults

#         # Ensure there's enough data for a split and multiple classes for stratification
#         # Minimum samples for split: e.g., if test_size=0.2, need at least 1/0.2 = 5 samples per class for stratify
#         # For simplicity, just check if total samples is enough for at least 1 validation sample per class.
#         min_samples_for_split = num_classes * max(2, int(1/test_size)) # Heuristic

#         if len(labels_np) < min_samples_for_split or len(np.unique(labels_np)) < num_classes:
#             print(f"  Warning: Not enough samples ({len(labels_np)}) or distinct classes ({len(np.unique(labels_np))} found vs {num_classes} expected for full {test_size*100}% val split and stratification). Training on all available data without validation split.")
#             X_train_np, y_train_np = features_np, labels_np
#             # X_val_np, y_val_np remain empty
#         else:
#              try:
#                  X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
#                      features_np, labels_np, test_size=test_size, random_state=42, stratify=labels_np
#                  )
#                  print(f"  Split data: {len(X_train_np)} train, {len(X_val_np)} validation samples.")
#              except ValueError as e:
#                  print(f"  Warning: Stratified split failed ({e}). Using non-stratified split.")
#                  X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
#                      features_np, labels_np, test_size=test_size, random_state=42
#                  )
#                  print(f"  Split data (non-stratified): {len(X_train_np)} train, {len(X_val_np)} validation samples.")
        
#         if len(X_train_np) == 0:
#             print("  Error: Training set is empty after split. Cannot proceed.")
#             return None

#         # --- Calculate Class Weights using ONLY y_train_np ---
#         class_weights_tensor = None
#         unique_train_labels, counts_train_labels = np.unique(y_train_np, return_counts=True)
#         if len(unique_train_labels) > 1: # Need at least 2 classes to compute weights meaningfully
#             try:
#                 weights = compute_class_weight(
#                     class_weight='balanced',
#                     classes=unique_train_labels,
#                     y=y_train_np
#                 )
#                 class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
#                 print(f"  Calculated class weights for training: {dict(zip(unique_train_labels, weights))}")
#             except Exception as e:
#                 print(f"  Warning: Could not compute class weights: {e}. Proceeding without weights.")
#         else:
#             print("  Warning: Only one class (or no class) in training data. Skipping class weights.")

#         X_train = torch.tensor(X_train_np, dtype=torch.float32)
#         y_train = torch.tensor(y_train_np, dtype=torch.long)
#         train_dataset = TensorDataset(X_train, y_train)
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
#         val_loader = None
#         if len(X_val_np) > 0 and len(y_val_np) > 0 : # Ensure validation set is not empty
#             X_val = torch.tensor(X_val_np, dtype=torch.float32)
#             y_val = torch.tensor(y_val_np, dtype=torch.long)
#             val_dataset = TensorDataset(X_val, y_val)
#             val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#             print(f"  Created training DataLoader with {len(train_dataset)} samples.")
#             print(f"  Created validation DataLoader with {len(val_dataset)} samples.")
#         else:
#             print(f"  Created training DataLoader with {len(train_dataset)} samples. No validation set created.")


#     except Exception as e:
#         print(f"  Error preparing tensors or splitting data: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

#     # 4. Instantiate Model
#     model = SimpleMLP(input_dim, num_classes) # .to(device) is handled in train_classifier_internal
#     print(f"  Instantiated SimpleMLP with input_dim={input_dim}, num_classes={num_classes}")

#     # 5. Train Model
#     trained_model = train_classifier_internal(
#         model, train_loader, device, epochs, lr, class_weights=class_weights_tensor # Pass weights
#     )
#     trained_model.eval()

#     # 6. Evaluate on Validation Set
#     if val_loader:
#         print("\n  --- Validation Results ---")
#         all_val_preds = []
#         all_val_labels = []
#         with torch.no_grad():
#             for inputs, labels_val in val_loader: # Renamed labels to labels_val
#                 inputs, labels_val = inputs.to(device), labels_val.to(device)
#                 outputs = trained_model(inputs)
#                 _, predicted = torch.max(outputs.data, 1)
#                 all_val_preds.extend(predicted.cpu().numpy())
#                 all_val_labels.extend(labels_val.cpu().numpy())

#         if all_val_labels:
#             val_accuracy = accuracy_score(all_val_labels, all_val_preds)
#             print(f"  Validation Accuracy: {val_accuracy:.4f}")
#             print("  Validation Classification Report:")
#             try:
#                 # Ensure target_names match the order of labels 0, 1, 2...
#                 report = classification_report(all_val_labels, all_val_preds, target_names=class_names_for_report, zero_division=0, labels=list(range(num_classes)))
#                 print(report)
#             except ValueError as ve:
#                 print(f"Could not generate full classification report (some classes might be missing in y_true or y_pred for validation): {ve}")
#                 print(f"Unique true labels in validation: {np.unique(all_val_labels)}")
#                 print(f"Unique predicted labels in validation: {np.unique(all_val_preds)}")


#             print("  Validation Confusion Matrix (rows: true, cols: pred):")
#             try:
#                 cm = confusion_matrix(all_val_labels, all_val_preds, labels=list(range(num_classes)))
#                 print(cm)
#             except ValueError as ve:
#                  print(f"Could not generate confusion matrix: {ve}")
#         else:
#             print("  No samples in validation set to evaluate or labels missing.")
#     else:
#         print("  No validation set available for evaluation (e.g. due to small dataset size).")
#     print("  --- End Validation Results ---")

#     return trained_model

# il_utils.py

# ... (imports, helper functions including print_validation_report, SimpleMLP, train_classifier_internal) ...

def train_incremental_classifier(
    img_type: str,
    new_anomaly_class: str,
    currently_known_anomaly_classes: list,
    device,
    epochs: int = 200, # For MLP
    lr: float = 0.001, # For MLP
    batch_size: int = 8, # For MLP
    test_size: float = 0.2
) -> Optional[nn.Module]: # Still returns the MLP for the pipeline
    """
    Trains/Retrains MLP and ALSO trains SVM, XGBoost, AdaBoost for comparison.
    Logs metrics for all. Returns the trained MLP.
    """
    print(f"\n--- Training Incremental Classifiers ({img_type}) ---")
    print(f"   New anomaly class to learn/focus on: '{new_anomaly_class}'")
    print(f"   Previously known other anomaly classes for replay: {currently_known_anomaly_classes}")

    # 1. Determine all classes and create label map (as before)
    all_anomaly_classes = sorted(list(set([new_anomaly_class] + currently_known_anomaly_classes)))
    all_classes_for_training = ["Normal"] + all_anomaly_classes
    num_classes = len(all_classes_for_training)
    label_map = {name: i for i, name in enumerate(all_classes_for_training)}
    # class_names_for_report must be in order of labels 0, 1, 2...
    class_names_for_report = [name for name, idx in sorted(label_map.items(), key=lambda item: item[1])]
    print(f"   Training for {num_classes} classes: {all_classes_for_training}")
    print(f"   Label Map: {label_map}")

    if num_classes < 2:
        print("  Error: At least two classes are required for classification training.")
        return None

    # 2. Load Data for all required classes (as before)
    all_features_list: List[np.ndarray] = []
    all_labels_list: List[int] = []
    input_dim: int = -1
    # ... (Your existing robust data loading logic for normal_features,
    #      new_anomaly_features, and replay_features into all_features_list and all_labels_list.
    #      Ensure input_dim is correctly determined from normal_features[0].shape[0])
    # Code snippet for data loading (ensure this part is complete and correct from previous versions):
    # Start Data Loading Block
    normal_buffer_path = get_buffer_path("normal_replay", img_type)
    normal_features_loaded = load_buffer(normal_buffer_path)
    if not normal_features_loaded: print(f"E: Normal replay buffer empty for {img_type}."); return None
    current_normal_features = [f for f in normal_features_loaded if isinstance(f, np.ndarray)]
    if not current_normal_features: print(f"E: No valid ndarrays in normal replay for {img_type}."); return None
    all_features_list.extend(current_normal_features)
    all_labels_list.extend([label_map["Normal"]] * len(current_normal_features))
    input_dim = current_normal_features[0].shape[0]
    print(f"  Loaded {len(current_normal_features)} Normal samples (Input Dim: {input_dim}).")
    if input_dim <= 0: print(f"E: Invalid input_dim ({input_dim})."); return None

    focus_anomaly_buffer_path = get_buffer_path(f"{new_anomaly_class}_anomalies", img_type)
    focus_anomaly_data_tuples = load_buffer(focus_anomaly_buffer_path)
    current_focus_anomaly_features = [item[0] for item in focus_anomaly_data_tuples if isinstance(item, (tuple, list)) and len(item)>1 and item[1] == new_anomaly_class and isinstance(item[0], np.ndarray)]
    if not current_focus_anomaly_features: print(f"E: Buffer for focus class '{new_anomaly_class}' empty/invalid for {img_type}."); return None
    all_features_list.extend(current_focus_anomaly_features)
    all_labels_list.extend([label_map[new_anomaly_class]] * len(current_focus_anomaly_features))
    print(f"  Loaded {len(current_focus_anomaly_features)} '{new_anomaly_class}' samples.")

    for known_class in currently_known_anomaly_classes:
        if known_class == new_anomaly_class: continue
        replay_buffer_path = get_buffer_path(f"{known_class}_anomalies", img_type)
        replay_data_tuples = load_buffer(replay_buffer_path)
        current_replay_features = [item[0] for item in replay_data_tuples if isinstance(item, (tuple, list)) and len(item)>1 and item[1] == known_class and isinstance(item[0], np.ndarray)]
        if not current_replay_features: print(f"W: Replay buffer for '{known_class}' empty/invalid for {img_type}.")
        else:
            all_features_list.extend(current_replay_features)
            all_labels_list.extend([label_map[known_class]] * len(current_replay_features))
            print(f"  Loaded {len(current_replay_features)} '{known_class}' samples for replay.")
    # End Data Loading Block

    # 3. Prepare Data for PyTorch & Split (as before)
    if not all_features_list: print("  Error: No features collected for training."); return None
    try:
        features_np = np.array([np.asarray(f) for f in all_features_list])
        labels_np = np.array(all_labels_list)
        # ... (Your robust train_test_split logic with stratification fallbacks as before) ...
        # This results in X_train_np, y_train_np, X_val_np, y_val_np
        # Start Data Splitting Block (ensure complete and robust)
        X_train_np, X_val_np, y_train_np, y_val_np = features_np, np.array([]), labels_np, np.array([]) # Defaults
        min_samples_for_split = num_classes * max(2, int(1/test_size) if test_size > 0 else 5)
        if len(labels_np) < min_samples_for_split or len(np.unique(labels_np)) < num_classes and num_classes > 1 :
            print(f"  W: Not enough samples ({len(labels_np)}) or distinct classes ({len(np.unique(labels_np))} vs {num_classes} expected) for full val split. Training on all data.")
            X_train_np, y_train_np = features_np, labels_np
        elif len(np.unique(labels_np)) < 2 :
             print(f"  W: Only one class present ({np.unique(labels_np)}). Training on all data without val.")
             X_train_np, y_train_np = features_np, labels_np
        else:
             try:
                 X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(features_np, labels_np, test_size=test_size, random_state=42, stratify=labels_np)
             except ValueError as e:
                 print(f"  W: Stratified split failed ({e}). Using non-stratified split."); X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(features_np, labels_np, test_size=test_size, random_state=42)
        print(f"  Data Split: {len(X_train_np)} train, {len(X_val_np)} validation samples.")
        if len(X_train_np) == 0: print("E: Training set empty after split."); return None
        # End Data Splitting Block
        
        # --- Calculate Class Weights for MLP and potentially others (using ONLY y_train_np) ---
        class_weights_for_mlp = None
        unique_train_labels, counts_train_labels = np.unique(y_train_np, return_counts=True)
        if len(unique_train_labels) > 1:
            try:
                weights = compute_class_weight(class_weight='balanced', classes=unique_train_labels, y=y_train_np)
                class_weights_for_mlp = torch.tensor(weights, dtype=torch.float32).to(device)
                print(f"  Calculated class weights (for MLP): {dict(zip(unique_train_labels, weights))}")
            except Exception as e: print(f"  W: Could not compute class weights: {e}.")
        else: print("  W: Only one class in training data. Skipping class weights for MLP.")
        # --- End Calculate Class Weights ---
    except Exception as e: print(f"  Error preparing/splitting data: {e}"); return None


    trained_mlp_model = None # To store the MLP for returning

    # --- 4. SimpleMLP Training & Evaluation ---
    print("\n  --- Training SimpleMLP ---")
    try:
        X_train_torch = torch.tensor(X_train_np, dtype=torch.float32)
        y_train_torch = torch.tensor(y_train_np, dtype=torch.long)
        train_dataset = TensorDataset(X_train_torch, y_train_torch)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print(f"  MLP training DataLoader: {len(train_dataset)} samples.")

        mlp_model = SimpleMLP(input_dim, num_classes)
        trained_mlp_model = train_classifier_internal(
            mlp_model, train_loader, device, epochs, lr, class_weights=class_weights_for_mlp
        )
        trained_mlp_model.eval()

        if len(X_val_np) > 0:
            X_val_torch = torch.tensor(X_val_np, dtype=torch.float32).to(device)
            with torch.no_grad():
                val_outputs = trained_mlp_model(X_val_torch)
                _, y_val_pred_mlp = torch.max(val_outputs.data, 1)
            print_validation_report("SimpleMLP", y_val_np, y_val_pred_mlp.cpu().numpy(), class_names_for_report, num_classes)
        else:
            print("  SimpleMLP: No validation data to report on.")
    except Exception as e:
        print(f"  Error during SimpleMLP training/evaluation: {e}")
        import traceback; traceback.print_exc()


    # --- 5. SVM Training & Evaluation ---
    print("\n  --- Training SVM ---")
    try:
        # SVC(probability=True) is needed if you want predict_proba, but slower.
        # class_weight='balanced' helps with imbalanced datasets.
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced', random_state=42)
        print(f"  Fitting SVM on {len(X_train_np)} training samples...")
        svm_model.fit(X_train_np, y_train_np)
        if len(X_val_np) > 0:
            y_val_pred_svm = svm_model.predict(X_val_np)
            print_validation_report("SVM (RBF Kernel)", y_val_np, y_val_pred_svm, class_names_for_report, num_classes)
        else:
            print("  SVM: No validation data to report on.")
    except Exception as e:
        print(f"  Error during SVM training/evaluation: {e}")
        import traceback; traceback.print_exc()


    # --- 6. XGBoost Training & Evaluation ---
    print("\n  --- Training XGBoost ---")
    try:
        # For XGBoost, labels need to be 0 to num_classes-1, which they are.
        # use_label_encoder=False is recommended for newer XGBoost versions.
        # For multiclass, objective is 'multi:softmax' and you need 'num_class'.
        xgb_model = XGBClassifier(
            objective='multi:softmax',
            num_class=num_classes,
            use_label_encoder=False, # Deprecated, use enable_categorical=True for actual categorical features
            eval_metric='mlogloss',    # Common for multiclass
            random_state=42,
            n_estimators=100 # Default, can be tuned
        )
        print(f"  Fitting XGBoost on {len(X_train_np)} training samples...")
        xgb_model.fit(X_train_np, y_train_np)
        if len(X_val_np) > 0:
            y_val_pred_xgb = xgb_model.predict(X_val_np)
            print_validation_report("XGBoost", y_val_np, y_val_pred_xgb, class_names_for_report, num_classes)
        else:
            print("  XGBoost: No validation data to report on.")
    except Exception as e:
        print(f"  Error during XGBoost training/evaluation: {e}")
        import traceback; traceback.print_exc()


    # --- 7. AdaBoost Training & Evaluation ---
    print("\n  --- Training AdaBoost ---")
    try:
        # AdaBoost often uses shallow decision trees as base estimators.
        # Pass class_weight to the base estimator if it supports it.
        base_estimator = DecisionTreeClassifier(max_depth=1, class_weight='balanced', random_state=42)
        ada_model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=50, # Default, can be tuned
            random_state=42
        )
        print(f"  Fitting AdaBoost on {len(X_train_np)} training samples...")
        ada_model.fit(X_train_np, y_train_np)
        if len(X_val_np) > 0:
            y_val_pred_ada = ada_model.predict(X_val_np)
            print_validation_report("AdaBoost (DecisionTree base)", y_val_np, y_val_pred_ada, class_names_for_report, num_classes)
        else:
            print("  AdaBoost: No validation data to report on.")
    except Exception as e:
        print(f"  Error during AdaBoost training/evaluation: {e}")
        import traceback; traceback.print_exc()

    print(f"\n--- Finished training all comparison classifiers for {img_type} ---")
    # The pipeline still expects the MLP model to be returned for saving as 'current'
    return xgb_model



def get_training_stats_path(image_type: str):
    """Gets the path for the JSON file storing training statistics."""
    return os.path.join(IL_STATE_DIR, f"training_stats_{image_type}.json")

def load_training_stats(image_type: str) -> Dict[str, int]:
    """Loads training stats (e.g., {'cancer': 20, 'benign': 15}).
       Value represents the buffer count at which that class was last trained.
    """
    path = get_training_stats_path(image_type)
    if not os.path.exists(path):
        return {} # No stats yet
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {path} (training_stats). Returning empty dict.")
        return {}
    except Exception as e:
        print(f"Error loading training stats from {path}: {e}")
        return {}

def save_training_stats(image_type: str, training_stats: Dict[str, int]):
    """Saves the training statistics."""
    path = get_training_stats_path(image_type)
    try:
        with open(path, 'w') as f:
            json.dump(training_stats, f, indent=4)
        print(f"Saved training stats {training_stats} to {os.path.basename(path)}")
    except Exception as e:
        print(f"Error saving training stats to {path}: {e}")


# il_utils.py
# ... (after imports and other helper functions) ...

def print_validation_report(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str], # e.g., ["Normal", "Cancer"]
    num_expected_classes: int
):
    """Prints a standardized validation report for a classifier."""
    print(f"\n  --- Validation Results for {model_name} ---")
    if len(y_true) == 0 or len(y_pred) == 0:
        print("  Validation set empty or no predictions. Skipping report.")
        return

    val_accuracy = accuracy_score(y_true, y_pred)
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    print("  Validation Classification Report:")
    try:
        # Ensure labels parameter includes all possible class indices for a full report
        report = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0,
            labels=list(range(num_expected_classes)) # Ensures all classes appear
        )
        print(report)
    except ValueError as ve:
        print(f"  Could not generate full classification report (some classes might be missing in y_true or y_pred for validation): {ve}")
        print(f"  Unique true labels in validation: {np.unique(y_true)}")
        print(f"  Unique predicted labels in validation: {np.unique(y_pred)}")

    print("  Validation Confusion Matrix (rows: true, cols: pred):")
    try:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_expected_classes)))
        print(cm)
    except ValueError as ve:
         print(f"  Could not generate confusion matrix: {ve}")
    print(f"  --- End Validation Results for {model_name} ---")

# ... (SimpleMLP class definition, train_classifier_internal, etc.) ...
# --- End Dynamic Training Function ---