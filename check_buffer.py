# import numpy as np
# import os
# import sys

# print(f"Python executable: {sys.executable}")
# print(f"Current working directory: {os.getcwd()}")

# # --- Ensure IL_STATE_DIR is correct relative to this script ---
# IL_STATE_DIR = os.path.join(os.path.dirname(__file__), "il_state")
# print(f"Expected il_state directory: {IL_STATE_DIR}")

# filepath = os.path.join(IL_STATE_DIR, "buffer_normal_replay_lung.npz")


# print(f"Attempting to load: {filepath}")

# if not os.path.exists(IL_STATE_DIR):
#     print(f"ERROR: Directory '{IL_STATE_DIR}' does not exist!")
# elif not os.path.exists(filepath):
#     print(f"ERROR: File '{filepath}' does not exist!")
# else:
#     try:
#         data = np.load(filepath, allow_pickle=True)
#         print(f"\nFile loaded successfully.")
#         print(f"File keys found: {data.files}")

#         if 'features' in data:
#             features = data['features']
#             print(f"\nFound 'features' key.")
#             print(f"Data type loaded: {type(features)}")

#             if hasattr(features, '__len__'):
#                  print(f"Number of items/samples: {len(features)}")
#                  if len(features) > 0:
#                      first_item = features[0]
#                      print(f"First item type: {type(first_item)}")
#                      # Check if first item is the feature array (expected format)
#                      if isinstance(first_item, np.ndarray) and hasattr(first_item, 'shape'):
#                           print("First item looks like a feature array.")
#                           print(f"  Feature vector shape: {first_item.shape}")
#                      # Check if first item is tuple (older format?)
#                      elif isinstance(first_item, (tuple, list)) and len(first_item) == 2:
#                           print("First item looks like a (feature, label) tuple.")
#                      else:
#                           print("First item structure is unexpected.")
#             elif hasattr(features, 'shape'):
#                  print(f"Shape of loaded 'features' data: {features.shape}")
#             else:
#                  print("'features' data has no length or shape attribute.")

#         else:
#             print("\nError: 'features' key not found in the file.")

#     except Exception as e:
#         print(f"\nError loading or inspecting file '{filepath}': {e}")
import numpy as np
import os
import sys

print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")

IL_STATE_DIR = os.path.join(os.path.dirname(__file__), "il_state")
print(f"Expected il_state directory: {IL_STATE_DIR}")

# --- MODIFY THIS LINE TO CHECK THE CANCER BUFFER ---
buffer_to_check = "buffer_cancer_anomalies_lung.npz"
# ---
filepath = os.path.join(IL_STATE_DIR, buffer_to_check)
print(f"Attempting to load: {filepath}")

if not os.path.exists(IL_STATE_DIR):
    print(f"ERROR: Directory '{IL_STATE_DIR}' does not exist!")
elif not os.path.exists(filepath):
    print(f"ERROR: File '{filepath}' does not exist!")
else:
    try:
        data = np.load(filepath, allow_pickle=True)
        print(f"\nFile loaded successfully.")
        print(f"File keys found: {data.files}")

        if 'features' in data:
            items_in_buffer = data['features'] # This should be an array of your (feature, label) tuples
            print(f"\nFound 'features' key.")
            print(f"Data type loaded for 'features': {type(items_in_buffer)}")
            print(f"Is it a NumPy array? {isinstance(items_in_buffer, np.ndarray)}")
            if hasattr(items_in_buffer, 'shape'):
                print(f"Shape of 'features' array: {items_in_buffer.shape}")


            if hasattr(items_in_buffer, '__len__'):
                 print(f"Number of items/samples stored: {len(items_in_buffer)}")
                 if len(items_in_buffer) > 0:
                     first_item = items_in_buffer[0]
                     print(f"\n--- Inspecting first item ---")
                     print(f"First item type: {type(first_item)}")

                     if isinstance(first_item, (tuple, list)) and len(first_item) == 2:
                          print("First item IS a tuple/list of length 2.")
                          feature_vector = first_item[0]
                          label = first_item[1]
                          print(f"  Item[0] (feature) type: {type(feature_vector)}")
                          print(f"  Item[0] is np.ndarray? {isinstance(feature_vector, np.ndarray)}")
                          if hasattr(feature_vector, 'shape'):
                              print(f"    Feature vector shape: {feature_vector.shape}")

                          print(f"  Item[1] (label) type: {type(label)}")
                          print(f"  Item[1] (label) value: '{label}'") # Crucial: Check this value
                     else:
                          print("First item is NOT a tuple/list of length 2 as expected for anomaly buffers.")
            else:
                 print("'features' data has no length attribute.")
        else:
            print("\nError: 'features' key not found in the file.")
    except Exception as e:
        print(f"\nError loading or inspecting file '{filepath}': {e}")