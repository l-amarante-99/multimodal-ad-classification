"""
Configuration file for MRI CNN training.
Optimized for Linux environment with GPU access.
"""

import os

def _get_env_float(name, default):
    v = os.environ.get(name, "")
    try:
        return float(v) if v else default
    except ValueError:
        return default

def _get_env_int(name, default):
    v = os.environ.get(name, "")
    try:
        return int(v) if v else default
    except ValueError:
        return default

# Data paths
META_CSV = "/home/lude14/bachelorarbeit/MRI_CNN/MRI_DIAGNOSIS.csv"
DATA_DIR = "/sc-projects/sc-proj-ukb-cvd/projects/theses/data/adni/MRI/ADNI"
CACHE_PATH = "/home/lude14/bachelorarbeit/MRI_CNN/nii_cache.pkl"
SAVE_DIR = "/sc-projects/sc-proj-ukb-cvd/projects/theses/data/adni/RUNS/Imaging"

# Training hyperparameters
BATCH_SIZE = 8
NUM_WORKERS = 8 
N_SPLITS = 5
EPOCHS = 200
LEARNING_RATE = _get_env_float("LEARNING_RATE", 1e-4)
WEIGHT_DECAY = 1e-4  
PATIENCE = 15 
ACCUMULATE_GRAD_BATCHES = _get_env_int("ACCUMULATE_GRAD_BATCHES", 4)
TARGET_SHAPE = (128, 128, 128)  # Target shape for MRI volume resizing

# Cross-validation settings
CV_RANDOM_STATE = 42

# Logging configuration
WANDB_PROJECT = "mri-resnet-3D-Cross-Validation"
LOG_EVERY_N_STEPS = 5

# GPU optimization settings
PIN_MEMORY = True
PERSISTENT_WORKERS = True 

def validate_config():
    """Validate that required paths exist and create save directory if needed."""
    missing_paths = []
    
    if not os.path.exists(META_CSV):
        missing_paths.append(f"META_CSV: {META_CSV}")
    
    if not os.path.exists(DATA_DIR):
        missing_paths.append(f"DATA_DIR: {DATA_DIR}")

    cache_dir = os.path.dirname(CACHE_PATH)
    if not os.path.exists(cache_dir):
        missing_paths.append(f"CACHE_PATH directory: {cache_dir}")
    
    # Create save directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        try:
            os.makedirs(SAVE_DIR, exist_ok=True)
            print(f"Created save directory: {SAVE_DIR}")
        except OSError as e:
            missing_paths.append(f"SAVE_DIR (cannot create): {SAVE_DIR} - {e}")
    
    if missing_paths:
        raise FileNotFoundError(
            "Missing required files/directories:\n" + "\n".join(missing_paths)
        )
    
    return True

if __name__ == "__main__":
    try:
        validate_config()
    except FileNotFoundError as e:
        print(f"Configuration validation failed:\n{e}")
