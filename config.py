"""
Configuration file for Multimodal (Clinical + MRI) training.
Optimized for Linux environment with GPU access.

"""

import os

RUN_NUMBER = os.environ["RUN_NUMBER"]

def _get_env_float(name, default):
    """Get float value from environment variable with fallback."""
    v = os.environ.get(name, "")
    try:
        return float(v) if v else default
    except ValueError:
        return default

def _get_env_int(name, default):
    """Get integer value from environment variable with fallback."""
    v = os.environ.get(name, "")
    try:
        return int(v) if v else default
    except ValueError:
        return default

def _get_env_str(name, default):
    """Get string value from environment variable with fallback."""
    return os.environ.get(name, default)

MERGED_CSV = "/home/lude14/bachelorarbeit/MRI_CNN/merged_multimodal_maddi.csv"
DATA_DIR = "/sc-projects/sc-proj-ukb-cvd/projects/theses/data/adni/MRI/ADNI"
CACHE_PATH = "/home/lude14/bachelorarbeit/MRI_CNN/nii_cache.pkl"
SAVE_DIR = "/sc-projects/sc-proj-ukb-cvd/projects/theses/data/adni/RUNS/Multimodal"

# Training hyperparameters 
BATCH_SIZE = _get_env_int("BATCH_SIZE", 4)
NUM_WORKERS = _get_env_int("NUM_WORKERS", 4)
N_SPLITS = 5
EPOCHS = 200
LEARNING_RATE = _get_env_float("LEARNING_RATE", 1e-4)
WEIGHT_DECAY = _get_env_float("WEIGHT_DECAY", 1e-4)
PATIENCE = _get_env_int("PATIENCE", 20)
ACCUMULATE_GRAD_BATCHES = _get_env_int("ACCUMULATE_GRAD_BATCHES", 32)

# Data splits
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

# MRI processing parameters
TARGET_SHAPE = (96, 96, 96) # (D, H, W)

# Augmentation parameters
FLIP_PROB = 0.5
AFFINE_PROB = 0.4
NOISE_PROB = 0.2
DROPOUT_PROB = _get_env_float("DROPOUT_PROB", 0.3)

# Model architecture parameters
FUSION_HIDDEN_DIM = _get_env_int("FUSION_HIDDEN_DIM", 128)
CLINICAL_FEATURE_DIM = 32  # Output dimension from clinical model
MRI_FEATURE_DIM = 512      # Output dimension from MRI model

# Cross-validation settings
CV_RANDOM_STATE = 42

# Logging configuration
WANDB_PROJECT = "multimodal-adni-cross-validation"
LOG_EVERY_N_STEPS = 5

# GPU optimization settings
PIN_MEMORY = True
PERSISTENT_WORKERS = True

def validate_config():
    """Validate that required paths exist and create save directory if needed."""
    missing_paths = []
    
    if not os.path.exists(MERGED_CSV):
        missing_paths.append(f"MERGED_CSV: {MERGED_CSV}")
    
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
        print("Configuration validation successful!")
        print(f"Target shape: {TARGET_SHAPE}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Dropout probability: {DROPOUT_PROB}")
        print(f"Fusion hidden dimension: {FUSION_HIDDEN_DIM}")
        print(f"Run number: {RUN_NUMBER}")
        print(f"Save directory: {SAVE_DIR}")
    except FileNotFoundError as e:
        print(f"Configuration validation failed:\n{e}")
