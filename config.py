"""
Configuration for Hybrid Quantum-Driven TB Detection Framework.
"""
import os
from pathlib import Path

# Paths - adjust DATA_ROOT to your Kaggle dataset location
BASE_DIR = Path(__file__).resolve().parent
# Folder that contains Normal and Tuberculosis subfolders (use exact name from File Explorer)
DATA_ROOT = BASE_DIR / "Dataset of Tuberculosis Chest X-rays Images"
# If your folder has a different name, change the line above, e.g.:
# DATA_ROOT = BASE_DIR / "TB_Chest_Radiography_Database"
MODEL_SAVE_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results"

# Dataset - Tawsifur Rahman TB Chest X-Ray (7000 images: 3500 Normal, 3500 TB)
# Expected structure: DATA_ROOT/Tuberculosis/*.png, DATA_ROOT/Normal/*.png
# Or: DATA_ROOT/Train/Normal, DATA_ROOT/Train/Tuberculosis, etc.
CLASSES = ["Normal", "Tuberculosis"]
NUM_CLASSES = 2
IMG_SIZE = 160  # smaller = much faster on CPU (still works well)
SEED = 42

# CLAHE (from base paper - improves contrast)
CLAHE_CLIP_LIMIT = 3.0
CLAHE_GRID_SIZE = (8, 8)

# Training (CPU-friendly defaults; you can increase later)
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1
EARLY_STOPPING_PATIENCE = 15
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10
NUM_WORKERS = 4  # increase data loading speed on CPU
FREEZE_BACKBONE = True  # huge speedup: train only the head first

# Model
BACKBONE_NAME = "efficientnet_b0"  # fast on CPU: efficientnet_b0/b1/b3/b4
QUANTUM_INSPIRED_DIM = 128
DROPOUT = 0.3

# Augmentation
AUGMENT_BRIGHTNESS = 0.2
AUGMENT_CONTRAST = 0.2
AUGMENT_ROTATION = 15

def ensure_dirs():
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
