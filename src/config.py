"""
Configuration file for the Brain Tumor Classification project.
"""

from pathlib import Path

# ======================================================
# PROJECT PATHS
# ======================================================

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

TRAIN_DIR = RAW_DATA_DIR / "Training"
TEST_DIR = RAW_DATA_DIR / "Testing"

# Outputs
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# ======================================================
# MODEL CONFIGURATION
# ======================================================

IMAGE_SIZE = 224

NUM_CHANNELS = 3

NUM_CLASSES = 4

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# ======================================================
# TRAINING CONFIGURATION
# ======================================================

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 1e-4

RANDOM_SEED = 42

NUM_WORKERS = 4

PIN_MEMORY = False