"""Shared constants for the multimodal credit scoring pipeline."""

import os

# --- Reproducibility ---
SEED = 42

# --- Paths ---
# Directory where raw downloaded data (CSVs, images, checkpoints) is stored.
DATA_DIR = os.environ.get("MULTIMODAL_DATA_DIR", "data")
# Directory where trained checkpoints and generated plots are written.
OUTPUT_DIR = os.environ.get("MULTIMODAL_OUTPUT_DIR", "outputs")

# --- Time-series features (Freddie Mac monthly loan performance) ---
# Order matters: this is the channel order fed into TimeSeriesModel.
TIME_SERIES_COLS = [
    "CURRENT_ACTUAL_UPB",
    "LOAN_AGE",
    "REMAINING_MONTHS",
    "CURRENT_INTEREST_RATE",
    "CURRENT_NON_INTEREST_BEARING_UPB",
    "ESTIMATED_LOAN_TO_VALUE",
    "INTEREST_BEARING_UPB",
    "CURRENT_LOAN_DELINQUENCY_STATUS",
]

# --- Static (origination) features ---
STATIC_NUMERIC_COLS = [
    "CREDIT_SCORE",
    "CLTV",
    "DTI_RATIO",
    "ORIGINAL_UPB",
    "ORIGINAL_LOAN_TERM",
    "MI_PERCENTAGE",
]
STATIC_CATEGORICAL_COLS = [
    "FIRST_TIME_HOMEBUYER",
    "OCCUPANCY_STATUS",
    "PROPERTY_TYPE",
]

# --- Labeling window ---
# Loans are labeled delinquent (target=1) if they were ever delinquent during
# this window; rows in this window are then dropped from the model inputs so
# the model only sees data strictly before the prediction window.
LABEL_WINDOW_START = "2024-04-01"
LABEL_WINDOW_END = "2024-06-30"

# --- Dataset / DataLoader ---
IMAGE_SIZE = 224
TEXT_MAX_LENGTH = 512
TEXT_MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 128
NUM_WORKERS = 2

# Column index boundaries used by MultimodalDataset to slice a merged row into
# (time-series columns) | (static feature columns) | (LiDAR_File, ...).
# Time-series columns come first (one array-valued column per feature),
# followed by static features, followed by the LiDAR_File path column last.
TIME_SERIES_START = 0
TIME_SERIES_END = len(TIME_SERIES_COLS)  # 8
STATIC_START = TIME_SERIES_END
STATIC_END = -1  # everything up to (not including) the trailing LiDAR_File column

# --- Training ---
TEST_SIZE = 0.2  # held-out test split, stratified by target
VAL_SIZE = 0.33  # validation split taken from the remaining train_val set
LEARNING_RATE = 1e-5
DEFAULT_EPOCHS = 10
