"""PyTorch Dataset that assembles all four modalities per loan sample."""

import random

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.config import (
    IMAGE_SIZE,
    SEED,
    STATIC_END,
    STATIC_START,
    TEXT_MAX_LENGTH,
    TEXT_MODEL_NAME,
    TIME_SERIES_END,
    TIME_SERIES_START,
)


def build_fixed_text_tokens(text: str, tokenizer=None, max_length: int = TEXT_MAX_LENGTH) -> dict:
    """Tokenize the single fixed Fed speech once; every sample reuses it."""
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    return {key: val.squeeze(0) for key, val in tokens.items()}


class MultimodalDataset(Dataset):
    """Combines time-series, static, LiDAR image, and fixed text features.

    `dataframe` columns are expected in the order produced by the src.data
    pipeline: time-series columns (array-valued) | static feature columns |
    LiDAR_File (path string, last column).
    """

    def __init__(self, dataframe, y_values, fixed_text_tokens):
        self.dataframe = dataframe.reset_index(drop=True)
        self.y_values = torch.tensor(y_values, dtype=torch.float32)
        self.fixed_text_tokens = fixed_text_tokens
        self.normalize_transform = transforms.Normalize(mean=[0.5], std=[0.5])

    def load_image(self, image_path):
        """Load and preprocess a LiDAR image using OpenCV for speed."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        return self.normalize_transform(image)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        time_series = torch.tensor(
            np.array(row.iloc[TIME_SERIES_START:TIME_SERIES_END].tolist()), dtype=torch.float32
        )
        static_features = torch.tensor(
            np.array(row.iloc[STATIC_START:STATIC_END].astype(float).values.tolist()), dtype=torch.float32
        )
        image = self.load_image(row["LiDAR_File"])
        target = self.y_values[idx]

        # Every sample shares the same tokenized text (see README limitations).
        return {
            "time_series": time_series,
            "static_features": static_features,
            "image_data": image,
            "text_data": self.fixed_text_tokens,
            "target": target,
        }


def worker_init_fn(worker_id, seed=SEED):
    """Ensure each DataLoader worker has a distinct, reproducible seed."""
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
