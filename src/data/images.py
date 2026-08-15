"""LiDAR image mapping: attach a geography-level image path to each loan.

Images are linked at the metro-area/ZIP3 level (MSA + ZIP3), not per loan --
loans sharing a metro area and ZIP3 prefix share the same LiDAR image. This
is a known simplification of the original lab, not a per-loan photo.
"""

import pandas as pd


def load_image_mapping(csv_path: str) -> pd.DataFrame:
    """Load the MSA+ZIP3 -> LiDAR_File path mapping table."""
    return pd.read_csv(csv_path, dtype=str)


def merge_images(x_train_val: pd.DataFrame, x_test: pd.DataFrame, image_df: pd.DataFrame):
    """Join the LiDAR_File path onto each split by (MSA, ZIP3), then drop the
    join keys since they're not model features.
    """
    x_train_val = x_train_val.merge(image_df, on=["MSA", "ZIP3"], how="left")
    x_test = x_test.merge(image_df, on=["MSA", "ZIP3"], how="left")

    x_train_val = x_train_val.drop(columns=["MSA", "ZIP3"])
    x_test = x_test.drop(columns=["MSA", "ZIP3"])

    return x_train_val, x_test
