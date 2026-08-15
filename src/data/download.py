"""Named downloaders for the original lab's Google Drive-hosted assets.

The original notebook scattered `!gdown --fuzzy <url>` calls across many
cells. This module gives each asset a name and a destination path so the
rest of the pipeline can depend on file paths instead of Colab cell order.

None of this data is redistributed in this repository -- run the functions
below (or `python -m src.data.download`) to fetch it into `DATA_DIR` /
`OUTPUT_DIR` before running the pipeline.
"""

import os
import zipfile

import gdown

from src.config import DATA_DIR, OUTPUT_DIR

_ASSETS = {
    "time_series_csv": {
        "url": "https://drive.google.com/file/d/1cte8MKaSdt3LrqXSUo9oGTe2hnkz8JsU/view?usp=sharing",
        "filename": "Multimodal_Lab_sampled.csv",
    },
    "static_csv": {
        "url": "https://drive.google.com/file/d/1lF1n0Ue-emaflJabOfmQI6CT0D3qAIYK/view?usp=sharing",
        "filename": "Multimodal_Lab_Origin_sampled.csv",
    },
    "lidar_images_zip": {
        "url": "https://drive.google.com/file/d/1cHLNcnTLdaKtedM_Uz-xmYaXXml0Ws-D/view?usp=sharing",
        "filename": "usgs_lidar.zip",
    },
    "image_mapping_csv": {
        "url": "https://drive.google.com/file/d/1QyzNtHVurvrXvHG1j2oeq7rwfC6vnA1h/view?usp=sharing",
        "filename": "msa_zip3_mapping_cleaned.csv",
    },
    "fed_speeches_csv": {
        "url": "https://drive.google.com/file/d/1uVt9BC2tgr-MWrFZvYvA_I8IzTabNZtL/view?usp=sharing",
        "filename": "fed_speeches.csv",
    },
}

_CHECKPOINTS = {
    "concat": {
        "url": "https://drive.google.com/file/d/1pCaOYSHK1kCiytdU_giHhQ6WoA8mrtCe/view?usp=sharing",
        "filename": "best_multimodal_model.pth",
    },
    "cross_attention": {
        "url": "https://drive.google.com/file/d/1UZOm-LHx1_Bu-_aUjCYGuJELrx6j_VbY/view?usp=sharing",
        "filename": "best_multimodal_cross_attention_model.pth",
    },
}


def _download_asset(key, data_dir=DATA_DIR):
    asset = _ASSETS[key]
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, asset["filename"])
    if not os.path.exists(out_path):
        gdown.download(asset["url"], out_path, fuzzy=True, quiet=False)
    return out_path


def download_time_series_csv(data_dir=DATA_DIR):
    return _download_asset("time_series_csv", data_dir)


def download_static_csv(data_dir=DATA_DIR):
    return _download_asset("static_csv", data_dir)


def download_image_mapping_csv(data_dir=DATA_DIR):
    return _download_asset("image_mapping_csv", data_dir)


def download_fed_speeches_csv(data_dir=DATA_DIR):
    return _download_asset("fed_speeches_csv", data_dir)


def download_lidar_images(data_dir=DATA_DIR):
    """Download and extract the LiDAR image archive, returning the extraction dir."""
    zip_path = _download_asset("lidar_images_zip", data_dir)
    extract_dir = os.path.join(data_dir, "usgs_lidar")
    if not os.path.isdir(extract_dir):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    return extract_dir


def download_pretrained_checkpoint(model_name, output_dir=OUTPUT_DIR):
    """model_name: 'concat' or 'cross_attention'."""
    asset = _CHECKPOINTS[model_name]
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, asset["filename"])
    if not os.path.exists(out_path):
        gdown.download(asset["url"], out_path, fuzzy=True, quiet=False)
    return out_path


def download_all(data_dir=DATA_DIR):
    """Fetch every raw data asset (not the pretrained checkpoints)."""
    return {
        "time_series_csv": download_time_series_csv(data_dir),
        "static_csv": download_static_csv(data_dir),
        "lidar_images_dir": download_lidar_images(data_dir),
        "image_mapping_csv": download_image_mapping_csv(data_dir),
        "fed_speeches_csv": download_fed_speeches_csv(data_dir),
    }


if __name__ == "__main__":
    paths = download_all()
    for name, path in paths.items():
        print(f"{name}: {path}")
