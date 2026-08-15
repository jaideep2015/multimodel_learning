"""Data assembly, training loop, and CLI entry point.

Usage:
    python -m src.train --model cross_attention --epochs 10
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    DEFAULT_EPOCHS,
    LEARNING_RATE,
    NUM_WORKERS,
    OUTPUT_DIR,
    SEED,
    STATIC_END,
    STATIC_START,
    TIME_SERIES_COLS,
    VAL_SIZE,
)
from src.data.download import download_all
from src.data.images import load_image_mapping, merge_images
from src.data.static_features import load_static_features, merge_time_series_with_static, prepare_static_features
from src.data.text import get_fixed_speech
from src.data.time_series import prepare_time_series
from src.dataset import MultimodalDataset, build_fixed_text_tokens, worker_init_fn
from src.models import build_model


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def prepare_dataframes(data_dir=DATA_DIR):
    """Run the full src.data pipeline: download -> label -> scale -> merge
    static features -> merge images -> load fixed speech text.
    """
    paths = download_all(data_dir)

    ts_data = prepare_time_series(paths["time_series_csv"])
    static_df = load_static_features(paths["static_csv"])
    static_train_final, static_test_final = prepare_static_features(
        static_df, ts_data.loan_numbers_train, ts_data.loan_numbers_test
    )
    x_train_val, x_test = merge_time_series_with_static(
        ts_data.x_train_val,
        ts_data.x_test,
        ts_data.loan_numbers_train,
        ts_data.loan_numbers_test,
        static_train_final,
        static_test_final,
    )

    image_df = load_image_mapping(paths["image_mapping_csv"])
    x_train_val, x_test = merge_images(x_train_val, x_test, image_df)

    fixed_text = get_fixed_speech(paths["fed_speeches_csv"])

    return x_train_val, x_test, ts_data.y_train_val, ts_data.y_test, ts_data.pos_weight, fixed_text


def build_dataloaders(x_train_val, x_test, y_train_val, y_test, fixed_text_tokens, seed=SEED):
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=VAL_SIZE, random_state=seed, stratify=y_train_val
    )

    set_seed(seed)
    train_dataset = MultimodalDataset(x_train, y_train, fixed_text_tokens)
    val_dataset = MultimodalDataset(x_val, y_val, fixed_text_tokens)
    test_dataset = MultimodalDataset(x_test, y_test, fixed_text_tokens)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader


def train_model(
    model, train_loader, val_loader, criterion, optimizer, device, num_epochs=DEFAULT_EPOCHS, checkpoint_path="best_model.pth"
):
    """Train with mixed precision on CUDA, checkpointing on best val loss."""
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            optimizer.zero_grad()

            time_series = batch["time_series"].to(device)
            static_features = batch["static_features"].to(device)
            image_data = batch["image_data"].to(device)
            text_inputs = {key: val.to(device) for key, val in batch["text_data"].items()}
            targets = batch["target"].to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(time_series, static_features, text_inputs, image_data).squeeze()
                    loss = criterion(outputs, targets.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(time_series, static_features, text_inputs, image_data).squeeze()
                loss = criterion(outputs, targets.float())
                loss.backward()
                optimizer.step()

            total_train_loss += loss.detach()
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            total_correct += (predictions == targets).sum()
            total_samples += targets.size(0)

        train_acc = 100 * total_correct.cpu().item() / total_samples
        avg_train_loss = total_train_loss.item() / len(train_loader)

        model.eval()
        total_val_loss = 0
        correct, total = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                time_series = batch["time_series"].to(device)
                static_features = batch["static_features"].to(device)
                image_data = batch["image_data"].to(device)
                text_inputs = {key: val.to(device) for key, val in batch["text_data"].items()}
                targets = batch["target"].to(device)

                outputs = model(time_series, static_features, text_inputs, image_data).squeeze()
                loss = criterion(outputs, targets.float())
                total_val_loss += loss.detach()

                probs = torch.sigmoid(outputs)
                predictions = (probs > 0.5).float()
                correct += (predictions == targets).sum()
                total += targets.size(0)

        val_acc = 100 * correct.cpu().item() / total
        avg_val_loss = total_val_loss.item() / len(val_loader)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved at epoch {epoch + 1} with val loss: {avg_val_loss:.4f}")

    print("Training complete.")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Train a multimodal credit delinquency model.")
    parser.add_argument("--model", choices=["concat", "cross_attention"], default="cross_attention")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    set_seed()
    x_train_val, x_test, y_train_val, y_test, pos_weight, fixed_text = prepare_dataframes(args.data_dir)
    fixed_text_tokens = build_fixed_text_tokens(fixed_text)

    train_loader, val_loader, _ = build_dataloaders(x_train_val, x_test, y_train_val, y_test, fixed_text_tokens)

    num_time_series_features = len(TIME_SERIES_COLS)
    seq_length = len(x_train_val.iloc[0, 0])
    num_static_features = len(x_train_val.iloc[0, STATIC_START:STATIC_END])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, num_time_series_features, seq_length, num_static_features, dropout=0.3).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_name = (
        "best_multimodal_model.pth" if args.model == "concat" else "best_multimodal_cross_attention_model.pth"
    )
    checkpoint_path = os.path.join(args.output_dir, checkpoint_name)

    train_model(
        model, train_loader, val_loader, criterion, optimizer, device, num_epochs=args.epochs, checkpoint_path=checkpoint_path
    )


if __name__ == "__main__":
    main()
