"""Model evaluation: inference loop, plots, and classification report.

Usage:
    python -m src.evaluate --model cross_attention --checkpoint outputs/best_multimodal_cross_attention_model.pth
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, DATA_DIR, NUM_WORKERS, OUTPUT_DIR, STATIC_END, STATIC_START, TIME_SERIES_COLS
from src.dataset import MultimodalDataset, build_fixed_text_tokens
from src.models import build_model
from src.train import prepare_dataframes


def evaluate_model(model, test_loader, device):
    """Run inference over the test set. Returns (labels, preds, probs)."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            time_series = batch["time_series"].to(device)
            static_features = batch["static_features"].to(device)
            text_inputs = {key: val.to(device) for key, val in batch["text_data"].items()}
            image_data = batch["image_data"].to(device)
            targets = batch["target"].to(device)

            logits = model(time_series, static_features, text_inputs, image_data).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(targets.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)
    return all_labels, all_preds, all_probs


def plot_confusion_matrix(all_labels, all_preds, output_path, cmap="Blues"):
    cm = confusion_matrix(all_labels, all_preds)
    cm_rates = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_rates,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=["No Delinquency", "Delinquent"],
        yticklabels=["No Delinquency", "Delinquent"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Normalized Confusion Matrix (Rates)")
    plt.savefig(output_path)
    plt.close()


def plot_roc_curve(all_labels, all_probs, output_path):
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = roc_auc_score(all_labels, all_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve - Delinquency Prediction")
    plt.legend(loc=4)
    plt.grid()
    plt.savefig(output_path)
    plt.close()
    return auc_score


def build_test_loader(x_test, y_test, fixed_text_tokens):
    test_dataset = MultimodalDataset(x_test, y_test, fixed_text_tokens)
    return DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained multimodal credit delinquency model.")
    parser.add_argument("--model", choices=["concat", "cross_attention"], default="cross_attention")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pth state dict for --model.")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    x_train_val, x_test, _, y_test, _, fixed_text = prepare_dataframes(args.data_dir)
    fixed_text_tokens = build_fixed_text_tokens(fixed_text)
    test_loader = build_test_loader(x_test, y_test, fixed_text_tokens)

    num_time_series_features = len(TIME_SERIES_COLS)
    seq_length = len(x_train_val.iloc[0, 0])
    num_static_features = len(x_train_val.iloc[0, STATIC_START:STATIC_END])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, num_time_series_features, seq_length, num_static_features, dropout=0.3).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    all_labels, all_preds, all_probs = evaluate_model(model, test_loader, device)

    os.makedirs(args.output_dir, exist_ok=True)
    cmap = "Blues" if args.model == "cross_attention" else sns.light_palette("seagreen", as_cmap=True)
    plot_confusion_matrix(all_labels, all_preds, os.path.join(args.output_dir, f"confusion_matrix_{args.model}.pdf"), cmap=cmap)
    auc_score = plot_roc_curve(all_labels, all_probs, os.path.join(args.output_dir, f"roc_curve_{args.model}.pdf"))

    print(f"ROC AUC: {auc_score:.4f}")
    print("Classification Report:\n", classification_report(all_labels, all_preds, digits=4))


if __name__ == "__main__":
    main()
