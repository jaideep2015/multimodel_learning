"""Per-modality encoders shared by both fusion architectures.

The original notebook defined these same five building blocks twice --
once inline in the concat-fusion model cell, once in the cross-attention
model cell -- with identical code in both places except for how they're
combined at the end. This module keeps a single copy.
"""

import math

import torch
import torch.nn as nn
from transformers import DistilBertModel

from src.config import TEXT_MODEL_NAME


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding, scaled down (0.1x) so it
    nudges rather than dominates the learned time-series embedding.
    """

    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + 0.1 * self.pe[:, : x.size(1), :]


class TimeSeriesModel(nn.Module):
    """Projects the per-month feature vector into embedding space, adds
    positional encoding, and runs a small Transformer encoder over the
    monthly sequence.
    """

    def __init__(self, num_time_series_features, seq_length, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(num_time_series_features, 64)
        self.positional_encoding = PositionalEncoding(embed_dim=64, max_len=seq_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Sequential(nn.Linear(seq_length * 64, 128), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, num_features, seq_len) -> (batch, seq_len, num_features)
        x = self.proj(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


class StaticModel(nn.Module):
    """MLP over static origination/borrower features."""

    def __init__(self, num_static_features, dropout=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_static_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.fc(x)


class TextModel(nn.Module):
    """DistilBERT encoder over the fixed Fed speech, using the [CLS] token."""

    def __init__(self, dropout=0.3, model_name=TEXT_MODEL_NAME):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.fc = nn.Sequential(nn.Linear(self.bert.config.hidden_size, 128), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, text_inputs):
        text_out = self.bert(**text_inputs).last_hidden_state[:, 0, :]
        return self.fc(text_out)


class LiDARCNN(nn.Module):
    """Small CNN over 224x224 grayscale LiDAR images."""

    def __init__(self, dropout=0.3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(nn.Linear(64 * 56 * 56, 128), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
