"""Intermediate fusion: concatenate all four modality embeddings.

See docs/images/architecture_concat_fusion.png. Test performance:
accuracy 92.0%, delinquent-class F1 0.30 (see README results table).
"""

import torch
import torch.nn as nn

from src.models.encoders import LiDARCNN, StaticModel, TextModel, TimeSeriesModel


class MultimodalDelinquencyModel(nn.Module):
    def __init__(self, num_time_series_features, seq_length, num_static_features, dropout=0.3):
        super().__init__()

        self.time_series_model = TimeSeriesModel(num_time_series_features, seq_length, dropout)
        self.static_fc = StaticModel(num_static_features, dropout)
        self.text_model = TextModel(dropout)
        self.image_model = LiDARCNN(dropout)

        self.fusion_layer = nn.Sequential(
            nn.Linear(128 + 128 + 128 + 128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, time_series, static_features, text_inputs, image_data):
        time_series_out = self.time_series_model(time_series)
        static_out = self.static_fc(static_features)
        text_out = self.text_model(text_inputs)
        image_out = self.image_model(image_data)

        fused = torch.cat([time_series_out, static_out, text_out, image_out], dim=1)
        return self.fusion_layer(fused)  # raw logits, for use with BCEWithLogitsLoss
