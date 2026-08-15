"""Cross-attention fusion: gated attention between modality pairs.

See docs/images/architecture_cross_attention.png. Test performance:
accuracy 93.5%, delinquent-class F1 0.345, same recall as the concat-fusion
model (72.3%) but better precision (see README results table).
"""

import torch
import torch.nn as nn

from src.models.encoders import LiDARCNN, StaticModel, TextModel, TimeSeriesModel


class CrossAttention(nn.Module):
    """Gated cross-attention: attends x1 (query) to x2 (key/value), then
    blends the attended output with the original x1 via a learned sigmoid
    gate, a learnable scale, and layer normalization.
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.gate = nn.Linear(embed_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.scale = nn.Parameter(torch.ones(1))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x1, x2):
        """
        Args:
            x1: Query features (batch, feature_dim).
            x2: Key/Value features (batch, feature_dim).
        Returns:
            Attended feature representation (batch, feature_dim).
        """
        gate_value = self.sigmoid(self.gate(torch.cat([x1, x2], dim=-1)))
        attn_output, _ = self.attention(x1.unsqueeze(0), x2.unsqueeze(0), x2.unsqueeze(0))
        return self.norm(self.scale * (gate_value * attn_output.squeeze(0) + (1 - gate_value) * x1))


class CrossAttentionFusionModel(nn.Module):
    def __init__(self, num_time_series_features, seq_length, num_static_features, dropout=0.3):
        super().__init__()

        self.time_series_model = TimeSeriesModel(num_time_series_features, seq_length, dropout)
        self.static_model = StaticModel(num_static_features, dropout)
        self.text_model = TextModel(dropout)
        self.image_model = LiDARCNN(dropout)

        # First-level cross-attention between modality pairs
        self.attn_text_time = CrossAttention(embed_dim=128, num_heads=4, dropout=dropout)
        self.attn_static_image = CrossAttention(embed_dim=128, num_heads=4, dropout=dropout)
        self.attn_time_static = CrossAttention(embed_dim=128, num_heads=4, dropout=dropout)

        # Second-level cross-attention between first-level outputs
        self.attn_fusion1 = CrossAttention(embed_dim=128, num_heads=4, dropout=dropout)
        self.attn_fusion2 = CrossAttention(embed_dim=128, num_heads=4, dropout=dropout)

        # Learnable attention-based fusion weights over the 3 final representations
        self.attn_weights = nn.Linear(128 * 3, 3)

        self.fusion_layer = nn.Sequential(nn.Linear(128, 1))

    def forward(self, time_series, static_features, text_inputs, image_data):
        time_series_out = self.time_series_model(time_series)
        static_out = self.static_model(static_features)
        text_out = self.text_model(text_inputs)
        image_out = self.image_model(image_data)

        attended_text_time = self.attn_text_time(text_out, time_series_out)
        attended_static_image = self.attn_static_image(static_out, image_out)
        attended_time_static = self.attn_time_static(time_series_out, static_out)

        attended_fusion1 = self.attn_fusion1(attended_text_time, attended_static_image)
        attended_fusion2 = self.attn_fusion2(attended_time_static, attended_static_image)

        weights = torch.softmax(
            self.attn_weights(torch.cat([attended_fusion1, attended_fusion2, attended_static_image], dim=1)), dim=1
        )
        fused_weighted = (
            weights[:, 0].unsqueeze(1) * attended_fusion1
            + weights[:, 1].unsqueeze(1) * attended_fusion2
            + weights[:, 2].unsqueeze(1) * attended_static_image
        )

        return self.fusion_layer(fused_weighted)  # raw logits, for use with BCEWithLogitsLoss
