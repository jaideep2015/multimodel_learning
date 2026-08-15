from src.models.fusion_concat import MultimodalDelinquencyModel
from src.models.fusion_cross_attention import CrossAttentionFusionModel

MODEL_REGISTRY = {
    "concat": MultimodalDelinquencyModel,
    "cross_attention": CrossAttentionFusionModel,
}


def build_model(name, num_time_series_features, seq_length, num_static_features, dropout=0.3):
    return MODEL_REGISTRY[name](num_time_series_features, seq_length, num_static_features, dropout=dropout)
