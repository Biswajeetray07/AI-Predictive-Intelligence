"""
Multi-Horizon Deep Fusion Model.

Extends the base DeepFusionModel with multi-horizon prediction heads
for 1-day, 5-day, and 30-day forecasting.

Architecture:
    NLP Sequence → Attention Block → Pooled NLP
    TS Features  → GMU Fusion ← Pooled NLP
                       │
                   MLP Backbone
                       │
              ┌────────┼────────┐
              │        │        │
           Head_1d  Head_5d  Head_30d
              │        │        │
           Pred_1d  Pred_5d  Pred_30d
"""

import torch
import torch.nn as nn
from src.models.fusion.fusion import NLPAttentionBlock, GatedModalityUnit


class MultiHorizonFusionModel(nn.Module):
    """
    Multi-horizon variant of DeepFusionModel.
    
    Shares a common backbone (NLP attention + GMU + MLP trunk) and branches
    into horizon-specific prediction heads for 1-day, 5-day, and 30-day forecasts.
    
    Inputs:
        nlp_sequence:   [batch, seq_len, 768]
        ts_features:    [batch, ts_dim]
    
    Output (dict):
        '1d':  [batch, 1] — 1-day prediction
        '5d':  [batch, 1] — 5-day prediction
        '30d': [batch, 1] — 30-day prediction
    """

    def __init__(
        self,
        nlp_dim: int = 768,
        ts_dim: int = 128,
        attention_heads: int = 4,
        mlp_hidden: list = [512, 256, 128],
        dropout: float = 0.3,
        horizons: list = [1, 5, 30],
    ):
        super().__init__()
        self.horizons = horizons

        # Shared NLP Temporal Attention Block
        self.nlp_attention = NLPAttentionBlock(
            embed_dim=nlp_dim,
            num_heads=attention_heads,
            dropout=dropout * 0.5,
        )

        # Shared Gated Fusion
        fusion_dim = 256
        self.gmu = GatedModalityUnit(
            nlp_dim=nlp_dim,
            ts_dim=ts_dim,
            out_dim=fusion_dim,
            dropout=dropout,
        )

        # Shared MLP Backbone (up to but not including final projection)
        layers = []
        in_dim = fusion_dim
        for hidden in mlp_hidden:
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden
        self.backbone = nn.Sequential(*layers)
        
        # Horizon-specific heads
        self.heads = nn.ModuleDict()
        for h in horizons:
            self.heads[f"head_{h}d"] = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 1),
            )

    def forward(self, nlp_sequence, ts_features):
        # 1. Attend to NLP sequence
        nlp_attended = self.nlp_attention(nlp_sequence)  # [batch, 768]

        # 2. Gated fusion
        fused = self.gmu(nlp_attended, ts_features)  # [batch, 256]

        # 3. Shared backbone
        backbone_out = self.backbone(fused)  # [batch, 128]

        # 4. Horizon-specific predictions
        predictions = {}
        for h in self.horizons:
            predictions[f"{h}d"] = self.heads[f"head_{h}d"](backbone_out)

        return predictions


if __name__ == "__main__":
    # Quick verification
    model = MultiHorizonFusionModel(nlp_dim=768, ts_dim=128)
    nlp = torch.randn(4, 60, 768)
    ts = torch.randn(4, 128)
    out = model(nlp, ts)
    
    for horizon, pred in out.items():
        print(f"{horizon} prediction shape: {pred.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("✅ Multi-horizon fusion model verified!")
