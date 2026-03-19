"""
Attention-Based Deep Fusion Model.

Architecture:
    NLP Embedding (768-D)
            │
    Multi-Head Self-Attention Layer
            │
            ▼
    Concatenate with Time Series Features
            │
    MLP Prediction Head (Dense + Dropout)
            │
    Final Prediction

This replaces simple concatenation, allowing the model to learn
which NLP signal dimensions are most predictive under different conditions.
"""

import torch
import torch.nn as nn
import math


class NLPAttentionBlock(nn.Module):
    """
    Multi-head self-attention over the NLP embedding temporal sequence.
    Treats the 768-D sequence over time to learn temporal importance.
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, nlp_sequence):
        """
        Args:
            nlp_sequence: [batch, seq_len, 768]
        Returns:
            attended: [batch, 768]
        """
        # Self-attention over the time dimension
        attn_out, _ = self.attention(nlp_sequence, nlp_sequence, nlp_sequence)

        # Gated residual connection
        gate = self.gate(attn_out)
        attended = self.layer_norm(gate * attn_out + (1 - gate) * nlp_sequence)
        
        # Mean pooling to get a single vector per batch element
        pooled = attended.mean(dim=1)
        
        return pooled


class GatedModalityUnit(nn.Module):
    """
    Adaptive gating mechanism to dynamically weigh NLP vs Time Series features based on their confidence.
    """
    def __init__(self, nlp_dim: int, ts_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        # Projections to common space
        self.nlp_proj = nn.Linear(nlp_dim, out_dim)
        self.ts_proj = nn.Linear(ts_dim, out_dim)
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, nlp_feat, ts_feat):
        # nlp_feat: [batch, nlp_dim], ts_feat: [batch, ts_dim]
        h_nlp = torch.tanh(self.nlp_proj(nlp_feat))
        h_ts = torch.tanh(self.ts_proj(ts_feat))
        
        # Calculate gate
        combined_features = torch.cat([h_nlp, h_ts], dim=-1)
        z = self.gate(combined_features)
        
        # Gated fusion
        fused = z * h_nlp + (1 - z) * h_ts
        return self.dropout(self.layer_norm(fused))


class DeepFusionModel(nn.Module):
    """
    Fuses Temporal NLP embedding sequences with downstream Time Series features via GMU.

    Inputs:
        nlp_sequence:   [batch, seq_len, 768]
        ts_features:    [batch, ts_dim]

    Output:
        prediction:     [batch, 1]
    """

    def __init__(self, nlp_dim: int = 768, ts_dim: int = 128,
                 attention_heads: int = 4, mlp_hidden: list = [512, 256, 128],
                 dropout: float = 0.3):
        super().__init__()

        # NLP Temporal Attention Block
        self.nlp_attention = NLPAttentionBlock(
            embed_dim=nlp_dim,
            num_heads=attention_heads,
            dropout=dropout * 0.5,
        )

        # Robust Gated Fusion
        fusion_dim = 256
        self.gmu = GatedModalityUnit(
            nlp_dim=nlp_dim,
            ts_dim=ts_dim,
            out_dim=fusion_dim,
            dropout=dropout
        )

        # MLP Prediction Head
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
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, nlp_sequence, ts_features):
        # 1. Apply temporal attention to sequence of NLP embeddings
        nlp_attended = self.nlp_attention(nlp_sequence)  # [batch, 768]

        # 2. GMU dynamically fuses the signals
        fused = self.gmu(nlp_attended, ts_features)  # [batch, 256]

        # 3. Predict
        prediction = self.mlp(fused)  # [batch, 1]

        return prediction


if __name__ == "__main__":
    # Quick verification
    model = DeepFusionModel(nlp_dim=768, ts_dim=128)
    nlp = torch.randn(4, 60, 768)  # Check 3D shape
    ts = torch.randn(4, 128)
    out = model(nlp, ts)
    print(f"Output shape: {out.shape}")  # [4, 1]
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
