"""
Transformer Model for Time Series Forecasting.

Architecture:
    Input [batch, seq_len, features]
        │
    Positional Encoding
        │
    Transformer Encoder (4 layers, causal mask)
        │
    Mean Pooling
        │
    Dense Layers → Prediction

Uses native PyTorch TransformerEncoder with causal masking
to prevent attending to future timesteps within a sequence.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence data."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])  # handle odd d_model safely
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]  # type: ignore[operator]
        return self.dropout(x)


class TransformerForecaster(nn.Module):
    """Transformer Encoder for time series forecasting with causal masking."""

    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 4, dropout: float = 0.2):
        super().__init__()

        # Project input features to d_model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate upper-triangular causal mask to prevent attending to future positions."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = self.input_projection(x)       # [batch, seq_len, d_model]
        x = self.pos_encoder(x)            # Add positional encoding

        # CRITICAL: causal mask prevents attending to future timesteps
        causal_mask = self._generate_causal_mask(x.size(1), x.device)
        x = self.transformer_encoder(x, mask=causal_mask)  # [batch, seq_len, d_model]

        # Mean pooling over time dimension
        x = self.layer_norm(x.mean(dim=1))  # [batch, d_model]

        out = self.fc(x)  # [batch, 1]
        
        # Return both prediction and latent vector
        return out, x

