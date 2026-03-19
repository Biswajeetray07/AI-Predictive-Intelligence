"""
GRU Model for Time Series Forecasting.

Architecture:
    Input [batch, seq_len, features]
        │
    Stacked GRU (2 layers)
        │
    Last Hidden State
        │
    Dense Layers → Prediction

GRU is lighter than LSTM with comparable performance on many tasks.
"""

import torch
import torch.nn as nn


class GRUForecaster(nn.Module):
    """Stacked GRU with residual connection for time series forecasting."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: [batch, seq_len, features]
        gru_out, h_n = self.gru(x)  # gru_out: [batch, seq_len, hidden]

        # Use the last time step's output
        last_out = gru_out[:, -1, :]  # [batch, hidden]
        last_out = self.layer_norm(last_out)

        out = self.fc(last_out)  # [batch, 1]
        
        # Return both prediction and latent vector
        return out, last_out
