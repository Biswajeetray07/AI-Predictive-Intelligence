"""
LSTM Model for Time Series Forecasting.

Architecture:
    Input [batch, seq_len, features]
        │
    Unidirectional LSTM (2 layers)
        │
    Attention Pooling
        │
    Dense Layers → Prediction

Note: Uses unidirectional (forward-only) LSTM to prevent look-ahead bias.
      Bidirectional LSTMs can see future timesteps within a sequence.
"""

import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    """Unidirectional LSTM with attention pooling for time series forecasting."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # CRITICAL: unidirectional to prevent look-ahead
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention mechanism over time steps
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Context projection (no *2 since unidirectional)
        self.context_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

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
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden]

        # Attention weights
        attn_weights = self.attention(lstm_out)           # [batch, seq_len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1) # normalize over time

        # Weighted sum
        raw_context = (lstm_out * attn_weights).sum(dim=1)    # [batch, hidden]

        # Project context
        context = self.context_projection(raw_context)  # [batch, hidden_dim]

        out = self.fc(context)  # [batch, 1]
        
        # Return both the prediction and the latent context vector for the fusion model
        return out, context

