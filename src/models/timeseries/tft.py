"""
Temporal Fusion Transformer (TFT) Model for Time Series Forecasting.

Simplified TFT architecture:
    Input [batch, seq_len, features]
        │
    Variable Selection Network
        │
    LSTM Encoder
        │
    Multi-Head Attention (Interpretable)
        │
    Gated Residual Network
        │
    Dense → Prediction

This is a lightweight version of the full TFT paper (Lim et al., 2019),
adapted for our single-target regression task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualNetwork(nn.Module):
    """GRN block: core building block of TFT."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

        # Skip connection projection if dims differ
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        h2 = self.fc2(h)
        gate = torch.sigmoid(self.gate(h))
        out = self.layer_norm(gate * h2 + residual)
        return out


class VariableSelectionNetwork(nn.Module):
    """Learns which input features matter most at each time step."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.grn = GatedResidualNetwork(input_dim, hidden_dim, input_dim, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [batch, seq_len, features]
        weights = self.grn(x)                    # [batch, seq_len, features]
        weights = self.softmax(weights)          # feature importance weights
        selected = x * weights                   # apply selection
        return selected, weights


class TFTForecaster(nn.Module):
    """Simplified Temporal Fusion Transformer for time series forecasting."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        # Variable Selection
        self.var_selection = VariableSelectionNetwork(input_dim, hidden_dim, dropout)

        # Temporal Processing (LSTM encoder)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Multi-Head Interpretable Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Post-attention GRN
        self.post_attention_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)

        # Final prediction head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: [batch, seq_len, features]

        # 1. Variable Selection
        selected, var_weights = self.var_selection(x)  # [batch, seq_len, features]

        # 2. LSTM Encoding
        lstm_out, _ = self.lstm(selected)  # [batch, seq_len, hidden]

        # 3. Multi-Head Attention (self-attention over time)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)  # [batch, seq_len, hidden]

        # 4. GRN post-processing with residual
        enriched = self.post_attention_grn(attn_out + lstm_out)  # [batch, seq_len, hidden]

        # 5. Use last time step for prediction
        context = enriched[:, -1, :]  # [batch, hidden] — 2D context vector for ensemble
        pred = self.fc(context)       # [batch, 1]

        # Return both prediction and 2D latent context vector (matches LSTM/GRU/Transformer)
        return pred, context
