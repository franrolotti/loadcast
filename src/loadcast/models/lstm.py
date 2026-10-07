"""Encoder-decoder LSTM with a direct (non-autoregressive) decoder.

See docs/models/lstm.md.
"""

from __future__ import annotations

import torch
from torch import nn

from loadcast.models.neural import NeuralForecaster


class Seq2SeqLSTM(nn.Module):
    def __init__(
        self,
        n_past: int,
        n_future: int,
        n_quantiles: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(n_past, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(n_future, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_size, n_quantiles)

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        # The encoder compresses the last week into its final (h, c) state; the decoder
        # starts from it and reads the known covariates of the target day.
        _, state = self.encoder(x_past)
        out, _ = self.decoder(x_future, state)
        return self.head(out)  # (B, horizon, n_quantiles)


class LSTM(NeuralForecaster):
    name = "lstm"

    def network(self, n_past: int, n_future: int, n_quantiles: int) -> nn.Module:
        return Seq2SeqLSTM(n_past, n_future, n_quantiles, **self.cfg.params["lstm"])
