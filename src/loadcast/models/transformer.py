"""A compact Transformer designed for day-ahead load forecasting.

Built from standard PyTorch layers, not from a forecasting library, so that every
design choice is explicit and can be ablated. See docs/models/transformer.md.

    x_past   (B, L, Fp) --VSN--> (B, L, d) --daily patches--> (B, L/P, d) --Encoder--> memory
    x_future (B, H, Ff) --VSN--> (B, H, d) + horizon embedding --Decoder(memory)--> quantiles

1. Variable selection networks (VSN, from the Temporal Fusion Transformer) learn
   input-dependent weights per variable. The weights are interpretable.
2. Daily patching (as in PatchTST): the 168h history becomes 7 tokens, one per day.
   Attention then compares whole days ("is this Monday like last Monday?") and costs
   O((L/P)^2) instead of O(L^2).
3. Direct multi-horizon decoding: 24 query tokens, one per target hour, built from
   the known covariates of that hour. No causal mask and no autoregression, so
   errors do not compound across the horizon.
4. Quantile head trained with the pinball loss (shared with the LSTM).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from loadcast.models.neural import NeuralForecaster


class GatedResidualNetwork(nn.Module):
    """GRN(x) = LayerNorm(skip(x) + GLU(W2 ELU(W1 x)))  (Lim et al., 2021)."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.gate = nn.Linear(d_out, 2 * d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.fc2(F.elu(self.fc1(x))))
        return self.norm(self.skip(x) + F.glu(self.gate(h), dim=-1))


class VariableSelection(nn.Module):
    """Embed each scalar variable separately, then mix them with learned softmax weights."""

    def __init__(self, n_vars: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_vars, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_vars, d_model))
        self.var_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.select_grn = GatedResidualNetwork(n_vars * d_model, d_model, n_vars, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = x.unsqueeze(-1) * self.weight + self.bias  # (B, T, V, d)
        weights = torch.softmax(self.select_grn(emb.flatten(-2)), dim=-1)  # (B, T, V)
        out = (weights.unsqueeze(-1) * self.var_grn(emb)).sum(-2)  # (B, T, d)
        return out, weights


class LoadTransformer(nn.Module):
    def __init__(
        self,
        n_past: int,
        n_future: int,
        n_quantiles: int,
        lookback: int,
        horizon: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        patch_len: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if lookback % patch_len:
            raise ValueError(f"lookback ({lookback}) must be a multiple of patch_len")
        self.n_patches = lookback // patch_len

        self.past_vsn = VariableSelection(n_past, d_model, dropout)
        self.future_vsn = VariableSelection(n_future, d_model, dropout)
        self.patch = nn.Linear(patch_len * d_model, d_model)
        self.patch_pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.horizon_pos = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)

        def layer(cls: type[nn.Module]) -> nn.Module:
            return cls(
                d_model,
                n_heads,
                4 * d_model,
                dropout,
                batch_first=True,
                norm_first=True,
            )

        self.encoder = nn.TransformerEncoder(
            layer(nn.TransformerEncoderLayer), n_layers, enable_nested_tensor=False
        )
        self.decoder = nn.TransformerDecoder(layer(nn.TransformerDecoderLayer), n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_quantiles))
        self.variable_weights: dict[str, torch.Tensor] = {}

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        past, w_past = self.past_vsn(x_past)
        tokens = self.patch(past.reshape(past.shape[0], self.n_patches, -1))
        memory = self.encoder(tokens + self.patch_pos)

        future, w_future = self.future_vsn(x_future)
        out = self.decoder(future + self.horizon_pos, memory)

        self.variable_weights = {
            "past": w_past.detach().mean((0, 1)),
            "future": w_future.detach().mean((0, 1)),
        }
        return self.head(out)  # (B, horizon, n_quantiles)


class Transformer(NeuralForecaster):
    name = "transformer"

    def network(self, n_past: int, n_future: int, n_quantiles: int) -> nn.Module:
        return LoadTransformer(
            n_past,
            n_future,
            n_quantiles,
            lookback=self.cfg.lookback_hours,
            horizon=self.cfg.horizon_hours,
            **self.cfg.params["transformer"],
        )
