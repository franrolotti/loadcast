"""Shared training and inference for the PyTorch models (LSTM and Transformer).

Both networks see exactly the same windows, loss, optimiser and early stopping, so
differences in results come from the architecture alone.
"""

from __future__ import annotations

import copy
import logging
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from loadcast.features import KNOWN_FUTURE, target_hours
from loadcast.models.base import Forecaster, qcol
from loadcast.windows import FUTURE, PAST, Scaler, Windows, make_windows

log = logging.getLogger(__name__)


def pinball_loss(pred: torch.Tensor, y: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """Quantile (pinball) loss averaged over batch, horizon and quantiles.

    pred: (B, H, Q), y: (B, H), quantiles: (Q,)
    """
    diff = y.unsqueeze(-1) - pred
    return torch.maximum(quantiles * diff, (quantiles - 1) * diff).mean()


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tensors(w: Windows) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(w.x_past), torch.from_numpy(w.x_future), torch.from_numpy(w.y)
    )


def train_network(
    net: nn.Module,
    train: Windows,
    valid: Windows,
    quantiles: list[float],
    batch_size: int,
    max_epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    seed: int,
) -> tuple[nn.Module, dict]:
    """AdamW + early stopping on validation pinball loss.

    Returns the best weights and the learning curve (mean train and validation pinball
    loss per epoch, in normalised units, and the epoch whose weights were kept).
    """
    torch.manual_seed(seed)
    device = _device()
    net.to(device)
    q = torch.tensor(quantiles, device=device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader = DataLoader(_tensors(train), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(_tensors(valid), batch_size=4 * batch_size)

    best_loss, best_state, bad_epochs = float("inf"), copy.deepcopy(net.state_dict()), 0
    curve = {"metric": "pinball (normalised)", "train": [], "valid": [], "best": 0}
    for epoch in range(max_epochs):
        net.train()
        train_losses = []
        for x_past, x_future, y in train_loader:
            optimiser.zero_grad()
            loss = pinball_loss(net(x_past.to(device), x_future.to(device)), y.to(device), q)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()
            train_losses.append(loss.item())

        net.eval()
        with torch.no_grad():
            valid_loss = np.mean(
                [
                    pinball_loss(net(xp.to(device), xf.to(device)), y.to(device), q).item()
                    for xp, xf, y in valid_loader
                ]
            )
        log.info("epoch %3d  valid pinball %.4f", epoch, valid_loss)
        curve["train"].append(float(np.mean(train_losses)))
        curve["valid"].append(float(valid_loss))
        if valid_loss < best_loss - 1e-5:
            best_loss, best_state, bad_epochs = (
                valid_loss,
                copy.deepcopy(net.state_dict()),
                0,
            )
            curve["best"] = epoch
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    net.load_state_dict(best_state)
    return net.cpu(), curve


class NeuralForecaster(Forecaster):
    """Builds windows, trains `network()` and maps outputs back to MW."""

    @abstractmethod
    def network(self, n_past: int, n_future: int, n_quantiles: int) -> nn.Module: ...

    def _windows(self, features, days, require_target: bool) -> Windows:
        return make_windows(
            features,
            days,
            self.scaler,
            lookback=self.cfg.lookback_hours,
            horizon=self.cfg.horizon_hours,
            issue_hour_utc=self.cfg.issue_hour_utc,
            require_target=require_target,
        )

    def fit(self, features, train_days, valid_days) -> None:
        train_hours = target_hours(train_days, self.cfg.horizon_hours)
        self.scaler = Scaler.fit(features.reindex(train_hours)[KNOWN_FUTURE])
        train = self._windows(features, train_days, require_target=True)
        valid = self._windows(features, valid_days, require_target=True)
        log.info(
            "%s: %d train / %d valid windows",
            self.name,
            len(train.days),
            len(valid.days),
        )

        opt = self.cfg.params["torch"]
        torch.manual_seed(opt["seed"])
        net = self.network(len(PAST), len(FUTURE), len(self.cfg.quantiles))
        self.net, self.curve = train_network(net, train, valid, self.cfg.quantiles, **opt)

    def predict(self, features: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
        w = self._windows(features, days, require_target=False)
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.from_numpy(w.x_past), torch.from_numpy(w.x_future)).numpy()
        out = np.sort(out, axis=-1) * w.scale[:, None, None] + w.loc[:, None, None]
        return pd.DataFrame(
            out.reshape(-1, out.shape[-1]),
            index=target_hours(w.days, self.cfg.horizon_hours),
            columns=[qcol(q) for q in self.cfg.quantiles],
        )
