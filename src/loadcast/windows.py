"""Turn the hourly feature table into (history, future covariates, target) tensors.

For each forecast day D:

    issue time t0 = D-1 at issue_hour_utc
    encoder   = [t0 - lookback, t0)   -> observed load + covariates
    decoder   = [D, D + horizon)      -> known covariates only
    target    = load on the decoder hours

The load is normalised per window with the mean/std of its own encoder history
(instance normalisation, "RevIN"), so the networks learn shapes rather than levels
and are robust to level shifts such as 2020 (COVID) or 2022 (energy crisis).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from loadcast.features import KNOWN_FUTURE

PAST = ["load", *KNOWN_FUTURE]
FUTURE = KNOWN_FUTURE


@dataclass
class Scaler:
    """Standardises covariates with statistics from the training period only."""

    mean: pd.Series
    std: pd.Series

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> Scaler:
        return cls(frame.mean(), frame.std().replace(0, 1.0))

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return (frame - self.mean) / self.std


@dataclass
class Windows:
    x_past: np.ndarray  # (N, lookback, len(PAST))
    x_future: np.ndarray  # (N, horizon, len(FUTURE))
    y: np.ndarray  # (N, horizon), normalised; NaN when unknown
    loc: np.ndarray  # (N,)
    scale: np.ndarray  # (N,)
    days: pd.DatetimeIndex


def make_windows(
    features: pd.DataFrame,
    days: pd.DatetimeIndex,
    scaler: Scaler,
    lookback: int,
    horizon: int,
    issue_hour_utc: int,
    require_target: bool,
) -> Windows:
    covariates = scaler.transform(features[KNOWN_FUTURE]).to_numpy(np.float32)
    load = features["load"].to_numpy(np.float32)

    # Row positions of each issue time and each target-day start (-1 if absent).
    issue_pos = features.index.get_indexer(
        days - pd.Timedelta("1D") + pd.Timedelta(hours=issue_hour_utc)
    )
    day_pos = features.index.get_indexer(days)
    in_range = (issue_pos >= lookback) & (day_pos >= 0) & (day_pos + horizon <= len(features))
    issue_pos, day_pos, days = issue_pos[in_range], day_pos[in_range], days[in_range]

    enc = issue_pos[:, None] + np.arange(-lookback, 0)  # (N, lookback) row positions
    dec = day_pos[:, None] + np.arange(horizon)  # (N, horizon) row positions

    complete = ~np.isnan(load[enc]).any(1)
    complete &= ~np.isnan(covariates[enc]).any((1, 2))
    complete &= ~np.isnan(covariates[dec]).any((1, 2))
    if require_target:
        complete &= ~np.isnan(load[dec]).any(1)
    enc, dec, days = enc[complete], dec[complete], days[complete]
    enc_load, dec_load = load[enc], load[dec]

    loc = enc_load.mean(1)
    scale = enc_load.std(1) + 1e-6
    norm_hist = (enc_load - loc[:, None]) / scale[:, None]
    return Windows(
        x_past=np.concatenate([norm_hist[..., None], covariates[enc]], axis=-1),
        x_future=covariates[dec],
        y=(dec_load - loc[:, None]) / scale[:, None],
        loc=loc,
        scale=scale,
        days=days,
    )
