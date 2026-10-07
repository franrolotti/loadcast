"""Point, probabilistic and significance metrics. See SPEC.md §7."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from loadcast.models.base import qcol


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mape(y: np.ndarray, p: np.ndarray) -> float:
    return float(100 * np.mean(np.abs((y - p) / y)))


def pinball(y: np.ndarray, pred: np.ndarray, quantiles: list[float]) -> float:
    """Mean pinball loss; pred has one column per quantile."""
    diff = y[:, None] - pred
    q = np.asarray(quantiles)[None, :]
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Share of observations inside [lower, upper], in %. Nominal 80% for q0.1-q0.9."""
    return float(100 * np.mean((y >= lower) & (y <= upper)))


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series) -> tuple[float, float]:
    """Diebold-Mariano test of equal accuracy on *daily* mean losses.

    Hourly errors within a day are strongly correlated, so following Ziel & Weron
    (2018) we average to one loss differential per day and run the test with the
    Harvey-Leybourne-Newbold small-sample correction (one-step horizon).
    Negative statistic -> model A is more accurate.
    """
    d = (loss_a - loss_b).groupby(loss_a.index.floor("D")).mean().dropna().to_numpy()
    n = len(d)
    statistic = d.mean() / np.sqrt(d.var(ddof=0) / n)
    statistic *= np.sqrt((n - 1) / n)  # HLN correction with h = 1
    p_value = 2 * stats.t.sf(np.abs(statistic), df=n - 1)
    return float(statistic), float(p_value)


def summarise(pred: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    """One row per model: point metrics on the median, interval metrics when available."""
    rows, median = [], qcol(0.5)
    reference = pred[pred["model"] == "tso"]
    for model, g in pred.groupby("model", sort=False):
        g = g.dropna(subset=["actual", median])
        y, p = g["actual"].to_numpy(), g[median].to_numpy()
        row = {
            "model": model,
            "MAE (MW)": mae(y, p),
            "RMSE (MW)": rmse(y, p),
            "MAPE (%)": mape(y, p),
        }
        qcols = [qcol(q) for q in quantiles]
        if all(c in g for c in qcols) and g[qcols].notna().all().all():
            row["Pinball (MW)"] = pinball(y, g[qcols].to_numpy(), quantiles)
            row["Coverage 80% (%)"] = coverage(y, g[qcols[0]].to_numpy(), g[qcols[-1]].to_numpy())
        if model != "tso" and len(reference):
            ref = reference.dropna(subset=["actual", median])
            common = g.index.intersection(ref.index)
            loss_a = (g.loc[common, "actual"] - g.loc[common, median]).abs()
            loss_b = (ref.loc[common, "actual"] - ref.loc[common, median]).abs()
            row["DM vs TSO"], row["p-value"] = diebold_mariano(loss_a, loss_b)
        rows.append(row)
    order = [
        "MAE (MW)",
        "RMSE (MW)",
        "MAPE (%)",
        "Pinball (MW)",
        "Coverage 80% (%)",
        "DM vs TSO",
        "p-value",
    ]
    table = pd.DataFrame(rows).set_index("model")
    return table[[c for c in order if c in table]]
