# On macOS xgboost and torch each bundle their own libomp. Two OpenMP runtimes in one
# process abort if torch initialises first ("OMP: Error #179") and deadlock if torch
# runs multi-threaded afterwards. Import xgboost first and keep torch single-threaded.
import sys  # isort: skip

from loadcast.models.xgb import XGBoost  # isort: skip

import torch  # isort: skip

if sys.platform == "darwin":
    torch.set_num_threads(1)

from collections.abc import Iterable

from loadcast.models.base import Forecaster
from loadcast.models.baselines import TSO, SeasonalNaive, VanillaMLR
from loadcast.models.lstm import LSTM
from loadcast.models.transformer import Transformer

MODELS: dict[str, type[Forecaster]] = {
    m.name: m for m in (SeasonalNaive, TSO, VanillaMLR, XGBoost, LSTM, Transformer)
}

BASELINES = ("seasonal_naive", "tso")


def with_baselines(models: Iterable[str]) -> list[str]:
    """Every run and backtest also scores the floor and the TSO benchmark (SPEC §6),
    so cards stay comparable and the Diebold–Mariano test always has a reference."""
    return list(dict.fromkeys(BASELINES + tuple(models)))


__all__ = ["BASELINES", "MODELS", "Forecaster", "with_baselines"]
