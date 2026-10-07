import numpy as np
import pandas as pd
import pytest

from loadcast import metrics
from loadcast.data.weather import population_weighted


def test_pinball_of_median_is_half_mae():
    y, p = np.array([1.0, 2.0, 3.0]), np.array([[2.0], [2.0], [2.0]])
    assert metrics.pinball(y, p, [0.5]) == metrics.mae(y, p[:, 0]) / 2


def test_coverage():
    y = np.array([1.0, 5.0, 10.0])
    assert metrics.coverage(y, np.zeros(3), np.full(3, 6.0)) == pytest.approx(200 / 3)


def test_diebold_mariano_detects_better_model():
    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01", periods=24 * 200, freq="1h", tz="UTC")
    good = pd.Series(np.abs(rng.normal(0, 1, len(index))), index=index)
    bad = pd.Series(np.abs(rng.normal(0, 2, len(index))), index=index)
    stat, p = metrics.diebold_mariano(good, bad)
    assert stat < 0 and p < 0.01


def test_population_weighting_ignores_missing_cities():
    index = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
    a = pd.DataFrame({"t": [10.0, np.nan]}, index=index)
    b = pd.DataFrame({"t": [20.0, 20.0]}, index=index)
    out = population_weighted([a, b], [3.0, 1.0])
    assert out["t"].tolist() == [12.5, 20.0]
