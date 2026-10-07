# loadcast

**Day-ahead electricity load forecasting on public European data: XGBoost vs LSTM vs a
purpose-built Transformer, benchmarked against the official TSO forecast.**

[![CI](https://github.com/franrolotti/loadcast/actions/workflows/ci.yml/badge.svg)](https://github.com/franrolotti/loadcast/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

- **One API token.** Load comes from ENTSO-E. Weather comes from Open-Meteo, which needs no key.
- **One command.** `make all` downloads the data, runs the backtest and writes every table and figure.
- **Readable.** Each model is a single file of a few hundred lines in plain PyTorch, scikit-learn API or numpy, with no forecasting framework in between.
- **Honest.** The information set matches a real day-ahead forecast, a test checks for look-ahead leakage, and every model is tested for statistical significance against the TSO.

---

## The question

At 09:00 UTC on day *D-1*, before the day-ahead market closes, forecast the load for
each hour of day *D*, together with an 80% prediction interval.

| | Model | What it is |
|---|---|---|
| Baselines | Seasonal naive | Same hour last week |
| | **TSO** | The operator's published day-ahead forecast (Red Eléctrica, RTE, German TSOs) |
| | Vanilla MLR | Hong's GEFCom2012 regression: calendar × cubic temperature, no lags |
| ML | **XGBoost** | Multi-quantile gradient boosting on calendar, weather and load lags |
| | **LSTM** | Seq2seq encoder–decoder with a direct (non-autoregressive) decoder |
| | **Transformer** | Custom design: variable selection (TFT) + daily patches (PatchTST) + direct multi-horizon decoder + instance normalisation |

The full problem definition, assumptions and evaluation protocol are in
**[SPEC.md](SPEC.md)**. Each model has a page with equations, assumptions and
limitations in [docs/models/](docs/models/).

## Results

> Pending the first full run (ES, DE, FR; test years 2022–2024).
> `make report` writes [`results/RESULTS.md`](results/) with the metric tables
> (MAE, RMSE, MAPE, pinball loss, interval coverage, Diebold–Mariano vs TSO) and figures.

## Quick start

```bash
git clone https://github.com/franrolotti/loadcast && cd loadcast
cp .env.example .env          # paste your ENTSO-E key (see below)
make install                  # uv sync: creates .venv with pinned dependencies
make all                      # download → build → backtest → report
```

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.11–3.13. A GPU is optional.

**Getting the ENTSO-E key.** Register at
[transparency.entsoe.eu](https://transparency.entsoe.eu), then email
`transparency@entsoe.eu` with the subject *"Restful API access"*. Approval usually
takes a few days. Weather needs no key.

**Partial runs.**

```bash
uv run loadcast download                       # cached per country-year in data/raw/
uv run loadcast build
uv run loadcast backtest --country ES --model xgboost --model transformer
uv run loadcast report
```

Everything else (countries, dates, issue time, hyper-parameters) is in
[`config.yaml`](config.yaml). To add a country, add its ENTSO-E code, time zone and a
few cities with populations.

## Data and exogenous variables

| Source | Variables |
|---|---|
| ENTSO-E Transparency Platform (`entsoe-py`) | Actual total load, TSO day-ahead forecast (benchmark only) |
| Open-Meteo archive (ERA5) | Temperature, humidity, wind, solar radiation, population-weighted over the 7 largest metro areas |
| `holidays` | National public holidays |

Derived features: local hour, weekday and month; annual cycle; weekend, holiday and
**bridge days**; heating and cooling degree-hours; a 24h moving average of
temperature (building inertia); **solar radiation as a proxy for behind-the-meter
PV**; and load lags of at least 48h. The rationale for each, and which ones were
left out, is in [SPEC.md §4](SPEC.md#4-features-exogenous-variables).

**Main assumption.** Realised weather stands in for the weather forecast ("perfect
forecast"). This is standard practice, but it favours our models over the TSO. See
[SPEC.md §5](SPEC.md#5-assumptions) and the roadmap for removing it.

## Repository layout

```
config.yaml            every experiment setting
SPEC.md                problem definition, assumptions, evaluation protocol
src/loadcast/
  data/                ENTSO-E + Open-Meteo download and assembly
  features.py          calendar / weather / lag features, information-set guard
  windows.py           tensors for the sequence models
  models/              baselines, xgb, lstm, transformer, shared torch training
  backtest.py          expanding-window folds
  metrics.py           point, probabilistic and Diebold–Mariano metrics
  report.py            tables and figures
docs/models/           one page per model
tests/                 run on synthetic data, no key needed
```

## Development

```bash
make test     # pytest, including the look-ahead leakage test, on synthetic data
make lint     # ruff
```

On macOS, torch runs single-threaded because torch and xgboost each bundle their
own OpenMP runtime (see `models/__init__.py`).

## Acknowledgements

Inspired by [DemandCast](https://github.com/open-energy-transition/demandcast)
(Open Energy Transition), which addresses the complementary long-term problem.
This repository is an independent implementation.

## License

MIT
