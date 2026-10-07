# loadcast — Specification

Status: v0.1 (skeleton implemented, results pending). This document defines *what* the
repository does and *why*. It is the reference for every design decision; the code
should not contradict it.

---

## 1. Goal and non-goals

**Goal.** Provide a minimal, readable and fully reproducible benchmark of day-ahead
electricity load forecasting models on public European data, answering three
questions:

1. How do gradient-boosted trees (XGBoost), a recurrent network (LSTM) and an
   attention-based network (Transformer) compare on the *same* information set?
2. Do any of them beat the official forecast published by the transmission system
   operator (TSO)? Is the difference statistically significant?
3. Which exogenous variables matter, and how much?

**Design principles.**

- *One token.* A user needs only an ENTSO-E API key. Weather comes from a
  key-less API.
- *One command.* `make all` reproduces every number and figure in `results/`.
- *Readable over clever.* Each model is a single file of a few hundred lines,
  with no forecasting framework in between. The networks are written in plain
  PyTorch so that every architectural choice is explicit.
- *Honest evaluation.* No look-ahead, no tuning on the test set, and
  significance tests instead of bare leaderboards.

**Non-goals.** Real-time operation or serving, sub-national forecasts, long-term
(multi-year) projections (see [DemandCast](https://github.com/open-energy-transition/demandcast)
for those), and squeezing out the last 0.1% of MAPE by heavy hyper-parameter search.

---

## 2. Forecasting problem

| Item | Definition |
|---|---|
| Target | Hourly *actual total load* (MW) reported to ENTSO-E (article 6.1.a), averaged to hourly resolution |
| Forecast day | UTC day *D*, hours 00:00–23:00 UTC (24 values) |
| Issue time | 09:00 UTC on *D-1*, i.e. before the day-ahead market gate closure (12:00 CET) |
| Lead times | 15h (first hour of *D*) to 38h (last hour of *D*) |
| Output | Quantiles 0.1, 0.5, 0.9 for the ML models; a point forecast for the baselines |
| Scope | One model per country (ES, DE, FR by default; any ENTSO-E area can be added in `config.yaml`) |

**Information set at the issue time.**

| Information | Available? | How it is used |
|---|---|---|
| Load up to 08:00–09:00 UTC on *D-1* | yes | Sequence models: 168h history window ending at the issue time. Tabular models: lags ≥ 48h only, which are known for all 24 target hours. |
| Calendar for *D* (hour, weekday, holidays, bridges) | yes | Known future covariate |
| Weather for *D* | as a forecast | **Approximated by the realised weather** (§5, assumption A1) |
| TSO forecast for *D* | yes (published on *D-1*) | Benchmark only, never an input |

The guard `features.check_information_set` refuses configurations where the longest
lead time exceeds the minimum lag. `tests/test_models.py::test_no_leakage`
multiplies every load value after the issue time by 10 and asserts that no model's
forecast changes.

**Why UTC days?** It removes daylight-saving ambiguities (23h/25h local days). The
calendar features are computed in local time, because demand follows local clocks.

---

## 3. Data sources

| Source | Variables | Access | Module |
|---|---|---|---|
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu) via [`entsoe-py`](https://github.com/EnergieID/entsoe-py) | Actual total load, TSO day-ahead total load forecast | API key (free) | `data/entsoe.py` |
| [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) (ERA5 / ERA5-Land reanalysis) | 2m temperature, relative humidity, 10m wind speed, shortwave radiation | No key | `data/weather.py` |
| [`holidays`](https://github.com/vacanza/holidays) | National public holidays | Python package | `features.py` |

**Processing.**

- ENTSO-E data come at 15/30/60-minute resolution in local time. They are converted
  to UTC and averaged to hourly resolution.
- Weather is a **population-weighted average over the 7 largest metropolitan areas**
  of each country (coordinates and weights in `config.yaml`). This is a transparent
  proxy for the temperature "felt" by electricity consumers, the role played by
  population-weighted gridded data in DemandCast.
- Each download is cached per (source, country, year) as parquet under `data/raw/`.
  Re-running is free and interrupted downloads resume.
- `data/processed/{country}.parquet` is a complete hourly UTC index. Gaps stay as
  NaN, and each model drops the rows or windows it cannot use. Missing shares are
  logged by `loadcast build`.

**Why entsoe-py directly.** It is the de-facto client, well maintained, and returns
pandas objects. A wrapper of about 40 lines with caching is all that is needed.

---

## 4. Features (exogenous variables)

| Group | Variable | Rationale |
|---|---|---|
| Calendar (local time) | hour, day of week, month | Daily and weekly activity cycles |
| | day-of-year sin/cos | Smooth annual cycle (daylight, seasonal activity) |
| | `is_weekend`, `is_holiday` | Non-working days have a different load profile |
| | `is_bridge` | Working day between two days off (often taken off). **New compared with DemandCast.** |
| Weather | temperature | Main driver of heating and cooling demand |
| | HDD = max(0, 15.5 − T), CDD = max(0, T − 22) | Make the U-shaped temperature response piecewise-linear for linear and tree models |
| | 24h exponential moving average of T | Thermal inertia of buildings |
| | relative humidity, wind speed | Apparent temperature (heat index, wind chill) |
| | shortwave radiation | **Behind-the-meter PV**: rooftop solar lowers measured load on sunny hours. Increasingly important in ES, DE and NL. **New.** |
| Load lags (tabular models) | t−48h, t−72h, t−168h, t−336h, mean of [t−71h, t−48h] | Persistence and weekly seasonality, respecting §2 |
| Load history (sequence models) | 168 hours ending at the issue time | Lets the network learn its own lag structure |

**Deliberately excluded.** GDP and per-capita annual consumption (used by
DemandCast) are close to constant within a country-year and add nothing at a
day-ahead horizon.

**Candidates for future experiments** (§8): electricity price (endogenous, so it
needs care), school holidays, daylight duration, a COVID stringency index for
2020–21, and installed PV capacity × radiation.

---

## 5. Assumptions

| ID | Assumption | Consequence |
|---|---|---|
| A1 | **Perfect weather forecast.** Realised (reanalysis) weather stands in for the weather forecast available on *D-1*. | Standard in the literature ("ex-post" evaluation), but it **flatters our models relative to the TSO**, which used a real forecast. Comparisons with the TSO are therefore an *upper bound* on our skill. Removing A1 with archived forecasts is on the roadmap (§8). |
| A2 | The TSO forecast is the one published on ENTSO-E for *D*, issued on *D-1*. | Its exact issue time varies by TSO, but it is always before *D*. |
| A3 | Seven cities with fixed population weights represent national weather. | Coarse. It can be replaced by gridded population weighting without changing the interface. |
| A4 | National holidays only. | Regional holidays (e.g. German Länder, Spanish autonomous communities) are ignored. They partly enter the error. |
| A5 | One model per country, retrained once per test year. | The models do not adapt within the test year, which is conservative. |

---

## 6. Models

All models implement `Forecaster.fit(features, train_days, valid_days)` and
`Forecaster.predict(features, days)` (`models/base.py`). Full write-ups with
equations are in `docs/models/`.

| Model | Type | Inputs | Output | Why it is here |
|---|---|---|---|---|
| `seasonal_naive` | Baseline | load at t−168h | point | Floor: any model must beat it |
| `tso` | Baseline | published TSO forecast | point | The professional benchmark |
| `vanilla_mlr` | Econometric | trend, month, weekday×hour, cubic temperature × month/hour | point | Hong's GEFCom2012 benchmark. Shows how far domain-driven linear modelling goes without lags |
| `xgboost` | Gradient-boosted trees | calendar + weather + lags | quantiles | Strong tabular baseline |
| `lstm` | Seq2seq LSTM | 168h history + known future covariates | quantiles | Classic deep-learning approach for load |
| `transformer` | Custom Transformer | same as LSTM | quantiles | Attention model designed for this problem (§6.1) |

**Fairness.** The LSTM and the Transformer share windows, normalisation, loss,
optimiser, early stopping and seed (`models/neural.py`). Only the network differs.

### 6.1 Transformer design (`models/transformer.py`)

The Transformer is built from PyTorch primitives rather than taken from a
forecasting library. Each component targets a property of load data:

| Component | Origin | Motivation for load |
|---|---|---|
| Variable selection networks with gated residual units | TFT (Lim et al., 2021) | Per-input, per-time-step importance weights. They give interpretability (which variables drive the forecast) and suppress noisy inputs. |
| Daily patching (patch = 24h) | PatchTST (Nie et al., 2023) | A token is one day, so attention compares days rather than hours. The 168h history becomes 7 tokens, which cuts the attention cost 576× and reduces overfitting. |
| Direct multi-horizon decoder | — | 24 queries, one per target hour, built from that hour's known covariates plus a learned horizon embedding. Errors do not compound, unlike autoregressive decoding. |
| Instance normalisation of load (RevIN-style) | Kim et al. (2022) | The network forecasts the *shape* relative to the last week. It is robust to level shifts (COVID 2020, energy crisis 2022). |
| Quantile head + pinball loss | — | Probabilistic forecasts, evaluated by calibration |
| Pre-LayerNorm blocks | Xiong et al. (2020) | More stable training without warm-up |

**Ablations** (planned, each a one-line config change):
`patch_len: 1` (hourly tokens) vs `24`; VSN vs a plain linear embedding; with and
without instance normalisation; lookback of 168h vs 336h.

---

## 7. Evaluation protocol

- **Expanding-window backtest.** For each test year *Y* ∈ {2022, 2023, 2024}: train
  on all data before *Y*, keep the last 90 days of that period for early stopping,
  and forecast every day of *Y*. Hyper-parameters are fixed in `config.yaml` and are
  never chosen using test data.
- **Point metrics** (on the median): MAE, RMSE, MAPE.
- **Probabilistic metrics**: mean pinball loss over the quantiles, and the empirical
  coverage of the 10–90% interval (nominal 80%).
- **Significance**: a Diebold–Mariano test of every model against the TSO forecast.
  Hourly errors within a day are strongly correlated, so the test uses the daily
  mean absolute-error differential with the Harvey–Leybourne–Newbold correction
  (Ziel & Weron, 2018).
- **Breakdowns**: error by hour of day, by test year (2022 = energy crisis), and on
  holidays vs regular days.
- **Outputs**: `results/predictions/{country}.parquet` (all forecasts),
  `results/metrics_{country}.csv`, `results/RESULTS.md` and `results/figures/`.

---

## 8. Roadmap

| Phase | Content | Status |
|---|---|---|
| 0 | Repository skeleton, data pipeline, 6 models, backtest, report, CI, tests (incl. leakage) | done |
| 1 | First full run on ES/DE/FR 2018–2024. Publish `results/RESULTS.md` and summarise it in the README | next |
| 2 | Transformer ablations (§6.1) and a feature-group ablation (calendar / weather / lags / PV) | |
| 3 | Remove assumption A1 with archived weather forecasts (Open-Meteo Previous Runs / Historical Forecast API), giving a like-for-like comparison with the TSO | |
| 4 | Optional: zero-shot foundation model (Chronos / TimesFM) as a 7th competitor; conformal calibration of the intervals | |

---

## 9. Repository layout

```
config.yaml              single source of truth for an experiment
src/loadcast/
  data/                  entsoe.py, weather.py, dataset.py (download + build)
  features.py            calendar, weather and lag features, information-set guard
  windows.py             tensors for the sequence models, instance normalisation
  models/                base, baselines, xgb, neural (shared training), lstm, transformer
  backtest.py            expanding-window folds
  metrics.py             MAE/RMSE/MAPE, pinball, coverage, Diebold–Mariano
  report.py              tables and figures
  cli.py                 loadcast {download,build,backtest,report}
docs/models/             one page per model: equations, assumptions, limitations
tests/                   synthetic data; features, metrics, models, leakage
```

## 10. References

- Hong, T. (2010). *Short term electric load forecasting*. PhD thesis, NCSU. Vanilla benchmark.
- Hong, T., Pinson, P., Fan, S. (2014). Global Energy Forecasting Competition 2012. *IJF* 30(2).
- Lim, B., Arık, S., Loeff, N., Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *IJF* 37(4).
- Nie, Y., Nguyen, N., Sinthong, P., Kalagnanam, J. (2023). A time series is worth 64 words (PatchTST). *ICLR*.
- Kim, T. et al. (2022). Reversible instance normalization for accurate time-series forecasting against distribution shift. *ICLR*.
- Xiong, R. et al. (2020). On layer normalization in the Transformer architecture. *ICML*.
- Ziel, F., Weron, R. (2018). Day-ahead electricity price forecasting with high-dimensional structures. *Energy Economics* 70.
- Diebold, F., Mariano, R. (1995). Comparing predictive accuracy. *JBES* 13(3).
- Chen, T., Guestrin, C. (2016). XGBoost: a scalable tree boosting system. *KDD*.
- Hochreiter, S., Schmidhuber, J. (1997). Long short-term memory. *Neural Computation* 9(8).
