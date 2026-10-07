# XGBoost

Code: [`src/loadcast/models/xgb.py`](../../src/loadcast/models/xgb.py)

## Model

An additive ensemble of $K$ regression trees fitted by gradient boosting:

$$\hat y_t = \sum_{k=1}^{K} f_k(\mathbf x_t), \qquad f_k \in \mathcal F_{\text{trees}}$$

Each tree $f_k$ is fitted to the gradient of the loss of the current ensemble, with
shrinkage (learning rate), row and column subsampling, and a depth limit as
regularisation (Chen & Guestrin, 2016).

**Probabilistic output.** We use XGBoost's multi-quantile objective
(`reg:quantileerror`). One model, with one output per quantile
$\tau \in \{0.1, 0.5, 0.9\}$, minimises the pinball loss

$$L_\tau(y, \hat y) = \max\big(\tau (y-\hat y),\ (\tau-1)(y-\hat y)\big).$$

Quantiles fitted independently can cross, so predictions are sorted per row.

## Inputs

One row per target hour: the calendar and weather of that hour plus load lags
of 48, 72, 168 and 336 hours and the mean of the 24 hours ending 48 hours before.
The minimum lag of 48h guarantees that every hour of day *D* is computable at the
issue time (SPEC §2). The *hour* feature lets a single model serve all 24 horizons.

## Training

Early stopping on the validation pinball loss (last 90 days before the test year).
Hyper-parameters are in `config.yaml` and are not tuned on test data.

## Assumptions and limitations

- **Rows are exchangeable.** Temporal structure enters only through lag and calendar
  features. The model cannot learn lag patterns that are not given as inputs.
- **No extrapolation.** Trees predict piecewise-constant values within the training
  range. A load level or a temperature never seen in training (a record heatwave,
  a structural drop as in 2022) is clamped to the edge of the training range.
  Lags mitigate this partly.
- Strong with little data, fast, and interpretable through feature importance and
  SHAP values.
