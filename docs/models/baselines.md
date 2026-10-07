# Baselines

Code: [`src/loadcast/models/baselines.py`](../../src/loadcast/models/baselines.py)

A forecasting result only means something next to a reference. We use three, from
trivial to professional.

## Seasonal naive

$$\hat y_t = y_{t-168}$$

The load at the same hour one week earlier. It captures daily and weekly seasonality
with no parameters at all. It fails on holidays, weather changes and the days after
a holiday. **Any model that does not clearly beat it is not useful.**

## TSO forecast

The day-ahead total load forecast that each transmission system operator publishes on
ENTSO-E (Red Eléctrica, the four German TSOs, RTE). TSOs use proprietary models,
real weather forecasts and operator expertise. **This is the bar the repository is
really about.**

Caveat: the TSO uses *forecast* weather, while our models use *realised* weather
(assumption A1 in [SPEC §5](../../SPEC.md#5-assumptions)). Beating the TSO under A1
is necessary but not sufficient evidence of a better operational model.

## Vanilla MLR (Hong, 2010)

The benchmark of the Global Energy Forecasting Competition 2012. It is a linear
regression that encodes decades of utility practice:

$$
y_t = \beta_0\,\text{Trend}_t + \alpha_{M_t} + \gamma_{W_t H_t}
    + f(T_t) + f(T_t)\,\delta_{M_t} + f(T_t)\,\eta_{H_t} + \varepsilon_t,
\qquad f(T) = \theta_1 T + \theta_2 T^2 + \theta_3 T^3
$$

- $M_t$ is the month, $W_t$ the weekday, $H_t$ the hour (local time). The 168
  weekday×hour dummies $\gamma$ give every hour of the week its own intercept.
- The cubic $f(T)$ allows the U-shaped (heating and cooling) response. Interacting it
  with month and hour lets the slope vary over the year (e.g. lighting vs heating)
  and over the day (people at home vs at work).
- Estimated by OLS (`numpy.linalg.lstsq`) on training plus validation data, with
  about 470 coefficients.

**Assumptions.** Linearity in the listed terms, homoskedastic errors, and no
autocorrelation modelling (no lags). The linear trend extrapolates the past growth
rate.

**Why include it.** It shows how much accuracy comes from domain knowledge alone,
with no load lags, so that the gains of the ML models can be attributed to what
they add: lags, non-linearities and interactions learned from data.
