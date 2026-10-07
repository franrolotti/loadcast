# Encoder–decoder LSTM

Code: [`src/loadcast/models/lstm.py`](../../src/loadcast/models/lstm.py),
shared training in [`neural.py`](../../src/loadcast/models/neural.py),
inputs in [`windows.py`](../../src/loadcast/windows.py)

## Model

A Long Short-Term Memory cell (Hochreiter & Schmidhuber, 1997) keeps a memory
$c_t$ that is updated through gates:

$$
\begin{aligned}
i_t, f_t, o_t &= \sigma(W x_t + U h_{t-1} + b)\ \ \text{(input, forget, output gates)}\\
\tilde c_t &= \tanh(W_c x_t + U_c h_{t-1} + b_c)\\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde c_t, \qquad h_t = o_t \odot \tanh(c_t)
\end{aligned}
$$

The additive memory path lets gradients flow over long sequences, which is what made
recurrent networks practical for time series.

**Architecture (seq2seq, direct).**

1. The **encoder** LSTM reads the last 168 hours up to the issue time: the
   normalised load plus all covariates. Its final state $(h, c)$ summarises the
   recent week.
2. The **decoder** LSTM starts from that state and reads the known covariates
   (calendar and weather) of the 24 target hours.
3. A linear head maps each decoder output to 3 quantiles.

The decoder does **not** feed its own predictions back in (non-autoregressive). This
avoids exposure bias and error accumulation over the horizon. The gap of 15 hours
between the issue time and the target day is absorbed by the encoder state.

## Inputs and normalisation

- Covariates are standardised with training-period statistics.
- The load in each window is normalised by the mean and standard deviation of its
  own 168h history (instance normalisation). The network predicts the deviation
  from the recent level, and predictions are mapped back to MW. This makes the model
  robust to slow level changes, such as efficiency gains or the 2022 demand drop.

## Training

The pinball loss over 3 quantiles × 24 hours, AdamW, gradient clipping at 1.0, and
early stopping on the validation loss with the best weights restored. All
hyper-parameters are in `config.yaml`. The setup is identical for the Transformer.

## Assumptions and limitations

- The whole history is compressed into a fixed-size state: a bottleneck for long
  lookbacks, and the 168h history is read strictly sequentially.
- No explicit mechanism for "look at the same hour last week". The network must
  learn to carry that information in its state. This is what attention addresses.
- Needs more data and tuning than XGBoost. The results are seed-dependent, so
  report means over seeds when comparing small differences.
