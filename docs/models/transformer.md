# LoadTransformer

Code: [`src/loadcast/models/transformer.py`](../../src/loadcast/models/transformer.py)

A Transformer written for day-ahead load forecasting from PyTorch building blocks
(`nn.TransformerEncoderLayer` / `nn.TransformerDecoderLayer`), without a forecasting
library. It combines ideas from the Temporal Fusion Transformer and PatchTST, and
keeps only what fits this problem.

```
x_past   (B, 168, Fp) ─ VSN ─► (B, 168, d) ─ reshape into 7 daily patches ─► Linear ─► (B, 7, d)
                                                                   + day position ─► Encoder ─► memory
x_future (B, 24, Ff)  ─ VSN ─► (B, 24, d) + horizon embedding ─► Decoder (cross-attends memory)
                                                                   ─► LayerNorm ─► Linear ─► (B, 24, 3 quantiles)
```

## 1. Variable selection network (VSN)

Each scalar input $x^{(j)}_t$ gets its own embedding $e^{(j)}_t = x^{(j)}_t w_j + b_j$.
A gated residual network (GRN) looks at all embeddings together and outputs softmax
weights $v_t \in \Delta^{V}$:

$$
\text{GRN}(a) = \text{LayerNorm}\big(a' + \text{GLU}(W_2\,\text{ELU}(W_1 a))\big), \qquad
\tilde x_t = \sum_{j} v^{(j)}_t\,\text{GRN}(e^{(j)}_t)
$$

- **Why:** load inputs mix very informative variables (hour, temperature) with weak
  ones (wind). Soft selection that depends on the input lets the network ignore
  noise, for example radiation at night.
- **Bonus:** averaging $v_t$ over the test set gives a *global variable
  importance*, stored in `model.variable_weights` after each forward pass.

## 2. Daily patching

The 168 hourly embeddings are grouped into 7 patches of 24 consecutive hours, and
each patch is projected to one token:

$$z_k = W_p\,[\tilde x_{24k}, \dots, \tilde x_{24k+23}] + p_k,\qquad k=0..6$$

- **Why:** a single hour is a poor semantic unit; one day is a natural one for load.
  With daily tokens, self-attention directly compares days ("last Monday" with
  "yesterday"). The attention matrix shrinks from 168² to 7² entries, which
  regularises the model and speeds it up.
- `patch_len` is a config value. `patch_len: 1` recovers hourly tokens for the
  ablation.

## 3. Direct multi-horizon decoder

The decoder receives **24 query tokens**, one per target hour. Each is the VSN
embedding of that hour's known covariates plus a learned horizon embedding $q_h$.
The queries attend to each other (self-attention) and to the 7 day-tokens of the
history (cross-attention). There is **no causal mask and no autoregression**: all
24 hours are predicted jointly, so errors do not compound and inference is a single
forward pass.

## 4. Output, normalisation and loss

- The quantile head outputs 3 values per hour, which are sorted at inference to
  prevent quantile crossing.
- The load is instance-normalised per window (RevIN-style, see
  [`windows.py`](../../src/loadcast/windows.py)). The network learns *shapes*, and
  the level is restored afterwards.
- Pre-LayerNorm blocks (`norm_first=True`) give stable training without
  learning-rate warm-up.
- The loss, optimiser and early stopping are shared with the LSTM, so the comparison
  isolates the architecture.

## Assumptions and limitations

- Attention has no built-in notion of order. Positions come from the learned patch
  and horizon embeddings, so ordering must be learned from data.
- With roughly 1,500 training days per country this is a *small-data* regime for a
  Transformer. That is why it is small (d_model = 64, 2 layers) and why patching and
  instance normalisation matter more than depth. Expect XGBoost to be competitive.
  Whether attention pays off is an empirical question this repository answers.
- Results vary with the seed. Report means and spreads over several seeds before
  drawing conclusions from small differences.

## Planned ablations

| Variant | Config change | Question |
|---|---|---|
| Hourly tokens | `patch_len: 1` | Does patching help, or only speed things up? |
| No VSN | replace VSN with `nn.Linear` | Is soft variable selection worth its parameters? |
| No instance norm | global scaling of load | How much of the robustness to 2022 comes from RevIN? |
| Two-week history | `lookback_hours: 336` | Does more history help once it is cheap (14 tokens)? |
