# loadcast · live record

Written by the [daily forecast workflow](../../blob/main/.github/workflows/daily.yml); do not edit by hand.

| Path | Content |
|---|---|
| `forecasts/{country}/{day}.csv` | Forecasts for `day`, one row per hour × model (`q0.1`, `q0.5`, `q0.9`), with issue timestamp. Append-only. |
| `actuals/{country}/{day}.csv` | ENTSO-E actual load and TSO day-ahead forecast. The last 7 days are refreshed daily. |
| `runs/{day}.json` | Status per country, duration and commit of each daily run. |

Rendered at https://franrolotti.github.io/loadcast/.
