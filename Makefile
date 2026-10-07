.PHONY: install data backtest report all train publish forecast dashboard test lint

# Optional filters, e.g. make train COUNTRIES="ES FR" MODELS="xgboost transformer"
COUNTRY_ARGS = $(foreach c,$(COUNTRIES),--country $(c))
MODEL_ARGS = $(foreach m,$(MODELS),--model $(m))

install:  ## Install dependencies into .venv
	uv sync --group dev

data:  ## Download ENTSO-E load + Open-Meteo weather and build the processed dataset
	uv run loadcast download $(COUNTRY_ARGS) $(MODEL_ARGS)
	uv run loadcast build $(COUNTRY_ARGS)

backtest:  ## Expanding-window backtest, folds in parallel; RUN=id stores it in that run's cards
	uv run loadcast backtest $(if $(RUN),--run $(RUN)) $(COUNTRY_ARGS) $(MODEL_ARGS)

report:  ## Compute metrics, significance tests and figures into results/
	uv run loadcast report

all: data backtest report

train:  ## A new training run on all data up to yesterday (models/{run}/); see README
	uv run loadcast download --end yesterday $(COUNTRY_ARGS)
	uv run loadcast build --end yesterday $(COUNTRY_ARGS)
	uv run loadcast train --end yesterday $(if $(RUN),--run $(RUN)) $(COUNTRY_ARGS) $(MODEL_ARGS)

publish:  ## Upload a training run as a GitHub release (again after a backtest): make publish RUN=id
	uv run loadcast publish --run $(RUN)

forecast:  ## Forecast tomorrow with the runs in live.yaml (history/)
	uv run loadcast fetch
	uv run loadcast forecast

dashboard:  ## Build the static dashboard into site/
	uv run loadcast dashboard

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .
