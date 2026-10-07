.PHONY: install data backtest report all train publish forecast dashboard test lint

install:  ## Install dependencies into .venv
	uv sync --group dev

data:  ## Download ENTSO-E load + Open-Meteo weather and build the processed dataset
	uv run loadcast download
	uv run loadcast build

backtest:  ## Expanding-window backtest, folds in parallel; RUN=id stores it in that run's cards
	uv run loadcast backtest $(if $(RUN),--run $(RUN))

report:  ## Compute metrics, significance tests and figures into results/
	uv run loadcast report

all: data backtest report

train:  ## A new training run on all data up to yesterday (models/{run}/); see README
	uv run loadcast download --end yesterday
	uv run loadcast build --end yesterday
	uv run loadcast train --end yesterday

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
