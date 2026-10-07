.PHONY: install data backtest report all train forecast dashboard test lint

install:  ## Install dependencies into .venv
	uv sync --group dev

data:  ## Download ENTSO-E load + Open-Meteo weather and build the processed dataset
	uv run loadcast download
	uv run loadcast build

backtest:  ## Train and evaluate every model on every test year
	uv run loadcast backtest

report:  ## Compute metrics, significance tests and figures into results/
	uv run loadcast report

all: data backtest report

train:  ## A new training run on all data up to yesterday (models/{run}/); see README
	uv run loadcast download --end yesterday
	uv run loadcast build --end yesterday
	uv run loadcast train --end yesterday

forecast:  ## Forecast tomorrow with the runs in live.yaml (history/)
	uv run loadcast fetch
	uv run loadcast forecast

dashboard:  ## Build the static dashboard into site/
	uv run loadcast dashboard

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .
