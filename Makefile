.PHONY: install data backtest report all test lint

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

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .
