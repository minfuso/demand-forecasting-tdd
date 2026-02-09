![CI](https://github.com/minfuso/demand-forecasting-tdd/actions/workflows/tests.yml/badge.svg)

# Demand forecasting – minimal TDD project

This project is a minimal and clean implementation of a demand forecasting
pipeline designed for retail / supply-chain use cases.

The goal is not to build a complex model, but to demonstrate:

- clean domain-driven structure
- test-driven development
- explicit data pipeline
- reproducibility (conda)
- separation between domain, data, features and models

## Code quality

This project enforces basic code quality standards:

- Ruff for linting
- Black for code formatting
- Pytest for automated tests

All checks are executed automatically in GitHub Actions.

## Scope

- historical daily sales
- per-product forecasting
- simple machine learning model
- explicit feature engineering (lags)
- extensible API for future uncertainty estimation

## Project structure

src/
- domain      : business objects
- data        : dataset access layer
- features    : feature engineering
- models      : ML models (wrapped)
- pipelines   : training / evaluation pipelines

tests/
- unit tests for each layer

## Philosophy

This project focuses on:
- maintainability
- testability
- clarity of responsibilities

Not on:
- deep learning
- automated feature generation
- notebook-driven workflows

## Run

### Environment

This project uses a Conda environment.
It is recommended to use Miniforge (as in the CI workflow).

```bash
conda env create -f environment.yml
conda activate demand_forecast
```

### Run tests

```bash
pytest -q
```

### Train from a CSV file (CLI)

A minimal comand-line interface is provided

```bash
python -m src.pipelines.train data/sample_sales.csv \
  --lags 1 7 \
  --horizon 1 \
  --output training_results.csv
```

The input CSV must contain the following columns:

- `date`
- `product_id`
- `sales`

#### Output

The command produces a CSV file with one row per product, containing:

- training status (`ok`, `skipped`, or `error`)
- MAE on the test split
- number of training samples
- number of test samples

Products with insufficient history for the selected lags and horizon are automatically skipped.