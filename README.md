# Demand forecasting – minimal TDD project

This project is a minimal and clean implementation of a demand forecasting
pipeline designed for retail / supply-chain use cases.

The goal is not to build a complex model, but to demonstrate:

- clean domain-driven structure
- test-driven development
- explicit data pipeline
- reproducibility (conda + docker)
- separation between domain, data, features and models

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
