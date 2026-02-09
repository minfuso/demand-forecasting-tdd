from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.data.dataset import SalesDataset
from src.features.build_features import build_features
from src.models.demand_model import DemandModel


@dataclass(frozen=True)
class ProductTrainResult:
    product_id: str
    status: str  # "ok" | "skipped" | "error"
    mae: float | None
    n_train: int
    n_test: int
    message: str | None = None


def _time_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1")

    n = len(X)
    if n < 2:
        raise ValueError("Not enough rows to split")

    n_test = max(1, int(np.ceil(n * test_size)))
    n_train = n - n_test

    if n_train < 1:
        raise ValueError("Not enough training rows after split")

    # We explicitly drop non-feature columns here
    X_train = X.iloc[:n_train].drop(columns=["date", "product_id"])
    y_train = y.iloc[:n_train]

    X_test = X.iloc[n_train:].drop(columns=["date", "product_id"])
    y_test = y.iloc[n_train:]

    return X_train, X_test, y_train, y_test


def train_per_product(
    df: pd.DataFrame,
    lags: list[int] = [1, 7],
    horizon: int = 1,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> pd.DataFrame:
    ds = SalesDataset(df)

    results: list[ProductTrainResult] = []

    for pid in ds.products():
        try:
            # Extract clean history for one product
            hist = ds.for_product(pid)

            # Feature engineering
            X, y = build_features(hist, lags=lags, horizon=horizon)

            # Temporal split (no shuffling)
            X_train, X_test, y_train, y_test = _time_split(X, y, test_size=test_size)

            # Train model
            model = DemandModel(random_state=random_state)
            model.fit(X_train, y_train)

            # Evaluate
            preds = model.predict(X_test)
            mae = float(mean_absolute_error(y_test, preds))

            results.append(
                ProductTrainResult(
                    product_id=pid,
                    status="ok",
                    mae=mae,
                    n_train=len(X_train),
                    n_test=len(X_test),
                    message=None,
                )
            )

        except ValueError as e:
            # Typical case: history too short, impossible split, etc.
            results.append(
                ProductTrainResult(
                    product_id=pid,
                    status="skipped",
                    mae=None,
                    n_train=0,
                    n_test=0,
                    message=str(e),
                )
            )

        except Exception as e:
            results.append(
                ProductTrainResult(
                    product_id=pid,
                    status="error",
                    mae=None,
                    n_train=0,
                    n_test=0,
                    message=str(e),
                )
            )

    out = pd.DataFrame([r.__dict__ for r in results])

    # Make mae a proper float column (NaN instead of None)
    out["mae"] = out["mae"].astype(float)

    return out


def train_from_csv(
    path: str,
    lags: list[int] = [1, 7],
    horizon: int = 1,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> pd.DataFrame:
    # Load raw data from CSV
    df = pd.read_csv(path)

    # Reuse the main per-product training pipeline
    return train_per_product(
        df=df,
        lags=lags,
        horizon=horizon,
        test_size=test_size,
        random_state=random_state,
    )


if __name__ == "__main__":
    import argparse

    # Minimal command line interface
    parser = argparse.ArgumentParser(description="Train a demand model per product.")

    parser.add_argument("csv_path", type=str, help="Path to input CSV file")

    parser.add_argument("--lags", type=int, nargs="+", default=[1, 7])
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="training_results.csv",
        help="Path to output CSV file",
    )

    args = parser.parse_args()

    results = train_from_csv(
        path=args.csv_path,
        lags=args.lags,
        horizon=args.horizon,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Save results to disk
    results.to_csv(args.output, index=False)

    print(f"Results written to {args.output}")
