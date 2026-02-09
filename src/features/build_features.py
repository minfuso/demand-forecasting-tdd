from __future__ import annotations

from typing import Iterable

import pandas as pd


def build_features(
    df: pd.DataFrame,
    lags: Iterable[int] = (1, 7),
    horizon: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build lag features and a future target (horizon steps ahead) for a single product history.

    Assumptions:
    - df contains: date, product_id, sales
    - df is a single product history OR already filtered; if not, we still group by product_id.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    required = {"date", "product_id", "sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    dff = df.copy()
    dff["date"] = pd.to_datetime(dff["date"], errors="raise")
    dff = dff.sort_values(["product_id", "date"]).reset_index(drop=True)

    min_points = max(lags) + horizon + 1
    counts = dff.groupby("product_id").size()
    too_short = counts[counts < min_points]

    # Minimum history constraint check
    if not too_short.empty:
        offenders = ", ".join([f"{pid}({n})" for pid, n in too_short.items()])
        raise ValueError(
            "History too short for lag/horizon settings: "
            f"min_points={min_points}. Offenders: {offenders}"
        )

    # Target: future sales
    dff["target"] = dff.groupby("product_id")["sales"].shift(-horizon)

    # Lag features
    lags = list(lags)
    for lag in lags:
        if lag < 1:
            raise ValueError("lags must be >= 1")
        dff[f"lag_{lag}"] = dff.groupby("product_id")["sales"].shift(lag)

    feature_cols = ["date", "product_id"] + [f"lag_{lag}" for lag in lags]

    # Drop rows where target or any feature is NaN (start of series + end due to horizon)
    out = dff.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)

    X = out[feature_cols].copy()
    y = out["target"].astype(
        float
    )  # float is fine for regression; later we can keep int if desired

    return X, y
