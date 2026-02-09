import pandas as pd
import pytest

from src.features.build_features import build_features


def test_build_features_creates_lags_and_target_without_leakage():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                    "2024-01-07",
                    "2024-01-08",
                    "2024-01-09",
                ]
            ),
            "product_id": ["A"] * 9,
            "sales": [10, 11, 12, 13, 14, 15, 16, 17, 18],
        }
    )

    X, y = build_features(df, lags=[1, 7], horizon=1)

    # Expected columns
    assert set(["lag_1", "lag_7"]).issubset(set(X.columns))

    # X/y alignement
    assert len(X) == len(y)
    assert X.index.equals(y.index)

    # Leakage check: sales should not be in X
    assert "sales" not in X.columns

    # One row verification for date 2024-01-08:
    row = X.loc[X["date"] == pd.Timestamp("2024-01-08")].iloc[0]
    assert row["lag_1"] == 16  # sales for 2024-01-07
    assert row["lag_7"] == 10  # sales for 2024-01-01

    # Target check: for date 2024-01-08, target should be sales for 2024-01-09
    # which is 18
    y_0801 = y.loc[X["date"] == pd.Timestamp("2024-01-08")].iloc[0]
    assert y_0801 == 18


def test_build_features_raises_if_history_too_short_for_lags_and_horizon():
    # Here we have 8 points, but with lags=[1,7] and horizon=1, we need at least 9 points to have one valid row (the last one).
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                    "2024-01-07",
                    "2024-01-08",
                ]
            ),  # 8 points -> insuffisant
            "product_id": ["A"] * 8,
            "sales": [10, 11, 12, 13, 14, 15, 16, 17],
        }
    )

    with pytest.raises(ValueError) as err:
        build_features(df, lags=[1, 7], horizon=1)

    assert "history too short" in str(err.value).lower()
    assert "min_points=9" in str(err.value)
