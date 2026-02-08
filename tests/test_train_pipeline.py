import pandas as pd

from src.pipelines.train import train_per_product


def test_train_per_product_runs_end_to_end_and_returns_metrics():
    # Product A: 12 days -> OK for lag_7 + horizon=1
    dates_a = pd.date_range("2024-01-01", periods=12, freq="D")
    df_a = pd.DataFrame({
        "date": dates_a,
        "product_id": ["A"] * len(dates_a),
        "sales": list(range(10, 22)),  # 10..21
    })

    # Product B: 8 days -> insufficient history (min_points = 9)
    dates_b = pd.date_range("2024-01-01", periods=8, freq="D")
    df_b = pd.DataFrame({
        "date": dates_b,
        "product_id": ["B"] * len(dates_b),
        "sales": list(range(5, 13)),  # 5..12
    })

    df = pd.concat([df_a, df_b], ignore_index=True)

    results = train_per_product(
        df=df,
        lags=[1, 7],
        horizon=1,
        test_size=0.25,
        random_state=0,
    )

    # One row per product
    assert set(results["product_id"]) == {"A", "B"}

    # Product A: successfully trained
    row_a = results.loc[results["product_id"] == "A"].iloc[0]
    assert row_a["status"] == "ok"
    assert row_a["mae"] >= 0.0
    assert row_a["n_train"] > 0
    assert row_a["n_test"] > 0

    # Product B: skipped (history too short)
    row_b = results.loc[results["product_id"] == "B"].iloc[0]
    assert row_b["status"] == "skipped"
    assert pd.isna(row_b["mae"])
