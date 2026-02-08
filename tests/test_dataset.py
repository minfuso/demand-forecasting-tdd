import pandas as pd
import pytest

from src.data.dataset import SalesDataset


def test_dataset_exposes_products():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"]),
        "product_id": ["A", "A", "B"],
        "sales": [10, 12, 5],
    })

    ds = SalesDataset(df)

    assert set(ds.products()) == {"A", "B"}


def test_dataset_rejects_missing_columns():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]),
        "product_id": ["A"],
        # missing "sales"
    })

    with pytest.raises(ValueError):
        SalesDataset(df)


def test_dataset_coerces_date_to_datetime():
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],  # strings on purpose
        "product_id": ["A", "A"],
        "sales": [1, 2],
    })

    ds = SalesDataset(df)

    assert pd.api.types.is_datetime64_any_dtype(ds.df["date"])
