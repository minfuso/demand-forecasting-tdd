from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {"date", "product_id", "sales"}


@dataclass(frozen=True)
class SalesDataset:
    df: pd.DataFrame

    def __post_init__(self) -> None:
        missing = REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        # We copy to avoid mutating user-provided DF unexpectedly
        df = self.df.copy()

        # Coerce date to datetime
        df["date"] = pd.to_datetime(df["date"], errors="raise")

        # Put back the normalized df (dataclass is frozen => use object.__setattr__)
        object.__setattr__(self, "df", df)

    def products(self) -> list[str]:
        return sorted(self.df["product_id"].dropna().unique().tolist())
