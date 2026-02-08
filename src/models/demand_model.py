from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class DemandModel:
    def __init__(self, random_state: int | None = None):
        self._model = RandomForestRegressor(
            n_estimators=50,
            random_state=random_state,
        )
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, y)
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        return self._model.predict(X)
