import numpy as np
import pandas as pd
import pytest

from src.models.demand_model import DemandModel


def test_demand_model_can_fit_and_predict():
    X = pd.DataFrame(
        {
            "lag_1": [10, 11, 12, 13],
            "lag_7": [3, 4, 5, 6],
        }
    )

    y = pd.Series([11, 12, 13, 14])

    model = DemandModel(random_state=0)

    model.fit(X, y)
    preds = model.predict(X)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X),)


def test_demand_model_raises_if_predict_before_fit():
    model = DemandModel()

    X = pd.DataFrame(
        {
            "lag_1": [1],
            "lag_7": [1],
        }
    )

    with pytest.raises(RuntimeError):
        model.predict(X)
