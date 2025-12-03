"""
Docstring for tests.test_model
"""

import joblib
import pandas as pd


def test_model_prediction():
    """
    Docstring for test_model_prediction
    """
    model = joblib.load("models/battery_model.pkl")
    sample = pd.DataFrame([{
        "last_battery": 65, 
        "mean_battery": 35,
        "min_battery": 11,
        "max_battery": 80,
        "count_events": 232,
        "std_battery": 123
    }])
    pred = model.predict(sample)
    assert pred.shape == (1,)
