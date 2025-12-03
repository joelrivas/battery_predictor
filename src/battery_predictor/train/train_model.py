"""
Docstring for train.train_model
"""

import os
import argparse
import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def train(inp, model_out):
    """
    Docstring for train

    :param inp: Description
    :param model_out: Description
    """

    seed = 23
    df = pd.read_parquet(inp)
    X = df[
        [
            "last_battery",
            "mean_battery",
            "min_battery",
            "max_battery",
            "count_events",
            "std_battery",
        ]
    ]
    y = df["target_minutes"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    model = LGBMRegressor(n_estimators=200, learning_rate=0.05)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = root_mean_squared_error(y_val, y_pred)

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)

    print(f"-- MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    return mae, rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--model-out", required=True)
    args = parser.parse_args()
    train(args.inp, args.model_out)
