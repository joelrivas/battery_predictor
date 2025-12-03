"""
Docstring for train.eval
"""

import argparse
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def evaluate(model_path, test_path):
    """
    Docstring for evaluate
    
    :param model_path: Description
    :param test_path: Description
    """

    model = joblib.load(model_path)
    df = pd.read_parquet(test_path)
    X = df[["last_battery", "mean_battery", "min_battery", "max_battery", "count_events", "std_battery"]]
    y = df["target_minutes"]

    y_pred = model.predict(X)
    print("-- Eval: MAE", mean_absolute_error(y, y_pred))
    print("-- Eval: RMSE", root_mean_squared_error(y, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test", required=True)
    args = parser.parse_args()
    evaluate(args.model, args.test)
