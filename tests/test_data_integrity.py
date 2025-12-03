"""Docstring for tests.test_data_integrity"""

import pandas as pd


def test_battery_range():
    """
    Docstring for test_battery_range
    """
    df = pd.read_parquet("data/sample_events.parquet")
    assert df["battery_pct"].between(0, 100).all()


def test_ordered_timestamps():
    """
    Docstring for test_ordered_timestamps
    """
    df = pd.read_parquet("data/sample_events.parquet")
    assert df.sort_values(["user_id", "ts"]).equals(df)
