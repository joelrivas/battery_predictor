"""Docstring for tests.test_featurizer"""

import pandas as pd
from battery_predictor.features.featurize import FeatureEngineer # type: ignore


def test_featurizer_outputs_expected_columns():
    """
    Docstring for test_featurizer_outputs_expected_columns
    """
    df = pd.read_parquet("data/sample_events.parquet")
    fe = FeatureEngineer(battery_threshold=0.1, resample="1min")
    feat = fe.featurize(df)

    expected = ['ts', 'user_id', 'battery_pct', 'hr', 'steps', 'usage_minutes',
       'sync_event', 'acc_x', 'acc_y', 'acc_z', 'consumption_pct', 'hour',
       'dayofweek', 'battery_mean_5m', 'battery_std_5m', 'battery_min_5m',
       'battery_max_5m', 'hr_mean_5m', 'hr_std_5m', 'sync_sum_5m',
       'battery_mean_15m', 'battery_std_15m', 'battery_min_15m',
       'battery_max_15m', 'hr_mean_15m', 'hr_std_15m', 'sync_sum_15m',
       'battery_mean_30m', 'battery_std_30m', 'battery_min_30m',
       'battery_max_30m', 'hr_mean_30m', 'hr_std_30m', 'sync_sum_30m',
       'battery_mean_60m', 'battery_std_60m', 'battery_min_60m',
       'battery_max_60m', 'hr_mean_60m', 'hr_std_60m', 'sync_sum_60m',
       'battery_lag_1m', 'hr_lag_1m', 'sync_lag_1m', 'battery_lag_5m',
       'hr_lag_5m', 'sync_lag_5m', 'battery_lag_10m', 'hr_lag_10m',
       'sync_lag_10m', 'battery_lag_30m', 'hr_lag_30m', 'sync_lag_30m',
       'target_minutes']

    for col in expected:
        assert col in feat.columns
