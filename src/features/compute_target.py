"""Compute target variable: time until battery percentage drops below threshold."""

import pandas as pd
import numpy as np

def compute_time_to_threshold(df_events, threshold=10):
    """
    df_events: DataFrame ordenado por ts, columnas: user_id, ts, battery_pct
    Returns per-row the minutes until battery <= threshold using forward simulation.
    For each record, find next timestamp where battery <= threshold; compute delta minutes.
    If never reaches threshold in available history, set np.nan or large value.
    """
    df = df_events.sort_values(["user_id", "ts"]).reset_index(drop=True)
    df["target_minutes"] = np.nan
    for _, g in df.groupby("user_id"):
        battery_arr = g["battery_pct"].values
        ts_arr = g["ts"].values
        n = len(battery_arr)

        # for each i find j>i where battery <= threshold
        # efficient approach: for each j where battery<=threshold, mark prior indices
        threshold_idxs = np.where(battery_arr <= threshold)[0]
        if len(threshold_idxs) == 0:
            continue
        # for each i, find first threshold_idx > i
        jpos = 0
        for i in range(n):
            while jpos < len(threshold_idxs) and threshold_idxs[jpos] <= i:
                jpos += 1
            if jpos < len(threshold_idxs):
                j = threshold_idxs[jpos]
                delta = (pd.to_datetime(ts_arr[j])-pd.to_datetime(ts_arr[i])).total_seconds()/60.0
                df.loc[g.index[i], "target_minutes"] = int(delta)
            else:
                df.loc[g.index[i], "target_minutes"] = np.nan
    return df
