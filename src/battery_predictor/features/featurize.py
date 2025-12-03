"""Feature Engineering Module."""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    Feature engineering for battery discharge prediction.
    """

    def __init__(self, battery_threshold=0.20, resample="1min"):
        self.battery_threshold = battery_threshold
        self.resample = resample

    def _calculate_target(self, df):
        """
        target_minutes = tiempo futuro (en minutos) hasta que la batería cruce battery_threshold.
        Si nunca cruza, target = np.nan o un valor grande configurable.
        """
        battery = df["battery_pct"].values
        times = df["ts"].values.astype("datetime64[m]")
        threshold = self.battery_threshold

        # buscar para cada punto cuándo se cruza el threshold
        result = []

        for i in range(len(df)):
            current_time = times[i]
            # buscar en el futuro
            future_idx = np.where(battery[i:] <= threshold)[0]

            if len(future_idx) == 0:
                result.append(np.nan)
            else:
                t_cross = times[i + future_idx[0]]
                diff = (t_cross - current_time).astype(int)
                result.append(diff)

        df["target_minutes"] = result
        return df

    def _time_features(self, df):
        df["hour"] = df["ts"].dt.hour
        df["dayofweek"] = df["ts"].dt.dayofweek
        return df

    def _rolling_features(self, df):
        windows = [5, 15, 30, 60]  # minutos

        for w in windows:
            df[f"battery_mean_{w}m"] = df["battery_pct"].rolling(w).mean()
            df[f"battery_std_{w}m"] = df["battery_pct"].rolling(w).std()
            df[f"battery_min_{w}m"] = df["battery_pct"].rolling(w).min()
            df[f"battery_max_{w}m"] = df["battery_pct"].rolling(w).max()

            df[f"hr_mean_{w}m"] = df["hr"].rolling(w).mean()
            df[f"hr_std_{w}m"] = df["hr"].rolling(w).std()

            df[f"sync_sum_{w}m"] = df["sync_event"].rolling(w).sum()

        return df

    def _lag_features(self, df):
        lags = [1, 5, 10, 30]

        for lag in lags:
            df[f"battery_lag_{lag}m"] = df["battery_pct"].shift(lag)
            df[f"hr_lag_{lag}m"] = df["hr"].shift(lag)
            df[f"sync_lag_{lag}m"] = df["sync_event"].shift(lag)

        return df

    def featurize(self, df):
        """
        df debe traer: user_id, ts, battery_pct, hr, sync_event, temperature (opcional)
        """

        # asegurar orden
        df = df.sort_values(["user_id", "ts"])

        # resample por cada dispositivo
        out = []

        for user_id, g in df.groupby("user_id"):
            g = g.set_index("ts").resample(self.resample).ffill().reset_index()
            g["user_id"] = user_id

            g = self._time_features(g)
            g = self._rolling_features(g)
            g = self._lag_features(g)
            g = self._calculate_target(g)

            out.append(g)

        df_feats = pd.concat(out).reset_index(drop=True)

        return df_feats
