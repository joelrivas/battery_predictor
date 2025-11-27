"""Generate synthetic battery usage data for multiple users over time."""

import os
import argparse
from datetime import datetime, timedelta
import numpy as np
# pylint: disable=no-name-in-module
from numpy.random import RandomState
import pandas as pd


def simulate_user_stream(user_id, start_ts, n_minutes, base_battery=100):
    """Simulate a time series of battery usage and sensor data for a single user."""

    # pylint: disable=no-name-in-module
    rng = RandomState(user_id)
    # base consumption rate per minute in % when idle (very small)
    base_consumption = rng.uniform(0.1, 0.01)  # % per minute
    # sensitivity multipliers (how much extra consumption per activity unit)
    hr_sens = rng.uniform(0.0005, 0.002)   # per bpm above baseline
    usage_sens = rng.uniform(0.02, 0.08)   # per minute screen-on
    steps_sens = rng.uniform(0.0002, 0.001) # per step
    sync_sens = rng.uniform(0.05, 0.15)    # per sync event (costly)

    rows = []
    battery = base_battery
    # baseline heart rate per user
    hr_baseline = rng.randint(55, 75)
    for i in range(n_minutes):
        ts = start_ts + timedelta(minutes=i)
        # stochastic behavior
        is_active = rng.rand() < 0.12  # 12% chance user is active this minute
        steps = rng.randint(0, 30) if is_active else 0
        usage_minutes = rng.rand()*1.0 if (rng.rand() < 0.08) else 0.0  # screen on fraction
        # heart rate fluctuates more when active
        hr = hr_baseline + (rng.randint(0,30) if is_active else rng.randint(-3,3))
        sync_event = 1 if (rng.rand() < 0.005) else 0  # rare sync events
        # accelerometer magnitude (proxy for movement)
        acc_x = rng.normal(0, 1) * (2 if is_active else 0.1)
        acc_y = rng.normal(0, 1) * (2 if is_active else 0.1)
        acc_z = rng.normal(0, 1) * (2 if is_active else 0.1)

        # instantaneous consumption %
        hr_component = max(0, (hr - hr_baseline)) * hr_sens
        usage_component = usage_minutes * usage_sens
        steps_component = steps * steps_sens
        sync_component = sync_event * sync_sens
        noise = rng.normal(0, 0.02)

        consumption_pct = base_consumption + hr_component + usage_component + steps_component \
                          + sync_component + noise
        # ensure battery reduces but not negative in small steps
        battery = max(0.0, battery - consumption_pct)
        # record
        rows.append({
            "user_id": int(user_id),
            "ts": ts,
            "battery_pct": float(battery),
            "hr": int(hr),
            "steps": int(steps),
            "usage_minutes": float(round(usage_minutes,3)),
            "sync_event": int(sync_event),
            "acc_x": float(acc_x),
            "acc_y": float(acc_y),
            "acc_z": float(acc_z),
            "consumption_pct": float(consumption_pct)
        })
        if battery <= 0.0:
            break
    return pd.DataFrame(rows)

def generate_population(n_users=200, minutes_per_user=60*24*7, start_ts=None,
                        out_path="data/synthetic_events.parquet"):
    """Generate synthetic battery usage data for multiple users over time."""

    start_ts = start_ts or datetime.now() - timedelta(days=7)
    all_dfs = []
    for uid in range(1, n_users+1):
        # each user we simulate a shorter slice to keep dataset size reasonable
        df = simulate_user_stream(user_id=uid, start_ts=start_ts, n_minutes=minutes_per_user,
                                  base_battery=np.random.uniform(30,100))
        all_dfs.append(df)
    full = pd.concat(all_dfs, ignore_index=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    full.to_parquet(out_path, index=False)
    print(f"Wrote {len(full)} rows to {out_path}")
    return full

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=200)
    parser.add_argument("--minutes", type=int, default=60*24)  # 1 day per user default
    parser.add_argument("--out", type=str, default="data/synthetic_events.parquet")
    args = parser.parse_args()
    generate_population(n_users=args.users, minutes_per_user=args.minutes, out_path=args.out)
