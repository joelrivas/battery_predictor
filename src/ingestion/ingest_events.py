"""Ingest events"""

import argparse
from datetime import datetime, timedelta
import pandas as pd
# pylint: disable=no-name-in-module
from numpy.random import RandomState

def generate_events(n=10000):
    """
    Generate random events recreating battery usage of devices.
    
    :param n: Number of events to generate
    """
    random_state = RandomState(seed=856)
    start = datetime.now()
    user_ids = random_state.randint(1, 1000, size=n)
    timestamps = []
    for x in random_state.randint(0, 60*24*14, size=n):
        timestamps.append(start - timedelta(minutes=int(x)) )
    event_type = random_state.choice(["hardware1", "app1", "usage"], size=n, p=[0.6, 0.2, 0.2])
    battery = random_state.randint(3, 100, size=n)
    df = pd.DataFrame({
        "user_id": user_ids,
        "timestamp": timestamps,
        "event_type": event_type,
        "battery": battery
    })
    return df

def main(out, n):
    """
    Generate events and save them on output path.
    
    :param out: Output path for parquet file
    :param n: Number of events to generate
    """
    df = generate_events(n)
    df.to_parquet(out, index=False)
    print(f"-- Wrote {len(df)} events to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, 
                        help="Output path for events parquet file")
    parser.add_argument("--n", type=int, default=10000, 
                        help="Number of events to generate")

    args = parser.parse_args()
    main(args.out, args.n)
