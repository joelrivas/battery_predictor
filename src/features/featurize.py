"""Feature engineering: create features from raw events."""

import argparse
import pandas as pd


def featurize(df):
    """
    create window and features.
    
    :param df: DataFrame of events
    """

    df["ts"] = pd.to_datetime(df["ts"])
    max_ts = df["ts"].max()
    window_start = max_ts - pd.Timedelta(hours=24)
    dfw = df[df["ts"] >= window_start]
    agg = dfw.groupby("user_id").agg(
        last_battery=("battery", "last"),
        mean_battery=("battery", "mean"),
        min_battery=("battery", "min"),
        max_battery=("battery", "max"),
        count_events=("battery", "count"),
        std_battery=("battery", "std")
    ).reset_index().fillna(0)

    # time to discharge (synthetic)
    agg["target_minutes"] = (agg["mean_battery"] / 100) * 24 * 60
    return agg

def main(inp, out):
    """
    Generate features and save them on output path.
    
    :param inp: path to parquet file
    :param out: path to save features in parquet format
    """

    df = pd.read_parquet(inp)
    feats = featurize(df)
    feats.to_parquet(out, index=False)
    print(f"Wrote {len(feats)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.inp, args.out)
