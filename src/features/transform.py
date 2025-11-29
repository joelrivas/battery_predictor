"""Feature engineering: create features from raw events."""

import os
import sys
import argparse
import pandas as pd

root_path = os.getcwd()
sys.path.append(root_path)

from src.features.featurize import FeatureEngineer


def transform(inp, out, battery_threshold=0.10):
    """
    Generate features and save them on output path.
    
    :param inp: path to parquet file
    :param out: path to save features in parquet format
    """

    input_path = f"data/{inp}.parquet"
    df = pd.read_parquet(input_path)
    fe = FeatureEngineer(battery_threshold=battery_threshold, resample="1min")
    feats = fe.featurize(df)
    output_path = f"data/{out}.parquet"
    feats.to_parquet(output_path, index=False)
    print(f"Wrote {len(feats)} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--battery_threshold", type=float, default=0.10)
    args = parser.parse_args()
    transform(args.inp, args.out)
