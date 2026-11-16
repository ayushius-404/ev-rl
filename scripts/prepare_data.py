"""Raw sessions -> processed per-day data + chronological split.   python scripts/prepare_data.py"""
import argparse
import json

from evrl.config import load_config
from evrl.data.preprocess import preprocess

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    meta = preprocess(load_config(a.config))
    print(json.dumps({k: v for k, v in meta.items() if k != "splits"}, indent=2))
    print({k: f"{len(v)} days ({v[0]} .. {v[-1]})" if v else "0 days" for k, v in meta["splits"].items()})
