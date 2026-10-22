"""Command line:

    python -m smogcast fetch      # download history (needs API keys in .env)
    python -m smogcast train      # train + evaluate + save model
    python -m smogcast forecast   # live 72-hour forecast -> outputs/latest.json
    python -m smogcast demo       # whole pipeline on SYNTHETIC data (no keys; code check only)
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    p = argparse.ArgumentParser(prog="smogcast")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--start", default=None)
    t = sub.add_parser("train")
    t.add_argument("--issue-step", type=int, default=12, help="hours between training issue times")
    t.add_argument("--max-iter", type=int, default=500)
    sub.add_parser("forecast")
    sub.add_parser("demo")
    a = p.parse_args()

    if a.cmd == "fetch":
        from . import config, pipeline
        pipeline.fetch_history(start=a.start or config.HISTORY_START)
    elif a.cmd == "train":
        from . import pipeline, train
        train.run(*pipeline.load_all(), issue_step=a.issue_step, max_iter=a.max_iter)
    elif a.cmd == "forecast":
        from . import forecast
        forecast.run()
    elif a.cmd == "demo":
        import joblib
        import pandas as pd

        from . import config, synthetic, train
        from .forecast import make_forecast
        print("*** SYNTHETIC DATA: checks the code runs; the numbers mean nothing ***")
        from .clean import clean_pm25
        st, pm_raw, wx, fires = synthetic.make()
        pm = clean_pm25(pm_raw)
        train.run(st, pm, wx, fires, issue_step=12, max_iter=100, test_start="2025-10-01", tag="synthetic")
        bundle = joblib.load(config.MODEL_DIR / "smogcast.joblib")
        now = pd.Timestamp("2025-11-10T06:00", tz="UTC")
        out = make_forecast(bundle, pm_raw[pm_raw["time"] <= now], wx, fires[fires["time"] <= now], now=now)
        (config.OUTPUT_DIR / "latest.json").write_text(json.dumps(out, indent=1))
        print(f"Wrote outputs/latest.json ({len(out['stations'])} stations)")


if __name__ == "__main__":
    main()
