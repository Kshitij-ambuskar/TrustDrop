"""Train, evaluate, select, calibrate and save. Usage (from repo root):
    python scripts/train.py                      # needs data/data.csv (real UCI data)
    python scripts/train.py --allow-synthetic    # DEVELOPMENT ONLY: uses data/synthetic_dev.csv"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.ml.training.train import main  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = main(allow_synthetic=a.allow_synthetic, n_splits=a.folds)
    print(f"\n[{out['provenance']}]\nSelected model: {out['best']}")
    for n, r in out["results"].items():
        print(f"  {n:20s} CV macro-F1 {r['cv']['mean']['f1_macro']:.3f} | test macro-F1 {r['test']['f1_macro']:.3f}")
