"""Validation-only stopping-rule check for the weighting control (never touches test).

Re-runs a config with a different early-stopping patience and reports the best validation epoch,
so B0 and B0-UW can be compared under one common stopping rule.

    python experiments/patience_check.py --config configs/b0_uw.yaml --seed 42 --patience 100
"""
import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from experiments.run_b0 import load_all, run_training  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--patience", type=int, required=True)
args = p.parse_args()
cfg, _, split, _ = load_all(args.config)
c = copy.deepcopy(cfg)
c["training"]["patience"] = args.patience
_, history, best = run_training(c, split, args.seed, log=lambda *_: None)
name = cfg.get("output_prefix", "b0")
out = ROOT / "results" / "raw" / "weighting_control" / f"{name}_seed{args.seed}_patience{args.patience}.csv"
pd.DataFrame(history).to_csv(out, index=False, float_format="%.6f")
stopped = "max_epochs" if len(history) >= c["training"]["max_epochs"] else "early_stopping"
print(f"{name} seed {args.seed} patience {args.patience}: best epoch {best['epoch']}, epochs run {len(history)}, "
      f"stopped by {stopped}, best val ndcg@10 {best['ndcg@10']:.6f}")
