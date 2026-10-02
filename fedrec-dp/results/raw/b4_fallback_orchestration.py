import sys; sys.path.insert(0, ".")
import pandas as pd
import experiments.run_b4 as rb4
from src.utils import load_config
cfg = load_config(rb4.ROOT / rb4.CONFIG)
jobs = []
for r in cfg["model"]["ranks"]:
    pr = rb4._per_rank(cfg, r)
    for lr in (2.5, 5.0):
        for slr in (0.5, 1.0):
            same = (float(pr["local_lr"]), float(pr["clip_norm"]), float(pr["server_lr"])) == (lr, 1.0, slr)
            for s in cfg["seeds"]:
                for e in [None] + list(cfg["privacy"]["target_epsilons"]):
                    jobs.append(rb4.job(r, e, s, kind="stability") if same else
                                rb4.job(r, e, s, clip=1.0, server_lr=slr, local_lr=lr, kind="fallback"))
df = rb4.run_many(jobs, 18, keep_checkpoint=True)
df.to_csv(rb4.RESULTS / "b4_fallback_screen.csv", index=False, float_format="%.17g")
print(len(df), (df.status != "ok").sum())
