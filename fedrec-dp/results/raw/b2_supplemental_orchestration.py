# B2 supplemental seeds 7/99: EXACT frozen B2 protocol, separate output roots (core B2 outputs untouched).
import sys; sys.path.insert(0, ".")
import experiments.run_b2 as rb2
rb2.RAW = rb2.ROOT / "results" / "raw" / "supplemental_b2"              # history/rounds/result/per-user files
rb2.CHECKPOINTS = rb2.ROOT / "checkpoints" / "supplemental_b2"          # checkpoints
rb2.RAW.mkdir(parents=True, exist_ok=True); rb2.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
cfg = rb2.load_config(rb2.ROOT / rb2.CONFIG); rb2.require_frozen(cfg)
levels = [None] + list(cfg["privacy"]["target_epsilons"])
jobs = [rb2.job(e, s, evaluate_test=True) for s in (7, 99) for e in levels]
if float(cfg["federated"]["server_lr"]) != rb2.B1_SERVER_LR:
    jobs += [rb2.job(None, s, server_lr=rb2.B1_SERVER_LR, evaluate_test=True) for s in (7, 99)]
rb2.run_many(jobs, 12)                                                  # NOT sweep(): analyse() would write core results/
