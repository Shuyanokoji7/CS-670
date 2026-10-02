"""Non-learned sanity baselines, evaluated with the shared full-ranking protocol.

    python -m src.baselines random
    python -m src.baselines popularity
    python -m src.baselines all
"""

import argparse

import numpy as np
import pandas as pd

from src.data import candidate_counts, load_processed
from src.evaluate import evaluate
from src.utils import load_config, project_path


def random_scorer(n_items, seed):
    """Uniform random scores; one generator consumed in user order -> reproducible."""
    rng = np.random.default_rng(seed)
    return lambda users: rng.random((len(users), n_items))


def item_popularity(train, n_items):
    """Number of training positives per item. Uses TRAIN ONLY."""
    return np.bincount(train["item"].to_numpy(), minlength=n_items).astype(np.float64)


def popularity_scorer(train, n_items):
    pop = item_popularity(train, n_items)
    return lambda users: np.broadcast_to(pop, (len(users), n_items))


def run(model, cfg, split):
    ev = cfg["evaluation"]
    k, ties = ev["k"], ev["tie_policy"]
    rows = []
    for target in ("validation", "test"):
        if model == "random":
            scorer = random_scorer(split["n_items"], cfg["seed"])
        elif model == "popularity":
            scorer = popularity_scorer(split["train"], split["n_items"])
        else:
            raise ValueError(model)
        summary, _ = evaluate(scorer, split, target, k=k, tie_policy=ties)
        row = {"model": model, **summary, "seed": cfg["seed"] if model == "random" else None,
               "protocol": f"full_ranking/{ev['candidates']}/tie={ties}",
               "split_seed": cfg["split"]["split_seed"]}
        if model == "random":
            # Closed-form expectation of HR@k under uniform random ranking.
            c = candidate_counts(split, target)
            row[f"expected_hr@{k}"] = float(np.mean(np.minimum(k, c) / c))
        rows.append(row)
    df = pd.DataFrame(rows)
    out = project_path(cfg["paths"]["results_dir"]) / f"{model}_baseline.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    return df, out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=["random", "popularity", "all"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    split = load_processed(cfg)
    k = cfg["evaluation"]["k"]
    cols = ["model", "split", "n_users", f"ndcg@{k}", f"ndcg@{k}_se", f"hr@{k}", f"recall@{k}", f"mrr@{k}", "mean_rank"]
    for model in (["random", "popularity"] if args.model == "all" else [args.model]):
        df, out = run(model, cfg, split)
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"-> {out}\n")


if __name__ == "__main__":
    main()
