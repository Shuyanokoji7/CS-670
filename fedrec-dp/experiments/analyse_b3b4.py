"""Predeclared B3/B4 test analysis (RESEARCH_LOG "B3/B4 protocol", corrections 5; frozen B3 rev2 / B4).

Five seeds 42/123/2026/7/99. Per model: average each user's test NDCG@10 over the five seeds, then a paired user
bootstrap (100,000 resamples, seed 2026, common resample indices for every contrast, batched). Conditional on these
training runs. Primary families: B3 rank - B1 (4 contrasts, simultaneous 98.75% Bonferroni CIs); B4 rank/eps - B2
at the same eps (16 contrasts, simultaneous 99.6875%); ordinary 95% for all. Secondary contrasts: 95% only.
Reads frozen artifacts only; writes results/b34_*.csv. No tuning, no training, no new test evaluation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES, RAW = ROOT / "results", ROOT / "results" / "raw"
SEEDS = [42, 123, 2026, 7, 99]
LEVELS = ["nodp", "eps8", "eps4", "eps2", "eps1"]
K = "ndcg@10"
B, BATCH, BOOT_SEED = 100_000, 10_000, 2026


METRIC_COLS = ["ndcg@10", "hr@10", "mrr@10"]
CHECKS = {"files": 0, "user_ids": None}


def per_user(path, col=K):
    """Per-user test metrics with input checks: unique user ids, identical sorted ids across ALL files, finite."""
    df = pd.read_csv(path)
    ids = df["user"].to_numpy()
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate user ids")
    if not np.isfinite(df[METRIC_COLS].to_numpy()).all():
        raise ValueError(f"{path}: non-finite metric values")
    srt = np.sort(ids)
    if CHECKS["user_ids"] is None:
        CHECKS["user_ids"] = srt
    elif not np.array_equal(CHECKS["user_ids"], srt):
        raise ValueError(f"{path}: user ids differ from the other models/seeds")
    CHECKS["files"] += 1
    return df.sort_values("user").set_index("user")[col]


def b1_files():
    return {s: (RAW / f"b1_per_user_test_seed{s}.csv" if s in (42, 123, 2026)
                else RAW / "supplemental_b1" / f"b1_per_user_test_seed{s}.csv") for s in SEEDS}


def b2_dir(s):
    return RAW / "b2_runs" if s in (42, 123, 2026) else RAW / "supplemental_b2" / "b2_runs"


def b2_tag(lvl, s):
    return f"{lvl}_T50_{'CNone' if lvl == 'nodp' else 'C1.0'}_slr1.0_seed{s}"


def main():
    models, seedmeans, rows = {}, [], []

    def add(name, paths, meta):
        paths = list(paths)
        tabs = [per_user(p, METRIC_COLS) for p in paths]
        assert len(tabs) == 5 and all(t.index.equals(tabs[0].index) for t in tabs), name
        models[name] = np.mean([t[K].to_numpy() for t in tabs], axis=0)
        row = {"model": name, **meta, "n_seeds": 5}
        for m, lab in (("ndcg@10", "ndcg"), ("hr@10", "hr"), ("mrr@10", "mrr")):
            sm = [t[m].mean() for t in tabs]
            row.update({f"test_{lab}_mean": float(np.mean(sm)), f"test_{lab}_seed_sd": float(np.std(sm, ddof=1))})
        seedmeans.append(row)

    add("B1", b1_files().values(), {"family": "B1"})
    for lvl in LEVELS:
        add(f"B2_{lvl}", [b2_dir(s) / f"per_user_test_{b2_tag(lvl, s)}.csv" for s in SEEDS], {"family": "B2", "level": lvl})
    b3 = pd.read_csv(RES / "b3_final_results.csv")
    for r, g in b3.groupby("rank"):
        assert sorted(g.seed) == sorted(SEEDS)
        add(f"B3_r{r}", [RAW / "b3_runs" / f"per_user_test_{t}.csv" for t in g.sort_values("seed").tag],
            {"family": "B3", "rank": r})
    b4 = pd.read_csv(RES / "b4_final_results.csv")
    for (role, r, lvl), g in b4.groupby(["role", "rank", "privacy_level"]):
        assert sorted(g.seed) == sorted(SEEDS) and (g.status == "ok").all()
        add(f"B4_{role}_r{r}_{lvl}", [RAW / "b4_runs" / f"per_user_test_{t}.csv" for t in g.tag],
            {"family": f"B4_{role}", "rank": r, "level": lvl})

    sel = pd.read_csv(RES / "b4_rank_selection.csv").set_index("privacy_level")["rank"]
    con = []                                                     # (family, name, a, b, simultaneous level)
    for r in (4, 8, 16, 32):
        con.append(("B3_vs_B1", f"r{r}-B1", f"B3_r{r}", "B1", 1 - 0.05 / 4))
    for r in (4, 8, 16, 32):
        for lvl in ("eps8", "eps4", "eps2", "eps1"):
            con.append(("B4_vs_B2", f"r{r}_{lvl}-B2_{lvl}", f"B4_main_r{r}_{lvl}", f"B2_{lvl}", 1 - 0.05 / 16))
    for lvl in LEVELS:
        con.append(("sec_selected_rank", f"sel_r{sel[lvl]}_{lvl}-B2_{lvl}", f"B4_main_r{sel[lvl]}_{lvl}", f"B2_{lvl}", None))
    for r in (4, 8, 16, 32):
        con.append(("sec_nodp_capacity", f"r{r}_nodp-B2_nodp", f"B4_main_r{r}_nodp", "B2_nodp", None))
        for lvl in ("eps8", "eps4", "eps2", "eps1"):
            con.append(("sec_own_dp_loss", f"r{r}_{lvl}-r{r}_nodp", f"B4_main_r{r}_{lvl}", f"B4_main_r{r}_nodp", None))
        con.append(("sec_fixed_diag_eps4", f"fixed_r{r}_eps4-B2_eps4", f"B4_fixeddiag_r{r}_eps4", "B2_eps4", None))
    for lvl in ("eps8", "eps4", "eps2", "eps1"):
        con.append(("ref_B2_dp_loss", f"B2_{lvl}-B2_nodp", f"B2_{lvl}", "B2_nodp", None))

    D = np.stack([models[a] - models[b] for _, _, a, b, _ in con])          # contrasts x users
    n = D.shape[1]
    rng = np.random.default_rng(BOOT_SEED)
    boots = np.empty((len(con), B))
    for start in range(0, B, BATCH):                                          # common user resamples for all contrasts
        idx = rng.integers(0, n, size=(BATCH, n))
        boots[:, start:start + BATCH] = np.stack([d[idx].mean(axis=1) for d in D])
    for (fam, name, a, b, sim), d, bt in zip(con, D, boots):
        row = {"family": fam, "contrast": name, "a": a, "b": b, "diff": float(d.mean()),
               "ci95_lo": float(np.quantile(bt, 0.025)), "ci95_hi": float(np.quantile(bt, 0.975)),
               "frac_boot_gt0": float((bt > 0).mean())}
        if sim is not None:
            row.update({"sim_level": sim, "sim_lo": float(np.quantile(bt, (1 - sim) / 2)),
                        "sim_hi": float(np.quantile(bt, 1 - (1 - sim) / 2))})
        rows.append(row)
    pd.DataFrame(seedmeans).to_csv(RES / "b34_test_means.csv", index=False, float_format="%.6f")
    pd.DataFrame(rows).to_csv(RES / "b34_contrasts.csv", index=False, float_format="%.6f")

    # perturbation diagnostics (per-run means over the 5 seeds)
    def b2_res(lvl, s):
        return pd.read_csv(b2_dir(s) / f"result_{b2_tag(lvl, s)}.csv").iloc[0]
    diag = []
    qn = 0.1 * 943
    for lvl in LEVELS:
        x = pd.DataFrame([b2_res(lvl, s) for s in SEEDS])
        diag.append({"model": f"B2_{lvl}", "D": int(x.shared_coordinates_D.iloc[0]), "sigma": x.noise_multiplier.iloc[0],
                     "C": x.clip_norm.iloc[0], "server_lr": x.server_lr.iloc[0],
                     "typical_noise_norm": x.typical_noise_norm.mean(), "noise_norm_mean": x.noise_norm_mean.mean(),
                     "server_step_noise_norm": (x.server_lr * x.noise_norm_mean / qn).mean(),
                     "clipped_aggregate_norm": x.clipped_aggregate_norm_mean.mean(), "snr": x.signal_to_noise_mean.mean(),
                     "frac_clipped": x.frac_clipped_mean.mean(), "final_item_norm": x.final_item_norm_mean.mean()})
    for (role, r, lvl), x in b4.groupby(["role", "rank", "privacy_level"]):
        diag.append({"model": f"B4_{role}_r{r}_{lvl}", "D": int(x.shared_coordinates_D.iloc[0]),
                     "sigma": x.privacy_noise_multiplier.iloc[0], "C": x.privacy_clip_norm.iloc[0],
                     "server_lr": x.server_lr.iloc[0], "typical_noise_norm": x.typical_noise_norm.mean(),
                     "noise_norm_mean": x.noise_norm_mean.mean(),
                     "server_step_noise_norm": (x.server_lr * x.noise_norm_mean / qn).mean(),
                     "clipped_aggregate_norm": x.clipped_aggregate_norm_mean.mean(), "snr": x.signal_to_noise_mean.mean(),
                     "frac_clipped": x.frac_clipped_mean.mean(), "final_item_norm": x.final_item_norm_mean.mean()})
    pd.DataFrame(diag).to_csv(RES / "b34_perturbation.csv", index=False, float_format="%.6g")

    # communication: per-client per-round payload AND total volume (to best / all rounds)
    com = []
    b1 = pd.concat([pd.read_csv(RAW / f"b1_seed{s}.csv" if s in (42, 123, 2026) else RAW / "supplemental_b1" / f"b1_seed{s}.csv").iloc[[0]]
                    for s in SEEDS])
    com.append({"model": "B1", "bytes_per_client_round": int(b1.bytes_per_client_per_round.iloc[0]),
                "rounds_to_best_mean": b1.best_round.mean(), "rounds_run_mean": b1.rounds_run.mean(),
                "GB_to_best_mean": b1.comm_bytes_to_best_round.mean() / 1e9, "GB_all_rounds_mean": b1.comm_bytes_total_run.mean() / 1e9})
    b3v = pd.read_csv(RES / "b3_final_validation.csv")
    for r, x in b3v.groupby("rank"):
        com.append({"model": f"B3_r{r}", "bytes_per_client_round": int(x.bytes_per_client_per_round.iloc[0]),
                    "rounds_to_best_mean": x.best_round.mean(), "rounds_run_mean": x.rounds_run.mean(),
                    "GB_to_best_mean": x.comm_bytes_to_best_round.mean() / 1e9, "GB_all_rounds_mean": x.comm_bytes_total_run.mean() / 1e9})
    x = pd.DataFrame([b2_res(l, s) for l in LEVELS for s in SEEDS])
    com.append({"model": "B2_all_levels", "bytes_per_client_round": 2 * 4 * int(x.shared_coordinates_D.iloc[0]),
                "rounds_to_best_mean": 50, "rounds_run_mean": 50, "GB_to_best_mean": x.comm_bytes_total.mean() / 1e9,
                "GB_all_rounds_mean": x.comm_bytes_total.mean() / 1e9})
    for r, x in b4[b4.role == "main"].groupby("rank"):
        com.append({"model": f"B4_r{r}_all_levels", "bytes_per_client_round": int(x.bytes_per_client_per_round.iloc[0]),
                    "rounds_to_best_mean": 50, "rounds_run_mean": 50, "GB_to_best_mean": x.comm_bytes_total.mean() / 1e9,
                    "GB_all_rounds_mean": x.comm_bytes_total.mean() / 1e9})
    c = pd.DataFrame(com)
    c["payload_vs_B1"] = c.bytes_per_client_round / c.bytes_per_client_round.iloc[0]
    c["volume_to_best_vs_B1"] = c.GB_to_best_mean / c.GB_to_best_mean.iloc[0]
    c.to_csv(RES / "b34_communication.csv", index=False, float_format="%.4f")
    print(f"input checks: {CHECKS['files']} per-user test files; {len(CHECKS['user_ids'])} unique user ids, identical "
          f"sorted ids across all models/seeds; all metrics finite (HR@10 = Recall@10 under leave-one-out)")
    print("wrote b34_test_means, b34_contrasts, b34_perturbation, b34_communication")


if __name__ == "__main__":
    main()
