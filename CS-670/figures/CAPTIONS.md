# Report figure captions and sources

Each numbered figure is available as PNG for preview and PDF/SVG for report
export. All are produced by [make_figures.py](../reproducibility/make_figures.py)
from curated tables, with no model evaluation. Seed SD indicates descriptive
variation, not a confidence interval or multiplicity adjustment. Use the
separate paired-contrast tables for primary statistical claims.

## 1. Nonprivate controls

[Preview](01_nonprivate_controls.png) · [PDF](01_nonprivate_controls.pdf)

Left: matched three-seed B0, B0-UW and B1 full-ranking test NDCG@10. Right:
five-seed converged B1 and B3 at ranks 4,8,16,32. Error bars show training-seed
SD. The two panels use different seed sets and should not be merged as one
matched comparison. Centralized/user-weighting point differences have overall
paired intervals including zero; the four adjusted B3-versus-B1 contrasts are
negative. Sources: centralized_federated_3seeds and final_5seed_means.

## 2. Historical private utility

[Preview](02_historical_private_utility.png) · [PDF](02_historical_private_utility.pdf)

Five-seed test NDCG@10 ± seed SD for B2 and B4, all at T=50, against target
epsilon (larger means weaker privacy). The dotted line is the historical
**nonprivate** raw-popularity reference. Historical tuning budgets were unequal.
Four of sixteen adjusted B4-versus-B2 comparisons are positive; do not infer
significance from visual error-bar overlap. Sources: final_5seed_means and
nonprivate_popularity. The line interpolates discrete experiments.

## 3. Controlled E1 utility

[Preview](03_effective_noise_utility.png) · [PDF](03_effective_noise_utility.pdf)

Full, Two, FixedB and one-release DP popularity test NDCG@10 ± five-seed SD.
For Two/FixedB each rank is selected on validation separately at each level.
The primary E1 tests instead compare matched ranks. Collaboration uses 50
rounds, popularity one full-population bounded-sensitivity release, with
separate calibration at the same target epsilon/delta. All collaborative point
means at epsilon at most four fall below DP popularity; no family-wide paired
significance claim against popularity is implied. Sources: test_means,
rank_selection; selected_rank_utility is the derived plotting table.

## 4. Effective noise, local scale and clipping

[Preview](04_noise_and_clipping.png) · [PDF](04_noise_and_clipping.pdf)

All E1 ranks at target epsilon 1. Left: each run's square root of mean squared
Frobenius norm of the immediate effective-item perturbation, averaged over
seeds. Center: mean immediate
score-shock RMSE, with a logarithmic axis to retain the huge finite Two-r32
outlier. Right: mean fraction of selected client updates clipped. These are
diagnostic aggregates, with no error bars; seed-level values are retained in
geometry_per_run. Different selected step sizes, local norms and clipping
geometry prevent interpreting the bars as a controlled dimension-only effect.
Matrix norms and score RMSE have different units. Source: geometry_means.

## 5. Local-state intervention trajectories

[Preview](05_local_state_memory.png) · [PDF](05_local_state_memory.pdf)

E2/E2b exploratory top-10 set disagreement with the reference after a retained
pulse, shared reset or local reset. Starting checkpoint epsilon is 1; pulse
amplitude is one model-specific per-round noise SD. Means and SD use five
checkpoint seeds after averaging two pulse replicates within seed. All panels
share a linear vertical scale. Curves can overlap. Large raw-coordinate Two
distances shrink under aligned coupling. Shared reset leaves a smaller
local-state effect. These are prediction disagreements on training-unseen
candidates, not relevance measurements or deployable interventions. Sources:
noise_memory/by_seed and noise_memory_aligned/by_seed, after_reset rows.

## 6. Factor-coordinate coupling control

[Preview](06_coupling_control.png) · [PDF](06_coupling_control.pdf)

Two-r8 shared-reset disagreement ten rounds after the intervention, alpha=1,
under raw-coordinate and orthogonally aligned coupling. Diamonds and whiskers
show five-seed means ± seed SD; open circles show every seed value (each is an
average of two pulse replicates). The vertical axis is logarithmic and values
are percentages of top-10 set entries that disagree. Mean differences of
17.8546% versus .4204% at epsilon 1, and 9.3960% versus .1083% at epsilon 2,
concern joint diagnostic coupling, not an improvement in either marginal
training algorithm. E2b was a declared adaptive follow-up after E2. Source:
the same seed tables as Figure 5, restricted to lag=10 and shared_reset.

## 7. Independent-data and balancing control

[Preview](07_balancing_replication.png) · [PDF](07_balancing_replication.pdf)

E3 Two-r8 shared-reset top-10 disagreement after ten rounds on both datasets
at starting epsilon 1/2. Diamonds show means, open circles all five seed
values, after averaging two pulse draws per seed. Whiskers show seed SD;
lower whiskers stop at zero where needed. The symmetric-log axis is linear
below .02 percentage points so exact zeros remain visible. Raw/aligned
coupling differs strongly with SVD balancing but has identical churn without
balancing in the saved seed table. Disabling balancing applies only to the
continuation from a balanced checkpoint. Neither axis nor intervention shows
a relevance improvement. Source: noise_replication/endpoint_by_seed. Produced
by [make_followup_figures.py](../reproducibility/make_followup_figures.py).

## 8. MovieLens-1M validation transfer

[Preview](08_ml1m_validation.png) · [PDF](08_ml1m_validation.pdf)

Full-population validation NDCG@10 on ML-1M with frozen transferred settings.
Bars and whiskers are five-seed means and SD; dots show each seed. Collaboration
uses 50 rounds and popularity a separately calibrated single bounded release.
The point differences are not paired significance claims. No ML-1M test target
was scored. Source: noise_replication/validation_by_seed; follow-up figure script.

## 9. Rejected ranking-tail candidate

[Preview](09_ranking_tail_screen.png) · [PDF](09_ranking_tail_screen.pdf)

Maximum absolute flip-probability error of variance-matched and linearized
Gaussian approximations against numerical evaluation of the known conditional
product-normal law. Each bar includes 320 prescribed state-user-pair records
(five seeds, sixteen users, four rank pairs). Logarithmic vertical scale.
The dashed line marks .01 error per pair; advancement required exceeding it
on at least 5% of the two primary pair types in both datasets. Even the maxima
over all four pair types fall far below it. This is a label-free, one-step
noise diagnostic, not accuracy or an estimate of cumulative training damage.
Source: ranking_tails/screen_summary; follow-up figure script.
