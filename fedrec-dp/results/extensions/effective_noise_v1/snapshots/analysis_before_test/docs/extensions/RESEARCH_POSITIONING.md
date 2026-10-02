# Positioning the effective-noise study

Primary sources checked on 2026-10-02. This is a contribution audit, not a claim
that the current study is already a publication-ready result.

| Work | Existing contribution | Implication for E1 |
|---|---|---|
| [Sun et al., FFA-LoRA (2024)](https://arxiv.org/abs/2403.12313) | Fixes a random factor; analyzes factor-aggregation mismatch and amplified Gaussian noise. | E1 cannot claim invention of fixed-factor private federation or the bilinear noise decomposition. Its setting has trainable item factors and private local user embeddings. |
| [Wen et al., FedASK (2025)](https://arxiv.org/abs/2507.09990) | Uses double sketching to train both low-rank adapters while addressing aggregation/noise problems. | A stronger related method than freezing alone. A direct recommender adaptation would require a separately specified whole-user mechanism and accounting, not imported LLM privacy claims. |
| [Zheng et al., From Bilinear to Linear (2026)](https://arxiv.org/abs/2609.26091) | Introduces a unified low-dimensional linear parameterization and a projection informed by warm-up statistics. | Even general linear parameterization is prior art. Private warm-up information needs accounting in this project's threat model. |
| [Curmei et al., public item features (2023)](https://arxiv.org/abs/2309.11516) | Uses collective matrix factorization of sensitive feedback and public item information to improve private recommendation. | Public metadata alone is an established direction. E1's random latent basis has no item semantics; do not present it as a metadata method. |
| [Bietti et al., personalized FL (2022)](https://proceedings.mlr.press/v162/bietti22a.html) | Studies local/global optimization with user-level joint DP and gives privacy–accuracy generalization guarantees. | Local user state plus a private global model is established. Our conditional score/margin diagnostics specialize this setting to ranking, rather than replacing their theory. |
| [Balle and Wang, analytic Gaussian mechanism (2018)](https://arxiv.org/abs/1805.06530) | Calibrates Gaussian noise using the exact Gaussian CDF condition. | E1-DPPop uses this established mechanism, with its own bounded entire-user vector and one-release accounting. |

## What could be worth a paper

The most defensible candidate is a mechanism study: demonstrate how latent
parameterization, clipping and persistent local personalization shape ranking
noise, then show when a linear shared model helps or fails under a finite
communication horizon. The evidence should distinguish three quantities:

1. Coordinate-noise norm, which counts private parameters.
2. Effective item-matrix disturbance, which depends on factor scale and bilinear noise.
3. Score/margin disturbance, which additionally depends on the local user vectors.

The conditional moment identities are straightforward consequences of Gaussian
moments. They are tools for the study, not sufficient theorem novelty by
themselves. The orthonormal FixedB equivalence to ordinary rank-r BPR also limits
algorithmic novelty. A useful negative finding would be that a cleaner noise
geometry loses because the model trains too slowly at T=50, or that clipped DP
popularity remains stronger. Such outcomes should guide the next experiment,
not prompt an expanded grid to chase a win.

## Gaps to address before a strong submission

- Replicate on an independently reserved MovieLens-1M test, with a declared
  validation budget and no transfer of historical test-based decisions.
- Add a communication-budget experiment: varying T changes privacy composition,
  so solve sigma for each method's permitted T. Common T and smaller payload
  establish payload savings, not superiority at an equal total budget.
- Verify the scope and implement an appropriate published private recommender
  comparison, such as [Private Alternating Least Squares](https://proceedings.mlr.press/v139/chien21a.html),
  with an explicitly compatible user-level threat model and clearly different
  objective. DP popularity is essential but does not replace that comparison.
- Use at least one control that varies effective optimization speed or clips
  effective item updates, because fixing B removes several mechanisms at once.
- Consider score-aware or matrix-size-weighted factor scaling only under a new
  bounded protocol that includes clipping/optimization consequences. A gauge
  sweep alone cannot justify a better training method.
- Metadata-plus-private-residual learning remains a distinct direction 3
  extension. Cold-target analysis on ML-100K has only fourteen test targets;
  report it descriptively and seek a larger replication for subgroup claims.

Keep the eventual paper centered on the mechanism established by the evidence.
Do not write its abstract around a presumed improvement or treat implementation
of a known fixed-factor method as a sufficient novel contribution.
