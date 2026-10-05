# Prior-art audit and claim boundaries

Audit date: 2026-10-03. This is a targeted primary-source review, not an exhaustive
systematic review. A missing matching paper is not evidence that no such paper
exists. Publication-quality novelty remains unestablished. The earlier E1 audit
covered fewer works; the additional geometry and trajectory papers below narrow
its possible contribution further.

## Closest checked work

| Citation / primary source | What overlaps with this project | Reading depth / boundary |
|---|---|---|
| [FFA-LoRA, Sun et al. 2024](https://arxiv.org/abs/2403.12313) | Fixed random factor, two-factor noise multiplication and aggregation mismatch. | Full HTML, especially Sections 3–4. Fixed-factor learning and the bilinear decomposition are established. |
| [FedASK, Wen et al. 2025](https://arxiv.org/abs/2507.09990) | Double sketching to address private federated factor updates and aggregation. | Full HTML methods/privacy sections inspected. Its local DP-SGD accounting cannot simply be relabeled whole-user DP. |
| [PRISM, Wang and Zhang 2026](https://arxiv.org/abs/2606.00944) | Intrinsic tangent-space clipping/noising, gauge-dependent factor perturbations, effective-noise moments, stable adaptation. | Full HTML formulation and method inspected, appendix topics checked. This directly blocks claiming gauge-aware private low-rank geometry as a new general idea. |
| [FLoRG, Meng et al. 2026](https://arxiv.org/abs/2602.17095) | Nonunique decomposition and Procrustes alignment across federated rounds. | Full HTML inspected. Alignment and decomposition drift are established. |
| [FedRot-LoRA, Zhang et al. 2026](https://arxiv.org/abs/2602.23638) | Orthogonal alignment of client factors before aggregation. | Author abstract inspected; full method not independently audited. Additional direct overlap for alignment-based proposals. |
| [FedGSA, Zheng et al. 2026](https://arxiv.org/abs/2608.03267) | Basis-invariant subspace aggregation after private low-rank learning. | Full HTML method inspected. Geometric aggregation under privacy is established. |
| [From Bilinear to Linear, Zheng et al. 2026](https://arxiv.org/abs/2609.26091) | Low-dimensional linear private federated parameterization. | Author abstract inspected. General linearization is prior art; bibliographic page has an inconsistent displayed submission date versus identifier, so retain identifier/year and recheck before submission. |
| [NoiseCurve, Gu et al. 2025](https://arxiv.org/abs/2510.05416) | Earlier privacy noise changes later gradients; curvature informs correlated noise. | Full HTML, especially Sections 2–3, inspected. Cumulative propagation and Hessian-based explanations are established. |
| [GeoDP, Duan et al. 2025](https://arxiv.org/abs/2504.05618) | Separates effects of perturbing gradient direction and magnitude. | Full HTML introduction, formulation and conclusions inspected. Direction-versus-norm analysis alone is insufficient novelty. |
| [DiSK, Zhang et al. 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/c7793beb7f32b559df55d48370f2b8ae-Paper-Conference.pdf) | Uses gradient dynamics and a simplified Kalman filter to reduce private optimization noise. | Primary proceedings PDF, Sections 3.1–3.2 inspected. Filtering noisy optimization histories is established. |
| [Bietti et al. 2022](https://proceedings.mlr.press/v162/bietti22a.html) | Joint user-level DP with local/global personalization and relative learning-rate geometry. | Proceedings paper, setup, algorithm and local-SGD discussion inspected. Local state, separate step sizes and personalization/privacy tradeoffs are established. |
| [Li et al. 2022](https://www.vldb.org/pvldb/vol15/p900-li.pdf) | Federated matrix factorization, user/rating privacy, embedding clipping and secure aggregation. | Primary PDF, setup and privacy definitions inspected. Merely clipping embeddings or adapting MF to federation is insufficient. |
| [Private ALS, Chien et al. 2021](https://proceedings.mlr.press/v139/chien21a.html) | Practical matrix completion under joint user-level privacy. | Main algorithm, practical modifications, privacy section and supplementary privacy proof inspected in the E3/E4 continuation. Not implemented here; see the separate implementation audit. |
| [PrivateRec, Liu et al.](https://arxiv.org/abs/2204.08146) | Private federated recommendation, public basis decomposition and private serving. | Author abstract and [KDD 2023 publication](https://doi.org/10.1145/3580305.3599889) checked; attempted HTML full text unavailable. Training/serving privacy unit needs full audit before comparison. |
| [Curmei et al. 2023](https://arxiv.org/abs/2309.11516) | Combines private collaborative feedback with public item features. | Full primary HTML algorithm, privacy scope and evaluation inspected in the E3/E4 continuation. Public item metadata is a baseline idea, not sufficient innovation. |
| [Zhang et al. 2022, clipping](https://proceedings.mlr.press/v162/zhang22b.html) | Clipping bias and client-update distributions in private FedAvg. | Proceedings abstract and related discussion checked. Clipping is an established mechanism, not a new discovery here. |
| [Wang et al. 2026, prototype degradation](https://arxiv.org/abs/2604.27833) | Noise degradation in personalized federated prototype learning. | Abstract screened only. Requires full comparison if a prototype-based direction is pursued. |
| [Gaunt, product-normal law](https://arxiv.org/abs/2408.04101) | Gaussian products as differences of independent noncentral chi-square variables, including sign probabilities and cumulants. | Full primary HTML, theorem and sign-probability results checked. E4 uses known probability theory; neither its law nor substitution into a ranking margin is a new theorem. |
| [Krichene et al. 2023, distribution skew](https://arxiv.org/abs/2302.07975) | User privacy-budget allocation among uneven tasks and recommendation items. | Primary PDF Algorithm 1, Theorem 3.3 and privacy proof inspected. Adaptive item-frequency weighting is established; full utility proofs not audited. |
| [Urmian, Liu and Khalil, noisy slate selection](https://arxiv.org/abs/2609.29453) | Score-to-ranking stability, margin certificates, privacy scope and correlated-noise experiments. | Primary HTML methods, assumptions and experiments inspected. Generic noisy-ranking stability is established; their explicit sub-Gaussian assumptions need not cover factor-product noise. E4 found no consequential approximation gap in our sampled states. |

Use [references.bib](references.bib) for verified core citations. Preprint status
is retained unless a primary proceedings source was checked. Search-engine
timestamps, third-party summaries and reported acceptance labels were not used
as technical evidence. This review did not validate every proof in these papers.

Additional primary sources screened during E3/E4: [Fed-LPAR](https://doi.org/10.1016/j.knosys.2026.116587)
already advertises personalized partitions and adaptive local regularization;
[DPBPRMF](https://doi.org/10.1016/j.neucom.2025.131871) is a private BPR baseline
candidate; [Federated Reconstruction](https://papers.nips.cc/paper/2021/file/5d44a2b0d85aa1a4dd3f218be6422c66-Paper.pdf)
studies reconstruction of local parameters. These were screened at abstract,
introduction or indexed-excerpt depth only, not audited for a matched whole-user
privacy comparison. They do not establish that the proposed next experiment
is novel. Third-party generated summaries were excluded as evidence.

## Claims rejected as standalone contributions

1. Adding noise changes model weights and can lower recommendation accuracy.
2. Smaller/fewer trainable parameters may reduce noise or communication.
3. Fixing one factor removes the product of two independent noise matrices.
4. Norms alone are inadequate; gradient direction and factor geometry matter.
5. Earlier noise affects subsequent gradients through optimization dynamics.
6. Procrustes alignment or intrinsic low-rank geometry fixes basis ambiguity.
7. Local personalization, embedding clipping, public metadata or filtering helps DP.

These are prior-art concepts. Empirical replication can be useful for a course
report, but a paper needs an additional specific result.

## Narrow candidate left open

**A measurement study of persistent local personalization and coupling-dependent
noise attribution in private federated ranking.** Its testable object is a
paired intervention, not a general noise-reduction algorithm. E2/E2b show why
raw-coordinate shared-noise coupling can strongly change apparent persistence
of prediction damage after an oracle shared-state reset.

This combination was not located in the checked sources. That is a search
observation, not a priority claim. The algebra behind it and the alignment tool
are known. A paper would need independent replication, robust counterfactual
couplings and an actionable implication beyond this simulator. The current
evidence is promising methodology and a modest local-memory effect, not proof
of a large new failure mode or an improved recommender.

## Independent review still required

- Forward/backward citation tracing from PRISM, FLoRG, NoiseCurve and Bietti.
- Full text of PrivateRec, recent personalized-DP work and stability/coupling
  studies in stochastic optimization; abstract screening is insufficient to
  assert their absence of a feature.
- Explicit differentiation from orthogonal alignment as a training method:
  E2b changes a diagnostic joint coupling, not the marginal noise scale.
- An independent dataset and a published user-level private recommendation
  baseline before any performance or generality claim.
