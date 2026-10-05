# Literature-search record

Date: 2026-10-03; continuation of the 2026-10-02 E1 review. Purpose: find and
falsify candidate novelty claims in private/federated recommendation. Web search
was followed by primary arXiv, proceedings and author/publisher documents.
Only links, bibliographic information and short original notes are retained;
the bundle does not redistribute papers or third-party generated summaries.

## Search families and representative queries actually issued

| Family | Queries | Outcome |
|---|---|---|
| Broad noise/weights | `differential privacy noise weight space geometry parameter perturbation model utility representation learning DP-SGD paper`; `federated learning differential privacy noise injected gradients model parameters utility weight perturbation analysis paper` | Broad direction already populated; identified GeoDP and optimization work. |
| Recommendation | `differentially private recommender systems user-level DP factorization noise analysis item embeddings paper`; `"federated recommendation" "noise" "user embeddings" differential privacy` | PrivateRec and private MF require comparison; no claim that private recommenders are new. |
| Local-state mechanism | `"differential privacy" "persistent" "local state" personalization`; `"personalized federated" "noise propagation"`; `"federated" "privacy noise" "local" "memory"` | No exact reset-based recommender diagnostic located; substantial personalization prior art found. |
| Dynamics | `"differentially private" "noise" "trajectory" "Hessian"`; `"Correlating Cross-Iteration Noise"`; `"DiSK" "Differentially Private" arxiv` | NoiseCurve and DiSK block generic trajectory/denoising novelty. |
| Clipping / established MF | `"Federated Matrix Factorization with Privacy Guarantee"`; `"Understanding Clipping for Federated Learning"` | Whole-user/rating distinction and embedding clipping already established. |
| Gauge confound (after E2) | `"LoRA" "gauge" "noise"`; `"federated" "Procrustes" "LoRA"`; `"differential privacy" "gauge"` | Identified FLoRG, FedRot-LoRA and PRISM; ruled out alignment itself as novel. |
| Coupling / attribution (after E2b) | `"differential privacy" "synchronous coupling" optimization`; `"noise attribution" "LoRA"`; `"common random numbers" "gauge"`; `"counterfactual" "noise" "federated" "personalization"` | No exact matching recommender reset study located; keyword absence is not evidence of priority. General coupling and counterfactual ideas are established. |
| Final specificity check | `"matrix factorization" "synchronous coupling"`; `"differential privacy" "counterfactual" "noise attribution"`; `"LoRA" "common random numbers"`; `"personalized federated learning" "noise propagation" "local"` | Results included general personalized alignment, unrelated LoRA applications and unverified generated research pages. No new exact-match primary source identified; those generated pages are not used as evidence. |
| Verification | `"FedGSA" "Geometry-Consistent"`; `"FedRot-LoRA" site:arxiv.org`; `"PRISM" "Gauge-Invariant" "LoRA"` | Located primary papers instead of relying on secondary search summaries. |
| Foundational citations | `site:arxiv.org "BPR: Bayesian Personalized Ranking"`; `site:proceedings.mlr.press "Communication-Efficient Learning of Deep Networks"`; `site:arxiv.org "Numerical Composition of Differential Privacy" Gopi` | Verified report bibliography. |

The full closest-work table records the reading depth for each source. Broad
search results include unsuitable or weakly specified DP work; their presence
is not evidence that their privacy claims are valid. We neither import those
guarantees nor use them to support our own claims. Failure to find an exact
match with a few keyword families is a weak novelty signal.

The search changed the investigation: the first pilot's large Two-factor
trajectory differences triggered an alignment control; the associated search
then uncovered additional close prior art. This adaptation is documented in
E2b's separate pre-output protocol rather than presented as preplanned E2.

## E3/E4 continuation — 2026-10-03

The following further searches preceded the ranking-tail screen or accompanied
the larger-data replication. Author/proceedings sources supplied the technical
evidence. Unrelated results and automatically generated summaries were rejected.

| Focus | Queries issued | Outcome |
|---|---|---|
| Practical private recommendation | `"differentially private" recommendation noise popularity personalized`; `"differential privacy" "recommender" "debias" noise`; `"differentially private" "matrix factorization" "regularization" noise user`; `"private recommendation" "popularity" "residual"` | No basis for claiming a new generic residual or popularity method. |
| Noise and ranking tails | `"LoRA" "ranking" "noise" "Gaussian"`; `"differential privacy" "bilinear" "tail"`; `"product of" "normal" "noncentral chi-square" difference distribution`; `"recommendation" "noise" "margin" "differential privacy"` | Found Gaunt's known distributional identity and Urmian et al.'s noisy-ranking stability paper; both constrain the candidate. |
| Personalized stability | `"differential privacy" "personalization" "instability"`; `"federated recommendation" "spectral" "noise"`; `"federated reconstruction" "differential privacy" recommendation`; `"private" "matrix factorization" "user embeddings" "clipping"` | Screened Fed-LPAR, Federated Reconstruction and embedding-clipping work. Local refitting/regularization is not a new idea. |
| Exact-match tail check | `"LoRA" "noise" "noncentral"`; `"differential privacy" "recommendation" "rank flips"`; `"matrix factorization" "Gaussian" "ranking stability"`; `"factor" "noise" "bilinear" "ranking" differential privacy` | Many unrelated LoRa radio and generic ranking results. No exact matching primary study located; this is not evidence of novelty. |
| Broader tail check | `"low-rank adaptation" "noise" "tail" differential privacy`; `"recommender" "noncentral" Gaussian`; `"matrix factorization" "rank" "perturbation" "probability" privacy`; `"recommendation" "Gaussian approximation" "privacy"` | Primary-domain filters used for technical evidence. Existing perturbation and probability theory makes an elementary calculation insufficient. |
| Coupling consequence | `"counterfactual" "coupling" "neural" "noise" identifiability`; `"common random numbers" "parameterization" neural`; `"federated" "noise" "Procrustes" "counterfactual"` | General stochastic sensitivity/coupling literature surfaced. No demonstrated external scientific consequence for our particular diagnostic yet. |
| Dataset | `site:grouplens.org datasets movielens 1m` | Verified official archive, checksum and research-use README. Stored local inputs outside the report. |

Full primary HTML was inspected for Gaunt's theorem/sign-probability treatment
and Urmian et al.'s assumptions, method and experiments. PRISM was rechecked.
An attempt to access the MovieLens ACM DOI page returned403; its existing
author PDF and official dataset README provide the citation/provenance.
The empirical E4 gate subsequently failed, so absence of an exact-match paper
was not used to rescue an unsupported novelty claim.

The continuation then inspected DPALS's proceedings algorithm, practical
modifications and supplementary privacy proof, and DP-CMF's primary HTML
algorithm/privacy/evaluation. Following its bibliography identified
[Multi-Task DP Under Distribution Skew](https://arxiv.org/abs/2302.07975);
the primary PDF's Algorithm 1, Theorem 3.3 and proof were read. This adds a
direct overlap constraint on item-frequency-aware privacy allocation.
Searches for `"Private Alternating Least Squares"` combined with `correction`,
`erratum`, `github`, and `"2σ" privacy`, plus
`site:github.com/google-research "dpam" "privacy"` and `"2107.09802" "code"`,
did not establish a checked implementation or a published correction. Several
direct HTML/code links were unavailable. No prior-paper error is claimed.
