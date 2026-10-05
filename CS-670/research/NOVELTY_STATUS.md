# Current novelty assessment — 3 October 2026

**Paper-level novelty is not yet established.** Substantial controlled evidence
exists, including a diagnostic limitation replicated on a second dataset. No
improved method or new privacy theorem has been demonstrated. The user has
authorized autonomous work toward a paper, with an open research direction;
this does not justify relaxing novelty or evidence standards.

| Proposed claim | Evidence now available | Defensible assessment |
|---|---|---|
| Fewer noisy coordinates improve recommendation | B4 has four positive adjusted comparisons but many losses; E1 controls tuning. | Conditional benchmark result, not a general principle. |
| Fixing one factor is a new solution | Prior FFA-LoRA/FedASK and E1's zero positive primary contrasts. | Known baseline; unsupported as our improvement. |
| The product of factor noises explains the failures | Small E1 energy share; E4 finds negligible Gaussian ranking-risk error in a fixed sample. | Rejected as the demonstrated explanation in these states. |
| Noise causes large persistent local-personalization damage | Large E2 raw distances; E3 aligned/no-balancing distances are small. | Broad interpretation rejected. |
| Paired noise-attribution measurements can be sensitive to factor-basis changes | Null example, E2/E2b, and E3 controls on two datasets. | Supported narrow diagnostic finding; paper novelty still unconfirmed. |
| Our private learner beats a strong simple reference | E1 and transferred E3 means trail user-level DP popularity in the relevant regimes. | Not established. |

The most defensible present contribution is a careful benchmark and a
diagnostic caution: a shared random seed does not ensure corresponding
effective perturbations across changing factor bases. Existing Procrustes,
gauge geometry and stochastic coupling ideas already cover the constituent
tools. We have not established that another published recommendation result
changes when this control is applied, or that this diagnostic chooses a useful
intervention better than simple alternatives. That missing consequence is the
main obstacle to a methods-paper claim.

## Next work that would materially change this assessment

1. Audit the published private ALS baseline and implement a faithful comparator
   before interpreting weak 50-round BPR utility as a fundamental privacy limit.
   Privacy unit, implicit-feedback objective and communication must be stated.
   The [implementation audit](PUBLISHED_BASELINE_AUDIT.md) now records the
   primary algorithm/privacy passages and the requirements for this comparison.
2. Establish metadata-only local personalization as a zero-shared-learning
   control. Then assess whether a small private collaborative residual adds
   value under a fixed total communication/privacy budget, especially for
   low-activity users and cold items. Public features and residual learning
   themselves are known; a specific unexpected boundary or mechanism is needed.
3. Retain the coupling result as a possible analysis contribution only if it
   changes a consequential conclusion beyond our own pilot. Freeze that target
   before further computation; more similar probes are insufficient.

The ML-1M test is still unscored, so a future genuinely frozen comparison can
reserve it. Validation has been inspected and cannot be described as untouched.
Do not select whichever new experiment looks best and call it confirmatory.
The formal course report can already present this complete, honest research
trajectory, including rejected hypotheses.
