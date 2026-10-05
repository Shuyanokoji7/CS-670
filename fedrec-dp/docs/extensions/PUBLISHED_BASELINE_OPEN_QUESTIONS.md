# Internal baseline-audit questions — not research findings

3 October 2026. These are implementation checks to resolve before coding a
published private ALS comparator. Keep them distinct from the completed
E3/E4 results and do not present them as paper novelty or an accusation.

1. The extracted text of DPALS's clipping definition displays a maximum where
   projection onto a norm ball requires a minimum. Confirm the rendered PDF
   and author implementation before copying notation. The browser screenshot
   result did not expose viewable image content in this session. An extraction
   artifact or a typographical issue is not an algorithmic discovery.
2. For our own hypothetical release of two independent noisy sufficient
   statistics, whiten their joint contribution before accounting. In a scalar
   example with two unit sensitivities and independent SD sigma, releasing
   both coordinates has Gaussian RDP alpha/sigma^2, not alpha/(2*sigma^2).
   This is elementary Gaussian composition. It does not by itself disprove
   a bound for a subsequently projected/inverted/factor-normalized output.
3. The DPALS supplementary proof and the later task-weighted theorem should
   be reconciled at the level of the exact mechanism and constants before a
   faithful implementation. The later paper explicitly composes matrix and
   vector releases. Private/public task counts and regularization scaling
   also need to match the specified adjacency. Do not silently assume all
   variants share one noise multiplier.

Sources inspected:
- https://proceedings.mlr.press/v139/chien21a/chien21a.pdf
- https://proceedings.mlr.press/v139/chien21a/chien21a-supp.pdf
- https://arxiv.org/pdf/2302.07975

No authors were contacted, no issue was filed, and no result was published.
If the implementation uses a conservative independently derived calibration,
label any deviation from the published algorithm and compare fairly.
