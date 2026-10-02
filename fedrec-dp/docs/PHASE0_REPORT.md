# Privacy-Preserving Federated Recommendation Systems

### Improving the Privacy–Utility–Communication Trade-off through Low-Dimensional Shared Representations

**Report 1: Research Setup (Phase 0)**
28 September 2026

---

## Summary

This report covers the first phase of the project. No recommendation model has been trained
yet. No federated training, differential privacy mechanism or low-rank model has been
implemented. What has been built is the experimental foundation that all later models will
be measured on. That foundation includes:

- a fixed MovieLens-100K data pipeline;
- a chronological train/validation/test split that is frozen and fingerprinted;
- a candidate-set definition that avoids recommending items the user has already rated;
- a single full-ranking evaluator with unit-tested metrics;
- two sanity baselines;
- a written threat model.

The aim was to make sure that when the actual comparisons start, any difference between
methods comes from the methods themselves and not from differences in data handling or
evaluation.

---

## 1. Introduction

Recommender systems learn from what users do: which items they click, watch, buy or rate.
That interaction history is exactly what makes the recommendations useful, and it is also
personal. A viewing history can reveal a lot about someone, even when each individual
rating looks harmless.

Federated learning is one way to reduce this exposure. Instead of sending raw interaction
data to a central server, each user's device trains on its own data and sends back only
model updates. This keeps the raw data local, but it does not by itself make the system
private. Model updates are computed from the user's data and can leak information about it,
and a server that sees each update individually can try to infer things from them. For that
reason the project plans to add differential privacy (DP) at the user level, which gives a
formal bound on how much the trained model can reveal about any one user.

DP comes at a cost. Noise has to be added to the shared updates, and that noise lowers
recommendation quality. Federated learning also has its own cost in communication, since
every participating client has to send an update each round. The project's longer-term
question is whether the *size* of the shared representation affects how these costs trade off
against each other.

This report does not attempt to answer that question yet. Phase 0 was about building a setup
where the question can be answered fairly and reproducibly. In particular, every future model
has to be trained and evaluated on exactly the same data split and scored by exactly the same
code. Without that, a small improvement from one method could just as easily be caused by a
different split or a slightly easier evaluation.

---

## 2. Research Question and Planned Study

The research question is:

> **Can reducing the dimensionality of privacy-sensitive shared representations improve the
> privacy–utility–communication trade-off in differentially private federated recommender
> systems?**

The working hypothesis is that, **under the same formal DP guarantee**, a lower-dimensional
shared representation may give a better balance between recommendation quality and
communication cost when the privacy noise is high enough.

The intuition goes like this. In a federated recommender, some parameters are shared between
clients and the server (for example, item representations), and each client sends updates to
them. If the shared representation is large:

- the model has more capacity to fit user preferences;
- each client has to send more numbers per round;
- there are more shared coordinates that receive DP noise.

If it is small:

- communication is cheaper;
- the model has less capacity;
- there are fewer coordinates for the noise to spread over, which *might* make the model
  behave better under heavy noise.

So it is possible that an intermediate size works best once strong privacy is required. It is
also possible that it does not, and the full-size representation wins at every privacy level.
The evaluation has been designed so that either outcome can be detected. The research log
writes down in advance what result would count as evidence against the hypothesis.

One point needs to be stated clearly: **lowering the dimension does not automatically give a
smaller ε.** Under the standard Gaussian mechanism, the privacy guarantee depends on the
clipping norm, the noise multiplier, the client sampling rate, the number of rounds and the
privacy accountant. It does not depend on how many coordinates the update has. The planned
experiments will therefore fix the same (ε, δ), computed with the same accounting framework
and settings, and compare utility and communication across dimensions at that fixed
guarantee. Any advantage for a smaller representation would have to appear as better
recommendations at the same ε, not as a better ε.

The planned comparisons (referred to in the repository as B0–B4) are centralised
recommendation, federated recommendation, DP federated recommendation, low-rank federated
recommendation and low-rank DP federated recommendation. None of these exist yet.

---

## 3. Threat Model

The full threat model is in `docs/THREAT_MODEL.md`. It describes what later phases have to
implement. It is not a description of current code, since no federated or private training
exists yet.

### Federated client

Each user is one federated client. User *u* owns a private dataset
D_u = {(i, r_ui, t)}: the movies they rated, the rating and the time. The whole history is
treated as private.

### Private information

Two things are meant to stay on the user's device: the raw interaction history D_u, and the
user's own representation p_u (the user vector in a factorisation model), which will be stored
and updated locally. What leaves the device, in later phases, is a clipped update to the
*shared* parameters. The size of that shared part is what the project will vary.

### Server

The server is assumed to be **honest-but-curious**. It runs the protocol correctly, but it may
try to learn about users from anything it receives. Malicious servers or clients that deviate
from the protocol are out of scope.

### User-level differential privacy

The privacy unit is the whole user, not a single rating. Two datasets count as neighbours if
they differ by **one user's entire contribution**. Rating-level privacy would only hide
individual ratings, but a user's profile is revealed by the combination of their ratings. What
we want to protect is whether a particular person took part in training at all. This is
stronger, and it means each user's total contribution per round has to be bounded (by clipping
their update) and accounted for across all rounds they participate in.

### Secure aggregation

The threat model assumes the server never sees an individual client's update, only the sum
of the clipped updates, as if they were combined by a secure aggregation protocol. **In this
project secure aggregation is assumed/simulated, not cryptographically implemented.** In
practice this will mean the server-side code only ever receives the aggregate. There is no
actual cryptographic protocol, no handling of clients that drop out, and no protection against
a server that breaks the rules.

Secure aggregation and differential privacy do different jobs:

- **Secure aggregation** stops the server from looking at any single user's update directly.
- **Differential privacy** limits how much the aggregate, and the global model trained from it,
  can reveal about any one user's participation.

Secure aggregation alone gives no formal bound on what the aggregate leaks, which is why DP is
needed on top of it.

**No DP guarantee is claimed at this stage**, because DP has not been implemented.

---

## 4. Dataset Selection

The project uses **MovieLens-100K**, the official GroupLens release. The raw data, as read by
the pipeline, contains:

| | Count |
|---|---|
| Users | 943 |
| Items (movies) | 1,682 |
| Ratings | 100,000 |

The SHA-256 hash of the ratings file (`u.data`) is pinned in the code and checked every time the
data is loaded, so a modified or different version of the file is rejected. The raw files are
only ever read. A check before and after preprocessing confirmed they were unchanged.

MovieLens-100K was chosen for a few reasons:

- **It is a fixed benchmark.** Unlike `ml-latest-small`, which is regenerated periodically, the
  100K release does not change, so results stay comparable over the course of the project.
- **It is small enough for repeated experiments.** Private federated experiments need many runs
  across privacy levels, dimensions and random seeds, and ranking against the whole catalogue
  is cheap at this size.
- **Each user maps naturally onto a federated client.** Every rating belongs to one user.
- **It has timestamps**, which makes a chronological split possible.

The loader has also been written so that MovieLens-1M can be added later. It will only be used
as a second dataset to check whether the findings generalise, and only if the main pipeline
works on 100K. It is not used anywhere in the current results.

---

## 5. Implicit-Feedback Formulation

The project treats recommendation as a **Top-K ranking** problem rather than rating
prediction. In practice, a recommender's job is to decide which few items to show a user, not
to predict whether they would give a film 3.4 or 3.7 stars. Ranking metrics measure the first
task directly. The implicit-feedback formulation also fits the pairwise models planned for
later phases.

Ratings are converted to implicit feedback with a simple rule: **a rating of 4 or 5 counts as a
positive interaction.** This gives 55,375 positive interactions out of the 100,000 ratings.

Ratings of 1, 2 or 3 are handled in two different ways, and it is worth being precise about
this:

- They are **not positives**. They are not used as positive training signal and can never be a
  validation or test target. They are also not treated as explicit negatives, since a rating of
  3 does not clearly mean "dislike".
- They **do** count as *previously observed*. If a user has already rated a movie, even badly,
  they have clearly seen it. Recommending it later as if it were new would be unrealistic, so
  these items are removed from that user's future candidate sets (Section 8).

In short, "positive" and "already seen" are two separate ideas in the pipeline.

The ≥ 4 threshold is a modelling assumption, and it is listed as one in the limitations.

---

## 6. Train, Validation and Test Split

The data is split per user and in chronological order, using a **leave-two-out** scheme. For
each user, their positive interactions are sorted by time and then assigned as follows:

- the latest positive goes to **test**;
- the second-latest positive goes to **validation**;
- all earlier positives go to **training**.

A user needs at least three positives for this to work (one each for training, validation and
test), so users with fewer than three positives are removed. A higher threshold was considered
and rejected. It would mostly remove light users, who have little data and are exactly the users
likely to suffer most under DP noise, so dropping them would bias the study toward heavy users.

In practice only **one user was removed**: raw user ID 685, who has no ratings of 4 or above at
all. Items were not filtered. All 1,682 movies stay in the catalogue.

The resulting split:

| | Count |
|---|---|
| Users before filtering | 943 |
| Users retained | 942 |
| Users removed | 1 (0 positives) |
| Training positives | 53,491 |
| Validation positives | 942 |
| Test positives | 942 |

Training positives per user:

| Mean | Median | Min | Max |
|---|---|---|---|
| 56.78 | 37.5 | 1 | 376 |

Two retained users have only a single training positive. They are kept.

The split is checked automatically every time it is built or loaded. The checks are:

- no validation or test item appears in that user's training positives;
- the validation and test items are different;
- every user has at least one training positive;
- the ordering train → validation → test holds for every user.

If any check fails, the pipeline stops with an error.

Hyperparameters, including the choice of dimensionality, will be tuned on validation only. The
test set will be used once per final configuration. Models will not be retrained on
training + validation before testing.

---

## 7. Timestamp Tie Handling

MovieLens timestamps record when a rating was *entered*, not when the movie was watched. Users
often rated many films in one sitting, so many ratings share the same second. This matters for
the split. For **305 users** the validation and test positives have the same timestamp, and for
**316 users** the validation positive shares its timestamp with the last training positive. For
these users, the timestamp alone cannot decide which item is "latest".

**Within the same second there is no reliable ordering.** The data simply does not say which of
two tied ratings came first.

The first version of the pipeline broke ties by item ID. That was deterministic, but it imposed
a systematic order (lower IDs always "earlier"), and item IDs are not random. They could be
related to things like when a film was added to the catalogue. So this was replaced.

Ties are now broken by a fixed pseudo-random key. For each rating, the key is computed from a
SHA-256 hash of (user ID, item ID, split seed), with **split_seed = 2026**. Each user's ratings
are ordered by timestamp first and by this key only when timestamps are equal.

Some properties of this approach:

- **It is reproducible.** SHA-256 gives the same value on every machine and every run. Python's
  built-in `hash()` would not, because it is randomised per process.
- **Model-training seeds cannot change the split.** The split seed is stored separately in
  `configs/dataset.yaml`, and preprocessing never reads the training seed. A test rebuilds the
  split under several different training seeds and checks that it comes out identical.
- **Only tied interactions are affected.** Changing the tie-break changed the validation and/or
  test item for **315 of the 942 users** (284 validation targets and 205 test targets). A test
  confirms that every one of these changes happened at a timestamp tie.

This tie-break should not be read as true chronology. For tied ratings the order is arbitrary
but fixed, which is the best that can be done with this data. For roughly a third of users,
"the most recent positive" really means "one of the positives from their most recent rating
session".

After this change the split was regenerated and **frozen** for all future models. A SHA-256
fingerprint of its contents
(`6faed6fc3d47b5b6fa638adfeea83cd7409d50c39fa01f85c10379d0daef6989`) is recorded in the results
and checked in the test suite. Any change to the split will make the tests fail.

---

## 8. Candidate-Set Construction

At evaluation time, each user's held-out item has to be ranked against a set of candidate
items. The question is which items should count as candidates. The rule used is:

> Candidates = all 1,682 movies, minus every movie the user rated (with **any** rating)
> **before** the held-out event. The held-out target itself is always kept.

For **validation**, this removes everything the user rated before the validation event. That
includes their training positives and any low ratings from the same period.

For **test**, it removes everything the user rated before the test event. That includes all of
the above, plus the validation item and any low ratings between validation and test.

"Before" uses the same per-user ordering as the split, including the same tie-break. Ratings
that happen *after* an event are never used to filter its candidates. For example, the future
test item stays in the validation candidate set, because at validation time the user had not
rated it yet. Using later ratings to filter earlier candidate sets would be a form of leakage.

Low-rated items are removed for the reason given in Section 5: the user has already seen them.
Leaving them in would ask the model to "recommend" films the user has already watched, and a
model could be penalised for ranking them highly even though, in a real system, they would
never be shown.

The resulting candidate set sizes per user:

| | Validation | Test |
|---|---|---|
| Mean | 1,581.6 | 1,579.1 |
| Median | 1,621 | 1,619 |
| Min | 1,001 | 1,000 |
| Max | 1,676 | 1,671 |

On average about 100 previously rated items are removed per user (100.4 for validation, 102.9 for
test). About 44 (validation) and 45 (test) of those are movies rated below 4, which would have stayed in the candidate
set if only previous *positives* were removed.

---

## 9. Cold Targets

Some held-out targets are movies that never appear as a positive in the training data. No user
has a training positive for them, so a collaborative model has little to learn about them. In
the current split there are:

- **6** cold validation targets;
- **14** cold test targets.

These were **kept on purpose**. They are hard cases, and removing them would raise every model's
score for reasons that have nothing to do with the model. It would also make the benchmark less
honest about how the system performs on less popular items. If needed, a warm-target-only
analysis can be reported later as a secondary diagnostic, but the main evaluation will always
include all targets.

---

## 10. Evaluation Protocol

Evaluation uses **full ranking**. For each user, the model scores every item in their candidate
set (on average about 1,580 movies), and we look at where the held-out target lands.

A common shortcut is to rank the target against a small random sample of negatives, for
example 1 positive plus 99 random items. This was deliberately avoided:

- Ranking against 100 items is much easier than ranking against about 1,600, so scores come out
  inflated.
- The result depends on which negatives happened to be sampled.
- Sampled metrics have been shown to sometimes change the relative order of models (Krichene &
  Rendle, 2020), which is exactly what this project needs to measure correctly.

Two further details of the evaluator:

- **Score ties.** If other candidates get exactly the same score as the target, they are counted as
  ranked above it. This is a pessimistic choice, so a model that outputs constant or very coarse
  scores cannot get credit by luck.
- **One evaluator for everyone.** Every future model, B0 through B4, will be scored by the same code
  (`src/evaluate.py`), on the same 942 users, with the same targets and candidate sets. No model
  gets a different protocol.

---

## 11. Evaluation Metrics

All metrics are computed per user and then averaged over the 942 users. Standard errors are
reported alongside the means. Since each user has exactly one held-out item, the metrics are
simple functions of the target's rank *r* (1 = top of the list).

### NDCG@10 (primary metric)

NDCG@10 asks whether the held-out movie appears in the top 10, and rewards it more the closer it
is to the top. With one relevant item it is:

- 1 / log2(r + 1) if r ≤ 10,
- 0 otherwise.

So rank 1 scores 1.0, rank 2 scores about 0.63, rank 10 scores about 0.29, and anything below the
top 10 scores 0. It is the primary metric because it reflects both *whether* a relevant item was
recommended and *how prominently*.

### Hit Rate@10

HR@10 is 1 if the target is anywhere in the top 10, and 0 otherwise. It ignores position within
the top 10, so a hit at rank 1 and a hit at rank 10 count the same.

### Recall@10

Recall@10 is the fraction of a user's relevant items that appear in the top 10. It is implemented
generally, for any number of relevant items. Here, though, each user has exactly one held-out
item, so **Recall@10 is identical to HR@10**. It is reported for completeness but should not be
treated as a second, independent piece of evidence.

### MRR@10

MRR@10 is 1/r if the target is in the top 10, and 0 otherwise. Like NDCG it rewards higher
positions, but it drops off faster: rank 2 already only gets 0.5.

### Testing the metrics

All metrics are unit-tested on hand-constructed cases:

- rank 1 gives NDCG = 1;
- rank 2 gives NDCG = 1/log2(3);
- rank 11 gives NDCG = HR = 0;
- rank 5 gives HR = Recall = 1;
- multi-item recall and NDCG cases are checked.

The fast rank-based versions are checked against straightforward list-based implementations.

---

## 12. Sanity Baselines

Two simple baselines were run through the evaluator to check that the pipeline behaves
sensibly and to give a lower reference point for later models.

- **Random.** Every user's candidates are ranked at random (seed 42).
- **Popularity.** Items are ranked by how many positive *training* interactions they have across
  all users. Validation and test data are not used to compute popularity, and a test checks this.

| Model | Split | NDCG@10 (± SE) | HR@10 = Recall@10 | MRR@10 |
|---|---|---|---|---|
| Random | Validation | 0.0023 ± 0.0009 | 0.0064 | 0.0011 |
| Random | Test | 0.0038 ± 0.0015 | 0.0085 | 0.0024 |
| Popularity | Validation | 0.0464 ± 0.0056 | 0.0839 | 0.0351 |
| Popularity | Test | 0.0443 ± 0.0052 | 0.0870 | 0.0314 |

The random baseline works as a check on the evaluator. With about 1,580 candidates, a random
ranking should put the target in the top 10 roughly 10 / 1,580 ≈ 0.6% of the time. The exact
expected HR@10 given each user's candidate count is 0.0064. The observed values (0.0064 on
validation, 0.0085 on test) are within one standard error of that.

Popularity is more than ten times better than random on NDCG@10 (about 12× on test). This is expected, since a
non-personalised popularity ranking is a known reasonable baseline on MovieLens, but the
absolute numbers are still low. For comparison, under the earlier protocol, before already-rated
low-scored items were removed from the candidates, popularity scored 0.029 test NDCG@10. The
increase is most likely because popular films the user had already seen and rated poorly no
longer take up top-10 slots.

These numbers set a floor. A learned model, private or not, should clearly beat popularity. A DP
model that drops to popularity level would effectively have lost its personalisation.

---

## 13. Reproducibility and Testing

Reproducibility was a main goal of this phase, not an afterthought.

- **One configuration file.** Everything that affects preprocessing and evaluation is in
  `configs/dataset.yaml`: the dataset, the positive threshold, the split type, the minimum
  number of positives, the split seed, K, the candidate rule and the tie policy.
- **Deterministic output.** Running preprocessing and the baselines twice from scratch produces
  byte-identical processed files and result files.
- **Raw data protection.** The raw ratings file hash is checked on every load, and the raw
  directory was verified unchanged after all runs.
- **Frozen split.** The split's content fingerprint is pinned in the tests.
- **Research log.** `RESEARCH_LOG.md` is append-only. Every protocol decision is recorded with its
  reason, and the two corrections made during this phase (the tie-break and the candidate rule)
  were logged as a separate dated entry rather than edited into the original. Both corrections
  were made before any learned model existed, so they could not have been influenced by model
  results.

The test suite has **43 tests, all passing**. They cover:

- the metric definitions;
- the evaluator, including candidate exclusion, tie handling, and that a perfect scorer gets
  NDCG = 1 while a constant scorer is not rewarded;
- the split: leakage checks, chronological order, determinism under row shuffling and different
  training seeds, tie-break behaviour, the pinned fingerprint;
- synthetic leaky splits, to check that the validation actually catches them.

The whole phase can be reproduced with:

```bash
cd fedrec-dp
source .venv/bin/activate
python -m src.data
python -m src.baselines random
python -m src.baselines popularity
python -m pytest
```

### Statistical plan for later comparisons

This was decided now, before any model results exist. Headline results will be averaged over
several model-training seeds (at least five). Because every method is evaluated on the same 942
users and the same candidate sets, comparisons between methods will be **paired**: the
per-user difference between two methods will be analysed, with paired bootstrap confidence
intervals over users. This is more sensitive and more appropriate than comparing two means with
overlapping error bars. It matters here because the standard errors on MovieLens-100K are fairly
large; popularity's NDCG@10 has a standard error of about 0.005.

---

## 14. Limitations

- **Small dataset.** 942 evaluation users gives noisy metrics. Effects of the size likely in this
  study may be close to the noise level, which is why multiple seeds and paired analysis are
  planned.
- **The ≥ 4 threshold is an assumption.** Other thresholds would give a different set of
  positives.
- **Same-second ordering is unknown.** For about a third of users, the split boundary falls
  inside a tie and is decided by the fixed pseudo-random tie-break, not by real chronology.
- **Per-user split, not a global time split.** One user's training data can come later in calendar
  time than another user's test item, so some "future" popularity information is available during
  training. This is standard practice for leave-one-out style splits, but it is slightly
  optimistic. A global time split was deliberately not used.
- **One held-out item per user.** This makes HR@10 and Recall@10 identical and limits how much each
  user contributes to the evaluation.
- **Secure aggregation is simulated/assumed**, not cryptographically implemented.
- **Evaluation is centralised.** Offline evaluation is a research tool and is not part of the
  privacy protocol.
- **No privacy guarantee yet.** Differential privacy has not been implemented, and nothing in this
  phase should be read as a privacy claim.

---

## 15. Next Steps

The next phase is B0, a centralised (non-federated, non-private) recommendation model trained
on the frozen training split and evaluated with the existing evaluator. It will act as the
upper reference for the federated and private variants that follow. Every later model will use
the same split, candidate sets, metrics and statistical plan described in this report.

---

## 16. Conclusion

Phase 0 produced a fixed, tested and documented experimental base rather than any model
results. The main outcomes are:

- a chronological MovieLens-100K split that is reproducible and frozen, with an honest tie-break
  for same-second ratings;
- a candidate definition that separates "liked" from "already seen";
- a full-ranking evaluator shared by every future method;
- baseline numbers that confirm the evaluator behaves as expected;
- a threat model that is explicit about what is assumed (secure aggregation) and what is not yet
  provided (differential privacy).

With this in place, the later comparison between full-size and low-dimensional shared
representations under a fixed privacy budget can be carried out on equal terms. It should
produce a result that can be trusted whichever way it turns out.

---

### Reference

W. Krichene and S. Rendle. *On Sampled Metrics for Item Recommendation.* KDD 2020.
