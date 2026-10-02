# Threat Model

Status (2026-10-02): **federated training (B1) and the DP mechanism (B2: whole-client clipping, Gaussian noise on the clipped sum, PRV accounting) are implemented in simulation. Secure aggregation is still ASSUMED, not implemented.** Original status note: no federated training, secure aggregation or
differential privacy exists in the codebase yet. Nothing below is a claim about
the current code. It is the model that later phases must implement and be judged against.

## Federated client

Each user `u` is one federated client and owns a private dataset

    D_u = {(i, r_ui, t)}

where `i` is an item (movie), `r_ui` the rating and `t` the timestamp. The user's
complete interaction history is private.

## Local private information

The following never directly leave the client:

1. the raw interaction history `D_u`;
2. the local user representation `p_u` (stored and updated only on the device).

What *does* leave the client (in later phases) is a clipped update to the **shared**
parameters (e.g. item-side representations). Reducing the dimensionality of that
shared representation is the object of study.

## Server

The server is **honest-but-curious**: it follows the protocol correctly but may try
to infer information about users from everything it receives (aggregates, model
states across rounds, participation metadata).

## Differential privacy level: user-level DP

**Neighbouring datasets:** `D` and `D'` are neighbouring if they differ by the
addition or removal of **one complete user's contribution** (all of that user's
interactions).

Rating-level (event-level) adjacency is *not* the primary definition. It would only
hide a single rating, while a user's behavioural profile (which movies they
watch, and when) is revealed by the *combination* of their ratings. The question we
want to protect is *whether an entire user participated in federated training*,
which requires bounding each user's whole contribution per round (clipping the
user's update) and accounting for it across all rounds in which they participate.

The formal guarantee `(ε, δ)` will be determined by the mechanism (Gaussian noise
added to the clipped aggregate), the clipping norm, client sampling rate, number of
rounds, noise multiplier and the privacy accountant. **The dimensionality of the
shared representation does not by itself change ε under the standard Gaussian
mechanism**, because sensitivity is set by the L2 clipping norm, not by the number
of coordinates. Dimensionality can change *utility* and *communication* at a fixed ε,
and that is the research question.

## Secure aggregation (assumed / simulated)

The DP threat model assumes the server does **not** observe individual client
updates. Conceptually, clipped client updates are combined by secure aggregation:

    Δ_1, Δ_2, ..., Δ_n  ──secure aggregation──▶  Σ_u Δ_u

**Limitation:** in this project secure aggregation is **simulated/assumed, not
cryptographically implemented**. The simulation simply means the code path that
represents the server only receives the sum. There is no cryptographic protocol,
no dropout handling, and no protection against a server that deviates from the
protocol. Results must not be described as having secure aggregation implemented.

| Mechanism | What it provides |
|---|---|
| Secure aggregation | Prevents the server from directly observing any individual client update; it sees only the aggregate. |
| Differential privacy | A formal `(ε, δ)` bound on how much the released aggregate (and thus the trained global model) can reveal about any single user's participation. |

They are complementary. Secure aggregation alone gives no formal bound on what the
*aggregate* leaks. DP alone, with noise added by the server (central DP), would require
trusting the server with individual updates. Secure aggregation removes that need
under the assumption above.

Where the noise is added (by the trusted aggregation step versus distributed across
clients) is to be decided in the DP phase. The accounting must match whichever is
chosen.

## Architecture

```text
 User Device (client u)
 ------------------------------
 Private interactions D_u      (never leave)
 Private user vector p_u       (never leaves)
 Local training
   -> update Δ_u to SHARED params
   -> clip: ||Δ_u||_2 <= C
            |
            | clipped client update
            v
 Secure Aggregation            (ASSUMED / SIMULATED, not cryptographic)
            |
            | aggregate only: Σ Δ_u
            v
 DP Mechanism                  (Gaussian noise calibrated to C; accountant tracks ε, δ)
            |
            v
 Server Global Model           (honest-but-curious; sees only noisy aggregate)
```

## Out of scope (for now)

- Malicious (actively deviating) servers or clients, Sybil and poisoning attacks.
- Leakage through participation metadata (who was sampled, timing, update sizes).
- Privacy of the *evaluation* procedure: offline evaluation here is centralised
  over held-out data and is a research instrument, not part of the deployed protocol.
- Inference from the released global model by third parties beyond what DP bounds.

## B2 implementation note: simulator vs protocol (added 2026-10-02)

- **What the protocol assumes the server observes:** per round, only the noisy aggregate
  Σ̃_t = Σ_{u∈S_t} clip(ΔQ_u, C) + N(0, σ²C² I). Secure aggregation is assumed to hide individual clipped updates.
  Where the noise is generated (a trusted aggregation step, or distributed across clients) is abstracted away. The
  accounting only requires that Σ̃_t is what gets released.
- **What the simulator does:** for convenience it builds every client's ΔQ_u in one process, clips it, and sums the
  results. Individual updates exist in memory only as an implementation detail. That is **not** a claim that a real server
  could not see them, and no cryptographic secure aggregation is implemented.
- **Released output and guarantee:** the sequence of noisy aggregates, and hence the global Q, is covered by user-level
  (ε, δ)-DP under add/remove-one-user adjacency, for the fixed hyperparameters (q, T, C, σ, η_s, local training rule). Each user's
  local p_u and their own recommendations are computed on-device from public Q and the user's own data. They are not
  released.
- **Not covered:** hyperparameter selection (the clipping grid came from non-private B1 statistics, and C and η_s were chosen on
  validation), validation monitoring, the earlier non-private baselines, and the noise RNG. The RNG is a seeded PCG64 stream,
  chosen for reproducibility and not cryptographically secure.
