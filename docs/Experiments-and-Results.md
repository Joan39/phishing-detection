# Experiments and Results

All experiments run on the PhiUSIIL URL dataset under **domain-held-out**
partitions (training and evaluation never share a registrable domain).
Phishing is the positive class throughout.

## 1. Experiment catalog

| Module | Purpose | Output |
| --- | --- | --- |
| `backend.train_models` | Production classical training (dataset + hard-negative corpus) | `models/url/*.joblib` |
| `backend.quantum_live` | Train/adapt the live 4-qubit VQC | `models/quantum_live.joblib` |
| `backend.quantum_ensemble_experiment` | Classical+quantum ensemble evaluation | `results/quantum_ensemble_experiment.json` |
| `backend.three_way_evaluation` | Domain-disjoint LR vs ensemble vs VQC vs hybrid comparison | `results/three_way_evaluation.json` |
| `backend.quantum_rl_configuration_experiment` | Quantum-weight × RL-policy matrix under drift | `results/quantum_rl_configuration_experiment.json` |
| `backend.adaptation_experiment` | Threshold bandit and drift simulator (library + standalone run) | `results/adaptation_experiment.json` |
| `scripts/quantum_live_eval.py` | Held-out evaluation of a live quantum artifact | console |
| `scripts/realworld_battery.py` | Live 56-URL real-world set through `/api/scans` | `results/realworld_battery.json` |
| `scripts/fresh_probe.py` | Second, never-tuned 18-URL set (live and offline) | `results/fresh_probe.json` |
| `scripts/legit_negative_experiment.py` | Hard-negative corpus configurations, scored on fold + both real-world sets | `results/legit_negative_experiment.json` |
| `scripts/realworld_offline_score.py` | Score deployed artifacts offline on both real-world sets, hybrid or classical-only, `--policy` to re-score under a candidate policy | console, `--output` JSON |
| `scripts/novel_eval.py` | Never-trained-on technique strings and unseen legitimate hosts, offline with the live fusion, cap and bandit cut | `results/novel_eval.json` |
| `scripts/live_worked_examples.py` | The documented worked URLs measured through the running server | `results/live_worked_examples.json` |
| `scripts/fit_diagnostics.py` | Train-versus-held-out accuracy, log-loss, Brier and reliability deciles, plus the four-feature capacity control | `results/fit_diagnostics.log` |
| `scripts/a_lowbin_hybrid.py` | The family of rows where model A is confidently legitimate, and how many become hybrid misses | `results/a_lowbin_hybrid.summary.txt` |
| `scripts/build_full_report_docx.py` | Rebuild the complete poster/presentation report from the artifacts above | `docs/Research-Project-Poster_and_Presentation_Full_Report.docx` |
| `scripts/build_presentation_answers_docx.py` | Rebuild the six form answers on their own | `docs/Research-Project-Presentation-Form-Answers.docx` |
| `scripts/production_benchmark.py` | Refit the live recipe on a domain-held-out fold and score the full stack; the source of the Model Comparison panel | `results/first_experiment.json` |
| `scripts/adaptive_replay.py` | Static-policy vs adaptive-policy arms over a simulated technique shift, plus a refit arm | `results/adaptive_replay_experiment.json` |
| `scripts/replay_feedback_policy.py` | Rebuild the adaptive policy from the feedback rows already in the database (`--dry-run` supported) | `results/adaptive_policy.json` |
| `backend/adaptive_policy.py` | Per-class thresholds, component re-ranking, PSI drift, review routing (live state) | `results/adaptive_policy.json` |
| `scripts/generate_system_report.py` | Rebuild the .docx system report | `reports/…Report.docx` |

Run any of them with `python -m backend.<name> [--output path]`.

## 2. Deployed stack on domain-held-out rows

`scripts/production_benchmark.py` produces the only numbers the Model Comparison
panel shows. Protocol: `StratifiedGroupKFold` (5 splits, seed 42) on registrable
domain, fold 0 held out — 47,158 rows over 43,817 domains, none of which appear
in the 175,286 training domains. Models A and B are refit for the benchmark on
the fold-0 training rows plus the full legitimate hard-negative corpus (7,712
corpus URLs × 10 and 170 bare HTTP origins × 60 = 87,320 rows; 275,957 rows in
total), which is the recipe `fit_production_models` applies to the live models;
`models/` is never written. The quantum column is the deployed artifact, whose
own training rows are recomputed from its circuit metadata and asserted disjoint
from the scored rows. Everything is scored through the live path — 0.35/0.65
classical fusion, 0.75/0.25 hybrid fusion, the trusted-domain cap, the
redirect-payload scoring of §7.1, and the bandit threshold 0.40 — so these are
deployment numbers, not a separate research split. The fusion weights here are
the documented static defaults; the learned policy weights in
`results/adaptive_policy.json` only diverge from them after analyst feedback, so
they are measured separately in §7.2 and §7.3 rather than folded into this table.
`python -m scripts.production_benchmark --policy <file>` scores the same split
with the weights and per-class thresholds recorded in that policy file instead,
and writes the weights actually used into the report.

| Component | Accuracy | Precision | Recall | F1 | ROC-AUC | Legit FPR |
| --- | --- | --- | --- | --- | --- | --- |
| A — logistic regression | 0.8741 | 0.8533 | 0.8526 | 0.8529 | 0.9091 | 0.1098 |
| B — soft-voting ensemble | 0.9848 | 0.9892 | 0.9751 | 0.9821 | 0.9965 | 0.0079 |
| Classical, weighted (A 0.35 / B 0.65) | 0.9765 | 0.9731 | 0.9720 | 0.9725 | 0.9937 | 0.0201 |
| Quantum circuit alone | 0.6297 | 0.5821 | 0.4788 | 0.5254 | 0.6585 | 0.2573 |
| **Hybrid live (0.75 classical + 0.25 quantum)** | **0.9674** | **0.9542** | **0.9704** | **0.9622** | **0.9923** | **0.0349** |

At a 0.50 decision point the hybrid row reads 0.9617 accuracy, 0.9539 F1 and
0.0109 false-positive rate. Three readings follow:

- Model A is the weak link and is down-weighted for exactly that reason; the
  ensemble carries the classical path.
- The quantum term costs held-out accuracy (0.9765 → 0.9674, FPR 0.0201 →
  0.0349) and buys real-world behaviour: corpus training is what removed the
  0.14 probability floor described in §6.3. On this split it is a diversity and
  calibration term, not a competitor to B.
- Held-out numbers are optimistic relative to live traffic: the same stack
  reaches 0.731 / 0.833 on the two real-world legitimate sets (§6.2) while
  keeping phishing at 1.000, because ordinary content-shaped URLs are rarer in
  the corpus than in the wild.

Earlier protocol note: the first benchmark artifact came from
`backend.train_models.run_experiment`, which trains on dataset rows only with 18
features (no `IsTrustedRoot`) and scores at 0.50. It reported A 0.9964 and B
0.9975 — numbers measured on a distribution the deployed models no longer see.
`/api/models/comparison` now labels any artifact without a `training_recipe`
block as a legacy run instead of presenting it as deployed accuracy, and
`scripts.production_benchmark` overwrites it with the table above.

## 3. Quantum component (live artifact)

Configuration: 4 qubits, `ZZFeatureMap` + `RealAmplitudes` (2 reps, 12
parameters), StatevectorSampler, 512 shots, COBYLA 60 iterations, 1,000
balanced training rows per class — plus, in the deployed artifact, 1,500
legitimate corpus rows added to that pool.

Held-out accuracy of the live model, before and after strengthening the
configuration, and after the corpus change:

| Metric | Before (1 rep, 256 shots) | After (2 reps, 512 shots) | Corpus-trained (deployed) |
| --- | --- | --- | --- |
| Accuracy | 0.548 | 0.634 | 0.596 |
| F1 (phishing) | 0.515 | 0.550 | 0.512 |
| ROC-AUC | 0.602 | 0.707 | 0.637 |
| Real-world AUC (74 URLs) | — | 0.570 | 0.746 |

Held-out numbers are fold-1, 500 balanced rows (`scripts/quantum_live_eval.py`);
the deployed column is the same circuit trained with 1,500 legitimate
counter-examples added, which trades dataset-distribution accuracy for
real-world discrimination (§6.5).

The VQC alone remains far weaker than the classical ensemble; its role is
ensemble diversity inside the 75/25 blend, not standalone detection.

## 4. Quantum ensemble × RL configuration matrix

Seed 42; classical trained on 188,637 rows, quantum on 2,000 balanced rows,
balanced 1,000-row test subset. Drift axis: phishing test rows made
legitimate-looking (shorter, less obfuscated); bandit labels revealed in
64-row batches.

| Quantum weight | Static F1 (clean) | Static F1 (drift) | RL F1 (drift) | RL AUC | RL FPR |
| --- | --- | --- | --- | --- | --- |
| 0.00 (classical only) | 0.9919 | 0.9889 | 0.9940 | 0.9999 | 0.0040 |
| 0.15 | 0.9919 | 0.9889 | 0.9940 | 0.9999 | 0.0040 |
| 0.25 (live default) | 0.9909 | 0.9889 | 0.9940 | 0.9993 | 0.0060 |
| 0.40 | 0.9909 | 0.9889 | 0.9910 | 0.9987 | 0.0120 |

Findings:

- Under drift the fixed 0.50 threshold misses 10–11 phishing rows; the bandit
  recovers them (F1 0.9889 → 0.9940) at ≤2 false positives, converging to the
  0.30 action in 15 of 16 batches.
- Quantum weights up to 0.25 preserve classical clean performance and match
  the classical-only adaptive F1; 0.40 degrades both policies.
- Fit cost on CPU simulation: quantum 883 s vs classical 27 s.

## 5. Production false-positive fix

Dataset artifact found in live verification: legitimate rows are nearly all
bare domains (mean length 27, 100% HTTPS, p99 length 39) while phishing rows
average length 46 with parameters, so models learned "long URL with query =
phishing" and scored ordinary site pages at ~1.0. The fix is the legitimate
hard-negative corpus described in `docs/Architecture.md` §4, built in three
measured stages (§6 below): brand-root pages, then non-brand hosts crossed with
path shapes, then long content-shaped paths and heavier plain-HTTP weighting.

## 6. Real-world generalization

Held-out fold metrics cannot see this failure mode, because the fold draws
legitimate rows from the same bare-domain distribution the models were trained
on. Two curated real-world sets are scored instead, at the live decision
threshold 0.40, with hosts deliberately disjoint from every training URL:

| Set | Legitimate | Phishing |
| --- | --- | --- |
| `scripts/realworld_battery.py` | 26 ordinary sites (news, docs, gov, university, bare HTTP origins) | 30 lookalike/link-farm URLs |
| `scripts/fresh_probe.py` | 12 further sites, including long dated article slugs | 6 campaign-style URLs |

The probe set exists because the battery drove the corpus design: a gain only on
the battery could be tuning rather than generalization.

### 6.1 Corpus configurations (classical path, offline, fold 0)

`python -m scripts.legit_negative_experiment` fits models A and B on fold-0
training rows plus one candidate corpus and scores the held-out fold, the battery
and the probe. Quantum is excluded (it trains on dataset rows only), so the
classical path is the variable.

| Corpus | fold acc | legit FPR | phishing recall | battery legit | probe legit | phishing sets |
| --- | --- | --- | --- | --- | --- | --- |
| 402 brand-root URLs × 250 | 0.9849 | 0.0156 | — | 0.3846 | — | 1.0000 |
| + 85 hosts × 30 paths × 10 | 0.9785 | 0.0217 | — | 0.7308 | — | 1.0000 |
| + bare HTTP origins × 60 | 0.9787 | 0.0195 | — | 0.7692 | — | 1.0000 |
| + bare HTTP origins × 200 | 0.9686 | 0.0228 | — | 0.8077 | — | 1.0000 |
| **adopted: 44 path shapes, bare × 60** | **0.9748** | **0.0201** | **0.9680** | **0.7692** | **0.8333** | **1.0000** |
| adopted, corpus × 16 instead of × 10 | 0.9725 | 0.0260 | 0.9706 | 0.7692 | 0.7500 | 1.0000 |
| adopted, no HTTP path variants | 0.9830 | 0.0134 | 0.9781 | 0.6538 | 0.7500 | 1.0000 |

Readings:

- More repetitions of the same shapes do not buy generalization: ×16 lowered
  probe accuracy (0.833 → 0.750) and raised the held-out false-positive rate.
- Path *shape* diversity is what generalizes. The 14 long content shapes (dated
  news slugs, multi-segment docs paths, hyphenated article titles) left the
  battery unchanged at 0.7692 and moved the never-tuned probe to 0.8333.
- Plain-HTTP weighting is a real trade, not a free win: 200 replicas raised the
  battery to 0.8077 but cost 0.6 held-out accuracy points, so 60 is deployed.
- No configuration below let a phishing URL through: both phishing sets stay at
  1.0000, and held-out phishing recall stays above 0.968.

### 6.2 Live hybrid before/after

The same URLs scored through `POST /api/scans` (classical fusion, 4-qubit VQC at
0.25 weight, trusted-domain cap, bandit threshold 0.40):

| Stage | battery legit | of which HTTP | probe legit | battery phishing | probe phishing |
| --- | --- | --- | --- | --- | --- |
| Brand-root corpus only (baseline) | 0.3846 | 0.2500 | — | 1.0000 | — |
| + 85 hosts × 30 paths × 10 | 0.6923 | 0.5000 | 0.5000 | 1.0000 | 1.0000 |
| + 44 path shapes, bare HTTP × 60 | 0.7308 | 0.5000 | 0.6667 | 1.0000 | 1.0000 |
| + quantum retrained on the same corpus | 0.7308 | 0.5000 | 0.7500 | 1.0000 | 1.0000 |
| current live stack (November 9) | 0.7308 | 0.5000 | 0.8333 | 1.0000 | 1.0000 |

The last row is `python -m scripts.realworld_battery` and `python -m
scripts.fresh_probe` against the running server, i.e. the deployed artifacts as
they now stand after the circuit retrain and the feedback retrains, with the
adaptive policy inside its cold-start bounds (§7.2), so it is still the
documented static blend. One probe URL that used to score legitimate now clears
the threshold; the guardrail is not what moved it — §7.1 measures zero
probability change on these 74 rows.

`python -m scripts.realworld_offline_score` reproduces these numbers offline
from the deployed artifacts, and matched the live server row for row; the gap
between §6.1 (classical only) and §6.2 (hybrid) is entirely the quantum term,
quantified in §6.3.

### 6.3 The quantum floor

The live VQC reads four lexical features only (`URLLength`, `DomainLength`,
`NoOfSubDomain`, `SpacialCharRatioInURL`) and is near-chance on real-world
traffic: AUC 0.5702 over both sets, mean 0.562 on legitimate versus 0.572 on
phishing. A 0.25-weight component that outputs ~0.56 for everything adds a
~0.14 probability floor to every scan, which is enough to push borderline
legitimate URLs over the 0.40 threshold — it costs two of twelve probe URLs and
one of twenty-six battery URLs.

### 6.5 Quantum retraining on the legitimate corpus

`python -m backend.quantum_live --hard-negatives 1500` mixes 1,500 evenly
sampled legitimate counter-examples into the circuit's training pool (1,000
balanced dataset rows per class + 1,500 corpus rows, target 0), and
`backend.quantum_live.adapt_live_model` re-adds them on every
feedback-triggered retrain so the fix cannot be silently undone. Both the
circuit configuration and the row count are recorded in the artifact under
`circuit.extra_rows`.

| Metric | dataset-trained VQC | corpus-trained VQC |
| --- | --- | --- |
| Real-world quantum-only AUC (74 URLs) | 0.5702 | 0.7463 |
| Real-world mean P(phishing), legitimate | 0.562 | 0.410 |
| Real-world mean P(phishing), phishing | 0.572 | 0.597 |
| Held-out fold-1 accuracy (500 rows) | 0.6340 | 0.5960 |
| Held-out fold-1 F1 / ROC-AUC | 0.5504 / 0.7069 | 0.5121 / 0.6370 |
| Live probe legitimate accuracy | 0.6667 | 0.7500 |
| Live battery legitimate accuracy | 0.7308 | 0.7308 |
| Live battery legitimate mean probability | 0.327 | 0.300 |
| Live phishing accuracy (both sets) | 1.0000 | 1.0000 |

This is a trade, not a free win: the circuit gives up held-out accuracy on the
dataset's own distribution — where legitimate rows are bare domains, exactly the
pattern the corpus contradicts — and buys real discrimination on traffic that
looks like ordinary use. The deployed artifact is the corpus-trained one,
because the failure it fixes is the one users hit. `en.wikipedia.org/wiki/Main_Page`
makes the effect concrete: its quantum term fell from 0.586 to 0.211, so it now
scores 0.221 on its own and no longer depends on the trusted-domain cap to stay
clear of the threshold.

### 6.4 Operating point on the real-world sets

The same deployed artifacts, classical path only, scored at fixed cut-offs
(`python -m scripts.realworld_offline_score`, sweep printed per set):

| Threshold | battery legit | battery phishing | probe legit | probe phishing |
| --- | --- | --- | --- | --- |
| 0.40 (live bandit action) | 0.7308 | 1.0000 | 0.7500 | 1.0000 |
| 0.45 | 0.7692 | 1.0000 | 0.8333 | 1.0000 |
| 0.50 | 0.8077 | 1.0000 | 0.8333 | 1.0000 |
| 0.55 | 0.9231 | **0.9667** | 0.8333 | 1.0000 |

Up to 0.50 the classical score separates both sets completely on this traffic —
legitimate accuracy rises for free — and 0.55 starts to cost phishing recall
(`http://paypa1.com/secure/update` falls under it). The deployed threshold is
not pinned by hand: the bandit selected 0.40 from analyst feedback, and the
hybrid floor in §6.3 is what makes the low end of that range necessary in the
live path. Offline classical-only numbers are also slightly stricter than the
§6.1 experiment rows, because production fits every dataset row while the
experiment holds one domain group out.

## 7. Adaptation to unseen techniques

Two layers, measured separately: a guardrail that closes a bypass family with no
retraining (§7.1) and the adaptive policy that learns per-class thresholds and
component weights from analyst feedback (§7.2). Both sit inside the live scoring
path described in §2 of How-Detection-Works, so they are deployment behaviour,
not research scaffolding.

### 7.1 Open-redirect guardrail

Attack shape: a clean, often trusted brand host carrying the credential-capture
URL as a parameter payload, e.g.
`https://www.google.com/url?sa=t&url=https://refund-claim-irs-com/tax` and
`https://l.facebook.com.l.php?u=https://secure-appleid-verify.com/login`. The
outer morphology is legitimate-looking, so before the guardrail both fell below
the 0.40 cut and novel-technique recall on 14 curated never-trained-on strings
was 0.8571 (12 of 14).

`backend/url_features.redirect_targets` reads the query and fragment pairs
(`parse_qs`, so percent-encoded payloads decode), keeps only absolute `http(s)`
or protocol-relative targets whose registrable domain differs from the outer
host, and ignores same-brand hops
(`accounts.google.com/Login?continue=https://mail.google.com/mail/`) and
non-URL values (`q=url`, `/local/path`, `javascript:`, `data:`). Two consequences
in `scan()`: the trusted-domain cap refuses to fire on a trusted host that
carries such a payload, and `score_redirect_targets` scores the payload itself
through the same fusion and lets the maximum win — the analyst sees the
highest-risk URL in the chain, redacted, under
`explanations["open_redirect_chain"]`.

Measurements with the deployed artifacts at threshold 0.40:

- Novel phishing recall 0.8571 → **1.0000**. The two bypasses now score 0.882
  and 0.659; nothing else in that set moved because nothing else carries a
  foreign payload.
- The 74 battery+probe rows: **0 probabilities moved**, verdicts identical
  (battery 19/26 legit and 30/30 phishing, probe 10/12 legit and 6/6 phishing
  either way). The guardrail is a strict addition of reach, not a re-tune of the
  existing traffic.
- Unseen legitimate traffic is untouched at 0.7143 (5 of 7); its two misses are
  content-shaped URLs, the URL-only-features limitation in §8.
- On 8,000 randomly sampled held-out fold-0 rows scored twice from one set of
  component readings — with and without the payload step, so the delta is the
  guardrail alone and not 512-shot noise — 11 rows carry a foreign redirect
  target, exactly 1 probability moved (0.830 → 0.876, a phishing row already
  caught), and accuracy / F1 / FPR / recall are identical to four decimals
  (0.9741 / 0.9700 / 0.0269 / 0.9755).
- `python -m scripts.production_benchmark` runs the same chain over the 47,158
  held-out rows (`redirect_chain_probabilities`), so §2's hybrid row already
  includes it.

### 7.2 Static policy vs adaptive policy (replay)

`scripts/adaptive_replay.py` streams one URL sequence through three arms.
Protocol: 6 cycles of the 56-URL real-world battery (336 rows of traffic the
models already see), then 21 shift rows (14 novel technique strings + 7
legitimate hosts absent from the corpus), then a 20-row holdout (12 strings never
used anywhere + 8 legitimate). Every row is labelled with its true class, which
is the optimistic part of the setup: a real analyst mislabels and delays.

| Arm | familiar legit | familiar phishing | shift legit | shift phishing | holdout legit | holdout phishing |
| --- | --- | --- | --- | --- | --- | --- |
| static policy (documented weights + global 0.40) | 0.7308 (flat, 6 cycles) | 1.0000 | 0.8571 | 1.0000 | 0.7500 | 1.0000 |
| adaptive policy (bandit contexts + re-ranking) | 0.7308 → 0.8846 by cycle 4 | 1.0000 | 0.8571 | 0.9286 | 1.0000 | 1.0000 |
| classical refit on the shift labels | — | — | — | — | 0.8750 | 1.0000 |

The adaptive arm's trajectory is the interesting part: legitimate accuracy is
0.7308 on the first familiar cycle, dips to 0.6923 on the second, then climbs to
0.8462 and settles at 0.8846 for cycles 4–6, while phishing stays at 1.0000 in
every familiar cycle. By the end of the stream the per-class bandit had moved
`https/public` (227 samples) from 0.40 to 0.50, left `http/public` (114 samples),
`https/trusted` (24) and `http/trusted` (12) at 0.40, and the multiplicative-
weights re-ranker had drifted classical from 0.35/0.65 to 0.508/0.492 with
quantum 0.25 → 0.2897 (component accuracies A 0.9125, B 0.9045, quantum 0.6897)
— full confidence, because 95 distinct URLs had been labeled by then. 26 of the
377 rows were flagged for analyst review on component disagreement or
near-threshold scores.

The exchange rate is honest and worth stating: the same raise of the cut that
recovers the legitimate rows cost one shift phishing row —
`https://bit.ly/3xKp9Qm`, 0.576 offline, falls under 0.50 — so shift phishing
recall goes 1.0000 → 0.9286. Mean legitimate probability barely moves (0.2934 →
0.2874) while mean phishing probability falls 0.846 → 0.828; the gain is the
threshold, not the score.

Refitting the classical models on the shift labels recovers holdout legitimate
to 0.8750 without the adaptive arm's 1.0000 and costs far more compute (24.3 s
to fit 147,341 rows at a reduced, not-leakage-controlled budget), which is why
the policy layer is the first response and a retrain is the second.

**The label-free drift detector did not fire.** Over the whole 377-row stream the
maximum PSI against the 200-row reference histogram was 0.0831 — 0.0780 at the
moment the shift arrived — against a 0.25 trigger. Fourteen novel strings inside
336 rows of familiar traffic do not move a 10-bin probability histogram
noticeably. So in this replay the adaptation is carried by the reward-driven
mechanisms (per-class thresholds, component re-ranking) and the PSI trigger is
inert; it is kept because it is the only signal that needs no labels at all, but
its current calibration is documented as a limitation in §8.

### 7.3 Backfilling the policy from stored feedback, and what learning costs

The layer went live after the labels. `results/qphish.db` holds 25 feedback rows,
22 of them definite labels, while the live policy file carried exactly one
labeled sample when this was written: the system was adaptive in structure and
static in fact. `scripts/replay_feedback_policy.py` closes that gap by running
the stored labels through the same update functions the live path uses
(`update_context`, `update_component_weights`, `observe_distribution`), so
history is not thrown away. `--dry-run` prints the live state next to the
replayed one and the per-class threshold each would choose; the default output is
the live policy file, which is the install step.

Replaying is not automatically good. Those 22 definite labels cover only 10
distinct URLs, and they are easy ones: the legitimate set scores 0.06-0.25 and
the phishing set 0.74-0.94, so on this evidence model A is right 22 times out of
22, model B 19, and the circuit 9. The re-ranker reads that as "A is the useful
model" and moves classical from the documented 0.35/0.65 to 0.6331/0.3669 with
quantum 0.25 → 0.10, on one week of traffic no held-out row resembles.
`MIN_EVIDENCE_URLS = 20` distinct labeled URLs now shrink the learned weights
toward the documented blend (`weight_confidence` is that ratio, capped at 1.0),
so a small backlog moves them part of the way; 10 distinct URLs on file is half
an opinion, hence the gated replay lands at 0.4916/0.5084 and quantum 0.175.
Both replayed policies leave all four context bandits at 0.40 — the largest is
`https/public` with 14 samples and a $q$-value of 0.36 at 0.40 against −0.41 at
0.70 — so the thresholds are not what changes here.

Measured with `--policy` on `scripts.production_benchmark.py` (fold-0 held-out,
47,158 rows) and on `scripts.realworld_offline_score.py` (both curated sets at the
0.40 cut), same artifacts, same rows, only the policy swapped:

| policy in force | held-out classical | held-out hybrid | F1 | phishing recall | legit FPR | battery legit | battery phishing | probe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| documented static (A 0.35 / B 0.65, quantum 0.25) | 0.9765 | **0.9674** | **0.9622** | **0.9704** | **0.0349** | 0.7308 | 1.0000 | 0.8333 |
| replayed backlog, ungated (A 0.6331 / B 0.3669, quantum 0.10) | 0.9329 | 0.9320 | 0.9207 | 0.9223 | 0.0608 | 0.8077 | 1.0000 | 0.8333 |
| replayed backlog, gated (A 0.4916 / B 0.5084, quantum 0.175) | 0.9560 | 0.9562 | 0.9494 | 0.9584 | 0.0454 | 0.7692 | 1.0000 | 0.8333 |

The gate buys two-thirds of the safety back and gives up about half the
real-world gain: held-out accuracy −3.54 → −1.12 points, false-positive rate
+2.59 → +1.05 points, while legitimate accuracy on the 26-URL battery set goes
+7.7 → +3.8 points. Phishing recall on the battery and the probe stays 1.0000 in
all three rows, so the cost is entirely on the dataset's own distribution, where
A is the weakest learner (0.8741 alone against B's 0.9848, §2) and a weight
vector derived from ten easy URLs is a bad trade. The battery gain is exactly one
row, the CERN page verified against the running server below. The two probe
misses are policy-independent at the live cut:
`https://www.smithsonianmag.com/science-nature/` reads 0.5728
static and 0.5243 gated, `http://www.ctext.org/ctext/zh/3` 0.5562 and 0.5749 —
the weights move them, but never across 0.40. Only the threshold reaches those
rows, which is why the bandits rather than the re-ranker are the lever for them.

This is why the live policy is still at the documented defaults. Installing the
gated replay is a deliberate offer, not a default: it trades a 1.2-point drop in
held-out phishing recall for a 3.8-point gain on one legitimate battery, and the
recall floor set for this project is 0.96. Run
`python -m scripts.replay_feedback_policy` when the backlog is bigger and less
repetitive, and re-measure with the two `--policy` commands above before shipping
it; the numbers in this table are the ones that decided it this time.

The swap itself was verified against the running server rather than only
offline: with the gated file in place, `POST /api/scans` on
`https://home.cern/science/computing/large-hadron-collider` returned 0.3883
(A 0.0297, B 0.6957, VQC 0.4824 — 0.825 × (0.4916·A + 0.5084·B) + 0.175 × VQC)
and verdict legitimate; restoring the documented blend returned 0.4676 and
verdict phishing on the same URL, with the report page naming the weights it
actually used in both cases. The battery miss this exposes is one row of the 0.73
legitimate accuracy in §6.2, and the same URL is why the report flags
low confidence: A and the VQC disagree by 67% around the cut.

## 8. Limitations

- No demonstrated quantum advantage; the VQC contributes diversity only. Its
  held-out fold-0 accuracy (0.6297, §2) is far below the classical path by
  design, and the corpus retraining moved it further down on the dataset's own
  distribution while raising real-world discrimination (§6.5) — the two
  distributions are not the same problem.
- Drift is simulated, not observed production traffic.
- The label-free PSI trigger is currently insensitive: in the §7.2 replay it
  never fired (max 0.0831 against a 0.25 trigger) even though a 14-string
  technique change arrived inside the window. Adaptation in that experiment came
  from analyst rewards, not from unsupervised detection, and a real wave would
  need to be much larger — or the window/trigger recalibrated — before the drift
  path matters.
- The adaptive policy needs labels to learn. The replay gives every row its true
  class, so the observed gains assume complete, correct analyst coverage; with
  partial or delayed feedback the per-class bandits and the re-ranker stay in
  their cold-start state and the system behaves like the static arm.
- The re-ranker treats every labeled URL as equally informative. It counts
  component accuracy per labeled URL with no weight for difficulty or
  redundancy, so a backlog of ten distinct, easy-to-separate URLs — where model A
  happened to be right on all 22 labels — supports a weight vector the 47k-row
  held-out split contradicts (§7.3). The evidence-volume gate caps how far one
  such backlog can move the weights; it does not make the evidence
  representative, and nothing in the layer yet asks whether it is.
- The redirect guardrail only sees payloads inside the scanned string. A real
  HTTP 3xx chain, a `<meta http-equiv=refresh>` page or a JavaScript
  `location.href` handoff is not followed, so an open redirect whose payload is
  delivered by the server rather than the URL still scores as the outer host.
- Feedback retraining runs in background worker threads (full production
  budget since November 2026, so live models never silently downsize).
- Features are URL-morphology only (no host reputation or certificate data), so
  a legitimate 68-character dated slug and a phishing one of the same shape are
  not separable by these features; the remaining misses are exactly that case
  (`http://neverssl.com`, `http://calpoly.edu`, `http://example.org/path`).
- Plain-HTTP URLs on hosts absent from the hard-negative corpus still score
  high: dataset legitimate rows are 100% HTTPS, so HTTP is a genuine learned
  signal, not only an artifact. Weighting the bare HTTP origins 60× recovers
  about half of the legitimate HTTP traffic in the battery (0.25 → 0.50) with no
  phishing regression; 200× recovers more but costs held-out accuracy, so it was
  not adopted.
- Both real-world sets are curated, not sampled traffic: 74 URLs measure
  generalization behavior, not a production confusion matrix.
