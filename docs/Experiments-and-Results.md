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
| `scripts/realworld_offline_score.py` | Score deployed artifacts offline on both real-world sets, hybrid or classical-only | console |
| `scripts/generate_system_report.py` | Rebuild the .docx system report | `reports/…Report.docx` |

Run any of them with `python -m backend.<name> [--output path]`.

## 2. Classical detectors (domain-held-out fold)

Fold protocol: `GroupShuffleSplit` on domain, 10% held out (23,661 rows),
seed 42; each model is trained on the remaining 90% of PhiUSIIL **plus the
realistic legitimate hard-negative corpus**, then scored at the 0.5 decision
point.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| A — logistic regression | 0.9261 | 0.9369 | 0.8825 | 0.9089 | 0.9614 |
| B — soft-voting ensemble | 0.9909 | 0.9963 | 0.9818 | 0.9890 | 0.9967 |
| Classical, equal average | 0.9717 | 0.9822 | 0.9495 | 0.9656 | 0.9947 |
| **Classical, weighted (A 0.35 / B 0.65)** | **0.9884** | **0.9914** | **0.9807** | **0.9860** | **0.9959** |

The weighted row is the live fusion in `backend/main.py`
(`CLASSICAL_WEIGHTS`, overridable via `QPHISH_CLASSICAL_WEIGHTS`). Weighting
by held-out strength adds +1.7 accuracy points over equal averaging, mainly by
curbing model A's false positives.

## 3. Quantum component (live artifact)

Configuration: 4 qubits, `ZZFeatureMap` + `RealAmplitudes` (2 reps, 12
parameters), StatevectorSampler, 512 shots, COBYLA 60 iterations, 1,000
balanced training rows per class.

Held-out accuracy of the deployed live model, before vs after strengthening
the configuration:

| Metric | Before (1 rep, 256 shots) | After (2 reps, 512 shots) |
| --- | --- | --- |
| Accuracy | 0.548 | 0.634 |
| F1 (phishing) | 0.515 | 0.550 |
| ROC-AUC | 0.602 | 0.707 |

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

## 7. Limitations

- No demonstrated quantum advantage; the VQC contributes diversity only, and on
  real-world traffic it currently costs accuracy (§6.3).
- Drift is simulated, not observed production traffic.
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
