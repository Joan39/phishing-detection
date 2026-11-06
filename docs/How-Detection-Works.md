# How Detection Works

This document explains, step by step, how QPhish decides whether a URL is
**phishing** or **legitimate**. It complements
[Architecture.md](Architecture.md), which describes the system layout.

Everything here operates on the URL **string only**. QPhish never opens,
fetches, or renders the submitted website.

## 1. The signal: URL morphology

Phishing pages must borrow the look of a trusted brand while living on an
attacker-controlled address. That pressure leaves traces in the URL itself —
long random domains, brand names in the wrong position, deep subdomain
chains, numeric paths, percent-encoded credentials, IP hosts, missing HTTPS.
Legitimate site traffic rarely accumulates all of those at once.

QPhish turns this observation into 19 numeric features
(`backend/url_features.py`):

| Feature group | Features | What it catches |
| --- | --- | --- |
| Size | `URLLength`, `DomainLength`, `TLDLength` | Random long domains and stuffed paths used to bury the real host |
| Host structure | `NoOfSubDomain`, `IsDomainIP`, `IsTrustedRoot` | Lookalike chains like `paypal.com.login-verify.xyz` and raw-IP hosts |
| Encoding | `HasObfuscation`, `NoOfObfuscatedChar`, `ObfuscationRatio` | `%40`-style tricks that hide credentials or redirect targets |
| Character mix | `NoOfLettersInURL`, `LetterRatioInURL`, `NoOfDegitsInURL`, `DegitRatioInURL` | Digit-heavy token domains (`paypa1`, `secure-bank-24`) |
| Query shape | `NoOfEqualsInURL`, `NoOfQMarkInURL`, `NoOfAmpersandInURL`, `NoOfOtherSpecialCharsInURL`, `SpacialCharRatioInURL` | Prefilled login parameters and unusual punctuation |
| Transport | `IsHTTPS` | Absence of TLS, historically correlated with phishing kits |

These are the raw material every model consumes. No word lists, no blocklists,
no network calls.

## 2. Model A — logistic regression

The 19 features are median-imputed, standardized (zero mean, unit variance),
and passed to a logistic regression trained with `class_weight="balanced"`.
The model learns one weight per feature; the decision is a linear boundary in
standardized feature space:

```
P(phishing) = σ(w·x + b)
```

Because the boundary is linear, A is stable and rarely confidently wrong, but
it under-uses feature interactions (e.g., "long *and* hyphen-heavy *and* no
HTTPS together"). On the domain-held-out fold it reaches 0.9261 accuracy.

## 3. Model B — soft-voting ensemble

B wraps three detectors and averages their probabilities with weights 1:2:2:

- the same logistic pipeline as A (linear view),
- a 200-tree random forest (axis-aligned interactions, "which combinations
  of thresholds co-occur"),
- a 200-iteration gradient-boosting stage (sequentially corrects the
  remaining errors, the strongest non-linear fitter of the three).

The forest and boosting stages let B capture the "several mediocre signals at
once" pattern A misses, at 0.9909 held-out accuracy and 0.9967 ROC-AUC.

**Classical fusion.** A and B are combined as
`P(classical) = 0.35·A + 0.65·B`. The weights follow the held-out ranking:
B dominates every metric, so it carries most of the vote; A stays in to keep
a simple linear reference that limits overconfident tree behavior.

## 4. The quantum model — a 4-qubit variational classifier

The VQC (`backend/quantum_live.py`) reads four complementary features:
`URLLength`, `DomainLength`, `NoOfSubDomain`, `SpacialCharRatioInURL`.

1. **Scaling** — each feature is standardized to [0, π] so it can encode an
   angle.
2. **Feature map** — `ZZFeatureMap` rotates four qubits by those angles and
   entangles neighbours with ZZ gates. The encoded state depends on
   *products* of the input values, not just sums: interaction information is
   physically built into the amplitudes.
3. **Variational circuit** — a `RealAmplitudes` ansatz (two repetitions,
   12 trainable rotation parameters) reshapes the state.
4. **Measurement** — 512 shots on the exact statevector sampler; the fraction
   of `1` outcomes maps to `P(phishing)`.
5. **Training** — COBYLA optimizes the 12 angles to separate the two classes
   on 1,000 balanced URL rows per class **plus 1,500 legitimate counter-example
   rows from the hard-negative corpus**. The corpus rows matter: on the dataset
   alone the four features teach "long URL with subdomains = phishing", because
   every legitimate dataset row is a bare domain, and the circuit then answers
   ~0.56 for all real traffic.

On CPU simulation the VQC is far weaker than the classical side standalone
(≈0.60 accuracy on held-out domains) — its role is *ensemble diversity*: its
errors come from a different hypothesis class, so blending it smooths the
classical probability rather than duplicating it. What it must not do is act as
a constant; measured on real-world URLs the corpus-trained circuit separates the
two classes at AUC 0.746, where the dataset-trained one managed 0.570.

## 5. Hybrid fusion

```
P(hybrid) = 0.75 · P(classical) + 0.25 · P(quantum)
```

`QUANTUM_WEIGHT` is capped at the 0.25 level validated in the quantum × RL
configuration matrix: at 0.25 the blend matches classical-only adaptive F1
under drift while keeping diversity; at 0.40 both policies degrade.

## 6. Trusted-domain calibration

A raw learned score can still overreact to realistic-but-unusual legitimate
traffic. Before the verdict, URLs whose host is under a configured trusted
root (e.g. `google.com`, `wikipedia.org`) are checked for *impersonation
markers*: IP host, `%XX` obfuscation, unusual special characters before the
fragment, or more than 2 subdomains. A clean trusted URL has its probability
capped at 0.25 — no trusted brand can be reported as phishing by morphology
alone, while `paypal.com.account-verify.info` style lookalikes fail the check
and keep their full score.

## 7. Adaptive decision threshold

```
verdict = phishing  if  P(hybrid) ≥ threshold
```

The threshold is not fixed. An epsilon-greedy bandit (`backend/main.py`,
state in `results/rl_state.json`) chooses among the actions
0.30 / 0.40 / 0.50 / 0.60 / 0.70. Each analyst feedback record is a reward:
correct verdict +1, wrong verdict −1, applied with a 0.2 learning rate to the
action that was in force. When new phishing waves arrive dressed as
legitimate traffic, the tuned threshold *drops* (the drift experiment settled
on 0.30 in 15 of 16 batches), catching attacks a static 0.50 cutoff misses.

## 8. Continuous adaptation

Every labeled correction:

1. updates the bandit immediately (threshold policy),
2. queues a full-budget classical retrain including the new example
   (background thread),
3. triggers periodic quantum retraining once 10+ new labels accumulate,
4. hot-reloads the new artifacts by mtime — no restart, no downtime.

## 9. A worked example (observed live)

`https://paypal-secure-refund.claim-center.net/login`

| Stage | Value | Why |
| --- | --- | --- |
| Model A | 0.977 | Brand-hyphen pattern + long host dominate the linear model |
| Model B | 0.979 | Trees agree; the domain is not a trusted root |
| Classical fusion | 0.978 | 0.35·0.977 + 0.65·0.979 |
| Quantum VQC | 0.492 | Encoded length/domain/subdomain state is undecided here |
| Hybrid | 0.857 | 0.75·0.978 + 0.25·0.492 |
| Calibration | n/a | Host is a lookalike chain, safeguard does not apply |
| Threshold | 0.40 (bandit) | 0.857 ≥ 0.40 → **phishing**, risk level *high* |

Two legitimate contrasts show how the score is assembled.
`https://en.wikipedia.org/wiki/Main_Page` gives A 0.069, B 0.308, VQC 0.211 →
hybrid 0.221, comfortably *legitimate* on its own merit; the 0.25 calibration
cap is available but no longer what keeps it clear of the threshold.
`https://www.economist.com/science-and-technology/2026-03-14/quantum-advantage`
is the harder case: A 0.165, B 0.234, VQC 0.561 → hybrid 0.297,
*legitimate/medium* with 0.103 of margin. The same URL scored 0.694 and was
flagged before the long-content negatives existed. It is passed on model
strength, not on a brand allowlist — which is the honest limit of URL
morphology on a 68-character dated slug.

## 10. What the system does *not* claim

- No demonstrated quantum advantage — the VQC is a diversity contributor.
- No content analysis: pages are never downloaded; a benign-looking URL on a
  newly registered malicious host is detected only through its morphology.
- Labels come from training data and analyst feedback, so unseen campaign
  names (typosquatting a brand never seen) rely on generic structural signals.
