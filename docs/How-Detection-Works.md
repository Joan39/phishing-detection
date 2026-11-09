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
HTTPS together"). On the domain-held-out fold it reaches 0.8741 accuracy.

## 3. Model B — soft-voting ensemble

B wraps three detectors and averages their probabilities with weights 1:2:2:

- the same logistic pipeline as A (linear view),
- a 200-tree random forest (axis-aligned interactions, "which combinations
  of thresholds co-occur"),
- a 200-iteration gradient-boosting stage (sequentially corrects the
  remaining errors, the strongest non-linear fitter of the three).

The forest and boosting stages let B capture the "several mediocre signals at
once" pattern A misses, at 0.9848 held-out accuracy and 0.9965 ROC-AUC.

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
(0.574 accuracy on held-out domains at the live threshold, 0.609 at 0.50) — its
role is *ensemble diversity*: its
errors come from a different hypothesis class, so blending it smooths the
classical probability rather than duplicating it. What it must not do is act as
a constant; measured on real-world URLs the corpus-trained circuit separates the
two classes at AUC 0.746, where the dataset-trained one managed 0.570.

## 5. Hybrid fusion

```
P(hybrid) = (1 - w_q) · P(classical) + w_q · P(quantum)
P(classical) = 0.35 · P(A) + 0.65 · P(B)
```

The static numbers above are the *default* policy: `QUANTUM_WEIGHT` is capped at
the 0.25 level validated in the quantum × RL configuration matrix (at 0.25 the
blend matches classical-only adaptive F1 under drift while keeping diversity; at
0.40 both policies degrade), and the 0.35/0.65 split follows the held-out
advantage of the voting ensemble over single logistic regression.

Those two weights are no longer frozen. `backend/adaptive_policy.py` re-ranks
them from the accuracy each detector has shown on recent analyst labels: a
component that keeps being right on current traffic gains influence, one that
keeps being wrong loses it, bounded so nothing is ever silenced (classical
weights stay in 0.20-0.80, the quantum weight in 0.10-0.40).

The re-ranking is gated by how much *independent* evidence it has. A handful of
feedback rows usually means the analyst rechecked the same few URLs, and the
accuracy gap between A and B on ten such rows is noise, not a verdict on which
detector is right about traffic. So the learned weights are shrunk toward the
documented blend in proportion to the number of distinct labeled URLs in the
policy (full weight at `MIN_EVIDENCE_URLS` = 20, and linearly below that). A
system that has not yet seen feedback scores exactly the documented static
blend - the cold-start guarantee is tested, not assumed.

`python -m scripts.replay_feedback_policy` rebuilds the policy from the feedback
rows already in the database, in the order they were recorded, through the same
update functions the endpoint uses. It exists because labels collected before
this layer existed otherwise leave the policy empty forever; `--dry-run` prints
what the replay would produce without touching the live file.

Backfilling is offered, not forced. Both `scripts.production_benchmark.py` and
`scripts.realworld_offline_score.py` take `--policy PATH`, so a candidate policy
can be scored against the live one on the same rows before it is installed. When
that was done for the current backlog, the gated replay raised legitimate
accuracy on the real-world battery (0.73 to 0.77) and lowered held-out phishing
recall by 1.2 points. The backlog is 22 labels over only 10 distinct URLs, and
they are easy separations, so the re-ranker found a weight vector that the
47,158-row held-out split disagrees with. The live policy therefore stays at the
documented blend until the evidence is larger and less repetitive; the numbers
and the decision rule are in `Experiments-and-Results.md` §7.3.

## 6. Trusted-domain calibration and the redirect guardrail

A raw learned score can still overreact to realistic-but unusual legitimate
traffic. Before the verdict, URLs whose host is under a configured trusted
root (e.g. `google.com`, `wikipedia.org`) are checked for *impersonation
markers*: IP host, `%XX` obfuscation, unusual special characters before the
fragment, more than 2 subdomains, or a redirect parameter that hands the
visitor to a different registrable domain. A clean trusted URL has its
probability capped at 0.25 - no trusted brand can be reported as phishing by
morphology alone, while `paypal.com.account-verify.info` style lookalikes fail
the check and keep their full score.

The last clause closes the one measured evasion the calibration itself was
helping: `google.com/url?sa=t&url=https://refund-claim-irs-com/tax`. A brand
page whose only content is a jump is not the brand's content, so it earns no
protection. Such payloads are also *scored on their own*:
`score_redirect_targets` extracts the foreign target (query and fragment,
percent-encoded or protocol-relative) and, if the target reads more phishing
than the wrapper, the target's score wins - which is what the user lands on.
Same-brand hops (`accounts.google.com` → `mail.google.com`) are ignored, and so
are values that are not absolute HTTP(S) URLs, so ordinary SSO return links are
untouched.

## 7. Adaptive decision thresholds

```
verdict = phishing  if  P(hybrid) ≥ threshold
```

Two bandits share this decision. The global one (`results/rl_state.json`)
chooses among 0.30 / 0.40 / 0.50 / 0.60 / 0.70; each analyst feedback record is a
reward - correct verdict +1, wrong verdict −1, applied with a 0.2 learning rate
to the action in force. When new phishing waves arrive dressed as legitimate
traffic, the tuned threshold *drops* (the drift experiment settled on 0.30 in 15
of 16 batches), catching attacks a static 0.50 cutoff misses.

The second bandit is per *traffic class* (`https/public`, `https/trusted`,
`http/public`, `http/trusted`). One cut cannot serve plain-HTTP lookalikes, deep
HTTPS content paths and trusted-brand hosts at once, which is exactly what the
measured false positives showed. A class keeps using the global threshold until
it has collected 8 verdicts of its own, then switches to the cut that has been
rewarded most for that class. `GET /api/scans` reports which one was used in
`threshold_source`, and `adaptive_context` names the class.

## 8. Continuous adaptation

Every labeled correction:

1. updates the global bandit and the class bandit immediately,
2. re-ranks the classical and quantum weights from each component's running
   accuracy against the label,
3. queues a full-budget classical retrain including the new example
   (background thread),
4. triggers periodic quantum retraining once 10+ new labels accumulate,
5. hot-reloads the new artifacts by mtime — no restart, no downtime.

Two of these do not wait for a label at all. The hybrid score histogram is
watched with a population-stability index over a rolling 200-scan window
(`results/adaptive_policy.json`); a shift of PSI ≥ 0.25 means the traffic has
changed shape, and the next feedback then triggers a refit and re-baselines the
window. Scans whose components disagree by ≥ 0.25 *and* sit within 0.10 of the
cut are returned with `review_recommended: true` - the verdict is honest about
being low-confidence, and those rows are the most informative training data.

## 9. A worked example (observed live)

`https://paypal-secure-refund.claim-center.net/login`

| Stage | Value | Why |
| --- | --- | --- |
| Model A | 0.976 | Brand-hyphen pattern + long host dominate the linear model |
| Model B | 0.980 | Trees agree; the domain is not a trusted root |
| Classical fusion | 0.979 | 0.35·0.976 + 0.65·0.980 |
| Quantum VQC | 0.307 | Encoded length/domain/subdomain state reads ordinary here |
| Hybrid | 0.811 | 0.75·0.979 + 0.25·0.307 |
| Calibration | n/a | Host is a lookalike chain, safeguard does not apply |
| Redirect payload | n/a | No cross-domain jump parameter to score |
| Threshold | 0.40 (global bandit; `https/public` context still cold) | 0.811 ≥ 0.40 → **phishing**, risk level *high* |

The evasion case, same live path and same threshold:

`https://www.google.com/url?sa=t&url=https://refund-claim-irs-com/tax`

| Stage | Value | Why |
| --- | --- | --- |
| Model A / B / VQC on the wrapper | 0.007 / 0.274 / 0.195 | It really is a `google.com` URL with a Google-shaped path |
| Hybrid on the wrapper | 0.184 | Would read *legitimate* — this is the bypass |
| Calibration | refused | The host is trusted, but a cross-domain jump parameter voids the protection |
| Model A / B / VQC on the payload | 0.506 / 0.886 / 0.377 | `refund-claim-irs-com` is a hyphen-dash brand lookalike |
| Hybrid, max of the chain | **0.659** | The payload's score wins, so the verdict is *phishing* |
| Reported | `redirect_target` (redacted), `explanations["open_redirect_chain"]` | The analyst sees which URL in the chain drove the decision |

The same happens to `https://l.facebook.com.l.php?u=https://secure-appleid-verify.com/login`:
0.270 on the wrapper, 0.882 on the payload. Those two were the measured
novel-technique bypasses; the guardrail is what closed them (§7.1 of
Experiments-and-Results).

Two legitimate contrasts show how the score is assembled.
`https://en.wikipedia.org/wiki/Main_Page` gives A 0.069, B 0.298, VQC 0.090 →
hybrid 0.186, comfortably *legitimate* on its own merit; the 0.25 calibration
cap is available but no longer what keeps it clear of the threshold.
`https://www.economist.com/science-and-technology/2026-03-14/quantum-advantage`
is the harder case: A 0.166, B 0.221, VQC 0.312 → hybrid 0.229,
*legitimate/medium* with 0.171 of margin. The same URL scored 0.694 and was
flagged before the long-content negatives existed. It is passed on model
strength, not on a brand allowlist — which is the honest limit of URL
morphology on a 68-character dated slug.

## 10. What the system does *not* claim

- No demonstrated quantum advantage — the VQC is a diversity contributor.
- No content analysis: pages are never downloaded; a benign-looking URL on a
  newly registered malicious host is detected only through its morphology.
- Labels come from training data and analyst feedback, so unseen campaign
  names (typosquatting a brand never seen) rely on generic structural signals.
- The redirect guardrail reads the scanned string only. A payload delivered by
  an HTTP 3xx response, a `<meta http-equiv=refresh>` tag or a JavaScript
  handoff is invisible to it, because nothing is fetched.
- The unsupervised drift signal is weak in practice: in the replayed technique
  shift the PSI window never crossed its 0.25 trigger (max 0.0997). Adaptation
  demonstrated so far is reward-driven, i.e. it needs analyst feedback.
