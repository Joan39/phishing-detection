# QPhish Architecture

## 1. System overview

```
 Browser (React 19 + Vite)
        │  HTTPS + bearer token
        ▼
 FastAPI backend (backend/main.py)
   ├── Auth          register / login / verify / password reset (SMTP)
   ├── Scan engine   features → classical → quantum → blend → calibration → threshold
   ├── Feedback      label extraction → RL bandit → model retraining
   ├── Storage       SQLite (local) or PostgreSQL (DATABASE_URL)
   └── /metrics      Prometheus counters
```

The frontend is a single-page workspace. All detection, training, and
persistence happen server-side; model files and datasets never reach the
browser.

## 2. Live scan pipeline

For `POST /api/scans`:

1. **Feature extraction** (`backend/url_features.py`) — 19 raw-URL features:
   length, letter/digit ratios, hyphen/underscore counts, subdomain depth,
   host/domain length, TLD length, `@`/`=`/`?`/`&` counts, unusual special
   characters, obfuscation tokens (`%XX`), HTTPS flag, IP-host flag, and a
   trusted-root flag.
2. **Classical models** (`models/url/model_A.joblib`, `model_B.joblib`) —
   A: logistic-regression pipeline; B: soft-voting ensemble (logistic
   regression + random forest + gradient boosting). Their probabilities are
   averaged.
3. **Quantum model** (`models/quantum_live.joblib`) — a 4-qubit variational
   quantum classifier (`ZZFeatureMap` + `RealAmplitudes`, 2 repetitions,
   StatevectorSampler, 512 shots, COBYLA 60 iterations) scored on four
   features: URL length, domain length, subdomain count, special-character
   ratio.
4. **Hybrid fusion** — `P = 0.75 · classical + 0.25 · quantum`
   (`QPHISH_QUANTUM_WEIGHT`).
5. **Trusted-domain calibration** — for hosts under a trusted root domain
   with no IP host, no `%XX` obfuscation, no unusual special characters
   before the fragment, and ≤2 subdomains, the probability is capped at 0.25.
   Lookalike domains are deliberately excluded from this safeguard.
6. **Adaptive threshold** — the RL bandit's current action (0.30–0.70,
   default 0.50, learned from analyst feedback) decides the verdict.

Models are cached in-process and reloaded automatically when their file
mtime/size changes, so retraining takes effect without a restart.

## 3. Feedback and adaptive learning

`POST /api/feedback`:

- Accepts a button label or typed text; explicit labels are extracted,
  conflicting labels are rejected, and unclear text stays *uncertain*
  (recorded but never trained on).
- The reward updates an epsilon-greedy `ThresholdBandit`
  (`backend/adaptation_experiment.py`); state persists in
  `results/rl_state.json`.
- Classical models retrain synchronously on the dataset plus all non-uncertain
  feedback; the quantum model retrains in a background worker using the full
  balanced training sample with a reduced optimizer budget
  (`backend/quantum_live.py::adapt_live_model`).

## 4. Training data conventions

- Dataset: PhiUSIIL URL dataset (`original/PhiUSIIL_Phishing_URL_Dataset.csv`).
- **Label convention:** the dataset encodes phishing as original label `0`;
  every training path maps this to target `1`, so *target 1 = phishing* and
  all metrics report phishing as the positive class.
- Production training (`backend/train_models.py`) adds a realistic-legitimate
  hard-negative corpus: 402 URL shapes (search paths, wiki articles, API
  routes, query strings, fragments, digits, deep subdomains, HTTP and HTTPS)
  generated from curated hosts in `backend/domain_policy.py`, each replicated
  250×. This corrects the dataset artifact where legitimate rows are almost
  all bare domains, which otherwise makes models flag any realistic URL.

## 5. Persistence

| Store | Contents |
| --- | --- |
| `results/qphish.db` (SQLite) or PostgreSQL | users, scans, feedback, metrics counters |
| `results/feedback.jsonl` | analyst feedback log |
| `results/rl_state.json` | bandit counts and selected threshold |
| `results/*.json` | saved experiment reports served by `/api/models/comparison` |
| `models/url/*.joblib`, `models/quantum_live.joblib` | live detectors |

## 6. Security posture

- Password hashing with a server-side secret; bearer tokens with expiry.
- Optional email verification and password reset over SMTP.
- CORS restricted to configured frontend origins; `QPHISH_ALLOWED_HOSTS`
  enforced; optional `QPHISH_REQUIRE_HTTPS`.
- Rate limiting on the API gateway (requests/window via env).
- URL query strings and fragments are redacted in stored/displayed results.
- `/metrics` is internal-only behind the deployment gateway.
