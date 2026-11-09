# QPhish — Quantum-Enhanced Adaptive Phishing Detection

QPhish is a URL phishing-detection workspace that fuses classical
machine-learning with a variational quantum classifier and adapts its
decision threshold with reinforcement learning from analyst feedback.
Submitted URLs are analyzed from their structure only — they are never
opened or fetched.

## How detection works

```
URL ──> 19 structural features ──> classical ensemble (LR + voting + boosting)
                                     │  75%
                                     ├──> hybrid probability ──> RL adaptive threshold ──> verdict
                                     │  25%        ▲
                  4-qubit VQC (Qiskit) ┘           ── analyst feedback loop
```

- **Classical detectors** — logistic regression and a soft-voting ensemble,
  trained on the PhiUSIIL dataset plus a realistic-legitimate hard-negative
  corpus. On 47,158 domain-held-out rows the ensemble reaches 0.9725 phishing
  F1 and the full hybrid stack 0.9622 (`python -m scripts.production_benchmark`).
- **Quantum detector** — 4-qubit VQC (`ZZFeatureMap` + `RealAmplitudes`,
  512 shots) contributing ensemble diversity to the hybrid score.
- **Real-world generalization** — a dataset-held-out fold cannot see the failure
  mode that matters in use (PhiUSIIL legitimate rows are nearly all bare
  domains), so two host-disjoint curated sets of 74 URLs are scored as well;
  legitimate accuracy went from 0.38 to 0.73 on the battery and 0.50 to 0.83 on
  the never-tuned probe, with phishing detection held at 1.00 on both
  (`docs/Experiments-and-Results.md` §6).
- **Adaptation** — an epsilon-greedy bandit tunes the decision threshold
  (0.30–0.70) from analyst verdicts; under simulated drift it lifts phishing
  F1 from 0.9889 (static) to 0.9940. `backend/adaptive_policy.py` extends the
  same feedback to a per-traffic-class threshold, a bounded re-ranking of the
  classical/quantum weights, a label-free PSI drift window and review routing.
  In a replayed technique shift that layer takes legitimate accuracy on familiar
  traffic from 0.73 to 0.88 and on never-seen holdout traffic from 0.75 to 1.00,
  at the cost of one novel shortener URL (§7.2). Weight re-ranking is shrunk
  toward the documented blend until 20 distinct URLs have been labeled, and
  `scripts.replay_feedback_policy` rebuilds the policy from feedback stored
  before the layer existed; `--policy` on the two scoring scripts measures any
  candidate against the live one. The current backlog is 22 labels over 10
  distinct, easy-to-separate URLs and replaying it is a net loss (+0.04
  real-world legitimate, −0.012 held-out phishing recall), so the live policy is
  still the documented static blend (§7.3).
- **Trusted-domain safeguard and redirect guardrail** — clean URLs on configured
  brand domains are calibration-capped; lookalikes are scored normally, and a
  trusted host loses its cap when its URL is only a jump to another registrable
  domain, because the payload is then scored on its own. That closed both
  measured novel-technique bypasses: recall on 14 unseen technique strings went
  from 0.857 to 1.00 with no retraining and no probability change on the 74
  curated real-world rows (§7.1).

## Quick start (development)

```bash
# Backend
python -m venv .venv && .venv/Scripts/activate
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload          # http://localhost:8000

# Frontend
npm --prefix frontend install
npm --prefix frontend run dev                        # http://localhost:5173

# Tests
python -m unittest discover -s tests
```

Train the live detectors with `python -m backend.train_models` and
`python -m backend.quantum_live`. See
[docs/Development-Setup.md](docs/Development-Setup.md) for the full setup and
[docs/Deployment.md](docs/Deployment.md) for the Docker Compose stack.

## Documentation

All guides live in [docs/](docs/README.md): user manual, how detection works,
architecture, API reference, development setup, experiments and results,
deployment. Word deliverables are generated from the measured artifacts, not
typed by hand: `python -m scripts.generate_system_report` for the system report,
`python -m scripts.build_presentation_answers_docx` for the poster-form answers
and `python -m scripts.build_full_report_docx` for the complete poster and
presentation report (answers plus architecture, operation, per-model detection,
results, comparison, limitations and the command that reproduces each number).

## Repository layout

```
backend/    FastAPI app, features, training, quantum + RL experiments
frontend/   React 19 + Vite workspace
docs/       full documentation set
scripts/    benchmarks, real-world batteries, adaptation replays, Word report builders
tests/      unit tests (features, training policy, label convention)
deploy/     Docker Compose stack, nginx gateway, Prometheus, preflight
```

Datasets (`original/`), trained models (`models/`), and experiment outputs
or databases (`results/`) are git-ignored and stay on the server.

## Research aims

This project experiments with combining **quantum ensemble learning** and
**reinforcement-learning adaptive thresholding** for phishing detection:
measuring how quantum fusion weights and threshold policies interact under
concept drift, and training the system so the hybrid detector stays
accurate on real-world URL morphology.
