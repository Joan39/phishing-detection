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
  F1 and the full hybrid stack 0.9467 (`python -m scripts.production_benchmark`).
- **Quantum detector** — 4-qubit VQC (`ZZFeatureMap` + `RealAmplitudes`,
  512 shots) contributing ensemble diversity to the hybrid score.
- **Real-world generalization** — a dataset-held-out fold cannot see the failure
  mode that matters in use (PhiUSIIL legitimate rows are nearly all bare
  domains), so two host-disjoint curated sets of 74 URLs are scored as well;
  legitimate accuracy went from 0.38 to 0.73 on the battery and 0.50 to 0.75 on
  the never-tuned probe, with phishing detection held at 1.00 on both
  (`docs/Experiments-and-Results.md` §6).
- **RL adaptation** — an epsilon-greedy bandit tunes the decision threshold
  (0.30–0.70) from analyst verdicts; under simulated drift it lifts phishing
  F1 from 0.9889 (static) to 0.9940.
- **Trusted-domain safeguard** — clean URLs on configured brand domains are
  calibration-capped; lookalike domains are scored normally.

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
deployment. The consolidated Word report is regenerated with
`python scripts/generate_system_report.py`.

## Repository layout

```
backend/    FastAPI app, features, training, quantum + RL experiments
frontend/   React 19 + Vite workspace
docs/       full documentation set
scripts/    report generator, live-model evaluation
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
