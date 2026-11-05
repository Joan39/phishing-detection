# Development Setup

## 1. Prerequisites

- Python 3.12+ (tested on 3.14)
- Node.js 20+ (tested on 24)
- The PhiUSIIL dataset at `original/PhiUSIIL_Phishing_URL_Dataset.csv`
  (git-ignored; obtain from the PhiUSIIL repository)

## 2. Backend

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows
# source .venv/bin/activate       # Linux/macOS

pip install -r backend/requirements.txt
pip install -r backend/requirements-quantum.txt   # Qiskit stack for quantum work

python -m uvicorn backend.main:app --reload        # http://localhost:8000
```

Configuration is environment-variable driven; copy `.env.example` to `.env`
and adjust. Local development works with defaults (SQLite fallback when
`DATABASE_URL` is unset).

### Tests

```bash
python -m unittest discover -s tests -v
```

### Useful CLIs

```bash
python -m backend.predict_url "https://example.com"       # score one URL
python -m backend.train_models                            # retrain classical A+B
python -m backend.quantum_live --output models/quantum_live.joblib
python -m backend.three_way_evaluation --output results/three_way_evaluation.json
python -m backend.quantum_rl_configuration_experiment --output \
    results/quantum_rl_configuration_experiment.json
python scripts/quantum_live_eval.py models/quantum_live.joblib
python scripts/generate_system_report.py                  # rebuild reports/*.docx
```

## 3. Frontend

```bash
npm --prefix frontend install
npm --prefix frontend run dev      # http://localhost:5173
npm --prefix frontend run build    # production bundle in frontend/dist
```

The app connects to `http://localhost:8000` unless `VITE_API_BASE_URL` is set
(create `frontend/.env` with `VITE_API_BASE_URL=http://host:port`).

## 4. Model artifacts

Live detectors must exist before scanning:

| Artifact | Producer |
| --- | --- |
| `models/url/model_A.joblib`, `model_B.joblib` | `python -m backend.train_models` |
| `models/quantum_live.joblib` | `python -m backend.quantum_live` |

The API caches models by file mtime/size and hot-reloads them after retraining.
`models/`, `original/`, and `results/` are git-ignored — never commit datasets,
databases, secrets, or model binaries.

## 5. Repository layout

```
backend/     FastAPI app, features, training, quantum, RL, experiments
frontend/    React 19 + Vite single-page workspace
scripts/     report generator and live-model evaluation
tests/       unit tests (features, training policy, label convention)
docs/        user manual, architecture, API, experiments, deployment
deploy/      Docker Compose stack, nginx gateway, Prometheus, preflight
reports/     generated .docx experiment reports (git-ignored)
original/    raw dataset (git-ignored)
models/      live detector artifacts (git-ignored)
results/     database, experiment JSON outputs, RL state (git-ignored)
```

## 6. Conventions

- Target `1` = phishing everywhere; the dataset's original label `0` is
  remapped during loading (unit tests enforce this).
- Experiment modules are runnable as `python -m backend.<name>` and write a
  JSON report consumed by `/api/models/comparison`.
- Style: type hints, `from __future__ import annotations`, no comments unless
  the *why* is non-obvious.
