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
| `scripts/generate_system_report.py` | Rebuild the .docx system report | `reports/…Report.docx` |

Run any of them with `python -m backend.<name> [--output path]`.

## 2. Classical detectors (domain-held-out fold)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| A — logistic regression | 0.9803 | 0.9868 | 0.9668 | 0.9767 | 0.9957 | 0.0097 |
| B — soft-voting ensemble | 0.9953 | 0.9968 | 0.9922 | 0.9945 | 0.9982 | 0.0024 |

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
bare domains (mean length 27, 100% HTTPS) while phishing rows average length
46 with parameters, so models learned "long URL with query = phishing" and
scored ordinary site pages at ~1.0. Fix: 402 realistic legitimate URL shapes
× 250 replicas as hard negatives (`backend/domain_policy.py`). Post-fix
verification: real-world legitimate URLs score 0.0003–0.21 while phishing
URLs remain ≥0.97; live API scans classify both correctly.

## 6. Limitations

- No demonstrated quantum advantage; the VQC contributes diversity only.
- Drift is simulated, not observed production traffic.
- Classical feedback retraining is synchronous in the request path.
- Features are URL-morphology only (no host reputation or certificate data).
