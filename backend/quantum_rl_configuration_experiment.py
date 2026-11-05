"""Compare quantum-ensemble and RL configurations for adaptive phishing detection.

One shared domain-held-out evaluation varies the detector along two axes:

  - the classical/quantum fusion weight (0.0, 0.15, 0.25, 0.40 by default),
  - the decision policy: a static 0.5 threshold versus the RL threshold bandit
    adapted under simulated legitimacy-mimicking drift.

The quantum component reuses the strengthened live defaults (balanced training
per class, circuit reps, shots, COBYLA budget) so the matrix reflects the
deployed configuration rather than a smoke test.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler

from .adaptation_experiment import ThresholdBandit, simulate_legitimacy_mimicking_drift
from .quantum_experiment import balanced_sample
from .quantum_live import (
    QUANTUM_CIRCUIT_REPS,
    QUANTUM_DEFAULT_MAXITER,
    QUANTUM_DEFAULT_SHOTS,
    QUANTUM_DEFAULT_TRAIN_PER_CLASS,
    QUANTUM_FEATURES,
    _build_classifier,
)
from .train_models import (
    DEFAULT_DATASET,
    make_baseline,
    make_ensemble,
    select_features,
)
from .train_models import load_dataset

THRESHOLD_ACTIONS = (0.30, 0.40, 0.50, 0.60, 0.70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "quantum_rl_configuration_experiment.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument(
        "--quantum-weights",
        type=float,
        nargs="+",
        default=[0.0, 0.15, 0.25, 0.40],
    )
    parser.add_argument("--train-per-class", type=int, default=QUANTUM_DEFAULT_TRAIN_PER_CLASS)
    parser.add_argument("--test-per-class", type=int, default=500)
    parser.add_argument("--maxiter", type=int, default=QUANTUM_DEFAULT_MAXITER)
    parser.add_argument("--shots", type=int, default=QUANTUM_DEFAULT_SHOTS)
    parser.add_argument("--circuit-reps", type=int, default=QUANTUM_CIRCUIT_REPS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--boosting-iterations", type=int, default=100)
    return parser.parse_args()


def metrics(
    target: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    legitimate = int((target == 0).sum())
    return {
        "accuracy": round(float(accuracy_score(target, predictions)), 4),
        "precision_phishing": round(
            float(precision_score(target, predictions, zero_division=0)), 4
        ),
        "recall_phishing": round(
            float(recall_score(target, predictions, zero_division=0)), 4
        ),
        "f1_phishing": round(float(f1_score(target, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(target, probabilities)), 4),
        "false_positive_rate": round(fp / legitimate, 4) if legitimate else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def adapt_thresholds(
    probabilities: np.ndarray,
    target: np.ndarray,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, list[float]]:
    """Run the epsilon-greedy threshold bandit over batches, return predictions."""
    agent = ThresholdBandit(THRESHOLD_ACTIONS, seed)
    predictions = np.zeros(len(target), dtype=int)
    selected: list[float] = []
    for start in range(0, len(target), batch_size):
        stop = min(start + batch_size, len(target))
        action, threshold = agent.choose()
        batch_predictions = (probabilities[start:stop] >= threshold).astype(int)
        reward = np.where(
            batch_predictions == target[start:stop],
            1.0,
            np.where(batch_predictions == 1, -2.0, -3.0),
        ).mean()
        agent.update(action, float(reward))
        predictions[start:stop] = batch_predictions
        selected.append(threshold)
    return predictions, selected


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    frame, target, groups = load_dataset(args.data)
    features = select_features(frame, "url")
    splits = list(
        StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=args.seed,
        ).split(frame[features], target, groups)
    )
    train_indices, test_indices = splits[args.fold]

    # Classical components trained on the full domain-held-out training split.
    x_train = frame.iloc[train_indices][features]
    y_train = target.iloc[train_indices]
    baseline = make_baseline(args.seed)
    ensemble = make_ensemble(
        args.seed,
        trees=args.trees,
        boosting_iterations=args.boosting_iterations,
    )
    started = time.perf_counter()
    baseline.fit(x_train, y_train)
    ensemble.fit(x_train, y_train)
    classical_fit_seconds = time.perf_counter() - started

    # Quantum component trained on a balanced subset (live defaults).
    q_splits = list(
        StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=args.seed,
        ).split(frame[QUANTUM_FEATURES], target, groups)
    )
    q_train_indices, _ = q_splits[args.fold]
    train_sample = balanced_sample(
        q_train_indices, target, args.train_per_class, args.seed
    )
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    xq_train = scaler.fit_transform(
        frame.iloc[train_sample][QUANTUM_FEATURES]
    )
    yq_train = target.iloc[train_sample].to_numpy()
    classifier = _build_classifier(
        maxiter=args.maxiter,
        shots=args.shots,
        seed=args.seed,
        reps=args.circuit_reps,
    )
    started = time.perf_counter()
    classifier.fit(xq_train, yq_train)
    quantum_fit_seconds = time.perf_counter() - started

    # Shared test sample: balanced rows, with a drift-simulated copy for RL.
    test_sample = balanced_sample(test_indices, target, args.test_per_class, args.seed)
    test_frame = frame.iloc[test_sample]
    y_test = target.iloc[test_sample].to_numpy()

    classical_frame = test_frame[features].copy()
    classical_probabilities = (
        baseline.predict_proba(classical_frame)[:, 1]
        + ensemble.predict_proba(classical_frame)[:, 1]
    ) / 2.0
    quantum_probabilities = np.asarray(
        classifier.predict_proba(scaler.transform(test_frame[QUANTUM_FEATURES]))
    )[:, 1]

    drifted_frame = simulate_legitimacy_mimicking_drift(
        classical_frame, target.iloc[test_sample]
    )
    drifted_classical = (
        baseline.predict_proba(drifted_frame)[:, 1]
        + ensemble.predict_proba(drifted_frame)[:, 1]
    ) / 2.0
    drifted_quantum = np.asarray(
        classifier.predict_proba(
            scaler.transform(drifted_frame[QUANTUM_FEATURES])
        )
    )[:, 1]

    configurations = []
    for quantum_weight in args.quantum_weights:
        fused = (1 - quantum_weight) * classical_probabilities + quantum_weight * quantum_probabilities
        drifted_fused = (
            (1 - quantum_weight) * drifted_classical
            + quantum_weight * drifted_quantum
        )

        static_predictions = (fused >= 0.5).astype(int)
        entry: dict[str, Any] = {
            "quantum_weight": quantum_weight,
            "static_threshold_0_5": metrics(y_test, static_predictions, fused),
        }

        rl_predictions, selected = adapt_thresholds(
            drifted_fused, y_test, args.seed, args.batch_size
        )
        entry["static_threshold_drift"] = metrics(
            y_test, (drifted_fused >= 0.5).astype(int), drifted_fused
        )
        entry["rl_adaptive_threshold_drift"] = metrics(y_test, rl_predictions, drifted_fused)
        entry["rl_selected_thresholds"] = {
            str(threshold): selected.count(threshold) for threshold in THRESHOLD_ACTIONS
        }
        configurations.append(entry)

    return {
        "experiment": "quantum-ensemble weights x RL threshold policy matrix",
        "seed": args.seed,
        "fold": args.fold,
        "train_rows": int(len(train_indices)),
        "quantum_train_rows": int(len(train_sample)),
        "test_rows": int(len(test_sample)),
        "quantum_features": QUANTUM_FEATURES,
        "circuit": {
            "feature_map": "ZZFeatureMap",
            "ansatz": "RealAmplitudes",
            "qubits": len(QUANTUM_FEATURES),
            "reps": args.circuit_reps,
            "sampler": "StatevectorSampler",
            "shots": args.shots,
            "maxiter": args.maxiter,
        },
        "drift": "phishing test rows made more legitimate-looking for the RL axis",
        "classical_fit_seconds": round(classical_fit_seconds, 2),
        "quantum_fit_seconds": round(quantum_fit_seconds, 2),
        "configurations": configurations,
    }


def main() -> None:
    args = parse_args()
    result = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
