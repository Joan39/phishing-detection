"""Train and evaluate a combined ensemble-quantum-RL phishing detector.

This is a single end-to-end prototype that fuses:

  - classical ensemble probabilities (B),
  - quantum probabilities (C), and
  - RL-based threshold adaptation (D)

The goal is to test whether a unified detector improves robustness when phishing
patterns shift across unseen domains.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
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

from adaptation_experiment import ThresholdBandit, simulate_legitimacy_mimicking_drift
from .quantum_experiment import QUANTUM_FEATURES, balanced_sample
from .train_models import (
    DEFAULT_DATASET,
    IDENTIFIER_COLUMNS,
    KNOWN_LEAKAGE_COLUMNS,
    make_ensemble,
    load_dataset,
    make_baseline,
    select_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "combined_experiment.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--quantum-weight", type=float, default=0.25)
    parser.add_argument("--train-per-class", type=int, default=80)
    parser.add_argument("--test-per-class", type=int, default=40)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--boosting-iterations", type=int, default=100)
    parser.add_argument("--maxiter", type=int, default=10)
    parser.add_argument("--shots", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def calculate_metrics(
    target: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    legitimate_count = int((target == 0).sum())
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision_phishing": float(
            precision_score(target, predictions, zero_division=0)
        ),
        "recall_phishing": float(recall_score(target, predictions, zero_division=0)),
        "f1_phishing": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "false_positive_rate": float(fp / legitimate_count)
        if legitimate_count
        else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 <= args.quantum_weight <= 1.0:
        raise ValueError("--quantum-weight must be between 0 and 1.")
    if args.train_per_class < 2 or args.test_per_class < 2:
        raise ValueError("Sampling per class must be at least 2 for adaptation evaluation.")

    frame, target, groups = load_dataset(args.data)
    features = select_features(frame, "url")
    splits = list(
        StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=args.seed,
        ).split(frame[features], target, groups)
    )
    train_indices, test_indices = splits[args.fold]

    x_train = frame.iloc[train_indices][features]
    x_test = frame.iloc[test_indices][features]
    y_train = target.iloc[train_indices]
    y_test = target.iloc[test_indices]

    ensemble = make_ensemble(
        args.seed,
        trees=args.trees,
        boosting_iterations=args.boosting_iterations,
    )
    started = time.perf_counter()
    ensemble.fit(x_train, y_train)
    ensemble_fit_seconds = time.perf_counter() - started
    classical_probabilities = ensemble.predict_proba(x_test)[:, 1]

    train_sample = balanced_sample(
        train_indices,
        target,
        args.train_per_class,
        args.seed,
    )
    test_sample = balanced_sample(
        test_indices,
        target,
        args.test_per_class,
        args.seed + 1,
    )

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    x_q_train = scaler.fit_transform(frame.iloc[train_sample][QUANTUM_FEATURES])
    x_q_test = scaler.transform(frame.iloc[test_sample][QUANTUM_FEATURES])
    y_q_train = target.iloc[train_sample].to_numpy()
    y_q_test = target.iloc[test_sample].to_numpy()

    feature_map = zz_feature_map(
        feature_dimension=len(QUANTUM_FEATURES),
        reps=1,
        entanglement="linear",
    )
    ansatz = real_amplitudes(
        num_qubits=len(QUANTUM_FEATURES),
        reps=1,
        entanglement="linear",
    )
    quantum_classifier = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=args.maxiter),
        sampler=StatevectorSampler(default_shots=args.shots, seed=args.seed),
    )
    started = time.perf_counter()
    quantum_classifier.fit(x_q_train, y_q_train)
    quantum_fit_seconds = time.perf_counter() - started
    quantum_probabilities = quantum_classifier.predict_proba(x_q_test)[:, 1]

    test_positions = pd.Series(
        np.arange(len(test_indices)),
        index=test_indices,
    ).loc[test_sample].to_numpy()
    classical_sample_probabilities = classical_probabilities[test_positions]

    fused_probabilities = (
        args.quantum_weight * quantum_probabilities
        + (1.0 - args.quantum_weight) * classical_sample_probabilities
    )
    fused_predictions = (fused_probabilities >= 0.5).astype(int)

    drifted_test = simulate_legitimacy_mimicking_drift(
        frame.iloc[test_sample][features].reset_index(drop=True),
        pd.Series(y_q_test).reset_index(drop=True),
    )
    drifted_x = scaler.transform(drifted_test[QUANTUM_FEATURES])
    drifted_quantum = quantum_classifier.predict_proba(drifted_x)[:, 1]
    drifted_classical = ensemble.predict_proba(drifted_test)[:, 1]
    drifted_fused = (
        args.quantum_weight * drifted_quantum
        + (1.0 - args.quantum_weight) * drifted_classical
    )
    static_drift_predictions = (drifted_fused >= 0.5).astype(int)

    adaptation_parts: list[np.ndarray] = []
    evaluation_parts: list[np.ndarray] = []
    for label in (0, 1):
        label_indices = np.flatnonzero(y_q_test == label)
        midpoint = len(label_indices) // 2
        adaptation_parts.append(label_indices[:midpoint])
        evaluation_parts.append(label_indices[midpoint:])
    adaptation_indices = np.concatenate(adaptation_parts)
    evaluation_indices = np.concatenate(evaluation_parts)
    agent = ThresholdBandit((0.30, 0.40, 0.50, 0.60, 0.70), args.seed)
    rewards: list[float] = []
    selected_thresholds: list[float] = []
    for start in range(0, len(adaptation_indices), args.batch_size):
        batch_indices = adaptation_indices[start : start + args.batch_size]
        action, threshold = agent.choose()
        batch_predictions = (drifted_fused[batch_indices] >= threshold).astype(int)
        batch_target = y_q_test[batch_indices]
        reward = np.where(
            batch_predictions == batch_target,
            1.0,
            np.where(batch_predictions == 1, -2.0, -3.0),
        ).mean()
        agent.update(action, float(reward))
        rewards.append(float(reward))
        selected_thresholds.append(threshold)

    learned_threshold = agent.thresholds[int(np.argmax(agent.q_values))]
    adaptive_predictions = (
        drifted_fused[evaluation_indices] >= learned_threshold
    ).astype(int)
    baseline_model = make_baseline(args.seed)
    baseline_model.fit(x_train, y_train)
    baseline_probabilities = baseline_model.predict_proba(x_test)[:, 1]
    baseline_predictions = (baseline_probabilities >= 0.5).astype(int)

    summary = {
        "experiment": "combined ensemble-quantum-RL phishing detector",
        "seed": args.seed,
        "fold": args.fold,
        "dataset": str(args.data),
        "rows": int(len(frame)),
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "sample_test_rows": int(len(test_sample)),
        "adaptation_rows": int(len(adaptation_indices)),
        "evaluation_rows": int(len(evaluation_indices)),
        "quantum_features": QUANTUM_FEATURES,
        "weights": {
            "quantum": args.quantum_weight,
            "classical": 1.0 - args.quantum_weight,
        },
        "classical_fit_seconds": ensemble_fit_seconds,
        "quantum_fit_seconds": quantum_fit_seconds,
        "fused_metrics": calculate_metrics(
            pd.Series(y_q_test),
            fused_predictions,
            fused_probabilities,
        ),
        "static_drift_threshold_metrics": calculate_metrics(
            pd.Series(y_q_test[evaluation_indices]),
            static_drift_predictions[evaluation_indices],
            drifted_fused[evaluation_indices],
        ),
        "adaptive_drift_threshold_metrics": calculate_metrics(
            pd.Series(y_q_test[evaluation_indices]),
            adaptive_predictions,
            drifted_fused[evaluation_indices],
        ),
        "rl_adaptation_threshold": learned_threshold,
        "component_metrics": {
            "A_baseline": calculate_metrics(
                y_test,
                baseline_predictions,
                baseline_probabilities,
            ),
            "B_ensemble": calculate_metrics(
                y_test,
                (classical_probabilities >= 0.5).astype(int),
                classical_probabilities,
            ),
            "C_quantum": calculate_metrics(
                pd.Series(y_q_test),
                (quantum_probabilities >= 0.5).astype(int),
                quantum_probabilities,
            ),
            "C_hybrid": calculate_metrics(
                pd.Series(y_q_test),
                (fused_predictions >= 0.5).astype(int),
                fused_probabilities,
            ),
            "B_full_ensemble_on_hybrid_sample": calculate_metrics(
                pd.Series(y_q_test),
                (classical_sample_probabilities >= 0.5).astype(int),
                classical_sample_probabilities,
            ),
        },
        "rl": {
            "actions": [0.30, 0.40, 0.50, 0.60, 0.70],
            "final_q_values": agent.q_values.tolist(),
            "mean_batch_reward": float(np.mean(rewards)),
            "selected_threshold_counts": {
                str(threshold): int(selected_thresholds.count(threshold))
                for threshold in agent.thresholds
            },
        },
        "excluded_columns": sorted(IDENTIFIER_COLUMNS | KNOWN_LEAKAGE_COLUMNS),
    }
    hybrid_metrics = summary["component_metrics"]["C_hybrid"]
    classical_metrics = summary["component_metrics"]["B_full_ensemble_on_hybrid_sample"]
    static_metrics = summary["static_drift_threshold_metrics"]
    adaptive_metrics = summary["adaptive_drift_threshold_metrics"]
    summary["comparisons"] = {
        "hybrid_minus_classical_f1": float(
            hybrid_metrics["f1_phishing"] - classical_metrics["f1_phishing"]
        ),
        "hybrid_minus_classical_roc_auc": float(
            hybrid_metrics["roc_auc"] - classical_metrics["roc_auc"]
        ),
        "rl_minus_static_drift_f1": float(
            adaptive_metrics["f1_phishing"] - static_metrics["f1_phishing"]
        ),
        "claim_supported_on_this_run": bool(
            hybrid_metrics["f1_phishing"] > classical_metrics["f1_phishing"]
            or adaptive_metrics["f1_phishing"] > static_metrics["f1_phishing"]
        ),
    }
    return summary


def main() -> None:
    args = parse_args()
    result = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
