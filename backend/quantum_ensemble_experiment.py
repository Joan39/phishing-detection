"""Evaluate a resource-aware hybrid classical-quantum phishing classifier.

The quantum circuit is trained on a balanced subset because a variational
quantum classifier is not practical for the full dataset. The classical
ensemble uses the same four features, but is trained on the full training
split by default. The final prediction uses a weighted average of both
probability estimates. Use ``--classical-training sample`` for a faster,
fully subset-based smoke test.
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

from .quantum_experiment import QUANTUM_FEATURES, balanced_sample
from .train_models import DEFAULT_DATASET, make_ensemble, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "quantum_ensemble_experiment.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--train-per-class", type=int, default=500)
    parser.add_argument("--test-per-class", type=int, default=250)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--boosting-iterations", type=int, default=100)
    parser.add_argument(
        "--classical-training",
        choices=("full", "sample"),
        default="full",
        help=(
            "Train the classical component on the full domain-held-out "
            "training split (recommended) or on the quantum sample."
        ),
    )
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--quantum-weight", type=float, default=0.25)
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

    frame, target, groups = load_dataset(args.data)
    missing = [column for column in QUANTUM_FEATURES if column not in frame]
    if missing:
        raise ValueError(f"Missing quantum features: {missing}")

    splits = list(
        StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=args.seed,
        ).split(frame[QUANTUM_FEATURES], target, groups)
    )
    train_indices, test_indices = splits[args.fold]
    train_sample = balanced_sample(
        train_indices, target, args.train_per_class, args.seed
    )
    test_sample = balanced_sample(
        test_indices, target, args.test_per_class, args.seed + 1
    )

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    x_train = scaler.fit_transform(frame.iloc[train_sample][QUANTUM_FEATURES])
    x_test = scaler.transform(frame.iloc[test_sample][QUANTUM_FEATURES])
    y_train = target.iloc[train_sample].to_numpy()
    y_test = target.iloc[test_sample].to_numpy()

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
    classifier = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=args.maxiter),
        sampler=StatevectorSampler(default_shots=args.shots, seed=args.seed),
    )
    classical = make_ensemble(
        args.seed,
        trees=args.trees,
        boosting_iterations=args.boosting_iterations,
    )

    started = time.perf_counter()
    classifier.fit(x_train, y_train)
    quantum_fit_seconds = time.perf_counter() - started
    classical_train_indices = (
        train_indices if args.classical_training == "full" else train_sample
    )
    classical_x_train = frame.iloc[classical_train_indices][QUANTUM_FEATURES]
    classical_y_train = target.iloc[classical_train_indices]
    started = time.perf_counter()
    classical.fit(classical_x_train, classical_y_train)
    classical_fit_seconds = time.perf_counter() - started

    quantum_probabilities = classifier.predict_proba(x_test)[:, 1]
    classical_probabilities = classical.predict_proba(
        frame.iloc[test_sample][QUANTUM_FEATURES]
    )[:, 1]
    hybrid_probabilities = (
        args.quantum_weight * quantum_probabilities
        + (1 - args.quantum_weight) * classical_probabilities
    )
    predictions = (hybrid_probabilities >= 0.5).astype(int)

    return {
        "experiment": "domain-held-out classical-quantum ensemble subset experiment",
        "seed": args.seed,
        "fold": args.fold,
        "rows": int(len(frame)),
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "train_sample_rows": int(len(train_sample)),
        "test_sample_rows": int(len(test_sample)),
        "train_sample_per_class": args.train_per_class,
        "test_sample_per_class": args.test_per_class,
        "classical_training": args.classical_training,
        "classical_train_rows": int(len(classical_train_indices)),
        "classical_train_domains": int(groups.iloc[classical_train_indices].nunique()),
        "quantum_train_domains": int(groups.iloc[train_sample].nunique()),
        "test_domains": int(groups.iloc[test_sample].nunique()),
        "domain_overlap": int(
            len(set(groups.iloc[train_indices]) & set(groups.iloc[test_indices]))
        ),
        "features": QUANTUM_FEATURES,
        "weights": {
            "quantum": args.quantum_weight,
            "classical": 1 - args.quantum_weight,
        },
        "circuit": {
            "feature_map": "ZZFeatureMap",
            "ansatz": "RealAmplitudes",
            "qubits": len(QUANTUM_FEATURES),
            "optimizer": "COBYLA",
            "maxiter": args.maxiter,
            "sampler": "StatevectorSampler",
            "shots": args.shots,
        },
        "quantum_fit_seconds": quantum_fit_seconds,
        "classical_fit_seconds": classical_fit_seconds,
        "metrics": calculate_metrics(
            pd.Series(y_test),
            predictions,
            hybrid_probabilities,
        ),
        "component_metrics": {
            "quantum": calculate_metrics(
                pd.Series(y_test),
                (quantum_probabilities >= 0.5).astype(int),
                quantum_probabilities,
            ),
            "classical": calculate_metrics(
                pd.Series(y_test),
                (classical_probabilities >= 0.5).astype(int),
                classical_probabilities,
            ),
        },
    }


def main() -> None:
    args = parse_args()
    result = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
