"""Run a small, reproducible quantum-classifier experiment.

The full dataset is intentionally not sent through a variational quantum
classifier: current simulators and hardware have a small practical circuit
budget. This experiment keeps the same domain-held-out split as the classical
models, samples balanced subsets, and reports the sample sizes and simulator
settings so the comparison is not mistaken for a full-dataset result.
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

from .train_models import DEFAULT_DATASET, load_dataset


QUANTUM_FEATURES = [
    "URLLength",
    "DomainLength",
    "NoOfSubDomain",
    "SpacialCharRatioInURL",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "quantum_experiment.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=500)
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--shots", type=int, default=512)
    return parser.parse_args()


def balanced_sample(
    indices: np.ndarray,
    target: pd.Series,
    per_class: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for label in (0, 1):
        candidates = indices[target.iloc[indices].to_numpy() == label]
        if len(candidates) < per_class:
            raise ValueError(
                f"Class {label} has only {len(candidates)} rows; "
                f"cannot sample {per_class}."
            )
        selected.append(rng.choice(candidates, size=per_class, replace=False))
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result


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
    sampler = StatevectorSampler(default_shots=args.shots, seed=args.seed)
    classifier = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=args.maxiter),
        sampler=sampler,
    )

    started = time.perf_counter()
    classifier.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictions = np.asarray(classifier.predict(x_test)).reshape(-1).astype(int)
    prediction_seconds = time.perf_counter() - started
    probabilities = classifier.predict_proba(x_test)[:, 1]

    return {
        "experiment": "domain-held-out quantum VQC subset experiment",
        "seed": args.seed,
        "fold": args.fold,
        "rows": int(len(frame)),
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "train_sample_rows": int(len(train_sample)),
        "test_sample_rows": int(len(test_sample)),
        "train_sample_per_class": args.train_per_class,
        "test_sample_per_class": args.test_per_class,
        "train_domains": int(groups.iloc[train_sample].nunique()),
        "test_domains": int(groups.iloc[test_sample].nunique()),
        "domain_overlap": int(
            len(
                set(groups.iloc[train_indices])
                & set(groups.iloc[test_indices])
            )
        ),
        "features": QUANTUM_FEATURES,
        "circuit": {
            "feature_map": "ZZFeatureMap",
            "ansatz": "RealAmplitudes",
            "qubits": len(QUANTUM_FEATURES),
            "optimizer": "COBYLA",
            "maxiter": args.maxiter,
            "sampler": "StatevectorSampler",
            "shots": args.shots,
        },
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "metrics": calculate_metrics(
            pd.Series(y_test),
            predictions,
            probabilities,
        ),
    }


def main() -> None:
    args = parse_args()
    result = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
