"""Evaluate all detector configurations on shared domain-separated partitions.

The outer split creates disjoint training and test domains. A second grouped
split creates validation domains from the outer training domains. The
validation partition selects the hybrid quantum weight and decision threshold;
the test partition is used once for the final comparison.
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
from .train_models import (
    DEFAULT_DATASET,
    build_feature_frame,
    make_baseline,
    make_ensemble,
    load_dataset,
    select_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "three_way_evaluation.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--validation-fold", type=int, default=0, choices=range(4))
    parser.add_argument("--validation-per-class", type=int, default=250)
    parser.add_argument("--test-per-class", type=int, default=250)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--boosting-iterations", type=int, default=100)
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--quantum-weight", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        target,
        predictions,
        labels=[0, 1],
    ).ravel()
    legitimate_count = int((target == 0).sum())
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision_phishing": float(
            precision_score(target, predictions, zero_division=0)
        ),
        "recall_phishing": float(recall_score(target, predictions, zero_division=0)),
        "f1_phishing": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "false_positive_rate": float(fp / legitimate_count),
        "threshold": threshold,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def choose_validation_operating_point(
    classical: np.ndarray,
    quantum: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float, dict[str, Any]]:
    best: tuple[float, float, dict[str, Any]] | None = None
    for quantum_weight in np.arange(0.0, 1.01, 0.25):
        fused = quantum_weight * quantum + (1.0 - quantum_weight) * classical
        for threshold in np.arange(0.30, 0.71, 0.05):
            result = metrics(target, fused, float(threshold))
            score = (
                result["f1_phishing"],
                result["recall_phishing"],
                -result["false_positive_rate"],
            )
            if best is None or score > best[2]["selection_score"]:
                result["selection_score"] = score
                best = (float(quantum_weight), float(threshold), result)
    if best is None:
        raise RuntimeError("Could not select a validation operating point.")
    best[2].pop("selection_score")
    return best


def main() -> None:
    args = parse_args()
    if args.validation_per_class < 2 or args.test_per_class < 2:
        raise ValueError("Validation and test samples must contain at least two rows per class.")

    frame, target, groups = load_dataset(args.data)
    feature_frame = build_feature_frame(frame, "url")
    features = select_features(feature_frame, "url")

    outer = list(
        StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=args.seed,
        ).split(feature_frame[features], target, groups)
    )
    outer_train, outer_test = outer[args.fold]
    outer_train_frame = feature_frame.iloc[outer_train].reset_index(drop=True)
    outer_train_target = target.iloc[outer_train].reset_index(drop=True)
    outer_train_groups = groups.iloc[outer_train].reset_index(drop=True)
    inner = list(
        StratifiedGroupKFold(
            n_splits=4,
            shuffle=True,
            random_state=args.seed + 1,
        ).split(outer_train_frame[features], outer_train_target, outer_train_groups)
    )
    inner_train, inner_validation = inner[args.validation_fold]

    train_groups = set(outer_train_groups.iloc[inner_train])
    validation_groups = set(outer_train_groups.iloc[inner_validation])
    test_groups = set(groups.iloc[outer_test])
    if train_groups & validation_groups or train_groups & test_groups or validation_groups & test_groups:
        raise RuntimeError("Domain leakage detected between train, validation, and test.")

    validation_sample = balanced_sample(
        inner_validation,
        outer_train_target,
        args.validation_per_class,
        args.seed + 2,
    )
    test_sample = balanced_sample(
        outer_test,
        target,
        args.test_per_class,
        args.seed + 3,
    )

    x_train = feature_frame.iloc[outer_train].iloc[inner_train][features]
    y_train = outer_train_target.iloc[inner_train]
    x_validation = feature_frame.iloc[outer_train].iloc[validation_sample][features]
    y_validation = outer_train_target.iloc[validation_sample].to_numpy()
    x_test = feature_frame.iloc[outer_test][features]
    y_test_full = target.iloc[outer_test].to_numpy()
    x_test_sample = feature_frame.iloc[test_sample][features]
    y_test = target.iloc[test_sample].to_numpy()

    models = {
        "A": make_baseline(args.seed),
        "B": make_ensemble(
            args.seed,
            trees=args.trees,
            boosting_iterations=args.boosting_iterations,
        ),
    }
    probabilities: dict[str, dict[str, np.ndarray]] = {"validation": {}, "test": {}}
    fit_seconds: dict[str, float] = {}
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds[name] = time.perf_counter() - started
        probabilities["validation"][name] = model.predict_proba(x_validation)[:, 1]
        probabilities["test"][name] = model.predict_proba(x_test_sample)[:, 1]

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    q_train = balanced_sample(
        inner_train,
        outer_train_target,
        max(args.validation_per_class * 2, 500),
        args.seed + 4,
    )
    x_q_train = scaler.fit_transform(
        outer_train_frame.iloc[q_train][QUANTUM_FEATURES]
    )
    q_y_train = outer_train_target.iloc[q_train].to_numpy()
    x_q_validation = scaler.transform(
        outer_train_frame.iloc[validation_sample][QUANTUM_FEATURES]
    )
    x_q_test = scaler.transform(frame.iloc[test_sample][QUANTUM_FEATURES])
    classifier = VQC(
        feature_map=zz_feature_map(
            feature_dimension=len(QUANTUM_FEATURES),
            reps=1,
            entanglement="linear",
        ),
        ansatz=real_amplitudes(
            num_qubits=len(QUANTUM_FEATURES),
            reps=1,
            entanglement="linear",
        ),
        optimizer=COBYLA(maxiter=args.maxiter),
        sampler=StatevectorSampler(default_shots=args.shots, seed=args.seed),
    )
    started = time.perf_counter()
    classifier.fit(x_q_train, q_y_train)
    quantum_fit_seconds = time.perf_counter() - started
    probabilities["validation"]["quantum"] = classifier.predict_proba(x_q_validation)[:, 1]
    probabilities["test"]["quantum"] = classifier.predict_proba(x_q_test)[:, 1]

    validation_classical = probabilities["validation"]["B"]
    if args.quantum_weight is None or args.threshold is None:
        selected_weight, selected_threshold, validation_metrics = (
            choose_validation_operating_point(
                validation_classical,
                probabilities["validation"]["quantum"],
                y_validation,
            )
        )
    else:
        selected_weight, selected_threshold = args.quantum_weight, args.threshold
        validation_metrics = metrics(
            y_validation,
            selected_weight * probabilities["validation"]["quantum"]
            + (1.0 - selected_weight) * validation_classical,
            selected_threshold,
        )

    test_classical = probabilities["test"]["B"]
    test_quantum = probabilities["test"]["quantum"]
    hybrid_test = selected_weight * test_quantum + (1.0 - selected_weight) * test_classical

    result = {
        "experiment": "three-way domain-separated shared evaluation",
        "seed": args.seed,
        "outer_fold": args.fold,
        "validation_fold": args.validation_fold,
        "rows": int(len(frame)),
        "partitions": {
            "train_rows": int(len(inner_train)),
            "validation_rows": int(len(inner_validation)),
            "test_rows": int(len(outer_test)),
            "validation_evaluated_rows": int(len(validation_sample)),
            "test_evaluated_rows": int(len(test_sample)),
            "train_domains": int(len(train_groups)),
            "validation_domains": int(len(validation_groups)),
            "test_domains": int(len(test_groups)),
            "train_validation_overlap": int(len(train_groups & validation_groups)),
            "train_test_overlap": int(len(train_groups & test_groups)),
            "validation_test_overlap": int(len(validation_groups & test_groups)),
        },
        "features": features,
        "quantum_features": QUANTUM_FEATURES,
        "quantum_training_rows": int(len(q_train)),
        "quantum_fit_seconds": quantum_fit_seconds,
        "classical_fit_seconds": fit_seconds,
        "selection": {
            "quantum_weight": selected_weight,
            "classical_weight": 1.0 - selected_weight,
            "threshold": selected_threshold,
            "validation_metrics": validation_metrics,
        },
        "metrics": {
            "logistic_regression": metrics(
                y_test,
                probabilities["test"]["A"],
                0.5,
            ),
            "soft_voting_ensemble": metrics(
                y_test,
                probabilities["test"]["B"],
                0.5,
            ),
            "quantum_vqc": metrics(
                y_test,
                test_quantum,
                0.5,
            ),
            "hybrid": metrics(y_test, hybrid_test, selected_threshold),
        },
        "full_test_partition_reference": {
            "soft_voting_ensemble": metrics(
                y_test_full,
                models["B"].predict_proba(feature_frame.iloc[outer_test][features])[:, 1],
                0.5,
            )
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
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
