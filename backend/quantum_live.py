"""Persisted VQC inference for the live URL scoring path."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

QUANTUM_FEATURES = [
    "URLLength",
    "DomainLength",
    "NoOfSubDomain",
    "SpacialCharRatioInURL",
]
QUANTUM_CIRCUIT_REPS = 2
QUANTUM_DEFAULT_MAXITER = 60
QUANTUM_DEFAULT_SHOTS = 512
QUANTUM_DEFAULT_TRAIN_PER_CLASS = 1000
# Feedback adaptation runs in a background worker, so it uses a smaller
# optimization budget than the strengthened full-training defaults.
QUANTUM_ADAPTATION_MAXITER = 20


def _build_classifier(*, maxiter: int, shots: int, seed: int, reps: int):
    from qiskit.circuit.library import real_amplitudes, zz_feature_map
    from qiskit.primitives import StatevectorSampler
    from qiskit_machine_learning.algorithms import VQC
    from qiskit_machine_learning.optimizers import COBYLA

    return VQC(
        feature_map=zz_feature_map(
            feature_dimension=len(QUANTUM_FEATURES),
            reps=reps,
            entanglement="linear",
        ),
        ansatz=real_amplitudes(
            num_qubits=len(QUANTUM_FEATURES),
            reps=reps,
            entanglement="linear",
        ),
        optimizer=COBYLA(maxiter=maxiter),
        sampler=StatevectorSampler(default_shots=shots, seed=seed),
    )


def train_quantum_model(
    dataset: Path,
    output: Path,
    *,
    seed: int = 42,
    fold: int = 0,
    train_per_class: int = QUANTUM_DEFAULT_TRAIN_PER_CLASS,
    maxiter: int = QUANTUM_DEFAULT_MAXITER,
    shots: int = QUANTUM_DEFAULT_SHOTS,
    reps: int = QUANTUM_CIRCUIT_REPS,
    quantum_weight: float = 0.25,
    feedback_urls: list[str] | None = None,
    feedback_targets: list[int] | None = None,
) -> None:
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import MinMaxScaler

    from .quantum_experiment import balanced_sample
    from .train_models import load_dataset
    from .url_features import extract_url_features_frame

    frame, target, groups = load_dataset(dataset)
    splits = list(
        StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        ).split(frame[QUANTUM_FEATURES], target, groups)
    )
    train_indices, _ = splits[fold]
    train_sample = balanced_sample(train_indices, target, train_per_class, seed)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    x_train = scaler.fit_transform(frame.iloc[train_sample][QUANTUM_FEATURES])
    targets = target.iloc[train_sample].to_numpy()
    if feedback_urls and feedback_targets:
        feedback_frame = extract_url_features_frame(feedback_urls)
        x_train = np.vstack(
            [x_train, scaler.transform(feedback_frame[QUANTUM_FEATURES])]
        )
        targets = np.concatenate([targets, np.asarray(feedback_targets, dtype=int)])
    classifier = _build_classifier(
        maxiter=maxiter,
        shots=shots,
        seed=seed,
        reps=reps,
    )
    classifier.fit(x_train, targets)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    joblib.dump(
        {
            "scaler": scaler,
            "model_weights": classifier.weights,
            "features": QUANTUM_FEATURES,
            "weights": {
                "quantum": quantum_weight,
                "classical": 1 - quantum_weight,
            },
            "circuit": {
                "feature_map": "ZZFeatureMap",
                "ansatz": "RealAmplitudes",
                "qubits": len(QUANTUM_FEATURES),
                "reps": reps,
                "sampler": "StatevectorSampler",
                "shots": shots,
                "maxiter": maxiter,
                "train_per_class": train_per_class,
            },
        },
        temporary,
    )
    temporary.replace(output)


def load_quantum_model(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Quantum live model not found: {path}")
    model = joblib.load(path)
    if not isinstance(model, dict) or not {"scaler", "model_weights"} <= model.keys():
        raise ValueError("Quantum live model has an invalid artifact format.")
    circuit = model.get("circuit", {})
    weights = np.asarray(model["model_weights"])
    qubits = int(circuit.get("qubits", len(QUANTUM_FEATURES)))
    if "reps" in circuit:
        reps = int(circuit["reps"])
    else:
        # Linear-entanglement RealAmplitudes uses qubits parameters per rep
        # plus one final layer; infer reps from artifacts predating metadata.
        params_per_rep = qubits
        reps = max(1, (weights.size - qubits) // params_per_rep)
    classifier = _build_classifier(
        maxiter=1,
        shots=int(circuit.get("shots", QUANTUM_DEFAULT_SHOTS)),
        seed=42,
        reps=reps,
    )
    classifier._fit_result = type(
        "FitResult", (), {"x": weights}
    )()
    model["classifier"] = classifier
    return model


def predict_quantum_probability(
    model: dict[str, Any],
    features: pd.DataFrame,
) -> float:
    missing = [name for name in QUANTUM_FEATURES if name not in features]
    if missing:
        raise ValueError(f"Quantum features are missing: {missing}")
    scaled = model["scaler"].transform(features[QUANTUM_FEATURES])
    probabilities = np.asarray(model["classifier"].predict_proba(scaled))
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Quantum classifier returned invalid probabilities.")
    return float(probabilities[0, 1])


def adapt_live_model(
    dataset: Path,
    output: Path,
    feedback_urls: list[str],
    feedback_targets: list[int],
    *,
    seed: int = 42,
    quantum_weight: float = 0.25,
) -> None:
    """Background retrain path used by the feedback endpoint.

    Keeps the strengthened circuit config but a smaller optimization budget,
    because this runs inside a worker thread while the API serves scans.
    """

    train_quantum_model(
        dataset,
        output,
        seed=seed,
        train_per_class=QUANTUM_DEFAULT_TRAIN_PER_CLASS,
        maxiter=QUANTUM_ADAPTATION_MAXITER,
        shots=QUANTUM_DEFAULT_SHOTS,
        reps=QUANTUM_CIRCUIT_REPS,
        quantum_weight=quantum_weight,
        feedback_urls=feedback_urls,
        feedback_targets=feedback_targets,
    )


def parse_args() -> argparse.Namespace:
    from .train_models import DEFAULT_DATASET

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models") / "quantum_live.joblib",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument(
        "--train-per-class", type=int, default=QUANTUM_DEFAULT_TRAIN_PER_CLASS
    )
    parser.add_argument("--maxiter", type=int, default=QUANTUM_DEFAULT_MAXITER)
    parser.add_argument("--shots", type=int, default=QUANTUM_DEFAULT_SHOTS)
    parser.add_argument("--circuit-reps", type=int, default=QUANTUM_CIRCUIT_REPS)
    parser.add_argument("--quantum-weight", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_quantum_model(
        args.data,
        args.output,
        seed=args.seed,
        fold=args.fold,
        train_per_class=args.train_per_class,
        maxiter=args.maxiter,
        shots=args.shots,
        reps=args.circuit_reps,
        quantum_weight=args.quantum_weight,
    )
    print(f"Live quantum model written to {args.output}")


if __name__ == "__main__":
    main()
