"""Persisted VQC inference for the live URL scoring path."""

from __future__ import annotations

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


def train_quantum_model(
    dataset: Path,
    output: Path,
    *,
    seed: int = 42,
    fold: int = 0,
    train_per_class: int = 500,
    maxiter: int = 30,
    shots: int = 256,
    feedback_urls: list[str] | None = None,
    feedback_targets: list[int] | None = None,
) -> None:
    from qiskit.circuit.library import real_amplitudes, zz_feature_map
    from qiskit.primitives import StatevectorSampler
    from qiskit_machine_learning.algorithms import VQC
    from qiskit_machine_learning.optimizers import COBYLA
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import MinMaxScaler

    from .quantum_experiment import balanced_sample
    from .train_models import load_dataset
    from .url_features import extract_url_features

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
        feedback_frame = pd.concat(
            [extract_url_features(url) for url in feedback_urls],
            ignore_index=True,
        )
        x_train = np.vstack(
            [x_train, scaler.transform(feedback_frame[QUANTUM_FEATURES])]
        )
        targets = np.concatenate([targets, np.asarray(feedback_targets, dtype=int)])
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
        optimizer=COBYLA(maxiter=maxiter),
        sampler=StatevectorSampler(default_shots=shots, seed=seed),
    )
    classifier.fit(x_train, targets)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    joblib.dump(
        {
            "scaler": scaler,
            "model_weights": classifier.weights,
            "features": QUANTUM_FEATURES,
            "weights": {"quantum": 0.25, "classical": 0.75},
            "circuit": {
                "feature_map": "ZZFeatureMap",
                "ansatz": "RealAmplitudes",
                "qubits": len(QUANTUM_FEATURES),
                "sampler": "StatevectorSampler",
                "shots": shots,
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
    from qiskit.circuit.library import real_amplitudes, zz_feature_map
    from qiskit.primitives import StatevectorSampler
    from qiskit_machine_learning.algorithms import VQC
    from qiskit_machine_learning.optimizers import COBYLA

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
        optimizer=COBYLA(maxiter=1),
        sampler=StatevectorSampler(
            default_shots=int(model.get("circuit", {}).get("shots", 256)),
            seed=42,
        ),
    )
    classifier._fit_result = type(
        "FitResult", (), {"x": np.asarray(model["model_weights"])}
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
