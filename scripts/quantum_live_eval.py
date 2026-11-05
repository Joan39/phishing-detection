"""Eval two live quantum artifacts on a held-out domain split (fold 1)."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.quantum_experiment import balanced_sample
from backend.quantum_live import (
    QUANTUM_FEATURES,
    load_quantum_model,
    predict_quantum_probability,
)
from backend.train_models import DEFAULT_DATASET, load_dataset


def evaluate(artifact: Path, fold: int = 1, per_class: int = 250) -> dict:
    frame, target, groups = load_dataset(DEFAULT_DATASET)
    splits = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(
            frame[QUANTUM_FEATURES], target, groups
        )
    )
    _, test_indices = splits[fold]
    test_sample = balanced_sample(test_indices, target, per_class, 42)
    x_test = frame.iloc[test_sample][QUANTUM_FEATURES]
    y_test = target.iloc[test_sample].to_numpy()

    model = load_quantum_model(artifact)
    scaled = model["scaler"].transform(x_test)
    probs = np.asarray(model["classifier"].predict_proba(scaled))[:, 1]
    preds = (probs >= 0.5).astype(int)
    circuit = model.get("circuit", {})
    return {
        "artifact": str(artifact),
        "circuit": circuit,
        "params": int(np.asarray(model["model_weights"]).size),
        "rows": int(len(y_test)),
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "f1_phishing": round(float(f1_score(y_test, preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        "mean_prob_legitimate": round(float(probs[y_test == 0].mean()), 4),
        "mean_prob_phishing": round(float(probs[y_test == 1].mean()), 4),
    }


if __name__ == "__main__":
    out = [evaluate(Path(p)) for p in sys.argv[1:]]
    print(json.dumps(out, indent=2))
