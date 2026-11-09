"""Diagnostic: is the deployed training recipe over- or under-fitting?

Mirrors scripts.production_benchmark's fold-0 split, then reports train-vs-held-out
gaps, calibration, and the accuracy ceiling of the four features the quantum
circuit is allowed to see.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from backend.quantum_live import QUANTUM_FEATURES
from backend.train_models import (
    DEFAULT_DATASET,
    make_baseline,
    make_ensemble,
    load_dataset,
    select_features,
)
from backend.url_features import extract_url_features_frame
from scripts.production_benchmark import (
    BOOSTING_ITERATIONS,
    FEATURE_PROFILE,
    SEED,
    TREES,
    fold_split,
    production_negative_rows,
)


def reliability(y_true, p) -> list[tuple[int, float, float]]:
    bins = np.floor(np.clip(p, 0, 0.999) * 10).astype(int)
    rows = []
    for b in range(10):
        mask = bins == b
        if mask.sum():
            rows.append((b, float(np.mean(p[mask])), float(np.mean(y_true[mask]))))
    return rows


def report(name: str, model, x_fit, y_fit, x_test, y_test) -> None:
    stage = time.perf_counter()
    model.fit(x_fit, np.asarray(y_fit))
    print(f"\n{name}: fitted in {time.perf_counter() - stage:.1f}s", flush=True)
    p_train = np.asarray(model.predict_proba(x_fit)[:, 1], dtype=float)
    p_test = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
    print(
        f"\n{name}: train acc={accuracy_score(y_fit, p_train >= 0.5):.4f} "
        f"auc={roc_auc_score(y_fit, p_train):.4f} logloss={log_loss(y_fit, p_train):.4f} "
        f"brier={brier_score_loss(y_fit, p_train):.4f}",
        flush=True,
    )
    print(
        f"{name}: held-out acc={accuracy_score(y_test, p_test >= 0.5):.4f} "
        f"auc={roc_auc_score(y_test, p_test):.4f} logloss={log_loss(y_test, p_test):.4f} "
        f"brier={brier_score_loss(y_test, p_test):.4f}",
        flush=True,
    )
    print(
        f"{name}: gap acc={accuracy_score(y_fit, p_train >= 0.5) - accuracy_score(y_test, p_test >= 0.5):+.4f} "
        f"auc={roc_auc_score(y_fit, p_train) - roc_auc_score(y_test, p_test):+.4f}",
        flush=True,
    )
    print(
        f"{name}: held-out reliability (bin, mean predicted, observed rate)",
        [
            (b, round(pm, 3), round(om, 3))
            for b, pm, om in reliability(np.asarray(y_test), p_test)
        ],
        flush=True,
    )


def main() -> None:
    started = time.perf_counter()
    frame, target, groups = load_dataset(DEFAULT_DATASET)
    stage = time.perf_counter()
    feature_frame = extract_url_features_frame(frame["URL"].astype(str).tolist())
    print(f"features extracted in {time.perf_counter() - stage:.1f}s", flush=True)
    features = select_features(feature_frame, FEATURE_PROFILE)
    train_index, test_index = fold_split(feature_frame, target, groups, 0, features)

    x_test = feature_frame.iloc[test_index][features].reset_index(drop=True)
    y_test = target.iloc[test_index].reset_index(drop=True)
    negatives, _ = production_negative_rows(features)
    x_fit = pd.concat(
        [feature_frame.iloc[train_index][features], negatives], ignore_index=True
    )
    y_fit = pd.concat(
        [target.iloc[train_index], pd.Series([0] * len(negatives), dtype=int)],
        ignore_index=True,
    )
    print(
        f"split: {len(x_fit)} training rows ({len(negatives)} hard negatives, "
        f"{np.mean(np.asarray(y_fit) == 0):.3f} legitimate share), {len(x_test)} held-out rows",
        flush=True,
    )

    report("A", make_baseline(SEED), x_fit, y_fit, x_test, y_test)
    report("B", make_ensemble(SEED, trees=TREES, boosting_iterations=BOOSTING_ITERATIONS), x_fit, y_fit, x_test, y_test)

    q_train = x_fit[QUANTUM_FEATURES]
    q_test = x_test[QUANTUM_FEATURES]
    report("classical fit on the circuit's 4 features", make_baseline(SEED), q_train, y_fit, q_test, y_test)
    print(f"\ntotal {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
