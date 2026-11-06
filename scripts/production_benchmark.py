"""Measure the deployed scoring stack on domain-held-out rows.

``results/first_experiment.json`` feeds the Model Comparison panel. It used to be
written by ``backend.train_models.run_experiment``, which fits the classical models on
dataset rows only with 18 features at a 0.50 threshold - none of which describes what
serves scans today: URL-derived features, the legitimate hard-negative corpus, 0.35/0.65
classical fusion, the corpus-trained VQC at 0.25, the trusted-domain cap and the bandit
threshold. This script refits under that live recipe on the same stratified domain split
and scores the held-out rows through the live path, so the panel shows the deployed stack
measured on domains it has never seen.

``models/`` is never written; the benchmark models live in memory only. The quantum
component is the deployed artifact, and its own training rows are recomputed and asserted
disjoint from the held-out rows before any hybrid number is reported.

Usage:
    python -m scripts.production_benchmark [--fold 0] [--skip-realworld]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
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

from backend.domain_policy import hard_negative_blocks, is_trusted_domain_url
from backend.main import (
    CLASSICAL_WEIGHTS,
    QUANTUM_WEIGHT,
    _active_threshold,
    _load_rl_state,
)
from backend.quantum_experiment import balanced_sample
from backend.quantum_live import QUANTUM_FEATURES, load_quantum_model
from backend.train_models import (
    DEFAULT_DATASET,
    load_dataset,
    make_baseline,
    make_ensemble,
    select_features,
)
from backend.url_features import (
    TRUSTED_DOMAIN_CAP,
    extract_url_features_frame,
    trusted_calibration_applies,
)
from scripts.realworld_offline_score import SETS, score, summarize

DEFAULT_OUTPUT = Path("results") / "first_experiment.json"
DEFAULT_QUANTUM = Path("models") / "quantum_live.joblib"
FEATURE_PROFILE = "url_raw"
SEED = 42
TREES = 200
BOOSTING_ITERATIONS = 200
QUANTUM_BATCH_ROWS = 2048
BENCHMARK_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quantum", type=Path, default=DEFAULT_QUANTUM)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument(
        "--skip-realworld",
        action="store_true",
        help="Omit the curated real-world sets, which need a full feature pass.",
    )
    return parser.parse_args()


def production_negative_rows(features: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """Rebuild the production hard-negative rows from the shared block recipe."""
    frames: list[pd.DataFrame] = []
    provenance: list[dict] = []
    for urls, replicas in hard_negative_blocks():
        block = extract_url_features_frame(urls)[features]
        frames.extend([block] * replicas)
        provenance.append({"urls": len(urls), "replicas": replicas, "rows": len(urls) * replicas})
    return pd.concat(frames, ignore_index=True), provenance


def fold_split(frame: pd.DataFrame, target: pd.Series, groups: pd.Series, fold: int, features: list[str]):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(frame[features], target, groups))
    train_index, test_index = splits[fold]
    overlap = set(groups.iloc[train_index]) & set(groups.iloc[test_index])
    if overlap:
        raise RuntimeError(f"Domain leakage detected in split: {len(overlap)} groups overlap.")
    return train_index, test_index


def assert_quantum_is_out_of_sample(
    artifact: dict[str, Any],
    target: pd.Series,
    train_index: np.ndarray,
    test_index: np.ndarray,
    fold: int,
) -> dict[str, Any]:
    """Prove the live VQC never saw the held-out rows this benchmark scores."""
    circuit = artifact.get("circuit", {})
    # Artifacts predating the "fold" field were written by the CLI default,
    # which is the same fold the benchmark uses; the assumption is recorded.
    declared_fold = circuit.get("fold")
    artifact_fold = fold if declared_fold is None else int(declared_fold)
    per_class = int(circuit.get("train_per_class", 1000))
    if artifact_fold != fold:
        raise RuntimeError(
            f"Quantum artifact trained on fold {artifact_fold}, benchmark uses fold {fold}."
        )
    trained = balanced_sample(train_index, target, per_class, SEED)
    overlap = int(len(set(int(index) for index in trained) & set(int(index) for index in test_index)))
    if overlap:
        raise RuntimeError(f"Quantum artifact trained on {overlap} held-out benchmark rows.")
    return {
        "artifact_fold": artifact_fold,
        "artifact_fold_declared": declared_fold is not None,
        "artifact_train_rows_per_class": per_class,
        "artifact_extra_rows": int(circuit.get("extra_rows", 0)),
        "dataset_rows_trained_on": int(len(trained)),
        "overlap_with_held_out_rows": overlap,
    }


def quantum_probabilities(artifact: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    """Batched form of backend.quantum_live.predict_quantum_probability."""
    scaled = artifact["scaler"].transform(features[QUANTUM_FEATURES])
    classifier = artifact["classifier"]
    values = np.empty(len(scaled), dtype=float)
    for start in range(0, len(scaled), QUANTUM_BATCH_ROWS):
        stop = min(start + QUANTUM_BATCH_ROWS, len(scaled))
        values[start:stop] = np.asarray(classifier.predict_proba(scaled[start:stop]))[:, 1]
    return values


def live_probability(
    classical: np.ndarray,
    quantum: np.ndarray | None,
    urls: list[str],
    features: pd.DataFrame,
) -> np.ndarray:
    """0.75/0.25 fusion then the trusted-domain cap, exactly as backend.main.scan does."""
    if quantum is None:
        probability = classical.copy()
    else:
        probability = (1 - QUANTUM_WEIGHT) * classical + QUANTUM_WEIGHT * quantum
    for index, url in enumerate(urls):
        # is_trusted_domain_url is the first clause of the live predicate, so
        # skipping it here cannot change which rows are capped.
        if not is_trusted_domain_url(url):
            continue
        if trusted_calibration_applies(url, features.iloc[[index]]):
            probability[index] = min(probability[index], TRUSTED_DOMAIN_CAP)
    return probability


def score_metrics(
    y_test: pd.Series,
    probabilities: np.ndarray,
    schemes: pd.Series,
    threshold: float,
    *,
    fit_seconds: float | None = None,
    prediction_seconds: float | None = None,
) -> dict[str, Any]:
    predictions = probabilities >= threshold
    true_negatives, false_positives, false_negatives, true_positives = (
        confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    )
    legitimate = (y_test == 0).to_numpy()
    phishing = (y_test == 1).to_numpy()
    http_phishing = phishing & (schemes == "http").to_numpy()
    https_phishing = phishing & (schemes == "https").to_numpy()
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_phishing": float(precision_score(y_test, predictions, zero_division=0)),
        "recall_phishing": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_phishing": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "false_positive_rate": float(false_positives / int(legitimate.sum())),
        "confusion_matrix_labels": ["legitimate", "phishing"],
        "confusion_matrix": [
            [int(true_negatives), int(false_positives)],
            [int(false_negatives), int(true_positives)],
        ],
        "mean_probability_legitimate": float(probabilities[legitimate].mean()),
        "mean_probability_phishing": float(probabilities[phishing].mean()),
        "recall_phishing_http": safe_ratio(
            (predictions & http_phishing).sum(), http_phishing.sum()
        ),
        "recall_phishing_https": safe_ratio(
            (predictions & https_phishing).sum(), https_phishing.sum()
        ),
        "at_0.50": {
            "accuracy": float(accuracy_score(y_test, probabilities >= 0.50)),
            "f1_phishing": float(f1_score(y_test, probabilities >= 0.50)),
            "false_positive_rate": float((probabilities[legitimate] >= 0.50).mean()),
        },
    }
    if fit_seconds is not None:
        metrics["fit_seconds"] = fit_seconds
    if prediction_seconds is not None:
        metrics["prediction_seconds"] = prediction_seconds
    return metrics


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if not denominator else round(float(numerator / denominator), 4)


def round_floats(payload: dict[str, Any]) -> dict[str, Any]:
    for key, value in payload.items():
        if isinstance(value, dict):
            round_floats(value)
        elif isinstance(value, float):
            payload[key] = round(value, 4)
    return payload


def classical_fusion(models: dict, features: pd.DataFrame) -> np.ndarray:
    """Vectorised backend.main.fuse_classical_probabilities over a whole frame."""
    scores = {
        name: np.asarray(model.predict_proba(features)[:, 1], dtype=float)
        for name, model in models.items()
    }
    weights = {name: CLASSICAL_WEIGHTS.get(name, 1.0) for name in scores}
    total = sum(weights.values())
    return sum(weights[name] * scores[name] for name in scores) / total


def main() -> None:
    args = parse_args()
    threshold = _active_threshold(_load_rl_state())
    started = time.perf_counter()

    frame, target, groups = load_dataset(args.data)
    feature_frame = extract_url_features_frame(frame["URL"].astype(str).tolist())
    features = select_features(feature_frame, FEATURE_PROFILE)
    train_index, test_index = fold_split(feature_frame, target, groups, args.fold, features)

    x_test = feature_frame.iloc[test_index][features].reset_index(drop=True)
    y_test = target.iloc[test_index].reset_index(drop=True)
    test_urls = frame["URL"].astype(str).iloc[test_index].tolist()
    test_schemes = (
        pd.Series(test_urls)
        .str.lower()
        .str.extract(r"^(https?)")[0]
        .reset_index(drop=True)
    )

    negatives, negative_blocks = production_negative_rows(features)
    x_train = pd.concat(
        [feature_frame.iloc[train_index][features], negatives], ignore_index=True
    )
    y_train = pd.concat(
        [
            target.iloc[train_index],
            pd.Series([0] * len(negatives), dtype=int),
        ],
        ignore_index=True,
    )
    print(
        f"fold {args.fold}: {len(x_train)} training rows "
        f"({len(negatives)} hard-negative), {len(x_test)} held-out rows, "
        f"threshold {threshold:.2f}",
        flush=True,
    )

    models: dict[str, Any] = {}
    fit_times: dict[str, float] = {}
    for name, model in (
        ("A", make_baseline(SEED)),
        ("B", make_ensemble(SEED, trees=TREES, boosting_iterations=BOOSTING_ITERATIONS)),
    ):
        stage = time.perf_counter()
        model.fit(x_train, y_train)
        models[name] = model
        fit_times[name] = time.perf_counter() - stage
        print(f"  fitted {name} in {fit_times[name]:.1f}s", flush=True)

    artifact = load_quantum_model(args.quantum)
    leakage = assert_quantum_is_out_of_sample(artifact, target, train_index, test_index, args.fold)
    print(
        f"  quantum artifact fold {leakage['artifact_fold']} trained on "
        f"{leakage['dataset_rows_trained_on']} dataset rows + "
        f"{leakage['artifact_extra_rows']} corpus rows, 0 held-out overlap",
        flush=True,
    )

    stage = time.perf_counter()
    model_scores = {
        name: np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
        for name, model in models.items()
    }
    classical = classical_fusion(models, x_test)
    quantum_values = quantum_probabilities(artifact, x_test)
    hybrid = live_probability(classical, quantum_values, test_urls, x_test)
    predict_seconds = time.perf_counter() - stage

    report = {
        "experiment": "production-recipe domain-held-out benchmark",
        "benchmark_version": BENCHMARK_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "dataset": str(args.data),
        "seed": SEED,
        "fold": args.fold,
        "rows": int(len(frame)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "train_domains": int(len(set(groups.iloc[train_index]))),
        "test_domains": int(len(set(groups.iloc[test_index]))),
        "domain_overlap": 0,
        "label_counts": {
            "legitimate": int((target == 0).sum()),
            "phishing": int((target == 1).sum()),
        },
        "phishing_definition": "original label == 0",
        "feature_profile": FEATURE_PROFILE,
        "features": features,
        "feature_count": len(features),
        "threshold": threshold,
        "training_recipe": {
            "hard_negative_blocks": negative_blocks,
            "hard_negative_rows": int(len(negatives)),
            "classical_weights": dict(CLASSICAL_WEIGHTS),
            "quantum_weight": QUANTUM_WEIGHT,
            "trusted_domain_cap": TRUSTED_DOMAIN_CAP,
            "trees": TREES,
            "boosting_iterations": BOOSTING_ITERATIONS,
            "quantum_artifact": str(args.quantum),
            "models_written": False,
        },
        "leakage_checks": {"quantum": leakage},
        "metrics": {
            name: round_floats(
                score_metrics(
                    y_test,
                    model_scores[name],
                    test_schemes,
                    threshold,
                    fit_seconds=fit_times[name],
                )
            )
            for name in models
        },
        "classical_ensemble": round_floats(
            score_metrics(y_test, classical, test_schemes, threshold, prediction_seconds=predict_seconds)
        ),
        "quantum_only": round_floats(
            score_metrics(y_test, quantum_values, test_schemes, threshold)
        ),
        "hybrid": round_floats(
            score_metrics(y_test, hybrid, test_schemes, threshold, prediction_seconds=predict_seconds)
        ),
    }

    if not args.skip_realworld:
        realworld = {"threshold": threshold, "sets": {}}
        for name, pairs in SETS.items():
            urls = [url for _, url in pairs]
            groups_label = [group for group, _ in pairs]
            values = extract_url_features_frame(urls)
            rows = score(models, artifact, values, urls, threshold=threshold)
            summary = summarize(rows, groups_label)
            realworld["sets"][name] = {
                "summary": summary,
                "mistakes": [
                    row
                    for row, group in zip(rows, groups_label)
                    if (row["verdict"] == "legitimate") != (group == "legitimate")
                ],
            }
            print(
                f"  {name}: legit "
                f"{summary['legitimate']['all']['correct_rate']:.4f} phish "
                f"{summary['phishing']['all']['correct_rate']:.4f}",
                flush=True,
            )
        report["realworld"] = realworld

    report["quantum_circuit"] = dict(artifact.get("circuit", {}))
    report["weights"] = {"classical": dict(CLASSICAL_WEIGHTS), "quantum": QUANTUM_WEIGHT}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "hybrid_accuracy": report["hybrid"]["accuracy"],
                "hybrid_f1": report["hybrid"]["f1_phishing"],
                "hybrid_auc": report["hybrid"]["roc_auc"],
                "hybrid_fpr": report["hybrid"]["false_positive_rate"],
                "hybrid_recall": report["hybrid"]["recall_phishing"],
                "classical_accuracy": report["classical_ensemble"]["accuracy"],
                "quantum_only_accuracy": report["quantum_only"]["accuracy"],
                "A_accuracy": report["metrics"]["A"]["accuracy"],
                "B_accuracy": report["metrics"]["B"]["accuracy"],
                "wrote": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
