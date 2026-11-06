"""Compare legitimate hard-negative corpora for the production models.

The PhiUSIIL corpus cannot answer the question this experiment asks. Its
legitimate rows are all short HTTPS bare domains, so a model trained on it
alone flags any unfamiliar path as phishing, while a real-world battery of
ordinary sites fails. Each configuration here is scored on both:

    held-out fold   - unseen domains from the dataset (does the corpus still
                      separate phishing from legitimate?)
    real-world set  - curated legit and phishing URLs the model has never seen
                      (does ordinary traffic still pass?)

Negatives for the real-world configuration use hosts deliberately disjoint
from the battery hosts, so a battery gain reflects shape generalization rather
than memorized brands. Quantum is excluded: it trains on dataset rows only and
is unchanged by this experiment, so the classical path is the variable.

Usage:
    python -m scripts.legit_negative_experiment
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from backend.domain_policy import (
    GENERAL_LEGITIMATE_HOSTS,
    legitimate_training_urls,
)
from backend.main import CLASSICAL_WEIGHTS
from backend.train_models import (
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
from scripts.fresh_probe import FRESH_LEGITIMATE, FRESH_PHISHING
from scripts.realworld_battery import LEGITIMATE_URLS, PHISHING_URLS

DEFAULT_OUTPUT = Path("results") / "legit_negative_experiment.json"
FOLD = 0
SEED = 42
TREES = 200
BOOSTING_ITERATIONS = 200
THRESHOLD = 0.40

CONFIGURATIONS = (
    {
        "name": "extended_paths_bare60",
        "replicas": 10,
        "general": True,
        "bare_http_uplift": 60,
    },
    {
        "name": "extended_paths_bare60_r16",
        "replicas": 16,
        "general": True,
        "bare_http_uplift": 60,
    },
    {
        "name": "extended_paths_bare60_https_only",
        "replicas": 10,
        "general": True,
        "bare_http_uplift": 60,
        "http_paths": 0,
    },
)

BATTERY_URLS = [("legitimate", url) for url in LEGITIMATE_URLS] + [
    ("phishing", url) for url in PHISHING_URLS
]

PROBE_URLS = [("legitimate", url) for url in FRESH_LEGITIMATE] + [
    ("phishing", url) for url in FRESH_PHISHING
]


def _features(urls: list[str]) -> pd.DataFrame:
    return extract_url_features_frame(urls)


def load_fold_data() -> dict:
    frame, target, groups = load_dataset(
        Path("original") / "PhiUSIIL_Phishing_URL_Dataset.csv"
    )
    feature_frame = _features(frame["URL"].astype(str).tolist())
    features = select_features(feature_frame, "url_raw")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_index, test_index = list(splitter.split(feature_frame[features], target, groups))[FOLD]
    battery = _features([url for _, url in BATTERY_URLS])
    probe = _features([url for _, url in PROBE_URLS])
    return {
        "x_train": feature_frame.iloc[train_index][features].reset_index(drop=True),
        "y_train": target.iloc[train_index].reset_index(drop=True),
        "x_test": feature_frame.iloc[test_index][features].reset_index(drop=True),
        "y_test": target.iloc[test_index].reset_index(drop=True),
        "test_schemes": frame["URL"].astype(str).str.lower().str.extract(
            r"^(https?)"
        )[0].iloc[test_index].reset_index(drop=True),
        "battery_features": battery[features].reset_index(drop=True),
        "battery_urls": [url for _, url in BATTERY_URLS],
        "battery_groups": [group for group, _ in BATTERY_URLS],
        "probe_features": probe[features].reset_index(drop=True),
        "probe_urls": [url for _, url in PROBE_URLS],
        "probe_groups": [group for group, _ in PROBE_URLS],
        "features": features,
    }


def fold_metrics(
    probabilities: np.ndarray,
    y_test: pd.Series,
    schemes: pd.Series,
) -> dict:
    predictions = probabilities >= THRESHOLD
    matrix = confusion_matrix(y_test, predictions, labels=[False, True])
    true_negatives, false_positives, false_negatives, true_positives = matrix.ravel()
    legitimate = (y_test == 0).to_numpy()
    phishing = (y_test == 1).to_numpy()
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1_phishing": round(float(f1_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "legitimate_false_positive_rate": round(
            float(false_positives / int(legitimate.sum())), 4
        ),
        "phishing_recall": round(float(true_positives / int(phishing.sum())), 4),
        "phishing_recall_http": round(
            float((predictions & phishing & (schemes == "http").to_numpy()).sum())
            / float((phishing & (schemes == "http").to_numpy()).sum()),
            4,
        ),
        "phishing_recall_https": round(
            float((predictions & phishing & (schemes == "https").to_numpy()).sum())
            / float((phishing & (schemes == "https").to_numpy()).sum()),
            4,
        ),
        "mean_probability_legitimate": round(float(probabilities[legitimate].mean()), 4),
        "mean_probability_phishing": round(float(probabilities[phishing].mean()), 4),
        "at_0.50": {
            "accuracy": round(float(accuracy_score(y_test, probabilities >= 0.50)), 4),
            "f1_phishing": round(float(f1_score(y_test, probabilities >= 0.50)), 4),
            "legitimate_false_positive_rate": round(
                float((probabilities[legitimate] >= 0.50).mean()), 4
            ),
        },
    }


def _predict_probabilities(model, features: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float)


def fuse_scores(scores: dict[str, np.ndarray]) -> np.ndarray:
    """Vectorised form of backend.main.fuse_classical_probabilities.

    The live endpoint fuses row by row; identical weights and normalisation are
    applied here so batch scoring agrees with the API to floating-point order.
    """
    weights = {name: CLASSICAL_WEIGHTS.get(name, 1.0) for name in scores}
    total = sum(weights.values())
    return sum(
        weights[name] * scores[name] for name in scores
    ) / total


def realworld_metrics(models, data: dict, prefix: str) -> dict:
    features = data[f"{prefix}_features"]
    scores = {
        name: _predict_probabilities(model, features) for name, model in models.items()
    }
    classical = fuse_scores(scores)
    rows = []
    for index, url in enumerate(data[f"{prefix}_urls"]):
        probability = float(classical[index])
        if trusted_calibration_applies(url, features.iloc[[index]]):
            probability = min(probability, TRUSTED_DOMAIN_CAP)
        rows.append(
            {
                "group": data[f"{prefix}_groups"][index],
                "url": url,
                "scheme": "http" if url.startswith("http://") else "https",
                "verdict": "phishing" if probability >= THRESHOLD else "legitimate",
                "probability": round(probability, 4),
            }
        )
    summary: dict[str, dict] = {}
    for row in rows:
        correct = (row["group"] == "legitimate") != (row["verdict"] == "phishing")
        for scope in ("all", row["scheme"]):
            bucket = summary.setdefault(row["group"], {}).setdefault(
                scope, {"n": 0, "correct": 0, "score_sum": 0.0}
            )
            bucket["n"] += 1
            bucket["correct"] += int(correct)
            bucket["score_sum"] += row["probability"]
    report = {
        group: {
            scope: {
                "n": bucket["n"],
                "correct_rate": round(bucket["correct"] / bucket["n"], 4),
                "mean_probability": round(bucket["score_sum"] / bucket["n"], 4),
            }
            for scope, bucket in scopes.items()
        }
        for group, scopes in summary.items()
    }
    misses = [row for row in rows if (row["group"] == "legitimate") == (row["verdict"] == "phishing")]
    return {"summary": report, "misses": misses}


def run() -> dict:
    started = time.perf_counter()
    data = load_fold_data()
    print(f"fold data ready in {time.perf_counter() - started:.1f}s", flush=True)
    results = {}
    for config in CONFIGURATIONS:
        urls = legitimate_training_urls(
            include_general_hosts=config["general"],
            http_paths_per_host=config.get("http_paths"),
        )
        negatives = _features(urls)
        negatives = pd.concat([negatives] * config["replicas"], ignore_index=True)[
            data["features"]
        ]
        uplift = config.get("bare_http_uplift", 0)
        if uplift:
            bare = _features(
                [f"http://{host}" for host in GENERAL_LEGITIMATE_HOSTS]
                + [f"http://{host}/" for host in GENERAL_LEGITIMATE_HOSTS]
            )[data["features"]]
            negatives = pd.concat([negatives] + [bare] * uplift, ignore_index=True)
        x_train = pd.concat([data["x_train"], negatives], ignore_index=True)
        y_train = pd.concat(
            [data["y_train"], pd.Series([0] * len(negatives), dtype=int)],
            ignore_index=True,
        )
        models = {
            "A": make_baseline(SEED),
            "B": make_ensemble(
                SEED, trees=TREES, boosting_iterations=BOOSTING_ITERATIONS
            ),
        }
        fitted: dict[str, object] = {}
        stage = time.perf_counter()
        for name, model in models.items():
            model.fit(x_train, y_train)
            fitted[name] = model
            print(f"  {config['name']} {name} fitted in {time.perf_counter() - stage:.1f}s", flush=True)
            stage = time.perf_counter()
        test_scores = {
            name: _predict_probabilities(model, data["x_test"])
            for name, model in fitted.items()
        }
        results[config["name"]] = {
            "negative_urls": len(urls),
            "negative_rows": int(len(negatives)),
            "training_rows": int(len(x_train)),
            "held_out_fold": fold_metrics(
                fuse_scores(test_scores), data["y_test"], data["test_schemes"]
            ),
            "realworld": realworld_metrics(fitted, data, "battery"),
            "fresh_probe": realworld_metrics(fitted, data, "probe"),
            "per_model": {
                name: fold_metrics(values, data["y_test"], data["test_schemes"])
                for name, values in test_scores.items()
            },
        }
        print(
            f"{config['name']}: fold={results[config['name']]['held_out_fold']['accuracy']} "
            f"legit_battery={results[config['name']]['realworld']['summary']['legitimate']['all']['correct_rate']} "
            f"phish_battery={results[config['name']]['realworld']['summary']['phishing']['all']['correct_rate']} "
            f"legit_probe={results[config['name']]['fresh_probe']['summary']['legitimate']['all']['correct_rate']} "
            f"phish_probe={results[config['name']]['fresh_probe']['summary']['phishing']['all']['correct_rate']}",
            flush=True,
        )
    return results


def main() -> None:
    results = run()
    DEFAULT_OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({
        name: {
            "fold_accuracy": payload["held_out_fold"]["accuracy"],
            "fold_legit_fpr": payload["held_out_fold"]["legitimate_false_positive_rate"],
            "fold_phish_recall": payload["held_out_fold"]["phishing_recall"],
            "battery_legit": payload["realworld"]["summary"]["legitimate"]["all"]["correct_rate"],
            "battery_legit_http": payload["realworld"]["summary"]["legitimate"]["http"]["correct_rate"],
            "battery_phishing": payload["realworld"]["summary"]["phishing"]["all"]["correct_rate"],
            "probe_legit": payload["fresh_probe"]["summary"]["legitimate"]["all"]["correct_rate"],
            "probe_phishing": payload["fresh_probe"]["summary"]["phishing"]["all"]["correct_rate"],
        }
        for name, payload in results.items()
    }, indent=2))


if __name__ == "__main__":
    main()
