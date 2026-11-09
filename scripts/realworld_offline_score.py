"""Score the deployed artifacts offline on both real-world sets.

The live battery needs a running server and one request per URL; this loads
``models/url/*.joblib`` and ``models/quantum_live.joblib`` directly and applies
the same fusion, trusted-domain cap and threshold, so an artifact swap can be
measured in seconds instead of minutes and compared with the live run.

Usage:
    python -m scripts.realworld_offline_score [--quantum PATH] [--policy PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.adaptive_policy import fuse_adaptive, load_policy
from backend.domain_policy import hard_negative_blocks
from backend.main import _load_policy, score_redirect_targets
from backend.quantum_live import QUANTUM_FEATURES, load_quantum_model
from backend.url_features import (
    TRUSTED_DOMAIN_CAP,
    extract_url_features_frame,
    trusted_calibration_applies,
)
from scripts.fresh_probe import FRESH_LEGITIMATE, FRESH_PHISHING
from scripts.realworld_battery import LEGITIMATE_URLS, PHISHING_URLS, _scheme

DEFAULT_MODEL_DIR = Path("models") / "url"
DEFAULT_QUANTUM = Path("models") / "quantum_live.joblib"
THRESHOLD = 0.40

SETS = {
    "battery": [
        ("legitimate", url) for url in LEGITIMATE_URLS
    ] + [("phishing", url) for url in PHISHING_URLS],
    "probe": [
        ("legitimate", url) for url in FRESH_LEGITIMATE
    ] + [("phishing", url) for url in FRESH_PHISHING],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--quantum", type=Path, default=DEFAULT_QUANTUM)
    parser.add_argument("--classical-only", action="store_true")
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Score with the weights and per-class thresholds in this policy "
        "file instead of the live one.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path; console output is always printed.",
    )
    return parser.parse_args()


def score(
    models: dict,
    quantum,
    features: pd.DataFrame,
    urls: list[str],
    threshold: float = THRESHOLD,
    policy: dict | None = None,
) -> list[dict]:
    """Reproduce the live fusion row by row, adaptive weights included."""
    policy = policy if policy is not None else _load_policy()
    classical_weights = policy["classical_weights"] or {
        name: 1.0 for name in models
    }
    rows = []
    for index, url in enumerate(urls):
        row = features.iloc[[index]]
        component_scores = {
            name: float(model.predict_proba(row)[:, 1][0])
            for name, model in models.items()
        }
        quantum_value = (
            None
            if quantum is None
            else float(
                np.asarray(quantum["classifier"].predict_proba(
                    quantum["scaler"].transform(row[QUANTUM_FEATURES])
                ))[0, 1]
            )
        )
        if quantum_value is not None:
            component_scores["quantum"] = quantum_value
        probability = fuse_adaptive(
            component_scores, classical_weights, policy["quantum_weight"]
        )
        capped = trusted_calibration_applies(url, row)
        if capped:
            probability = min(probability, TRUSTED_DOMAIN_CAP)
        chain = score_redirect_targets(url, models, quantum, policy)
        target_probability = None if chain is None else chain[1]
        if target_probability is not None and target_probability > probability:
            probability = target_probability
        rows.append(
            {
                "url": url,
                "scheme": _scheme(url),
                "probability": round(float(probability), 4),
                "A": round(float(models["A"].predict_proba(row)[:, 1][0]), 4),
                "B": round(float(models["B"].predict_proba(row)[:, 1][0]), 4),
                "quantum": None if quantum_value is None else round(quantum_value, 4),
                "trusted_cap": capped,
                "redirect_target_probability": (
                    None if target_probability is None else round(target_probability, 4)
                ),
                "verdict": "phishing" if probability >= threshold else "legitimate",
            }
        )
    return rows


def summarize(rows: list[dict], groups: list[str]) -> dict:
    summary: dict[str, dict] = {}
    for row, group in zip(rows, groups):
        correct = (row["verdict"] == "legitimate") == (group == "legitimate")
        for scope in ("all", row["scheme"]):
            bucket = summary.setdefault(group, {}).setdefault(
                scope, {"n": 0, "correct": 0, "score_sum": 0.0}
            )
            bucket["n"] += 1
            bucket["correct"] += int(correct)
            bucket["score_sum"] += row["probability"]
    return {
        group: {
            scope: {
                "n": values["n"],
                "correct_rate": round(values["correct"] / values["n"], 4),
                "mean_probability": round(values["score_sum"] / values["n"], 4),
            }
            for scope, values in scopes.items()
        }
        for group, scopes in summary.items()
    }


def main() -> None:
    args = parse_args()
    models = {
        name: joblib.load(args.model_dir / f"model_{name}.joblib") for name in ("A", "B")
    }
    quantum = None if args.classical_only else load_quantum_model(args.quantum)
    policy = load_policy(args.policy) if args.policy is not None else _load_policy()
    print(
        f"models={args.model_dir} quantum={'off' if quantum is None else args.quantum} "
        f"policy={'live' if args.policy is None else args.policy} "
        f"classical_weights={policy['classical_weights']} "
        f"quantum_weight={policy['quantum_weight']} "
        f"negatives={sum(len(urls) * replicas for urls, replicas in hard_negative_blocks())} rows"
    )
    report = {}
    if quantum is not None:
        from sklearn.metrics import roc_auc_score

        pairs = [pair for set_ in SETS.values() for pair in set_]
        urls = [url for _, url in pairs]
        features = extract_url_features_frame(urls)
        q_values = np.asarray(
            quantum["classifier"].predict_proba(
                quantum["scaler"].transform(features[QUANTUM_FEATURES])
            )
        )[:, 1]
        labels = np.asarray([1 if group == "phishing" else 0 for group, _ in pairs])
        print(
            f"\nquantum alone on both real-world sets: "
            f"AUC={roc_auc_score(labels, q_values):.4f} "
            f"mean_legit={q_values[labels == 0].mean():.3f} "
            f"mean_phish={q_values[labels == 1].mean():.3f}"
        )
    for name, pairs in SETS.items():
        urls = [url for _, url in pairs]
        groups = [group for group, _ in pairs]
        features = extract_url_features_frame(urls)
        rows = score(models, quantum, features, urls, policy=policy)
        summary = summarize(rows, groups)
        report[name] = {
            "rows": rows,
            "summary": summary,
            "misses": [
                row
                for row, group in zip(rows, groups)
                if (row["verdict"] == "legitimate") != (group == "legitimate")
            ],
        }
        print(f"\n{name}:")
        for group in ("legitimate", "phishing"):
            for scope, values in summary[group].items():
                print(
                    f"  {group:<10} {scope:<6} n={values['n']:<3} "
                    f"correct={values['correct_rate']:.3f} "
                    f"mean_prob={values['mean_probability']:.3f}"
                )
        for threshold in (0.40, 0.45, 0.50, 0.55):
            rates = {
                group: round(
                    float(
                        np.mean(
                            [
                                (row["probability"] < threshold)
                                == (row_group == "legitimate")
                                for row, row_group in zip(rows, groups)
                                if row_group == group
                            ]
                        )
                    ),
                    4,
                )
                for group in ("legitimate", "phishing")
            }
            print(
                f"  threshold {threshold:.2f}: legit={rates['legitimate']:.4f} "
                f"phishing={rates['phishing']:.4f}"
            )
        for row in report[name]["misses"]:
            print(
                f"  MISS {row['probability']:.3f} A={row['A']:.2f} B={row['B']:.2f} "
                f"Q={row['quantum']} {row['url']}"
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
