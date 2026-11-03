"""Evaluate an RL threshold adapter under a controlled URL-pattern drift.

This is a simulation, not a live-feedback claim. The base ensemble is trained
on the original training fold. Phishing examples in the held-out fold are
then changed to mimic more legitimate-looking URL patterns. Labels are
revealed in batches to a threshold bandit, representing delayed analyst or
user feedback.
"""

from __future__ import annotations

import argparse
import json
import time
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

from .train_models import (
    DEFAULT_DATASET,
    KNOWN_LEAKAGE_COLUMNS,
    IDENTIFIER_COLUMNS,
    load_dataset,
    make_ensemble,
    select_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "adaptation_experiment.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--boosting-iterations", type=int, default=100)
    return parser.parse_args()


def simulate_legitimacy_mimicking_drift(
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    drifted = features.reset_index(drop=True).copy()
    target = target.reset_index(drop=True)
    phishing = target.eq(1)
    for column, amount in (
        ("NoOfOtherSpecialCharsInURL", -1),
        ("NoOfSubDomain", -1),
        ("NoOfObfuscatedChar", -1),
        ("URLLength", -8),
    ):
        if column in drifted:
            drifted.loc[phishing, column] = (
                drifted.loc[phishing, column] + amount
            ).clip(lower=0)
    if "SpacialCharRatioInURL" in drifted:
        drifted.loc[phishing, "SpacialCharRatioInURL"] = (
            drifted.loc[phishing, "SpacialCharRatioInURL"] - 0.01
        ).clip(lower=0)
    return drifted


def metrics_from_predictions(
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


class ThresholdBandit:
    """A small epsilon-greedy RL agent whose actions are decision thresholds."""

    def __init__(
        self,
        thresholds: tuple[float, ...],
        seed: int,
        epsilon: float = 0.10,
        learning_rate: float = 0.20,
    ) -> None:
        self.thresholds = thresholds
        self.q_values = np.zeros(len(thresholds), dtype=float)
        self.rng = np.random.default_rng(seed)
        self.epsilon = epsilon
        self.learning_rate = learning_rate

    def choose(self) -> tuple[int, float]:
        if self.rng.random() < self.epsilon:
            index = int(self.rng.integers(len(self.thresholds)))
        else:
            index = int(np.argmax(self.q_values))
        return index, self.thresholds[index]

    def update(self, action: int, reward: float) -> None:
        self.q_values[action] += self.learning_rate * (
            reward - self.q_values[action]
        )


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    frame, target, groups = load_dataset(args.data)
    features = select_features(frame, "url")
    splits = list(
        StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=args.seed,
        ).split(frame[features], target, groups)
    )
    train_indices, test_indices = splits[args.fold]
    x_train = frame.iloc[train_indices][features]
    x_test = frame.iloc[test_indices][features]
    y_train = target.iloc[train_indices]
    y_test = target.iloc[test_indices]

    model = make_ensemble(
        args.seed,
        trees=args.trees,
        boosting_iterations=args.boosting_iterations,
    )
    started = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started

    drifted_test = simulate_legitimacy_mimicking_drift(x_test, y_test)
    probabilities = model.predict_proba(drifted_test)[:, 1]
    static_predictions = (probabilities >= 0.5).astype(int)
    agent = ThresholdBandit((0.30, 0.40, 0.50, 0.60, 0.70), args.seed)
    adaptive_predictions = np.zeros(len(y_test), dtype=int)
    rewards: list[float] = []
    selected_thresholds: list[float] = []

    for start in range(0, len(y_test), args.batch_size):
        stop = min(start + args.batch_size, len(y_test))
        action, threshold = agent.choose()
        batch_predictions = (probabilities[start:stop] >= threshold).astype(int)
        batch_target = y_test.iloc[start:stop].to_numpy()
        reward = np.where(
            batch_predictions == batch_target,
            1.0,
            np.where(batch_predictions == 1, -2.0, -3.0),
        ).mean()
        agent.update(action, float(reward))
        adaptive_predictions[start:stop] = batch_predictions
        rewards.append(float(reward))
        selected_thresholds.append(threshold)

    return {
        "experiment": "simulated legitimacy-mimicking drift with RL threshold adaptation",
        "seed": args.seed,
        "fold": args.fold,
        "batch_size": args.batch_size,
        "rows": int(len(frame)),
        "test_rows": int(len(y_test)),
        "domain_overlap": int(
            len(set(groups.iloc[train_indices]) & set(groups.iloc[test_indices]))
        ),
        "feature_profile": "url",
        "excluded_columns": sorted(IDENTIFIER_COLUMNS | KNOWN_LEAKAGE_COLUMNS),
        "drift": {
            "description": "Phishing test examples become shorter and less obfuscated.",
            "affected_columns": [
                "NoOfOtherSpecialCharsInURL",
                "NoOfSubDomain",
                "NoOfObfuscatedChar",
                "URLLength",
                "SpacialCharRatioInURL",
            ],
        },
        "base_model_fit_seconds": fit_seconds,
        "static_threshold_0_5": metrics_from_predictions(
            y_test,
            static_predictions,
            probabilities,
        ),
        "rl_adaptive_threshold": metrics_from_predictions(
            y_test,
            adaptive_predictions,
            probabilities,
        ),
        "rl": {
            "actions": [0.30, 0.40, 0.50, 0.60, 0.70],
            "final_q_values": agent.q_values.tolist(),
            "mean_batch_reward": float(np.mean(rewards)),
            "selected_threshold_counts": {
                str(threshold): int(selected_thresholds.count(threshold))
                for threshold in agent.thresholds
            },
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
