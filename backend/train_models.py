"""Train the first leakage-controlled phishing detection experiments.

This script intentionally implements configurations A and B first:

    A - balanced logistic-regression baseline
    B - soft-voting ensemble of logistic regression, random forest, and
        histogram gradient boosting

The test split is grouped by normalized Domain, so a domain cannot occur in
both training and testing. The default profile uses dataset URL features
directly computable from a raw URL; ``url_raw`` is the inference-compatible
subset that omits TLD. Dataset-derived URL probability features are available
only with ``--feature-profile url_dataset`` and page-content features only
with ``--feature-profile all``. URLSimilarityIndex is excluded because the
inspection audit identified it as potentially leaky.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import joblib
from sklearn.base import ClassifierMixin
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .domain_policy import hard_negative_blocks


DEFAULT_DATASET = Path("original") / "PhiUSIIL_Phishing_URL_Dataset.csv"
DEFAULT_OUTPUT = Path("results") / "first_experiment.json"
IDENTIFIER_COLUMNS = {"FILENAME", "URL", "Domain", "Title", "label"}
KNOWN_LEAKAGE_COLUMNS = {"URLSimilarityIndex"}
URL_DERIVED_FEATURES = (
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLD",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
    "IsTrustedRoot",
)
URL_DATASET_FEATURES = set(URL_DERIVED_FEATURES) | {
    "CharContinuationRate",
    "TLDLegitimateProb",
    "URLCharProb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        choices=range(5),
        help="Held-out fold from the five-fold stratified domain split.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("A", "B"),
        default=("A", "B"),
        help="Configurations to train during this first experiment.",
    )
    parser.add_argument(
        "--feature-profile",
        choices=("url", "url_raw", "url_dataset", "all"),
        default="url",
        help=(
            "Use dataset URL features (default), the raw-URL inference "
            "subset excluding TLD, URL features including dataset-derived "
            "probabilities, or all numeric features."
        ),
    )
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--boosting-iterations", type=int, default=100)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Optional directory in which to save fitted model files.",
    )
    return parser.parse_args()


def normalise_domain(value: object) -> str:
    """Return a stable group key without changing the model features."""
    return str(value).strip().lower().removeprefix("www.")


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    frame = pd.read_csv(path)
    required = {"label", "Domain"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    if frame["label"].isna().any():
        raise ValueError("The target column contains missing labels.")

    labels = frame["label"].astype(int)
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError(f"Expected binary labels 0/1, found {sorted(labels.unique())}")

    # In PhiUSIIL, 0 is phishing and 1 is legitimate. Make phishing the
    # positive class so recall, precision, and false-positive rate are clear.
    target = (labels == 0).astype(int)
    groups = frame["Domain"].map(normalise_domain)
    if (groups == "").any():
        raise ValueError("The domain grouping column contains empty values.")
    return frame, target, groups


def select_features(frame: pd.DataFrame, profile: str) -> list[str]:
    numeric_columns = frame.select_dtypes(include="number").columns.tolist()
    excluded = IDENTIFIER_COLUMNS | KNOWN_LEAKAGE_COLUMNS
    if profile in {"url", "url_raw", "url_dataset"}:
        allowed = (
            URL_DERIVED_FEATURES
            if profile == "url"
            else set(URL_DERIVED_FEATURES) - {"TLD"}
            if profile == "url_raw"
            else URL_DATASET_FEATURES
        )
        features = [
            column
            for column in numeric_columns
            if column in allowed and column not in excluded
        ]
    else:
        features = [column for column in numeric_columns if column not in excluded]
    if not features:
        raise ValueError("No numeric training features remain after exclusions.")
    return features


def build_feature_frame(frame: pd.DataFrame, profile: str) -> pd.DataFrame:
    if profile != "url_raw":
        return frame
    if "URL" not in frame.columns:
        raise ValueError("Raw URL features require the URL column.")

    from .url_features import extract_url_features_frame

    try:
        return extract_url_features_frame(frame["URL"])
    except ValueError as exc:
        raise ValueError(
            f"Could not extract features from the dataset URL values: {exc}"
        ) from exc


def build_production_training_data(
    path: Path = DEFAULT_DATASET,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build full-data production features with explicit legitimate hard negatives."""
    frame, target, _ = load_dataset(path)
    feature_frame = build_feature_frame(frame, "url_raw")
    blocks = hard_negative_blocks()
    negative_frames: list[pd.DataFrame] = []
    for urls, replicas in blocks:
        block = build_feature_frame(pd.DataFrame({"URL": urls}), "url_raw")
        negative_frames.extend([block] * replicas)
    hard_negative_frame = pd.concat(negative_frames, ignore_index=True)
    feature_frame = pd.concat(
        [feature_frame, hard_negative_frame],
        ignore_index=True,
    )
    target = pd.concat(
        [target, pd.Series([0] * len(hard_negative_frame), dtype=int)],
        ignore_index=True,
    )
    features = select_features(feature_frame, "url_raw")
    return feature_frame[features], target, features


def fit_production_models(
    model_dir: Path,
    *,
    seed: int = 42,
    trees: int = 200,
    boosting_iterations: int = 200,
) -> dict[str, int | str]:
    """Fit the live models on all labeled rows and trusted-root hard negatives."""
    x_train, target, features = build_production_training_data()
    models: dict[str, ClassifierMixin] = {
        "A": make_baseline(seed),
        "B": make_ensemble(
            seed,
            trees=trees,
            boosting_iterations=boosting_iterations,
        ),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        model.fit(x_train, target)
        temporary = model_dir / f"model_{name}.joblib.tmp"
        joblib.dump(model, temporary)
        temporary.replace(model_dir / f"model_{name}.joblib")
    return {
        "rows": len(target),
        "features": len(features),
        "legitimate_hard_negative_rows": sum(
            len(urls) * replicas for urls, replicas in hard_negative_blocks()
        ),
    }


def make_baseline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


def make_ensemble(seed: int, trees: int, boosting_iterations: int) -> VotingClassifier:
    return VotingClassifier(
        estimators=[
            ("logistic", make_baseline(seed)),
            (
                "forest",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "classifier",
                            RandomForestClassifier(
                                n_estimators=trees,
                                class_weight="balanced_subsample",
                                n_jobs=-1,
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "boosting",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "classifier",
                            HistGradientBoostingClassifier(
                                max_iter=boosting_iterations,
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            ),
        ],
        voting="soft",
        weights=[1, 2, 2],
        n_jobs=-1,
    )


def evaluate(
    model: ClassifierMixin,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    true_negatives, false_positives, false_negatives, true_positives = (
        confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    )
    legitimate_count = int((y_test == 0).sum())
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_phishing": float(precision_score(y_test, predictions, zero_division=0)),
        "recall_phishing": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_phishing": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "false_positive_rate": float(false_positives / legitimate_count)
        if legitimate_count
        else None,
        "confusion_matrix_labels": ["legitimate", "phishing"],
        "confusion_matrix": [
            [int(true_negatives), int(false_positives)],
            [int(false_negatives), int(true_positives)],
        ],
    }


def feature_separation_audit(
    frame: pd.DataFrame,
    target: pd.Series,
    features: list[str],
) -> list[dict[str, float]]:
    """Report univariate separation as a warning, not as model performance."""
    audit: list[dict[str, float]] = []
    for feature in features:
        values = frame[feature].fillna(frame[feature].median())
        auc = roc_auc_score(target, values)
        audit.append(
            {
                "feature": feature,
                "auc": float(auc),
                "distance_from_random": float(abs(auc - 0.5)),
            }
        )
    return sorted(audit, key=lambda item: item["distance_from_random"], reverse=True)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Dataset-only ablation over one domain-held-out fold.

    Writes no ``training_recipe``, so the comparison panel reports its output as a
    legacy run. Use ``scripts.production_benchmark`` for figures that describe the
    deployed stack.
    """
    frame, target, groups = load_dataset(args.data)
    feature_frame = build_feature_frame(frame, args.feature_profile)
    features = select_features(feature_frame, args.feature_profile)
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=args.seed,
    )
    splits = list(splitter.split(feature_frame[features], target, groups))
    train_indices, test_indices = splits[args.fold]

    train_groups = set(groups.iloc[train_indices])
    test_groups = set(groups.iloc[test_indices])
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(f"Domain leakage detected in split: {len(overlap)} groups overlap.")

    x_train = feature_frame.iloc[train_indices][features]
    x_test = feature_frame.iloc[test_indices][features]
    y_train = target.iloc[train_indices]
    y_test = target.iloc[test_indices]

    models: dict[str, ClassifierMixin] = {}
    if "A" in args.models:
        models["A"] = make_baseline(args.seed)
    if "B" in args.models:
        models["B"] = make_ensemble(
            args.seed,
            trees=args.trees,
            boosting_iterations=args.boosting_iterations,
        )

    metrics: dict[str, Any] = {}
    model_paths: dict[str, str] = {}
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        model_metrics = evaluate(model, x_test, y_test)
        model_metrics["prediction_seconds"] = time.perf_counter() - started
        model_metrics["fit_seconds"] = fit_seconds
        metrics[name] = model_metrics
        if args.model_dir is not None:
            args.model_dir.mkdir(parents=True, exist_ok=True)
            model_path = args.model_dir / f"model_{name}.joblib"
            joblib.dump(model, model_path)
            model_paths[name] = str(model_path)

    return {
        "experiment": "domain-held-out first experiment",
        "seed": args.seed,
        "fold": args.fold,
        "dataset": str(args.data),
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "train_domains": int(len(train_groups)),
        "test_domains": int(len(test_groups)),
        "domain_overlap": int(len(overlap)),
        "label_counts": {
            "legitimate": int((target == 0).sum()),
            "phishing": int((target == 1).sum()),
        },
        "phishing_definition": "original label == 0",
        "feature_profile": args.feature_profile,
        "excluded_columns": sorted(IDENTIFIER_COLUMNS | KNOWN_LEAKAGE_COLUMNS),
        "features": features,
        "model_paths": model_paths,
        "univariate_feature_separation_audit": feature_separation_audit(
            feature_frame,
            target,
            features,
        )[:10],
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    result = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
