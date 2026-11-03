"""Predict phishing from a raw URL using saved URL-only models."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from .url_features import extract_url_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models") / "url",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = extract_url_features(args.url)
    print(f"URL: {args.url}")
    print("Interpretation: 1 = phishing, 0 = legitimate")
    for name in ("A", "B"):
        model_path = args.model_dir / f"model_{name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Saved model not found: {model_path}. "
                "Train with --feature-profile url --model-dir models/url first."
            )
        model = joblib.load(model_path)
        probability = float(model.predict_proba(features)[:, 1][0])
        prediction = int(probability >= 0.5)
        print(
            f"{name}: prediction={prediction}, "
            f"phishing_probability={probability:.6f}"
        )


if __name__ == "__main__":
    main()
