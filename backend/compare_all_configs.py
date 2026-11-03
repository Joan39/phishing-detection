"""Run and compare all phishing-detector configurations A, B, C, and D."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path("results") / "unified_experiment_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the project’s A/B classical baseline and ensemble, the C quantum "
            "experiment, and the D drift-adaptation experiment under the same "
            "domain-held-out setup, then summarize their performance in one file."
        )
    )
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to store the unified JSON summary.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep the individual summary JSON files created for A/B/C/D.",
    )
    return parser.parse_args()


def run_script(script: str, *extra_args: str) -> dict[str, Any]:
    cmd = [sys.executable, str(ROOT / script), *extra_args]
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return json.loads((ROOT / "results" / (Path(script).stem + "_temp.json")).read_text(encoding="utf-8"))


def collect_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_metric_entry(label: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": label,
        "accuracy": metrics.get("accuracy"),
        "precision_phishing": metrics.get("precision_phishing"),
        "recall_phishing": metrics.get("recall_phishing"),
        "f1_phishing": metrics.get("f1_phishing"),
        "roc_auc": metrics.get("roc_auc"),
        "false_positive_rate": metrics.get("false_positive_rate"),
        "confusion_matrix": metrics.get("confusion_matrix"),
    }


def run_experiments(args: argparse.Namespace) -> dict[str, Any]:
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    a_output = results_dir / "A_unified.json"
    b_output = results_dir / "B_unified.json"
    c_output = results_dir / "C_unified.json"
    d_output = results_dir / "D_unified.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "train_models.py"),
            "--models",
            "A",
            "--fold",
            str(args.fold),
            "--seed",
            str(args.seed),
            "--feature-profile",
            "url",
            "--output",
            str(a_output),
        ],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "train_models.py"),
            "--models",
            "B",
            "--fold",
            str(args.fold),
            "--seed",
            str(args.seed),
            "--feature-profile",
            "url",
            "--output",
            str(b_output),
        ],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "quantum_experiment.py"),
            "--fold",
            str(args.fold),
            "--seed",
            str(args.seed),
            "--train-per-class",
            "200",
            "--test-per-class",
            "100",
            "--output",
            str(c_output),
        ],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "quantum_ensemble_experiment.py"),
            "--fold",
            str(args.fold),
            "--seed",
            str(args.seed),
            "--train-per-class",
            "200",
            "--test-per-class",
            "100",
            "--output",
            str(results_dir / "C_hybrid_unified.json"),
        ],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "adaptation_experiment.py"),
            "--fold",
            str(args.fold),
            "--seed",
            str(args.seed),
            "--output",
            str(d_output),
        ],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )

    a_data = collect_file(a_output)
    b_data = collect_file(b_output)
    c_data = collect_file(c_output)
    c_hybrid_data = collect_file(results_dir / "C_hybrid_unified.json")
    d_data = collect_file(d_output)

    unified = {
        "experiment": "unified_a_b_c_d_domain_held_out_comparison",
        "seed": args.seed,
        "fold": args.fold,
        "comparison": [
            summarize_metric_entry("A", a_data["metrics"]["A"]),
            summarize_metric_entry("B", b_data["metrics"]["B"]),
            summarize_metric_entry("C_quantum", c_data["metrics"]),
            summarize_metric_entry("C_hybrid", c_hybrid_data["metrics"]),
            summarize_metric_entry(
                "D_static_threshold",
                d_data["static_threshold_0_5"],
            ),
            summarize_metric_entry(
                "D_adaptive_threshold",
                d_data["rl_adaptive_threshold"],
            ),
        ],
    }
    unified["ranking_by_f1_phishing"] = sorted(
        unified["comparison"],
        key=lambda item: (item["f1_phishing"] or -1.0, item["roc_auc"] or -1.0),
        reverse=True,
    )
    unified["ranking_by_f1_phishing"] = [
        item["config"] for item in unified["ranking_by_f1_phishing"]
    ]

    if not args.keep_intermediate:
        for path in [a_output, b_output, c_output, results_dir / "C_hybrid_unified.json", d_output]:
            if path.exists():
                path.unlink()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(unified, indent=2), encoding="utf-8")
    return unified


def main() -> None:
    args = parse_args()
    summary = run_experiments(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
