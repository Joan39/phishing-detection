"""Second, independent real-world set: URLs never used to shape the corpus.

The first battery drove the hard-negative design, so a gain there could be
tuning rather than generalization. These hosts appear in neither the battery nor
backend/domain_policy.py, and this file is scored both offline (fitted models)
and live (the running API).

Usage:
    python -m scripts.fresh_probe
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from backend.domain_policy import legitimate_training_urls
from scripts.realworld_battery import (
    BASE,
    LEGITIMATE_URLS,
    PHISHING_URLS,
    _post,
    _scheme,
    LOGIN,
)

DEFAULT_OUTPUT = Path("results") / "fresh_probe.json"

FRESH_LEGITIMATE = [
    "https://www.worldwildlife.org/species/a-z",
    "https://www.nih.gov/news-events/news-releases",
    "https://www.pbs.org/wgbh/nova/article/quantum-computing",
    "https://www.smithsonianmag.com/science-nature/",
    "https://www.theguardian.com/technology/2026/mar/14/ai-models",
    "https://www.eff.org/issues/privacy",
    "https://docs.ansible.com/ansible/latest/index.html",
    "https://www.wikihow.com/Identify-a-Phishing-Email",
    "https://www.sciencemag.org/news/2026/03/cryptographers-crack-legacy-hash",
    "https://www.economist.com/science-and-technology/2026-03-14/quantum-advantage",
    "http://www.textfiles.com/historical/",
    "http://www.ctext.org/ctext/zh/3",
]

FRESH_PHISHING = [
    "https://paypa1.com.signin.secure-verify-account.cyou/update",
    "http://apple-support-lock-id.top/unlock?uid=44213",
    "https://146.112.234.45/portal/login.php",
    "http://netflix-login.billing-update.icu/recover",
    "https://dhl-parcel-redelivery.top/track?id=991",
    "http://www.chase-secure-login.click/signon",
]


def assert_independent() -> None:
    trained = {url.lower() for url in legitimate_training_urls()}
    battery = {url.lower() for url in LEGITIMATE_URLS + PHISHING_URLS}
    overlap = {
        url
        for url in FRESH_LEGITIMATE + FRESH_PHISHING
        if url.lower() in trained | battery
    }
    if overlap:
        raise SystemExit(f"probe set is not independent: {sorted(overlap)}")


def run_live(pacing_seconds: float = 1.05) -> list[dict]:
    assert_independent()
    token = _post("/api/auth/login", LOGIN).get("access_token")
    rows = []
    for group, urls in (
        ("legitimate", FRESH_LEGITIMATE),
        ("phishing", FRESH_PHISHING),
    ):
        for url in urls:
            time.sleep(pacing_seconds)
            result = _post("/api/scans", {"url": url}, token)
            rows.append(
                {
                    "group": group,
                    "url": url,
                    "scheme": _scheme(url),
                    "verdict": result["verdict"],
                    "probability": round(result["probability"], 4),
                    "A": round(result["model_scores"]["A"], 3),
                    "B": round(result["model_scores"]["B"], 3),
                    "quantum": round(result["model_scores"]["quantum"], 3),
                    "threshold": result["decision_threshold"],
                }
            )
    return rows


def summarize(rows: list[dict]) -> dict:
    report: dict[str, dict] = {}
    for row in rows:
        correct = (row["group"] == "legitimate") == (row["verdict"] == "legitimate")
        for scope in ("all", row["scheme"]):
            bucket = report.setdefault(row["group"], {}).setdefault(
                scope, {"n": 0, "correct": 0, "score_sum": 0.0}
            )
            bucket["n"] += 1
            bucket["correct"] += int(correct)
            bucket["score_sum"] += row["probability"]
    return {
        group: {
            scope: {
                "n": bucket["n"],
                "correct_rate": round(bucket["correct"] / bucket["n"], 4),
                "mean_probability": round(bucket["score_sum"] / bucket["n"], 4),
            }
            for scope, bucket in scopes.items()
        }
        for group, scopes in report.items()
    }


def main() -> None:
    rows = run_live()
    misses = [
        row
        for row in rows
        if (row["group"] == "legitimate") != (row["verdict"] == "legitimate")
    ]
    for row in rows:
        correct = (row["group"] == "legitimate") == (row["verdict"] == "legitimate")
        print(
            f"{'ok  ' if correct else 'MISS'} {row['group']:<10} {row['scheme']:<5} "
            f"{row['probability']:.3f} A={row['A']:.3f} B={row['B']:.3f} "
            f"Q={row['quantum']:.3f}  {row['url']}"
        )
    summary = summarize(rows)
    print(json.dumps(summary, indent=2))
    print(f"\nmistakes ({len(misses)}):")
    for row in misses:
        print(f"  {row['group']:<10} {row['verdict']:<10} {row['url']}")
    DEFAULT_OUTPUT.write_text(
        json.dumps({"rows": rows, "summary": summary, "mistakes": misses}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
