"""Real-world generalization battery for the live detection stack.

Held-out fold metrics only contain PhiUSIIL rows, and every legitimate row in
that corpus is HTTPS on a near-bare domain, so fold accuracy cannot show how
the system behaves on ordinary real traffic. This battery scores curated
legitimate and phishing URLs through the running API and reports the two rates
that matter operationally: false positives on legitimate traffic and recall on
phishing traffic, each split by URL scheme.

Usage:
    python -m scripts.realworld_battery            # live API battery
    python -m scripts.realworld_battery --dataset  # add dataset phishing recall
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = "http://127.0.0.1:8000"
LOGIN = {"email": "verify@qphish.dev", "password": "VerifyPass123!"}
RESULTS_PATH = Path("results") / "realworld_battery.json"

# Real sites reachable over the public web, chosen to cover the shapes ordinary
# users type: bare hosts, deep paths, query strings, both schemes, trusted and
# non-trusted brands.
LEGITIMATE_URLS = [
    "https://docs.python.org/3/library/os.html",
    "https://pypi.org/project/requests/",
    "https://arxiv.org/abs/2301.05427",
    "https://stackoverflow.com/questions/12345678/how-to-parse-a-url",
    "https://news.ycombinator.com/item?id=12345678",
    "https://www.gov.uk/guidance/na0-phishing",
    "https://www.rfc-editor.org/rfc/rfc9110",
    "https://home.cern/science/computing/large-hadron-collider",
    "https://huggingface.co/docs/transformers/index",
    "https://httpbin.org/get?section=account&page=2",
    "https://www.rug.nl/news/2026/03/quantum-machine-learning",
    "https://www.gnu.org/philosophy/free-sw.html",
    "https://example.com",
    "https://www.google.com/search?q=quantum+computing",
    "https://en.wikipedia.org/wiki/Main_Page",
    "https://github.com/login?return_to=%2Fsettings",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.microsoft.com/en-us/windows",
    "http://neverssl.com",
    "http://info.cern.ch/hypertext/WWW/TheProject.html",
    "http://calpoly.edu",
    "http://www.gnu.org/software/bash/",
    "http://example.org/path",
    "http://rss.slashdot.org/Slashdot/slashdotMain",
    "http://www.google.com",
    "http://en.wikipedia.org/wiki/Quantum_computing",
]

# Phishing shapes: brand lookalikes, credential paths, raw IP hosts, delegated
# subdomain tricks, encoding obfuscation and hyphen-heavy attack hosts. These
# are synthetic but match patterns that appear in live phishing feeds.
PHISHING_URLS = [
    "http://paypal-secure-refund.claim-center.net",
    "http://paypa1.com/secure/update",
    "http://192.168.1.1/admin",
    "https://146.112.234.45/phpMyAdmin/index.php",
    "https://appleid.apple.com.account-verify.systems/signin",
    "https://microsoft.com.sign-in.security-check.xyz/verify",
    "http://login-verify-account-support.tk/signin.php?email=user%40corp.com",
    "https://chase-verify-account.top/login?redirect=%2Fhome",
    "https://www-amazon.com.order-return-xyz.com/orders",
    "https://docs.google.com-forms.share/1xFAuthRequired",
    "https://secure-docx.com/view/document?id=%41%42%43",
    "https://update-your-account-information.com/id=12345",
    "http://xn--paypa1-bcd.com/login",
    "https://netflix-member-billing.com/resubscribe?card=4111",
    "http://dhl-delivery-tracking.icu/confirm.php?zip=90210",
    "https://www.dropbox.com.signed-file-review.work/paper.pdf",
    "http://internal-hr-portal.xyz/payroll/update-bank-details",
    "https://o365-mail-quarreled.com/authenticate?u=ceo@company.com",
    "http://166.88.44.12/owa/auth/logon.aspx",
    "https://metamask-io-wallet-restore.top/seed",
    "http://bit.ly.verify-account-now.click/3xR9kLm",
    "https://instagram-password-reset.club/confirm?id=8842",
    "http://service-appleid-locked.com/unlock.php",
    "https://fedex-delivery-issue.top/resolve?tracking=7788123",
    "https://login-secure-session.xyz/session.php?sid=4412",
    "http://paypal.com.login-verify-account.xyz",
    "https://adobe-id-document-viewer.top/signin",
    "http://icloud-find-my-phone-loss.com/lock",
    "https://bankofamerica-online-statement.click/login",
    "http://support-app-store-billing.com/refund-request",
]


def _post(path: str, payload: dict, token: str | None = None) -> dict:
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def _scheme(url: str) -> str:
    return "http" if url.startswith("http://") else "https"


def run_live_battery(token: str, pacing_seconds: float = 1.05) -> dict:
    """Scan the battery through the live API.

    ``pacing_seconds`` keeps the run under the 60-requests-per-minute rate
    limit so the battery measures detection, not throttling.
    """
    rows = []
    for group, urls in (("legitimate", LEGITIMATE_URLS), ("phishing", PHISHING_URLS)):
        for url in urls:
            time.sleep(pacing_seconds)
            try:
                result = _post("/api/scans", {"url": url}, token)
            except urllib.error.HTTPError as error:
                rows.append(
                    {
                        "group": group,
                        "url": url,
                        "scheme": _scheme(url),
                        "verdict": "rejected",
                        "probability": 1.0,
                        "threshold": 0.0,
                        "model_scores": {"A": 0.0, "B": 0.0, "quantum": 0.0},
                        "trusted_domain_calibration": False,
                        "http_error": error.code,
                    }
                )
                continue
            rows.append(
                {
                    "group": group,
                    "url": url,
                    "scheme": _scheme(url),
                    "verdict": result["verdict"],
                    "probability": result["probability"],
                    "threshold": result["decision_threshold"],
                    "model_scores": result["model_scores"],
                    "trusted_domain_calibration": result["explanations"][
                        "trusted_domain_calibration"
                    ],
                }
            )
    return summarize(rows)


def run_dataset_fit_check(limit_per_scheme: int = 6000) -> dict:
    """Check how the live classical models fit real PhiUSIIL rows.

    These rows are part of production training, so the numbers measure fit, not
    generalization; they are used as a guard against a retrain collapsing the
    dataset's own phishing patterns. The quantum model is left out because it
    trains on the dataset unchanged and per-shot sampling is far too slow for a
    sample of this size.
    """
    import pandas as pd

    from backend.main import (
        CLASSICAL_WEIGHTS,
        _models,
        fuse_classical_probabilities,
    )
    from backend.url_features import extract_url_features

    frame = pd.read_csv("original/PhiUSIIL_Phishing_URL_Dataset.csv")
    frame = frame.assign(
        scheme=frame["URL"].astype(str).str.lower().str.extract(r"^(https?)")[0]
    )
    models = _models()
    report: dict[str, dict] = {}
    for label, class_name in ((0, "phishing"), (1, "legitimate")):
        per_scheme: dict[str, dict] = {}
        for scheme in ("http", "https"):
            subset = frame[(frame["label"] == label) & (frame["scheme"] == scheme)]
            subset = subset.head(limit_per_scheme)
            if subset.empty:
                continue
            features = pd.concat(
                [extract_url_features(url) for url in subset["URL"]],
                ignore_index=True,
            )
            scores = {
                name: model.predict_proba(features)[:, 1]
                for name, model in models.items()
            }
            frame_rows = pd.DataFrame(scores)
            classical = np.array(
                [
                    fuse_classical_probabilities(row, CLASSICAL_WEIGHTS)
                    for row in frame_rows.to_dict("records")
                ]
            )
            caught = classical >= 0.40
            per_scheme[scheme] = {
                "rows": int(len(subset)),
                "classical_recall": round(float(caught.mean()), 4),
                "mean_classical_probability": round(float(classical.mean()), 4),
                "mean_A": round(float(frame_rows["A"].mean()), 4),
                "mean_B": round(float(frame_rows["B"].mean()), 4),
            }
        report[class_name] = per_scheme
    return report


def summarize(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    report: dict[str, dict] = {}
    for group, items in groups.items():
        buckets: dict[str, dict[str, float]] = defaultdict(
            lambda: {"n": 0, "wrong": 0, "score_sum": 0.0}
        )
        for row in items:
            flagged = row["verdict"] == "phishing"
            wrong = flagged if group == "legitimate" else not flagged
            for key in ("all", row["scheme"]):
                bucket = buckets[key]
                bucket["n"] += 1
                bucket["wrong"] += float(wrong)
                bucket["score_sum"] += row["probability"]
        report[group] = {
            scope: {
                "n": int(bucket["n"]),
                "error_rate": round(bucket["wrong"] / bucket["n"], 4),
                "correct_rate": round(1 - bucket["wrong"] / bucket["n"], 4),
                "mean_probability": round(bucket["score_sum"] / bucket["n"], 4),
            }
            for scope, bucket in buckets.items()
        }
    mistakes = [
        {
            "url": row["url"],
            "group": row["group"],
            "verdict": row["verdict"],
            "probability": round(row["probability"], 4),
            "A": round(row["model_scores"]["A"], 3),
            "B": round(row["model_scores"]["B"], 3),
            "quantum": round(row["model_scores"]["quantum"], 3),
            "trusted_cap": row["trusted_domain_calibration"],
        }
        for row in rows
        if (row["group"] == "legitimate") == (row["verdict"] == "phishing")
    ]
    return {"summary": report, "mistakes": mistakes, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="store_true",
        help="Also run the classical fit check on sampled PhiUSIIL rows.",
    )
    args = parser.parse_args()

    token = _post("/api/auth/login", LOGIN).get("access_token")
    report = run_live_battery(token)
    if args.dataset:
        report["dataset_fit_check"] = run_dataset_fit_check()
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for group, summary in report["summary"].items():
        for scope, metrics in summary.items():
            print(
                f"{group:<11} {scope:<6} n={metrics['n']:<3} "
                f"correct={metrics['correct_rate']:.3f} "
                f"mean_prob={metrics['mean_probability']:.3f}"
            )
    if report.get("dataset_fit_check"):
        print(json.dumps(report["dataset_fit_check"], indent=2))
    print(f"\nmistakes ({len(report['mistakes'])}):")
    for mistake in report["mistakes"]:
        print(
            f"  {mistake['group']:<11} {mistake['verdict']:<11} "
            f"{mistake['probability']:.3f} A={mistake['A']:.3f} "
            f"B={mistake['B']:.3f} cap={int(mistake['trusted_cap'])}  {mistake['url']}"
        )


if __name__ == "__main__":
    main()
