"""Evaluation on techniques the models were never trained on.

Scores curated URL strings offline with the deployed artifacts, using the live
fusion, trusted cap and bandit threshold: 14 novel phishing strings and 7
legitimate hosts absent from the hard-negative corpus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from backend.main import _active_threshold, _load_rl_state
from backend.quantum_live import load_quantum_model
from backend.domain_policy import legitimate_training_urls
from backend.url_features import extract_url_features_frame
from scripts.realworld_offline_score import score, summarize

import joblib

NOVEL_PHISHING = [
    "https://rnicrosoftonline.com/common/oauth2/v2.0/authorize?client_id=111122223333",
    "https://docs-comment-sharing.xyz/g/1aBcDeFgHiJkLmNoPqRsTuVw/edit?usp=sharing",
    "https://sharepoint-fileview.com/s/P9xQ2vLm8RnYtWqZ",
    "https://zoom-us02web.com/j/8712345678?pwd=Q2hpbmExMjM0",
    "https://bit.ly/3xKp9Qm",
    "https://l.facebook.com.l.php?u=https://secure-appleid-verify.com/login",
    "https://www.google.com/url?sa=t&url=https://refund-claim-irs-com/tax",
    "https://my-appleid-verify.com/",
    "https://xn--paypa1-4ef-support.com/",
    "https://hr-portal-benefits.openemployee-enrollment.com/2026/enrollment/step1",
    "https://invoice-dropbox-transfer.com/s/5829104372/invoice-2026.pdf",
    "https://mfa-reset-portal.com/device/verify?session=8f3a2b1c",
    "https://teams-invite.microsoftteamsjoin.com/join/MTA4LTIwMjY",
    "https://www.chase-secure-alert.com/activity/review/confirm",
]

FRESH_LEGIT = [
    "https://www.rust-lang.org/learn/road-rust",
    "https://stackoverflow.com/questions/78112345/how-to-stream-large-json",
    "https://arxiv.org/abs/2603.11842",
    "https://www.nist.gov/cyberframework",
    "https://docs.python.org/3/library/sqlite3.html",
    "https://www.ecb.europa.eu/press/mrl-pub/html/index.en.html",
    "https://www.who.int/publications/i/item/9789240000000",
    "https://archive.org/details/report-on-quantum-computing",
    "https://www.cdc.gov/nchs/data/index.htm",
    "https://canada.ca/en/services/immigration-citizenship.html",
    "https://openstreetmap.org/#map=13/51.5/-0.12",
    "https://githubstatus.com/",
]


def host(url: str) -> str:
    return (urlsplit(url if "://" in url else "//" + url).hostname or "").lower().removeprefix("www.")


def main() -> None:
    trained = {host(url) for url in legitimate_training_urls()}
    unseen = [url for url in FRESH_LEGIT if host(url) not in trained]
    dropped = [url for url in FRESH_LEGIT if host(url) in trained]
    print(f"dropped {len(dropped)} legit URLs already in the training corpus: {dropped}")

    models = {name: joblib.load(Path("models") / "url" / f"model_{name}.joblib") for name in ("A", "B")}
    quantum = load_quantum_model(Path("models") / "quantum_live.joblib")
    threshold = _active_threshold(_load_rl_state())

    pairs = [("legitimate", url) for url in unseen] + [("phishing", url) for url in NOVEL_PHISHING]
    urls = [url for _, url in pairs]
    groups = [group for group, _ in pairs]
    features = extract_url_features_frame(urls)
    rows = score(models, quantum, features, urls, threshold=threshold)
    summary = summarize(rows, groups)
    print(f"threshold {threshold}")
    for group in ("legitimate", "phishing"):
        values = summary[group]["all"]
        print(f"  {group:<10} n={values['n']:<3} correct={values['correct_rate']:.4f} mean_prob={values['mean_probability']:.3f}")
    print("\nBYPASSED novel phishing (scored legitimate):")
    for row, group in zip(rows, groups):
        if group == "phishing" and row["verdict"] == "legitimate":
            print(f"  {row['probability']:.3f} A={row['A']:.3f} B={row['B']:.3f} Q={row['quantum']:.3f} {row['url']}")
    print("\nFALSE POSITIVES on unseen legitimate:")
    for row, group in zip(rows, groups):
        if group == "legitimate" and row["verdict"] == "phishing":
            print(f"  {row['probability']:.3f} A={row['A']:.3f} B={row['B']:.3f} Q={row['quantum']:.3f} {row['url']}")
    print("\nCaught novel phishing:")
    for row, group in zip(rows, groups):
        if group == "phishing" and row["verdict"] == "phishing":
            print(f"  {row['probability']:.3f} A={row['A']:.3f} B={row['B']:.3f} Q={row['quantum']:.3f} {row['url']}")
    Path("results/novel_eval.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
