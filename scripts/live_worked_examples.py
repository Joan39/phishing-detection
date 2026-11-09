"""Score the documented worked examples through the running server.

The report quotes these values, so they are taken live rather than copied from the
markdown docs, and written to results/live_worked_examples.json for the builder to
read.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"

URLS = [
    "https://paypal-secure-refund.claim-center.net/login",
    "https://www.google.com/url?sa=t&url=https://refund-claim-irs-com/tax",
    "https://l.facebook.com.l.php?u=https://secure-appleid-verify.com/login",
    "https://en.wikipedia.org/wiki/Main_Page",
    "https://www.economist.com/science-and-technology/2026-03-14/quantum-advantage",
    "https://home.cern/science/computing/large-hadron-collider",
    "https://www.gov.uk/guidance/na0-phishing",
    "http://paypa1.com/secure/update",
]


def main() -> None:
    with httpx.Client(base_url=BASE, timeout=120.0) as client:
        token = client.post(
            "/api/auth/login",
            json={"email": "verify@qphish.dev", "password": "VerifyPass123!"},
        ).json()["access_token"]
        client.headers["Authorization"] = f"Bearer {token}"
        rows = []
        for url in URLS:
            result = client.post("/api/scans", json={"url": url}).json()
            scores = result.get("model_scores") or {}
            policy = result.get("scoring_policy") or {}
            rows.append(
                {
                    "url": url,
                    "A": scores.get("A"),
                    "B": scores.get("B"),
                    "quantum": scores.get("quantum"),
                    "hybrid": scores.get("hybrid"),
                    "probability": result["probability"],
                    "verdict": result["verdict"],
                    "risk_level": result.get("risk_level"),
                    "decision_threshold": result.get("decision_threshold"),
                    "threshold_source": result.get("threshold_source"),
                    "adaptive_context": result.get("adaptive_context"),
                    "review_recommended": result.get("review_recommended"),
                    "component_disagreement": result.get("component_disagreement"),
                    "redirect_target": result.get("redirect_target"),
                    "policy": policy,
                }
            )
            print(
                f"{result['probability']:.4f} {result['verdict']:<10} "
                f"A={scores.get('A')} B={scores.get('B')} Q={scores.get('quantum')} "
                f"cut={result.get('decision_threshold')} "
                f"disagree={result.get('component_disagreement')} {url}"
            )
        Path("results/live_worked_examples.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )
        print("wrote results/live_worked_examples.json")


if __name__ == "__main__":
    main()
