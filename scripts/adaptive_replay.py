"""Compare the static policy with the adaptive policy on a simulated shift in
phishing technique.

The question this answers is what the adaptive layer actually buys: the deployed
models and the redirect guardrail are held identical across the arms, so the
only difference is whether thresholds, component weights and retraining respond
to the traffic that arrives.

Three arms score the same stream of rows:

    static     - fixed 0.40 cut, frozen 0.35/0.65 + 0.25 quantum blend.
    adaptive   - contextual thresholds, feedback-re-ranked component weights and
                 label-free drift detection, learning as the stream arrives.
    retrained  - the static arm again, after the drift-triggered refit has seen
                 the shift rows, measured on technique families it never saw.

The shift block is made of novel technique strings the deployed models never
trained on; the holdout block is a second, disjoint set of families, so the
retrained arm cannot be credited for memorising the rows it was refit on.

Usage:
    python -m scripts.adaptive_replay
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.adaptive_policy import (
    DISTRIBUTION_WINDOW,
    choose_threshold,
    component_accuracy,
    default_policy,
    fuse_adaptive,
    needs_review,
    observe_distribution,
    policy_context,
    reset_distribution_reference,
    update_component_weights,
    update_context,
    weight_confidence,
)
from backend.main import CLASSICAL_WEIGHTS, QUANTUM_WEIGHT
from backend.quantum_live import QUANTUM_FEATURES, load_quantum_model
from backend.train_models import load_dataset, make_baseline, make_ensemble, select_features
from backend.url_features import (
    TRUSTED_DOMAIN_CAP,
    extract_url_features_frame,
    trusted_calibration_applies,
)
from scripts.fresh_probe import FRESH_LEGITIMATE
from scripts.production_benchmark import (
    FEATURE_PROFILE,
    SEED,
    classical_fusion,
    live_probability,
    production_negative_rows,
    quantum_probabilities,
    redirect_chain_probabilities,
)
from scripts.realworld_battery import LEGITIMATE_URLS, PHISHING_URLS

DEFAULT_OUTPUT = Path("results") / "adaptive_replay_experiment.json"
MODEL_DIR = Path("models") / "url"
QUANTUM_ARTIFACT = Path("models") / "quantum_live.joblib"
DATASET = Path("original") / "PhiUSIIL_Phishing_URL_Dataset.csv"
THRESHOLD = 0.40
TREES = 200
BOOSTING_ITERATIONS = 200
REFIT_SAMPLE_ROWS = 60_000
FAMILIAR_CYCLES = 6

# Novel families the deployed models have never seen; these drive the shift.
SHIFT_PHISHING = [
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
SHIFT_LEGITIMATE = FRESH_LEGITIMATE[:7]

# A disjoint second set of families, never used for training in any arm.
HOLDOUT_PHISHING = [
    "https://www.chase-verify-mobile-deposit.com/mobile/deposit-review/login",
    "https://secure-dhl-parcel-redelivery.net/pay-7.98?lang=en-GB",
    "https://usps-deliveryexception.com/redelivery?code=US-8842",
    "https://netflix-billing-update.com/recurring/payment-method",
    "https://hris-paystub-secure.com/employee/payroll/download-91827.pdf",
    "https://appleid-billing-review.com/signin/account?country=US",
    "https://adp-payroll-direct-deposit.com/employee/update-bank",
    "https://www.office-invoice-viewer.com/attachment/INVOICE-4471.pdf",
    "https://us-mfa-push-approval.com/device/approve?challenge=29f4",
    "https://refund-wellsfargo-alert.com/online/deposit/dispute",
    "https://install-python-package.com/pip/numpy-stable.whl",
    "https://wallet-airdrop-claim.io/connect-wallet?token=eth",
]

HOLDOUT_LEGITIMATE = [
    "https://pytorch.org/tutorials/beginner/basics/data_tutorial.html",
    "https://www.rfc-editor.org/rfc/rfc9110",
    "https://huggingface.co/docs/transformers/index",
    "https://arxiv.org/abs/2603.11842",
    "https://www.nist.gov/cyberframework",
    "https://docs.python.org/3/library/sqlite3.html",
    "https://www.who.int/publications/i/item/9789240000000",
    "https://www.ecb.europa.eu/press/mrl-pub/html/index.en.html",
]


def build_stream() -> list[dict]:
    """Familiar traffic first, then the technique shift, then the unseen holdout.

    The familiar block repeats because the drift baseline needs 200 observations
    before it can call anything a shift; real traffic arrives far faster than a
    curated battery, and shortening the window to make the demo fire would not be
    the mechanism under test.
    """
    stream = []
    for cycle in range(FAMILIAR_CYCLES):
        block = "familiar" if cycle == 0 else f"familiar{cycle + 1}"
        stream.extend(
            [{"block": block, "group": group, "url": url}
             for group, urls in (("legitimate", LEGITIMATE_URLS), ("phishing", PHISHING_URLS))
             for url in urls]
        )
    stream.extend(
        [{"block": "shift", "group": group, "url": url}
         for group, urls in (("legitimate", SHIFT_LEGITIMATE), ("phishing", SHIFT_PHISHING))
         for url in urls]
    )
    stream.extend(
        [{"block": "holdout", "group": group, "url": url}
         for group, urls in (("legitimate", HOLDOUT_LEGITIMATE), ("phishing", HOLDOUT_PHISHING))
         for url in urls]
    )
    return stream


def component_scores(
    stream: list[dict], models: dict, artifact: dict
) -> dict:
    urls = [row["url"] for row in stream]
    features = extract_url_features_frame(urls)
    selected = select_features(features, FEATURE_PROFILE)
    frame = features[selected]
    scores = {
        name: np.asarray(model.predict_proba(frame)[:, 1], dtype=float)
        for name, model in models.items()
    }
    scores["quantum"] = quantum_probabilities(artifact, frame)
    scores["redirect"] = redirect_chain_probabilities(models, artifact, urls, selected)
    scores["features"] = frame
    scores["urls"] = urls
    return scores


def blend(index: int, scores: dict, weights: dict, quantum_weight: float) -> float:
    """Deployed scoring for one row: fusion, guardrail payload, trusted cap."""
    probability = fuse_adaptive(
        {name: float(scores[name][index]) for name in ("A", "B", "quantum")},
        weights,
        quantum_weight,
    )
    url = scores["urls"][index]
    row = scores["features"].iloc[[index]]
    if trusted_calibration_applies(url, row):
        probability = min(probability, TRUSTED_DOMAIN_CAP)
    redirect = scores["redirect"][index]
    if not np.isnan(redirect) and redirect > probability:
        probability = float(redirect)
    return float(probability)


def context_of(index: int, scores: dict) -> str:
    url = scores["urls"][index]
    row = scores["features"].iloc[[index]]
    return policy_context(
        is_https=bool(row["IsHTTPS"].iloc[0]),
        is_trusted_root=trusted_calibration_applies(url, row),
    )


def score_group(rows: list[dict], group: str) -> dict:
    subset = [row for row in rows if row["group"] == group]
    if not subset:
        return {"n": 0}
    correct = sum(
        (row["verdict"] == "phishing") == (group == "phishing") for row in subset
    )
    return {
        "n": len(subset),
        "correct_rate": round(correct / len(subset), 4),
        "mean_probability": round(
            float(np.mean([row["probability"] for row in subset])), 4
        ),
    }


def summarise(rows: list[dict]) -> dict:
    return {
        block: {
            group: score_group([row for row in rows if row["block"] == block], group)
            for group in ("legitimate", "phishing")
        }
        for block in dict.fromkeys(row["block"] for row in rows)
    }


def run_static(scores: dict, stream: list[dict]) -> list[dict]:
    rows = []
    for index, meta in enumerate(stream):
        probability = blend(index, scores, CLASSICAL_WEIGHTS, QUANTUM_WEIGHT)
        rows.append(
            {
                **meta,
                "probability": round(probability, 4),
                "verdict": "phishing" if probability >= THRESHOLD else "legitimate",
                "threshold": THRESHOLD,
            }
        )
    return rows


def run_adaptive(scores: dict, stream: list[dict]) -> tuple[list[dict], dict]:
    """The policy learns from the analyst verdict for each row as it arrives."""
    policy = default_policy()
    rows = []
    drift_first_seen = None
    psi_at_shift_start = None
    max_psi = 0.0
    for index, meta in enumerate(stream):
        context = context_of(index, scores)
        probability = blend(
            index, scores, policy["classical_weights"], policy["quantum_weight"]
        )
        threshold, action, used_context = choose_threshold(policy, context, THRESHOLD)
        verdict = "phishing" if probability >= threshold else "legitimate"
        observed = {**meta, "probability": round(probability, 4), "verdict": verdict,
                    "threshold": threshold, "context": context,
                    "context_used": used_context,
                    "review": needs_review(
                        {name: float(scores[name][index]) for name in ("A", "B", "quantum")},
                        probability,
                        threshold,
                    )}
        # The analyst confirms the true label for every row in this replay.
        actual = 1 if meta["group"] == "phishing" else 0
        reward = 1.0 if (probability >= threshold) == bool(actual) else -1.0
        update_context(policy, context, action, reward)
        update_component_weights(
            policy,
            {name: float(scores[name][index]) for name in ("A", "B", "quantum")},
            actual,
            meta["url"],
        )
        observe_distribution(policy, probability)
        psi = float(policy["distribution"]["psi"])
        max_psi = max(max_psi, psi)
        if psi_at_shift_start is None and meta["block"] == "shift":
            psi_at_shift_start = psi
        if policy["distribution"]["drift"] and drift_first_seen is None:
            drift_first_seen = index
        if policy["distribution"]["drift"]:
            reset_distribution_reference(policy)
        rows.append(observed)
    report = {
        "drift_flagged_at_row": drift_first_seen,
        "max_psi": round(max_psi, 4),
        "psi_when_shift_arrived": psi_at_shift_start,
        "drift_trigger": 0.25,
        "final_classical_weights": policy["classical_weights"],
        "final_quantum_weight": policy["quantum_weight"],
        "evidence_urls": len(policy.get("evidence_urls", [])),
        "weight_confidence": round(weight_confidence(policy), 4),
        "component_accuracy": component_accuracy(policy),
        "contexts": {
            context: {
                "samples": entry["samples"],
                "threshold": float(
                    entry["thresholds"][int(np.argmax(entry["q_values"]))]
                ),
            }
            for context, entry in policy["contexts"].items()
        },
        "review_flagged": sum(row["review"] for row in rows),
        "window": DISTRIBUTION_WINDOW,
    }
    return rows, report


def refit_models(scores: dict, stream: list[dict]) -> dict:
    """Feedback-triggered refit: the labelled shift rows are now training rows.

    Mirrors what the feedback worker actually does - corpus rows plus the
    production hard-negative corpus plus the new labels - on a corpus sample so
    the replay stays a single-core job. This arm is not a leakage-controlled
    held-out measurement; its only claim is what the refit does to the families
    the stream never showed it.
    """
    frame, target, _groups = load_dataset(DATASET)
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(frame), size=min(REFIT_SAMPLE_ROWS, len(frame)), replace=False)
    features = extract_url_features_frame(
        frame["URL"].astype(str).iloc[sample].tolist()
    )
    selected = select_features(features, FEATURE_PROFILE)
    base = features[selected].reset_index(drop=True)
    base_target = target.iloc[sample].reset_index(drop=True)
    negatives, provenance = production_negative_rows(selected)

    shift_rows = [
        index for index, meta in enumerate(stream) if meta["block"] == "shift"
    ]
    extra = scores["features"].iloc[shift_rows].reset_index(drop=True)
    extra_target = pd.Series(
        [1 if stream[index]["group"] == "phishing" else 0 for index in shift_rows],
        dtype=int,
    )
    x = pd.concat([base, negatives, extra], ignore_index=True)
    y = pd.concat(
        [
            base_target,
            pd.Series([0] * len(negatives), dtype=int),
            extra_target,
        ],
        ignore_index=True,
    )
    started = time.perf_counter()
    models = {
        "A": make_baseline(SEED).fit(x, y),
        "B": make_ensemble(
            SEED, trees=TREES, boosting_iterations=BOOSTING_ITERATIONS
        ).fit(x, y),
    }
    return {
        "models": models,
        "fit_seconds": round(time.perf_counter() - started, 1),
        "training_rows": int(len(x)),
        "corpus_sample_rows": int(len(base)),
        "negative_rows": int(len(negatives)),
        "negative_blocks": provenance,
        "shift_rows": len(shift_rows),
    }


def run_retrained(scores: dict, stream: list[dict], refit: dict, artifact: dict) -> list[dict]:
    holdout = [index for index, meta in enumerate(stream) if meta["block"] == "holdout"]
    urls = [scores["urls"][index] for index in holdout]
    frame = scores["features"].iloc[holdout].reset_index(drop=True)
    models = refit["models"]
    classical = classical_fusion(models, frame)
    quantum = quantum_probabilities(artifact, frame)
    probabilities = live_probability(classical, quantum, urls, frame)
    chain = redirect_chain_probabilities(models, artifact, urls, list(frame.columns))
    probabilities = np.fmax(probabilities, chain)
    rows = []
    for position, index in enumerate(holdout):
        probability = float(probabilities[position])
        rows.append(
            {
                **stream[index],
                "probability": round(probability, 4),
                "verdict": "phishing" if probability >= THRESHOLD else "legitimate",
                "threshold": THRESHOLD,
            }
        )
    return rows


def main() -> None:
    started = time.perf_counter()
    stream = build_stream()
    models = {
        name: joblib.load(MODEL_DIR / f"model_{name}.joblib") for name in ("A", "B")
    }
    artifact = load_quantum_model(QUANTUM_ARTIFACT)
    scores = component_scores(stream, models, artifact)
    print(f"{len(stream)} stream rows scored in {time.perf_counter() - started:.1f}s", flush=True)

    static_rows = run_static(scores, stream)
    adaptive_rows, adaptive_report = run_adaptive(scores, stream)
    print("static and adaptive arms complete", flush=True)

    refit = refit_models(scores, stream)
    retrained_rows = run_retrained(scores, stream, refit, artifact)
    print(
        f"reduced-budget refit done in {refit['fit_seconds']}s on "
        f"{refit['training_rows']} rows",
        flush=True,
    )

    report = {
        "experiment": "static policy vs adaptive policy on a simulated technique shift",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "protocol": {
            "stream_blocks": {
                block: sum(row["block"] == block for row in stream)
                for block in ("familiar", "shift", "holdout")
            },
            "threshold": THRESHOLD,
            "guardrail": "redirect-payload scoring is active in every arm",
            "analyst": "every row is labelled with its true class in this replay",
            "retrain": (
                "reduced budget: 60,000 corpus rows + the full hard-negative "
                "corpus + the labelled shift rows; not leakage-controlled"
            ),
            "holdout_families": "never used for training in any arm",
            "familiar_cycles": FAMILIAR_CYCLES,
        },
        "arms": {
            "static": summarise(static_rows),
            "adaptive": summarise(adaptive_rows),
            "retrained_static": summarise(retrained_rows),
        },
        "adaptive_policy": adaptive_report,
        "refit": {
            key: value
            for key, value in refit.items()
            if key != "models"
        },
        "missed_by_arm": {
            name: sorted(
                {
                    row["url"]
                    for row in rows
                    if row["block"] in ("shift", "holdout")
                    and row["group"] == "phishing"
                    and row["verdict"] == "legitimate"
                }
            )
            for name, rows in (
                ("static", static_rows),
                ("adaptive", adaptive_rows),
                ("retrained_static", retrained_rows),
            )
        },
    }
    DEFAULT_OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["arms"], indent=2))
    print(json.dumps(report["adaptive_policy"], indent=2))


if __name__ == "__main__":
    main()
