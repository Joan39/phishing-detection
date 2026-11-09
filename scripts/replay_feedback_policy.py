"""Rebuild the adaptive policy from the analyst feedback already recorded.

The live endpoint learns from feedback as it arrives, so labels collected before
the policy layer existed leave the policy empty and the system keeps scoring on
the documented static split. This replays the stored feedback rows through the
same update functions, in the order they were recorded, so the per-class
thresholds and the component weights reflect every label the deployment already
has. The label-free drift window is not replayed — it belongs to live traffic.

Usage:
    python -m scripts.replay_feedback_policy [--dry-run] [--output path]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

from backend.adaptive_policy import (
    DEFAULT_POLICY_FILE,
    THRESHOLD_ACTIONS,
    choose_threshold,
    default_policy,
    load_policy,
    policy_context,
    policy_summary,
    save_policy,
    update_component_weights,
    update_context,
)
from backend.main import (
    DATABASE_FILE,
    _active_threshold,
    _load_rl_state,
    _stored_model_scores,
)
from backend.url_features import extract_url_features, trusted_calibration_applies


def _context_for(url: str) -> str:
    features = extract_url_features(url)
    return policy_context(
        is_https=bool(features["IsHTTPS"].iloc[0]),
        is_trusted_root=trusted_calibration_applies(url, features),
    )


def _feedback_rows() -> list[sqlite3.Row]:
    connection = sqlite3.connect(DATABASE_FILE)
    connection.row_factory = sqlite3.Row
    try:
        return list(
            connection.execute(
                "SELECT user_id, scan_url, analyst_verdict, decision_threshold, reward "
                "FROM feedback ORDER BY id"
            )
        )
    finally:
        connection.close()


def replay(rows: list[sqlite3.Row]) -> tuple[dict, int, int]:
    policy = default_policy()
    labeled = 0
    scored = 0
    for row in rows:
        if row["analyst_verdict"] == "uncertain":
            continue
        labeled += 1
        action = int(
            np.argmin(
                np.abs(np.asarray(THRESHOLD_ACTIONS) - float(row["decision_threshold"]))
            )
        )
        context = _context_for(row["scan_url"])
        update_context(policy, context, action, float(row["reward"]))
        component_scores = _stored_model_scores(row["scan_url"], row["user_id"])
        if component_scores:
            scored += 1
            update_component_weights(
                policy,
                component_scores,
                1 if row["analyst_verdict"] == "phishing" else 0,
                row["scan_url"],
            )
    return policy, labeled, scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_POLICY_FILE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what the replay would produce without touching the live policy file",
    )
    args = parser.parse_args()

    global_threshold = _active_threshold(_load_rl_state())
    rows = _feedback_rows()
    policy, labeled, scored = replay(rows)
    summary = policy_summary(policy, global_threshold)
    print(f"feedback rows: {len(rows)}; labeled {labeled}; with stored component scores {scored}")
    print(f"live policy before: {json.dumps(policy_summary(load_policy(args.output, use_cache=False), global_threshold), indent=2)}")
    print(f"replayed policy: {json.dumps(summary, indent=2)}")
    for context in ("https/public", "https/trusted", "http/public", "http/trusted"):
        threshold, _, used = choose_threshold(policy, context, global_threshold)
        print(f"  {context:<14} -> {threshold:.2f} ({'class bandit' if used else 'global bandit'})")
    if args.dry_run:
        print("dry run: live policy unchanged")
        return
    save_policy(policy, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
