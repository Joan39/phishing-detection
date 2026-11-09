"""Adaptive policy layer: context-specific thresholds, feedback-weighted
components, label-free drift detection and disagreement routing.

The live system was already adaptive in one narrow way - an epsilon-greedy
bandit moves a single global threshold among five values, and labelled feedback
retrains the models. That cannot follow a new phishing technique, because one
threshold has to serve plain-HTTP lookalikes, deep HTTPS content paths and
trusted-brand hosts at once, and because the 0.75/0.25 blend is frozen no matter
how the components behave on current traffic.

This module adds four mechanisms, each with a cold-start guard so nothing
deviates from the documented static policy until it has evidence:

``choose_threshold``   per-context bandit (scheme x trusted-root), falls back to
                       the global threshold while a context is cold.
``update_component_weights``
                       multiplicative-weights over the recent accuracy of A, B
                       and the VQC, bounded so no component is ever zeroed out
                       and shrunk toward the documented blend until enough
                       distinct URLs have been labeled.
``observe_distribution``
                       population shift of the hybrid score histogram; needs no
                       labels, so a technique change is noticed before feedback.
``disagreement``       spread between components, used to mark a scan as worth
                       analyst review instead of pretending to be confident.

State lives in ``results/adaptive_policy.json`` (gitignored, like the RL state).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_POLICY_FILE = Path("results") / "adaptive_policy.json"

THRESHOLD_ACTIONS = [0.30, 0.40, 0.50, 0.60, 0.70]
CLASSICAL_BOUNDS = (0.20, 0.80)
QUANTUM_WEIGHT_BOUNDS = (0.10, 0.40)
DEFAULT_QUANTUM_WEIGHT = 0.25
MIN_COMPONENT_SAMPLES = 10
MIN_CONTEXT_SAMPLES = 8
MIN_EVIDENCE_URLS = 20
EVIDENCE_URL_CAP = 5000
LEARNING_RATE = 0.2
COMPONENT_CUT = 0.5
DISTRIBUTION_WINDOW = 200
DRIFT_BINS = 10
DRIFT_PSI_TRIGGER = 0.25
DISAGREEMENT_REVIEW = 0.25
NEAR_THRESHOLD_REVIEW = 0.10
WEIGHT_TEMPERATURE = 4.0

_POLICY_CACHE: dict[str, tuple[float, dict]] = {}


def _histogram(values: list[float]) -> list[float]:
    counts, _ = np.histogram(np.clip(np.asarray(values, dtype=float), 0.0, 1.0), bins=DRIFT_BINS, range=(0.0, 1.0))
    total = float(counts.sum())
    return [count / total for count in counts] if total else [0.0] * DRIFT_BINS


def population_shift(reference: list[float], current: list[float]) -> float:
    """Population stability index; above ~0.25 the distributions have moved."""
    psi = 0.0
    for expected, actual in zip(reference, current):
        e = max(expected, 1e-6)
        a = max(actual, 1e-6)
        psi += (a - e) * math.log(a / e)
    return round(float(psi), 4)


def default_policy() -> dict[str, Any]:
    return {
        "classical_weights": {"A": 0.35, "B": 0.65},
        "quantum_weight": DEFAULT_QUANTUM_WEIGHT,
        "component_accuracy": {
            name: {"correct": 0, "total": 0} for name in ("A", "B", "quantum")
        },
        "evidence_urls": [],
        "contexts": {},
        "distribution": {"reference": [], "recent": [], "psi": 0.0, "drift": False},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def load_policy(path: Path = DEFAULT_POLICY_FILE, *, use_cache: bool = True) -> dict[str, Any]:
    if not path.exists():
        return default_policy()
    stamp = path.stat().st_mtime
    if use_cache:
        cached = _POLICY_CACHE.get(str(path))
        if cached is not None and cached[0] == stamp:
            return cached[1]
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Adaptive policy is unreadable: {exc}") from exc
    merged = default_policy()
    merged.update(state)
    merged["component_accuracy"] = {**default_policy()["component_accuracy"], **state.get("component_accuracy", {})}
    if use_cache:
        _POLICY_CACHE[str(path)] = (stamp, merged)
    return merged


def save_policy(state: dict[str, Any], path: Path = DEFAULT_POLICY_FILE) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)
    _POLICY_CACHE[str(path)] = (path.stat().st_mtime, state)


def policy_context(*, is_https: bool, is_trusted_root: bool) -> str:
    """Coarse traffic class the threshold is chosen for.

    Only three bits, because the whole point is that each class needs a
    different cut: plain HTTP is where the corpus is thinnest, and trusted-root
    HTTPS is where the calibration cap already suppresses the score.
    """
    scheme = "https" if is_https else "http"
    return f"{scheme}/{'trusted' if is_trusted_root else 'public'}"


def context_state(state: dict[str, Any], context: str) -> dict[str, Any]:
    return state["contexts"].get(
        context, {"thresholds": list(THRESHOLD_ACTIONS), "q_values": [0.0] * len(THRESHOLD_ACTIONS), "samples": 0}
    )


def choose_threshold(
    state: dict[str, Any],
    context: str,
    global_threshold: float,
) -> tuple[float, int, bool]:
    """Return (threshold, action index, whether the context was used)."""
    entry = context_state(state, context)
    samples = int(entry.get("samples", 0))
    q_values = [float(value) for value in entry.get("q_values", [])]
    if samples < MIN_CONTEXT_SAMPLES or not q_values or all(value == 0.0 for value in q_values):
        action = int(np.argmin(np.abs(np.asarray(THRESHOLD_ACTIONS) - global_threshold)))
        return THRESHOLD_ACTIONS[action], action, False
    thresholds = [float(value) for value in entry.get("thresholds", THRESHOLD_ACTIONS)]
    return float(thresholds[int(np.argmax(q_values))]), int(np.argmax(q_values)), True


def update_context(
    state: dict[str, Any],
    context: str,
    action: int,
    reward: float,
) -> None:
    entry = context_state(state, context)
    entry.setdefault("thresholds", list(THRESHOLD_ACTIONS))
    entry.setdefault("q_values", [0.0] * len(entry["thresholds"]))
    entry["q_values"][action] += LEARNING_RATE * (reward - entry["q_values"][action])
    entry["samples"] = int(entry.get("samples", 0)) + 1
    state["contexts"][context] = entry


def component_accuracy(state: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name, bucket in state["component_accuracy"].items():
        total = int(bucket.get("total", 0))
        out[name] = None if total < MIN_COMPONENT_SAMPLES else round(bucket["correct"] / total, 4)
    return out


def weight_confidence(state: dict[str, Any]) -> float:
    """How far the re-ranker has earned its way off the documented blend.

    Twenty distinct labeled URLs is the point where its own opinion is allowed
    full weight. Below that the learned weights are shrunk toward the defaults,
    because a handful of feedback rows - usually the few URLs an analyst happens
    to recheck - is not enough to judge which detector is right on traffic.
    """
    return min(1.0, len(state.get("evidence_urls", [])) / MIN_EVIDENCE_URLS)


def _record_evidence(state: dict[str, Any], url: str | None) -> None:
    if not url:
        return
    seen = list(state.get("evidence_urls", []))
    if url in seen:
        return
    seen.append(url)
    state["evidence_urls"] = seen[-EVIDENCE_URL_CAP:]


def update_component_weights(
    state: dict[str, Any],
    scores: dict[str, float],
    label: int,
    url: str | None = None,
) -> dict[str, Any]:
    """Score each component against the analyst label and re-weight them.

    Classical weights follow a temperature-scaled softmax over recent accuracy,
    so the better-fitting detector gains ground without ever being silenced.
    The quantum weight tracks how far the circuit is above chance, bounded in
    ``QUANTUM_WEIGHT_BOUNDS``. Both are shrunk toward the documented blend until
    ``weight_confidence`` says there is enough independent evidence (see
    :func:`weight_confidence`).
    """
    _record_evidence(state, url)
    for name in ("A", "B", "quantum"):
        value = scores.get(name)
        if value is None:
            continue
        bucket = state["component_accuracy"].setdefault(name, {"correct": 0, "total": 0})
        bucket["total"] += 1
        bucket["correct"] += int((float(value) >= COMPONENT_CUT) == bool(label))

    accuracy = component_accuracy(state)
    if all(value is None for value in accuracy.values()):
        return state

    baseline = default_policy()
    confidence = weight_confidence(state)

    classical = {name: accuracy[name] for name in ("A", "B") if accuracy[name] is not None}
    if len(classical) == 2:
        exponentials = {name: math.exp(WEIGHT_TEMPERATURE * value) for name, value in classical.items()}
        total = sum(exponentials.values())
        weights = {name: value / total for name, value in exponentials.items()}
        weights = {
            name: min(max(value, CLASSICAL_BOUNDS[0]), CLASSICAL_BOUNDS[1])
            for name, value in weights.items()
        }
        normalised = sum(weights.values())
        learned = {name: value / normalised for name, value in weights.items()}
        state["classical_weights"] = {
            name: round(
                (1 - confidence) * baseline["classical_weights"][name]
                + confidence * value,
                4,
            )
            for name, value in learned.items()
        }

    quantum = accuracy.get("quantum")
    if quantum is not None:
        span = QUANTUM_WEIGHT_BOUNDS[1] - QUANTUM_WEIGHT_BOUNDS[0]
        above_chance = min(max(quantum - 0.5, 0.0) / 0.3, 1.0)
        learned = QUANTUM_WEIGHT_BOUNDS[0] + span * above_chance
        state["quantum_weight"] = round(
            (1 - confidence) * baseline["quantum_weight"] + confidence * learned, 4
        )
    return state


def fuse_adaptive(
    scores: dict[str, float],
    classical_weights: dict[str, float],
    quantum_weight: float,
) -> float:
    """Live fusion with the adaptive weights; quantum drops out when absent."""
    present = {name: value for name, value in scores.items() if name in classical_weights}
    if not present:
        raise ValueError("Fusion requires at least one classical score.")
    weight_total = sum(classical_weights[name] for name in present)
    classical = sum(classical_weights[name] * present[name] for name in present) / weight_total
    quantum = scores.get("quantum")
    return classical if quantum is None else (1 - quantum_weight) * classical + quantum_weight * quantum


def disagreement(scores: dict[str, float]) -> float:
    values = [float(value) for value in scores.values() if value is not None]
    return 0.0 if len(values) < 2 else round(max(values) - min(values), 4)


def needs_review(scores: dict[str, float], probability: float, threshold: float) -> bool:
    """Flag scans the ensemble is not confident about, for analyst review.

    High spread plus a score sitting near the cut is where a mislabel is most
    likely, and those are also the most valuable training rows.
    """
    return (
        disagreement(scores) >= DISAGREEMENT_REVIEW
        and abs(probability - threshold) <= NEAR_THRESHOLD_REVIEW
    )


def observe_distribution(state: dict[str, Any], probability: float) -> dict[str, Any]:
    """Track the hybrid score histogram and flag drift without needing labels."""
    window = state["distribution"]
    recent = [float(value) for value in window.get("recent", [])] + [float(probability)]
    recent = recent[-DISTRIBUTION_WINDOW:]
    window["recent"] = recent
    if not window.get("reference"):
        window["reference"] = _histogram(recent) if len(recent) >= DISTRIBUTION_WINDOW else []
    if window.get("reference") and len(recent) >= max(20, DISTRIBUTION_WINDOW // 4):
        psi = population_shift(window["reference"], _histogram(recent))
        window["psi"] = psi
        window["drift"] = bool(psi >= DRIFT_PSI_TRIGGER)
    else:
        window["psi"] = 0.0
        window["drift"] = False
    state["distribution"] = window
    return state


def reset_distribution_reference(state: dict[str, Any]) -> dict[str, Any]:
    """Called after a retrain: the post-retrain traffic becomes the baseline."""
    recent = [float(value) for value in state["distribution"].get("recent", [])]
    state["distribution"] = {
        "reference": _histogram(recent) if recent else [],
        "recent": recent,
        "psi": 0.0,
        "drift": False,
    }
    return state


def policy_summary(state: dict[str, Any], global_threshold: float) -> dict[str, Any]:
    contexts = {
        context: {
            "threshold": float(entry["thresholds"][int(np.argmax(entry["q_values"]))])
            if entry.get("q_values") and any(entry["q_values"])
            else None,
            "samples": int(entry.get("samples", 0)),
        }
        for context, entry in state["contexts"].items()
    }
    baseline = default_policy()
    return {
        "classical_weights": state["classical_weights"],
        "quantum_weight": state["quantum_weight"],
        "component_accuracy": component_accuracy(state),
        "component_samples": {
            name: int(bucket.get("total", 0)) for name, bucket in state["component_accuracy"].items()
        },
        "contexts": contexts,
        "evidence_urls": len(state.get("evidence_urls", [])),
        "weight_confidence": round(weight_confidence(state), 4),
        "active_contexts": sum(
            1
            for entry in contexts.values()
            if entry["threshold"] is not None and entry["samples"] >= MIN_CONTEXT_SAMPLES
        ),
        "at_documented_defaults": (
            state["classical_weights"] == baseline["classical_weights"]
            and float(state["quantum_weight"]) == float(baseline["quantum_weight"])
        ),
        "global_threshold": global_threshold,
        "drift": {
            "psi": state["distribution"].get("psi", 0.0),
            "flagged": bool(state["distribution"].get("drift", False)),
            "window": len(state["distribution"].get("recent", [])),
        },
        "updated_at": state.get("updated_at"),
    }
