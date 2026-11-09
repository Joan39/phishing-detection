"""Unit tests for the adaptive policy layer and its cold-start guarantees."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.adaptive_policy import (
    CLASSICAL_BOUNDS,
    DEFAULT_POLICY_FILE,
    DISTRIBUTION_WINDOW,
    MIN_COMPONENT_SAMPLES,
    MIN_CONTEXT_SAMPLES,
    MIN_EVIDENCE_URLS,
    QUANTUM_WEIGHT_BOUNDS,
    THRESHOLD_ACTIONS,
    choose_threshold,
    component_accuracy,
    default_policy,
    disagreement,
    fuse_adaptive,
    load_policy,
    needs_review,
    observe_distribution,
    policy_context,
    policy_summary,
    population_shift,
    reset_distribution_reference,
    save_policy,
    update_component_weights,
    update_context,
    weight_confidence,
)


class ContextThresholdTests(unittest.TestCase):
    def test_cold_context_falls_back_to_the_global_threshold(self) -> None:
        state = default_policy()
        threshold, action, used = choose_threshold(state, "https/public", 0.40)
        self.assertEqual((threshold, action, used), (0.40, 1, False))

    def test_context_takes_over_once_it_has_been_rewarded(self) -> None:
        state = default_policy()
        for _ in range(MIN_CONTEXT_SAMPLES):
            update_context(state, "http/public", 0, 1.0)
        threshold, action, used = choose_threshold(state, "http/public", 0.70)
        self.assertTrue(used)
        self.assertEqual((threshold, action), (THRESHOLD_ACTIONS[0], 0))

    def test_another_context_is_unaffected(self) -> None:
        state = default_policy()
        for _ in range(MIN_CONTEXT_SAMPLES):
            update_context(state, "http/public", 0, 1.0)
        threshold, _, used = choose_threshold(state, "https/trusted", 0.50)
        self.assertFalse(used)
        self.assertEqual(threshold, 0.50)

    def test_reward_moves_the_preferred_action(self) -> None:
        state = default_policy()
        for _ in range(MIN_CONTEXT_SAMPLES):
            update_context(state, "https/public", 2, 1.0)
        for _ in range(30):
            update_context(state, "https/public", 4, 1.0)
            update_context(state, "https/public", 2, -1.0)
        threshold, _, _ = choose_threshold(state, "https/public", 0.30)
        self.assertEqual(threshold, THRESHOLD_ACTIONS[4])

    def test_context_label_separates_scheme_and_trusted_hosts(self) -> None:
        self.assertEqual(
            [
                policy_context(is_https=True, is_trusted_root=False),
                policy_context(is_https=True, is_trusted_root=True),
                policy_context(is_https=False, is_trusted_root=True),
            ],
            ["https/public", "https/trusted", "http/trusted"],
        )


class ComponentWeightTests(unittest.TestCase):
    def test_no_reweighting_before_the_minimum_samples(self) -> None:
        state = default_policy()
        for index in range(MIN_COMPONENT_SAMPLES - 1):
            update_component_weights(state, {"A": 0.9, "B": 0.1}, 1, f"https://a{index}.test/x")
        self.assertIsNone(component_accuracy(state)["A"])
        self.assertEqual(state["classical_weights"], {"A": 0.35, "B": 0.65})

    def test_better_component_gains_weight_within_bounds(self) -> None:
        state = default_policy()
        for index in range(40):
            url = f"https://traffic{index}.test/x"
            update_component_weights(state, {"A": 0.9, "B": 0.1}, 1, url)
            update_component_weights(state, {"A": 0.1, "B": 0.9}, 0, url)
        weights = state["classical_weights"]
        self.assertGreater(weights["A"], 0.65)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=4)
        self.assertTrue(
            all(CLASSICAL_BOUNDS[0] <= value <= CLASSICAL_BOUNDS[1] for value in weights.values())
        )

    def test_re_ranking_is_shrunk_until_enough_distinct_urls_are_labeled(self) -> None:
        state = default_policy()
        for _ in range(40):
            update_component_weights(state, {"A": 0.9, "B": 0.1}, 1, "https://same.test/x")
        self.assertEqual(weight_confidence(state), 1 / MIN_EVIDENCE_URLS)
        self.assertLess(state["classical_weights"]["A"], 0.40)
        first_round = state["classical_weights"]["A"]
        for index in range(MIN_EVIDENCE_URLS):
            update_component_weights(state, {"A": 0.9, "B": 0.1}, 1, f"https://{index}.test/x")
        self.assertEqual(weight_confidence(state), 1.0)
        self.assertGreater(state["classical_weights"]["A"], first_round + 0.2)

    def test_half_the_evidence_gives_half_the_opinion(self) -> None:
        state = default_policy()
        for index in range(40):
            update_component_weights(state, {"A": 0.9, "B": 0.1}, 1, f"https://{index}.test/x")
        halfway = round(0.35 * (1 - 10 / MIN_EVIDENCE_URLS) + state["classical_weights"]["A"] * (10 / MIN_EVIDENCE_URLS), 4)
        fresh = default_policy()
        for index in range(10):
            update_component_weights(fresh, {"A": 0.9, "B": 0.1}, 1, f"https://{index}.test/x")
        self.assertAlmostEqual(fresh["classical_weights"]["A"], halfway, places=4)

    def test_quantum_weight_tracks_accuracy_and_stays_bounded(self) -> None:
        strong = default_policy()
        for index in range(40):
            update_component_weights(strong, {"quantum": 0.9}, 1, f"https://s{index}.test/x")
        self.assertGreater(strong["quantum_weight"], 0.25)
        weak = default_policy()
        for index in range(40):
            update_component_weights(weak, {"quantum": 0.9}, 0, f"https://w{index}.test/x")
        self.assertEqual(weak["quantum_weight"], QUANTUM_WEIGHT_BOUNDS[0])
        self.assertLessEqual(strong["quantum_weight"], QUANTUM_WEIGHT_BOUNDS[1])

    def test_missing_component_score_is_skipped(self) -> None:
        state = default_policy()
        for index in range(20):
            update_component_weights(state, {"A": 0.9}, 1, f"https://m{index}.test/x")
        self.assertIsNone(component_accuracy(state)["B"])
        self.assertEqual(state["component_accuracy"]["B"]["total"], 0)


class FusionTests(unittest.TestCase):
    def test_default_policy_reproduces_the_documented_static_blend(self) -> None:
        state = default_policy()
        scores = {"A": 0.8, "B": 0.4, "quantum": 0.6}
        self.assertAlmostEqual(
            fuse_adaptive(scores, state["classical_weights"], state["quantum_weight"]),
            0.75 * (0.35 * 0.8 + 0.65 * 0.4) + 0.25 * 0.6,
        )

    def test_absent_quantum_score_reduces_to_the_classical_mean(self) -> None:
        self.assertAlmostEqual(
            fuse_adaptive({"A": 0.2, "B": 0.6}, {"A": 0.5, "B": 0.5}, 0.25), 0.4
        )

    def test_unlisted_component_is_excluded_from_normalisation(self) -> None:
        self.assertAlmostEqual(
            fuse_adaptive({"A": 0.2, "C": 0.9}, {"A": 0.5, "B": 0.5}, 0.25), 0.2
        )

    def test_no_classical_score_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fuse_adaptive({"quantum": 0.5}, {"A": 0.35, "B": 0.65}, 0.25)


class DisagreementTests(unittest.TestCase):
    def test_spread_is_the_widest_component_gap(self) -> None:
        self.assertAlmostEqual(
            disagreement({"A": 0.9, "B": 0.5, "quantum": 0.2}), 0.7
        )

    def test_single_component_cannot_disagree(self) -> None:
        self.assertEqual(disagreement({"A": 0.9}), 0.0)

    def test_review_is_requested_for_spread_near_the_cut(self) -> None:
        scores = {"A": 0.55, "B": 0.15, "quantum": 0.45}
        self.assertTrue(needs_review(scores, 0.38, 0.40))

    def test_confident_score_is_not_routed_for_review(self) -> None:
        scores = {"A": 0.95, "B": 0.10, "quantum": 0.85}
        self.assertFalse(needs_review(scores, 0.90, 0.40))

    def test_agreement_near_the_cut_is_not_routed_for_review(self) -> None:
        scores = {"A": 0.42, "B": 0.40, "quantum": 0.41}
        self.assertFalse(needs_review(scores, 0.41, 0.40))


class DriftTests(unittest.TestCase):
    def test_identical_histograms_have_no_shift(self) -> None:
        bins = [0.1] * 10
        self.assertEqual(population_shift(bins, bins), 0.0)

    def test_score_mass_moving_to_the_top_flags_drift(self) -> None:
        state = default_policy()
        for _ in range(DISTRIBUTION_WINDOW):
            observe_distribution(state, 0.10)
        self.assertFalse(state["distribution"]["drift"])
        self.assertEqual(len(state["distribution"]["reference"]), 10)
        for _ in range(120):
            observe_distribution(state, 0.90)
        self.assertTrue(state["distribution"]["drift"])
        self.assertGreater(state["distribution"]["psi"], 0.25)
        # The window is bounded, so observations cannot grow without limit.
        self.assertLessEqual(len(state["distribution"]["recent"]), DISTRIBUTION_WINDOW)

    def test_reset_makes_the_current_traffic_the_new_baseline(self) -> None:
        state = default_policy()
        for _ in range(DISTRIBUTION_WINDOW):
            observe_distribution(state, 0.90)
        for _ in range(120):
            observe_distribution(state, 0.10)
        self.assertTrue(state["distribution"]["drift"])
        reset_distribution_reference(state)
        self.assertFalse(state["distribution"]["drift"])
        self.assertEqual(state["distribution"]["psi"], 0.0)
        # The 120 recent low scores (bin [0.1, 0.2)) now define the baseline.
        self.assertAlmostEqual(sum(state["distribution"]["reference"]), 1.0)
        self.assertAlmostEqual(state["distribution"]["reference"][1], 0.6)

    def test_a_stable_stream_stays_below_the_trigger(self) -> None:
        state = default_policy()
        for index in range(400):
            observe_distribution(state, 0.20 if index % 2 else 0.21)
        self.assertFalse(state["distribution"]["drift"])


class PersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.path = Path(self._directory.name) / "adaptive_policy.json"

    def test_round_trip_keeps_learned_state(self) -> None:
        state = default_policy()
        for _ in range(MIN_CONTEXT_SAMPLES):
            update_context(state, "https/trusted", 3, 1.0)
        save_policy(state, self.path)
        reloaded = load_policy(self.path, use_cache=False)
        self.assertEqual(reloaded["contexts"], state["contexts"])
        self.assertIsNotNone(reloaded["updated_at"])

    def test_missing_file_yields_the_static_defaults(self) -> None:
        state = load_policy(self.path, use_cache=False)
        self.assertEqual(state["classical_weights"], {"A": 0.35, "B": 0.65})
        self.assertEqual(state["quantum_weight"], 0.25)
        self.assertEqual(state["contexts"], {})

    def test_partial_file_is_merged_over_the_defaults(self) -> None:
        self.path.write_text('{"quantum_weight": 0.18}', encoding="utf-8")
        state = load_policy(self.path, use_cache=False)
        self.assertEqual(state["quantum_weight"], 0.18)
        self.assertEqual(state["classical_weights"], {"A": 0.35, "B": 0.65})

    def test_summary_reports_contexts_and_drift(self) -> None:
        state = default_policy()
        for _ in range(MIN_CONTEXT_SAMPLES):
            update_context(state, "http/public", 0, 1.0)
        update_context(state, "https/trusted", 0, 1.0)
        summary = policy_summary(state, 0.40)
        self.assertEqual(summary["global_threshold"], 0.40)
        self.assertEqual(summary["contexts"]["http/public"]["threshold"], 0.30)
        self.assertEqual(summary["contexts"]["http/public"]["samples"], MIN_CONTEXT_SAMPLES)
        self.assertFalse(summary["drift"]["flagged"])
        self.assertTrue(summary["at_documented_defaults"])

    def test_summary_counts_only_contexts_that_have_taken_over(self) -> None:
        state = default_policy()
        update_context(state, "https/public", 0, 1.0)
        self.assertEqual(policy_summary(state, 0.40)["active_contexts"], 0)
        for _ in range(MIN_CONTEXT_SAMPLES):
            update_context(state, "https/public", 0, 1.0)
        self.assertEqual(policy_summary(state, 0.40)["active_contexts"], 1)

    def test_summary_reports_relearned_weights_as_not_default(self) -> None:
        state = default_policy()
        state["quantum_weight"] = 0.31
        self.assertFalse(policy_summary(state, 0.40)["at_documented_defaults"])

    def test_default_file_is_the_ignored_results_directory(self) -> None:
        self.assertEqual(DEFAULT_POLICY_FILE, Path("results") / "adaptive_policy.json")
        with patch("backend.main.ADAPTIVE_POLICY_FILE", self.path):
            from backend.main import _load_policy

            self.assertEqual(_load_policy()["classical_weights"], {"A": 0.35, "B": 0.65})


if __name__ == "__main__":
    unittest.main()
