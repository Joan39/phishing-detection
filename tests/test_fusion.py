"""Unit tests for the live classical/hybrid fusion helpers."""

from __future__ import annotations

import os
import unittest

from backend.main import (
    _classical_weights,
    fuse_classical_probabilities,
    fuse_hybrid_probability,
)


class ClassicalWeightParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous = os.environ.pop("QPHISH_CLASSICAL_WEIGHTS", None)

    def tearDown(self) -> None:
        os.environ.pop("QPHISH_CLASSICAL_WEIGHTS", None)
        if self._previous is not None:
            os.environ["QPHISH_CLASSICAL_WEIGHTS"] = self._previous

    def test_default_split_favours_the_ensemble(self) -> None:
        self.assertEqual(_classical_weights(), {"A": 0.35, "B": 0.65})

    def test_explicit_override(self) -> None:
        os.environ["QPHISH_CLASSICAL_WEIGHTS"] = "A=0.3, B=0.7"
        self.assertEqual(_classical_weights(), {"A": 0.3, "B": 0.7})

    def test_empty_value_means_equal_weighting(self) -> None:
        os.environ["QPHISH_CLASSICAL_WEIGHTS"] = ""
        self.assertEqual(_classical_weights(), {})

    def test_malformed_value_is_rejected(self) -> None:
        os.environ["QPHISH_CLASSICAL_WEIGHTS"] = "A=0.35,B=oops"
        with self.assertRaises(ValueError):
            _classical_weights()

    def test_negative_value_is_rejected(self) -> None:
        os.environ["QPHISH_CLASSICAL_WEIGHTS"] = "A=-1,B=2"
        with self.assertRaises(ValueError):
            _classical_weights()


class FusionTests(unittest.TestCase):
    def test_weighted_classical_mean(self) -> None:
        scores = {"A": 0.9, "B": 0.1}
        fused = fuse_classical_probabilities(scores, {"A": 0.35, "B": 0.65})
        self.assertAlmostEqual(fused, 0.35 * 0.9 + 0.65 * 0.1)

    def test_unknown_models_fall_back_to_equal_weight(self) -> None:
        scores = {"A": 0.2, "C": 0.6}
        fused = fuse_classical_probabilities(scores, {"A": 0.35, "B": 0.65})
        self.assertAlmostEqual(fused, (0.35 * 0.2 + 1.0 * 0.6) / 1.35)

    def test_empty_weights_reduces_to_plain_average(self) -> None:
        scores = {"A": 0.2, "B": 0.8}
        self.assertAlmostEqual(
            fuse_classical_probabilities(scores, {}), 0.5
        )

    def test_hybrid_blend(self) -> None:
        self.assertAlmostEqual(
            fuse_hybrid_probability(0.8, 0.4, 0.25), 0.75 * 0.8 + 0.25 * 0.4
        )

    def test_zero_quantum_weight_returns_classical(self) -> None:
        self.assertAlmostEqual(fuse_hybrid_probability(0.62, 0.1, 0.0), 0.62)

    def test_all_zero_classical_weights_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fuse_classical_probabilities({"A": 0.5, "B": 0.5}, {"A": 0.0, "B": 0.0})


if __name__ == "__main__":
    unittest.main()
