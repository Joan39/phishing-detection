"""Unit tests for how the comparison endpoint describes its benchmark source.

The panel used to present a dataset-only, 18-feature, 0.50-threshold run as the
deployed accuracy, so the labelling rules are tested rather than trusted.
"""

from __future__ import annotations

import unittest

from backend.main import _benchmark_description

LEGACY = {
    "fold": 0,
    "test_rows": 47158,
    "features": [f"f{i}" for i in range(18)],
    "metrics": {"A": {"accuracy": 0.9964}},
}

PRODUCTION = {
    "fold": 0,
    "test_rows": 47158,
    "threshold": 0.4,
    "feature_count": 19,
    "generated_at": "2026-11-06T10:00:00+00:00",
    "training_recipe": {
        "hard_negative_rows": 87320,
        "classical_weights": {"A": 0.35, "B": 0.65},
        "quantum_weight": 0.25,
    },
    "quantum_circuit": {"qubits": 4, "shots": 512},
    "realworld": {"sets": {"battery": {"summary": {}}}},
    "metrics": {"A": {"accuracy": 0.8741}},
}


class BenchmarkDescriptionTests(unittest.TestCase):
    def test_legacy_artifact_is_not_reported_as_deployed(self) -> None:
        description = _benchmark_description(LEGACY)
        self.assertFalse(description["deployed_recipe"])
        self.assertIn("Legacy", description["scope"])
        self.assertIn("not the deployed models", description["scope"])
        self.assertIn("dataset-only", description["protocol"])

    def test_production_artifact_names_the_protocol_and_decision_point(self) -> None:
        description = _benchmark_description(PRODUCTION)
        self.assertTrue(description["deployed_recipe"])
        self.assertIn("hard-negative corpus", description["protocol"])
        self.assertIn("threshold 0.40", description["scope"])
        self.assertIn("47,158 unseen rows", description["scope"])
        self.assertEqual(description["hard_negative_rows"], 87320)
        self.assertEqual(description["quantum_circuit"], {"qubits": 4, "shots": 512})

    def test_realworld_block_is_passed_through_for_the_panel(self) -> None:
        self.assertIn("sets", _benchmark_description(PRODUCTION)["realworld"])
        self.assertIsNone(_benchmark_description(LEGACY)["realworld"])


if __name__ == "__main__":
    unittest.main()
