import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from backend.threshold_calibration import (
    evaluate_threshold,
    select_zero_false_positive_threshold,
)
from backend.train_models import (
    KNOWN_LEAKAGE_COLUMNS,
    build_feature_frame,
    evaluate,
    load_dataset,
    make_baseline,
    normalise_domain,
    select_features,
)


class TrainModelTests(unittest.TestCase):
    def test_normalise_domain_removes_whitespace_and_www_prefix(self):
        self.assertEqual(normalise_domain("  WWW.Example.COM "), "example.com")

    def test_load_dataset_maps_original_zero_to_phishing(self):
        frame = pd.DataFrame(
            {
                "Domain": ["www.safe.test", "evil.test"],
                "label": [1, 0],
                "URLLength": [20, 80],
            }
        )

        path = self._write_frame(frame)
        try:
            loaded_frame, target, groups = load_dataset(path)
        finally:
            path.unlink()

        self.assertEqual(len(loaded_frame), 2)
        self.assertEqual(target.tolist(), [0, 1])
        self.assertEqual(groups.tolist(), ["safe.test", "evil.test"])

    def test_select_features_excludes_identifiers_and_known_leakage(self):
        frame = pd.DataFrame(
            {
                "URLLength": [10, 20],
                "TLDLength": [3, 3],
                "URLSimilarityIndex": [0.1, 0.9],
                "Domain": ["a.test", "b.test"],
                "label": [0, 1],
            }
        )

        features = select_features(frame, "url")

        self.assertIn("URLLength", features)
        self.assertIn("TLDLength", features)
        self.assertNotIn("URLSimilarityIndex", features)
        self.assertTrue(KNOWN_LEAKAGE_COLUMNS.isdisjoint(features))

    def test_raw_feature_profile_matches_training_features(self):
        frame = pd.DataFrame(
            {
                "URL": ["https://example.com", "http://bad.test/login?id=1"],
                "Domain": ["example.com", "bad.test"],
                "label": [1, 0],
            }
        )

        feature_frame = build_feature_frame(frame, "url_raw")
        features = select_features(feature_frame, "url_raw")

        self.assertEqual(len(features), 19)
        self.assertNotIn("TLD", features)
        self.assertEqual(feature_frame.shape, (2, 19))

    def test_evaluate_returns_binary_metrics_and_confusion_matrix(self):
        model = Pipeline(
            [
                ("classifier", LogisticRegression(random_state=42)),
            ]
        )
        x_train = pd.DataFrame({"feature": [-2, -1, 1, 2]})
        y_train = pd.Series([0, 0, 1, 1])
        x_test = pd.DataFrame({"feature": [-1.5, 1.5]})
        y_test = pd.Series([0, 1])
        model.fit(x_train, y_train)

        metrics = evaluate(model, x_test, y_test)

        self.assertEqual(metrics["confusion_matrix_labels"], ["legitimate", "phishing"])
        self.assertEqual(metrics["confusion_matrix"], [[1, 0], [0, 1]])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1_phishing"], 1.0)

    def test_zero_false_positive_threshold_uses_validation_legitimate_maximum(self):
        probabilities = np.asarray([0.1, 0.72, 0.31, 0.91])
        labels = np.asarray([0, 0, 1, 1])

        threshold = select_zero_false_positive_threshold(probabilities, labels)
        metrics = evaluate_threshold(probabilities, labels, threshold)

        self.assertGreater(threshold, 0.72)
        self.assertEqual(metrics["false_positives"], 0)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [1, 1]])

    @staticmethod
    def _write_frame(frame: pd.DataFrame) -> Path:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
            frame.to_csv(handle.name, index=False)
            return Path(handle.name)


if __name__ == "__main__":
    unittest.main()
