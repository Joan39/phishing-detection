import unittest
from pathlib import Path
from unittest.mock import patch

from backend import quantum_live as ql


class QuantumHardNegativeTests(unittest.TestCase):
    def test_strided_sample_bounds(self):
        urls = [str(index) for index in range(100)]

        self.assertEqual(ql.strided_sample(urls, 0), [])
        self.assertEqual(len(ql.strided_sample(urls, 10)), 10)
        self.assertEqual(ql.strided_sample(urls, 1000), urls)

    def test_hard_negative_rows_are_labelled_legitimate(self):
        urls, targets = ql.hard_negative_feedback_rows(50)

        self.assertEqual(len(urls), 50)
        self.assertEqual(set(targets), {0})
        self.assertTrue(all(url.startswith("http") for url in urls))

    def test_feedback_adaptation_readds_the_corpus(self):
        """One feedback retrain must not undo the legitimate counter-examples."""
        with patch.object(ql, "train_quantum_model") as fit:
            ql.adapt_live_model(
                Path("dataset.csv"),
                Path("model.joblib"),
                ["https://phish.example/x"],
                [1],
            )

        kwargs = fit.call_args.kwargs
        self.assertEqual(kwargs["feedback_urls"][0], "https://phish.example/x")
        self.assertEqual(kwargs["feedback_targets"][0], 1)
        self.assertEqual(set(kwargs["feedback_targets"][1:]), {0})
        self.assertEqual(
            kwargs["feedback_urls"][1:],
            ql.hard_negative_feedback_rows()[0],
        )
        self.assertEqual(
            len(kwargs["feedback_urls"]), len(kwargs["feedback_targets"])
        )


if __name__ == "__main__":
    unittest.main()
