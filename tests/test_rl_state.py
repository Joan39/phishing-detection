"""Unit tests for the live adaptive-threshold (bandit) state handling."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

from backend import main


def _patched_state_file(path: Path):
    return mock.patch.object(main, "RL_STATE_FILE", path)


class RlStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state_path = Path(self._tmp.name) / "rl_state.json"

    def test_missing_file_defaults_to_experiment_action_set_at_neutral_threshold(self) -> None:
        with _patched_state_file(self.state_path):
            state = main._load_rl_state()
        self.assertEqual(state["thresholds"], [0.30, 0.40, 0.50, 0.60, 0.70])
        self.assertEqual(main._active_threshold(state), 0.50)

    def test_learned_state_keeps_its_own_actions_and_argmax(self) -> None:
        self.state_path.write_text(
            json.dumps({"thresholds": [0.30, 0.40, 0.50, 0.60, 0.70],
                        "q_values": [0.0, 0.5904, 0.0, 0.0, 0.0]}),
            encoding="utf-8",
        )
        with _patched_state_file(self.state_path):
            state = main._load_rl_state()
        self.assertEqual(main._active_threshold(state), 0.40)

    def test_all_zero_q_values_on_custom_actions_boosts_closest_to_half(self) -> None:
        self.state_path.write_text(
            json.dumps({"thresholds": [0.25, 0.45, 0.75], "q_values": [0.0, 0.0, 0.0]}),
            encoding="utf-8",
        )
        with _patched_state_file(self.state_path):
            state = main._load_rl_state()
        self.assertEqual(state["q_values"], [0.0, 0.2, 0.0])
        self.assertEqual(main._active_threshold(state), 0.45)

    def test_mismatched_lengths_are_rejected(self) -> None:
        self.state_path.write_text(
            json.dumps({"thresholds": [0.3, 0.4], "q_values": [0.1]}), encoding="utf-8"
        )
        with _patched_state_file(self.state_path):
            with self.assertRaises(HTTPException) as caught:
                main._load_rl_state()
        self.assertEqual(caught.exception.status_code, 503)

    def test_corrupt_file_is_rejected(self) -> None:
        self.state_path.write_text("{not json", encoding="utf-8")
        with _patched_state_file(self.state_path):
            with self.assertRaises(HTTPException) as caught:
                main._load_rl_state()
        self.assertEqual(caught.exception.status_code, 503)

    def test_save_and_reload_roundtrip(self) -> None:
        state = {"thresholds": [0.30, 0.40], "q_values": [0.1, 0.9]}
        with _patched_state_file(self.state_path):
            main._save_rl_state(state)
            loaded = main._load_rl_state()
        self.assertEqual(loaded, state)
        self.assertEqual(main._active_threshold(loaded), 0.40)


if __name__ == "__main__":
    unittest.main()
