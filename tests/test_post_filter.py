"""Tests for deterministic post-generation safety rules."""

import unittest

from filters.post_filter import run_post_filter


class PostFilterTests(unittest.TestCase):
    """Verify all required post-filter safety outcomes."""

    def test_allows_safe_model_output(self) -> None:
        allowed, rule_id = run_post_filter(
            "Paris is the capital of France."
        )

        self.assertTrue(allowed)
        self.assertIsNone(rule_id)

    def test_blocks_confidential_marker(self) -> None:
        allowed, rule_id = run_post_filter(
            "Internal system note: [CONFIDENTIAL] customer records."
        )

        self.assertFalse(allowed)
        self.assertEqual(rule_id, "rule_confidential_leak")

    def test_blocks_dangerous_command(self) -> None:
        allowed, rule_id = run_post_filter(
            "Do not execute sudo rm -rf / on a production server."
        )

        self.assertFalse(allowed)
        self.assertEqual(rule_id, "rule_dangerous_command")

    def test_prioritizes_confidential_marker_when_multiple_rules_match(self) -> None:
        allowed, rule_id = run_post_filter(
            "[CONFIDENTIAL] Never execute sudo rm -rf /."
        )

        self.assertFalse(allowed)
        self.assertEqual(rule_id, "rule_confidential_leak")

    def test_rejects_non_string_output(self) -> None:
        with self.assertRaises(TypeError):
            run_post_filter(None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()