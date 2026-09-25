"""Tests for the OpenRouter Qwen HTTP client."""

import os
import unittest
from unittest.mock import Mock, patch

import requests

from client.openrouter import (
    OPENROUTER_URL,
    QWEN_MODEL,
    REQUEST_TIMEOUT_SECONDS,
    OpenRouterClientError,
    call_qwen_model,
)


class OpenRouterClientTests(unittest.TestCase):
    """Verify request construction and controlled API failures."""

    def setUp(self) -> None:
        self.original_api_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-api-key"

    def tearDown(self) -> None:
        if self.original_api_key is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = self.original_api_key

    @patch("client.openrouter.requests.post")
    def test_calls_pinned_free_qwen_model_with_expected_request(
        self,
        mock_post: Mock,
    ) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Paris is the capital of France."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        result = call_qwen_model("What is the capital of France?")

        self.assertEqual(result, "Paris is the capital of France.")
        mock_response.raise_for_status.assert_called_once()
        mock_post.assert_called_once_with(
            OPENROUTER_URL,
            headers={
                "Authorization": "Bearer test-api-key",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "Safe Qwen OpenRouter Gateway",
            },
            json={
                "model": QWEN_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the capital of France?",
                    }
                ],
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    def test_fails_when_api_key_is_missing(self) -> None:
        os.environ.pop("OPENROUTER_API_KEY", None)

        with self.assertRaises(OpenRouterClientError) as context:
            call_qwen_model("Hello")

        self.assertIn("OPENROUTER_API_KEY", str(context.exception))

    @patch("client.openrouter.requests.post")
    def test_wraps_network_errors(self, mock_post: Mock) -> None:
        mock_post.side_effect = requests.Timeout("Request timed out")

        with self.assertRaises(OpenRouterClientError) as context:
            call_qwen_model("Hello")

        self.assertIn("OpenRouter request failed", str(context.exception))

    @patch("client.openrouter.requests.post")
    def test_wraps_malformed_response(self, mock_post: Mock) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {"choices": []}
        mock_post.return_value = mock_response

        with self.assertRaises(OpenRouterClientError) as context:
            call_qwen_model("Hello")

        self.assertIn("invalid chat-completion response", str(context.exception))

    @patch("client.openrouter.requests.post")
    def test_rejects_empty_model_content(self, mock_post: Mock) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "   "}}]
        }
        mock_post.return_value = mock_response

        with self.assertRaises(OpenRouterClientError) as context:
            call_qwen_model("Hello")

        self.assertIn("empty or non-text response", str(context.exception))

    def test_rejects_non_string_prompt(self) -> None:
        with self.assertRaises(TypeError):
            call_qwen_model(123)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()