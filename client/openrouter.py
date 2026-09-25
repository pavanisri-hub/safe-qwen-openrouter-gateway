"""HTTP client for calling a pinned Qwen model through OpenRouter."""

import os
from typing import Any, Dict

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
QWEN_MODEL = "qwen/qwen-2.5-7b-instruct"
REQUEST_TIMEOUT_SECONDS = 30


class OpenRouterClientError(Exception):
    """Raised when the OpenRouter request or response cannot be used safely."""


def call_qwen_model(prompt: str) -> str:
    """
    Send a pre-filtered prompt to Qwen through OpenRouter.

    Raises:
        OpenRouterClientError: If configuration, networking, HTTP status,
        JSON parsing, or response shape is invalid.
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterClientError(
            "OPENROUTER_API_KEY is not configured in the environment."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv(
            "OPENROUTER_HTTP_REFERER",
            "http://localhost",
        ),
        "X-Title": os.getenv(
            "OPENROUTER_X_TITLE",
            "Safe Qwen OpenRouter Gateway",
        ),
    }

    payload: Dict[str, Any] = {
        "model": QWEN_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OpenRouterClientError(
            f"OpenRouter request failed: {exc}"
        ) from exc

    try:
        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise OpenRouterClientError(
            "OpenRouter returned an invalid chat-completion response."
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise OpenRouterClientError(
            "OpenRouter returned an empty or non-text response."
        )

    return content