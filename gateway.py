"""CLI pipeline for a Safe Qwen client through OpenRouter."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from client.openrouter import OpenRouterClientError, call_qwen_model
from filters.post_filter import run_post_filter
from filters.pre_filter import run_pre_filter

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "redteam.json"
RESULTS_PATH = BASE_DIR / "results.json"
AUDIT_PATH = BASE_DIR / "audit.jsonl"

SAFETY_REFUSAL = "Refused by safety policy."
API_ERROR_RESPONSE = "Error calling API"


def load_prompt_batch(
    input_path: Optional[Path] = None,
) -> List[Dict[str, str]]:  
    """
    Load and validate the red-team input batch.

    Raises:
        ValueError: If the JSON is invalid or does not match the required schema.
    """
    input_path = input_path or INPUT_PATH

    try:
        with input_path.open("r", encoding="utf-8") as input_file:
            data = json.load(input_file)
    except FileNotFoundError as exc:
        raise ValueError(f"Input file not found: {input_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {input_path}: {exc.msg}") from exc

    if not isinstance(data, list):
        raise ValueError("redteam.json must contain a JSON array.")

    validated_items: List[Dict[str, str]] = []
    seen_ids = set()

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {index} must be a JSON object.")

        required_fields = ("id", "prompt", "type")
        if any(field not in item for field in required_fields):
            raise ValueError(
                f"Item at index {index} must contain id, prompt, and type."
            )

        prompt_id = item["id"]
        prompt = item["prompt"]
        prompt_type = item["type"]

        if not all(
            isinstance(value, str) and value.strip()
            for value in (prompt_id, prompt, prompt_type)
        ):
            raise ValueError(
                f"Item at index {index} must use non-empty string values."
            )

        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt id found: {prompt_id}")

        seen_ids.add(prompt_id)
        validated_items.append(
            {
                "id": prompt_id,
                "prompt": prompt,
                "type": prompt_type,
            }
        )

    return validated_items


def append_audit_log(
    prompt_id: str,
    rule: Optional[str],
    action: str,
    audit_path: Optional[Path] = None,
) -> None:
    """Append one schema-compliant safety decision to the JSONL audit log."""
    audit_path = audit_path or AUDIT_PATH


    if action not in {"allowed", "blocked"}:
        raise ValueError("action must be either 'allowed' or 'blocked'")

    log_entry = {
        "prompt_id": prompt_id,
        "safety_rule": rule,
        "action": action,
    }

    with audit_path.open("a", encoding="utf-8") as audit_file:
        audit_file.write(json.dumps(log_entry) + "\n")


def write_results(
    results: List[Dict[str, str]],
    results_path: Optional[Path] = None,
) -> None:
    """Write the final prompt outcomes as a JSON array."""
    results_path = results_path or RESULTS_PATH

    
    with results_path.open("w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)
        results_file.write("\n")


def process_prompt(item: Dict[str, str]) -> Dict[str, str]:
    """Process one prompt through pre-filter, Qwen, and post-filter gates."""
    prompt_id = item["id"]
    prompt = item["prompt"]

    allowed, rule_id = run_pre_filter(prompt)
    if not allowed:
        append_audit_log(prompt_id, rule_id, "blocked")
        return {
            "id": prompt_id,
            "original_prompt": prompt,
            "response": SAFETY_REFUSAL,
        }

    try:
        model_output = call_qwen_model(prompt)
    except OpenRouterClientError as exc:
        print(f"Warning: prompt {prompt_id}: {exc}", file=sys.stderr)
        append_audit_log(prompt_id, "api_error", "blocked")
        return {
            "id": prompt_id,
            "original_prompt": prompt,
            "response": API_ERROR_RESPONSE,
        }

    allowed, rule_id = run_post_filter(model_output)
    if not allowed:
        append_audit_log(prompt_id, rule_id, "blocked")
        return {
            "id": prompt_id,
            "original_prompt": prompt,
            "response": SAFETY_REFUSAL,
        }

    append_audit_log(prompt_id, None, "allowed")
    return {
        "id": prompt_id,
        "original_prompt": prompt,
        "response": model_output,
    }


def run_pipeline() -> List[Dict[str, str]]:
    """Run the full non-interactive batch safety pipeline."""
    load_dotenv(BASE_DIR / ".env")
    prompt_batch = load_prompt_batch()

    AUDIT_PATH.write_text("", encoding="utf-8")

    results = [process_prompt(item) for item in prompt_batch]
    write_results(results)
    return results


def main() -> int:
    """Run the pipeline and return a process exit code."""
    try:
        results = run_pipeline()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Processed {len(results)} prompt(s). "
        f"Results: {RESULTS_PATH.name}; Audit: {AUDIT_PATH.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())