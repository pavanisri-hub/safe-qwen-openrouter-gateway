"""Pre-generation safety checks for user prompts."""

import re
from typing import Optional, Tuple

JAILBREAK_PHRASE = "IGNORE ALL PREVIOUS INSTRUCTIONS"
SSN_PATTERN = re.compile(r"\d{3}-\d{2}-\d{4}")


def run_pre_filter(prompt: str) -> Tuple[bool, Optional[str]]:
    """
    Evaluate a prompt against deterministic pre-generation safety rules.

    Returns:
        (True, None) when the prompt is safe.
        (False, rule_id) when a safety rule is triggered.
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")

    if re.search(JAILBREAK_PHRASE, prompt, flags=re.IGNORECASE):
        return False, "rule_jailbreak"

    if SSN_PATTERN.search(prompt):
        return False, "rule_pii"

    return True, None