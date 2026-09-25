"""Post-generation safety checks for model responses."""

from typing import Optional, Tuple

CONFIDENTIAL_MARKER = "[CONFIDENTIAL]"
DANGEROUS_COMMAND = "sudo rm -rf"


def run_post_filter(output_text: str) -> Tuple[bool, Optional[str]]:
    """
    Evaluate model output against deterministic post-generation safety rules.

    Returns:
        (True, None) when the output is safe.
        (False, rule_id) when a safety rule is triggered.
    """
    if not isinstance(output_text, str):
        raise TypeError("output_text must be a string")

    if CONFIDENTIAL_MARKER in output_text:
        return False, "rule_confidential_leak"

    if DANGEROUS_COMMAND in output_text:
        return False, "rule_dangerous_command"

    return True, None