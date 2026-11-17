"""Input guardrail: Validates length, detects prompt-injections/jailbreaks, and masks PII."""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Max permitted input length
MAX_INPUT_LENGTH = 1000

# Regex patterns for adversarial injections, jailbreaks, and harmful instructions
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"reveal\s+(?:your\s+|the\s+)?(?:system\s+)?prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a|an)?", re.IGNORECASE),
    re.compile(r"disregard\s+(?:all\s+)?prior\s+instructions", re.IGNORECASE),
    re.compile(r"system\s+override", re.IGNORECASE),
    re.compile(r"bypass\s+safety", re.IGNORECASE),
    re.compile(r"print\s+(?:the\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"drop\s+table", re.IGNORECASE),
    re.compile(r"delete\s+all\s+(?:orders|customers|products)", re.IGNORECASE),
    re.compile(r"truncate\s+table", re.IGNORECASE),
    re.compile(r"show\s+database\s+passwords?", re.IGNORECASE),
    re.compile(r"exfiltrate", re.IGNORECASE),
    re.compile(r";\s*drop\s+", re.IGNORECASE),
    re.compile(r";\s*delete\s+", re.IGNORECASE),
]

# PII regex patterns for data masking before logging or memory storage
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
CARD_REGEX = re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")


@dataclass
class InputGuardResult:
    is_safe: bool
    error_message: str | None
    masked_text: str


def mask_pii(text: str) -> str:
    """Masks email addresses, phone numbers, and card-like numbers in text."""
    masked = EMAIL_REGEX.sub("[EMAIL_MASKED]", text)
    masked = PHONE_REGEX.sub("[PHONE_MASKED]", masked)
    masked = CARD_REGEX.sub("[CARD_MASKED]", masked)
    return masked


def validate_input(user_input: str) -> InputGuardResult:
    """Validates user input against length constraints, emptiness, and prompt injection patterns.

    Returns InputGuardResult with is_safe, optional friendly rejection error, and PII-masked text.
    """
    if not user_input or not user_input.strip():
        return InputGuardResult(
            is_safe=False,
            error_message="Please enter a valid business question.",
            masked_text="",
        )

    text = user_input.strip()

    if len(text) > MAX_INPUT_LENGTH:
        return InputGuardResult(
            is_safe=False,
            error_message=f"Question is too long (exceeds {MAX_INPUT_LENGTH} characters). Please shorten your question.",
            masked_text=text[:MAX_INPUT_LENGTH],
        )

    # Check for prompt injection / jailbreak patterns
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning("Input guard blocked potential injection matching '%s'", pattern.pattern)
            return InputGuardResult(
                is_safe=False,
                error_message="I cannot process this request because it contains forbidden instructions or attempts to bypass security policies.",
                masked_text=mask_pii(text),
            )

    # Clean and mask any user-supplied PII for memory and logs
    sanitized = mask_pii(text)

    return InputGuardResult(
        is_safe=True,
        error_message=None,
        masked_text=sanitized,
    )
