"""Project-wide GlueLLM SDK guardrails configuration."""

from __future__ import annotations

import os

from gluellm.guardrails.config import (
    BlocklistConfig,
    GuardrailsConfig,
    MaxLengthConfig,
    PIIConfig,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def default_guardrails_config() -> GuardrailsConfig:
    """Return SDK-native guardrails config used across all GlueLLM clients."""
    enabled = _env_bool("PROJECT_GUARDRAILS_ENABLED", True)

    patterns_raw = os.getenv("PROJECT_GUARDRAILS_BLOCKLIST_PATTERNS", "")
    patterns = [p.strip() for p in patterns_raw.split(",") if p.strip()]
    blocklist = BlocklistConfig(patterns=patterns) if patterns else None

    pii = PIIConfig(
        redact_emails=_env_bool("PROJECT_GUARDRAILS_REDACT_EMAILS", True),
        redact_phones=_env_bool("PROJECT_GUARDRAILS_REDACT_PHONES", True),
        redact_ssn=_env_bool("PROJECT_GUARDRAILS_REDACT_SSN", True),
        redact_credit_cards=_env_bool("PROJECT_GUARDRAILS_REDACT_CREDIT_CARDS", True),
    )

    max_length = MaxLengthConfig(
        max_input_length=_env_int("PROJECT_GUARDRAILS_MAX_INPUT_LENGTH", 12000),
        max_output_length=_env_int("PROJECT_GUARDRAILS_MAX_OUTPUT_LENGTH", 4000),
        strategy=os.getenv("PROJECT_GUARDRAILS_MAX_LENGTH_STRATEGY", "block"),
    )

    return GuardrailsConfig(
        enabled=enabled,
        blocklist=blocklist,
        pii=pii,
        max_length=max_length,
        max_output_guardrail_retries=_env_int("PROJECT_GUARDRAILS_MAX_RETRIES", 2),
    )
