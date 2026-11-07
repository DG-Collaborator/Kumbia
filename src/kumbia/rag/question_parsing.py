"""
   Deterministic extraction of query parameters from a free-text question —
   the time window it asks about and an IPv4 address it names. Shared by any
   skill whose fixed query needs a window or a host: the regex reads the
   user's words, the LLM never does, so a parameter can't be invented.
"""
from __future__ import annotations

import ipaddress
import re

MAX_HOURS_BACK = 720  # 30 days

_UNIT_HOURS = {
    "minute": 1 / 60, "min": 1 / 60, "m": 1 / 60,
    "hour": 1, "hr": 1, "h": 1,
    "day": 24, "d": 24,
    "week": 168, "wk": 168, "w": 168,
    "month": 720,
}

_NUMBERED_WINDOW = re.compile(
    r"\b(?:last|past|previous|over|in)\s+(?:the\s+)?(\d+)\s*"
    r"(minutes?|mins?|m|hours?|hrs?|h|days?|d|weeks?|wks?|w|months?)\b",
    re.IGNORECASE,
)
_COMPACT_WINDOW = re.compile(r"\b(\d+)\s*(h|hrs?|d|w)\b", re.IGNORECASE)
_SINGLE_WINDOW = re.compile(
    r"\b(?:last|past|previous|this)\s+(minute|hour|day|week|month)\b", re.IGNORECASE
)
_NAMED_WINDOWS = {"today": 24, "yesterday": 48, "overnight": 12, "tonight": 12}

_IPV4_IN_TEXT = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.]*\d)")


def hours_from_question(question: str, default: int) -> int:
    """
       The trailing window the question asks about, in whole hours (rounded
       up, clamped to 1..MAX_HOURS_BACK), or default when it names none.
    """
    match = _NUMBERED_WINDOW.search(question) or _COMPACT_WINDOW.search(question)
    if match:
        return _clamp(int(match.group(1)) * _unit_hours(match.group(2)))

    match = _SINGLE_WINDOW.search(question)
    if match:
        return _clamp(_unit_hours(match.group(1)))

    lowered = question.lower()
    for word, hours in _NAMED_WINDOWS.items():
        if re.search(rf"\b{word}\b", lowered):
            return hours
    return _clamp(default)


def ipv4_from_question(question: str) -> str | None:
    """The first valid IPv4 address the question names, or None."""
    for candidate in _IPV4_IN_TEXT.findall(question):
        try:
            ipaddress.IPv4Address(candidate)
        except ValueError:
            continue
        return candidate
    return None


def _unit_hours(unit: str) -> float:
    unit = unit.lower()
    if unit in _UNIT_HOURS:
        return _UNIT_HOURS[unit]
    return _UNIT_HOURS[unit.rstrip("s")]


def _clamp(hours: float) -> int:
    whole = int(hours) if hours == int(hours) else int(hours) + 1
    return max(1, min(whole, MAX_HOURS_BACK))
