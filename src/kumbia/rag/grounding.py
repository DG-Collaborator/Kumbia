"""
   Deterministic hallucination guard, shared by every persona: finds the
   figures (numbers, percentages, IPv4 addresses) an LLM-written answer states
   that appear nowhere in the evidence it was given. Pure text work — no LLM,
   no persona knowledge — so any pipeline can check its own narration against
   whatever figures, rows, or remembered facts grounded it.

   Matching is deliberately tolerant where wording legitimately differs from
   the evidence (thousands separators, a sign dropped in prose, rounding to
   fewer decimals, an hour taken from a timestamp) and strict where it
   matters: an IP must appear verbatim, and a number must be in the evidence
   (or round from one in it) to count as supported.
"""
from __future__ import annotations

import re

_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.]*\d)")
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?")

# Bare integers this small are almost always counting words, list markers, or
# ordinals in prose ("the top 3", "1."), not data figures worth policing.
_SMALL_INTEGER_LIMIT = 10


def unsupported_figures(answer: str, *evidence: str) -> list[str]:
    """
       The figures in answer that no piece of evidence supports, in the order
       they appear (deduplicated). An empty list means every stated figure
       traces back to the evidence.
    """
    evidence_text = "\n".join(evidence)
    evidence_ips = set(_IPV4.findall(evidence_text))
    evidence_numbers = _numbers_in(_IPV4.sub(" ", evidence_text)) | _numbers_in(evidence_text)

    unsupported: list[str] = []
    for ip in _IPV4.findall(answer):
        if ip not in evidence_ips and ip not in unsupported:
            unsupported.append(ip)

    for raw in _NUMBER.findall(_IPV4.sub(" ", answer)):
        token = raw.rstrip(",")
        value = _to_float(token.rstrip("%"))
        if value is None:
            continue
        if "." not in token and not token.endswith("%") and value <= _SMALL_INTEGER_LIMIT:
            continue
        if not _is_supported(token, value, evidence_numbers) and token not in unsupported:
            unsupported.append(token)
    return unsupported


def withhold_unsupported(narration: str, figures: str, question: str) -> str:
    """
       Returns narration unchanged when every figure it states is in figures
       (or the question); otherwise replaces it with a note naming what was
       withheld, so only the deterministic figures are shown. Public so any
       persona pipeline that narrates computed figures can apply the same rule.
    """
    unsupported = unsupported_figures(narration, figures, question)
    if not unsupported:
        return narration
    return (
        "(The model's narration was withheld because it cited figures not in the data: "
        f"{', '.join(unsupported)}. The figures above are exact.)"
    )


def _numbers_in(text: str) -> set[float]:
    """Every number in text, plus its parts: a timestamp's year, month, hour, etc. each count on their own."""
    values: set[float] = set()
    for raw in _NUMBER.findall(text):
        value = _to_float(raw.rstrip(",%"))
        if value is not None:
            values.add(abs(value))
    for raw in re.findall(r"\d+(?:\.\d+)?", text):
        values.add(float(raw))
    return values


def _is_supported(token: str, value: float, evidence_numbers: set[float]) -> bool:
    """Exact match, or the evidence holds a number that rounds to this one at the answer's own precision."""
    digits = token.rstrip("%")
    decimals = len(digits.split(".", 1)[1]) if "." in digits else 0
    for candidate in evidence_numbers:
        if candidate == value or round(candidate, decimals) == value:
            return True
        if decimals == 0 and abs(candidate - value) < 1:  # "about 520" for 520.5
            return True
    return False


def _to_float(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None
