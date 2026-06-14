"""Pricing freshness guard.

Cost tracking is only as accurate as the price table. This test fails once the
catalogue is older than PRICING_MAX_AGE_DAYS so stale prices surface in CI
instead of silently misreporting spend. When it fails: re-verify every price in
evaluation/pricing.py against the providers' pricing pages, then bump
PRICING_AS_OF.
"""

from __future__ import annotations

from datetime import date

from evaluation.pricing import (
    PRICING_AS_OF,
    PRICING_MAX_AGE_DAYS,
    pricing_age_days,
)


def test_pricing_as_of_is_valid_iso_date():
    # Raises ValueError if malformed.
    parsed = date.fromisoformat(PRICING_AS_OF)
    assert parsed <= date.today(), "PRICING_AS_OF is in the future"


def test_pricing_not_stale():
    age = pricing_age_days()
    assert age <= PRICING_MAX_AGE_DAYS, (
        f"Pricing catalogue is {age} days old (max {PRICING_MAX_AGE_DAYS}). "
        f"Re-verify every price in evaluation/pricing.py against the providers' "
        f"pricing pages, then bump PRICING_AS_OF (currently {PRICING_AS_OF})."
    )
