"""Pearls AQI Predictor - Lahore Season Classifier.

Defines the project-level mapping from calendar month to Lahore atmospheric season.
This mapping is treated as a fixed project definition (not a scientific fact) and is
the single authoritative source for seasonal labels used in Phase 11 diagnostics.

Season Mapping (Lahore Climate):
    winter_smog : Nov, Dec, Jan, Feb — temperature inversions, crop burning, extreme PM2.5
    transition  : Mar, Apr, Oct      — rapid regime changes, mixed AQI behavior
    summer      : May, Jun           — rising ozone, heat-driven photochemistry
    monsoon     : Jul, Aug, Sep      — wet deposition suppresses PM2.5, high humidity
"""

from __future__ import annotations


# Single authoritative season mapping — documented here, never duplicated elsewhere.
LAHORE_MONTH_TO_SEASON: dict[int, str] = {
    1: "winter_smog",   # January
    2: "winter_smog",   # February
    3: "transition",    # March
    4: "transition",    # April
    5: "summer",        # May
    6: "summer",        # June
    7: "monsoon",       # July
    8: "monsoon",       # August
    9: "monsoon",       # September
    10: "transition",   # October
    11: "winter_smog",  # November
    12: "winter_smog",  # December
}

VALID_SEASONS = frozenset(LAHORE_MONTH_TO_SEASON.values())


class LahoreSeasonClassifier:
    """Maps calendar month integers to Lahore atmospheric season labels.

    The mapping is defined in LAHORE_MONTH_TO_SEASON and is the single
    project-level definition for all Phase 11 seasonal diagnostics.
    """

    def classify(self, month: int) -> str:
        """Return the season label for a given calendar month (1–12).

        Args:
            month: Integer month in range [1, 12].

        Returns:
            Season label: one of 'winter_smog', 'transition', 'summer', 'monsoon'.

        Raises:
            ValueError: If month is outside [1, 12].
        """
        if month not in LAHORE_MONTH_TO_SEASON:
            raise ValueError(f"Month must be in [1, 12], got {month!r}")
        return LAHORE_MONTH_TO_SEASON[month]

    def classify_series(self, months: list[int]) -> list[str]:
        """Classify a list of month integers into season labels.

        Args:
            months: List of integer months.

        Returns:
            List of season label strings in the same order.
        """
        return [self.classify(m) for m in months]
