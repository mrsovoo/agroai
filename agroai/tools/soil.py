"""Soil tool: classification and drainage/leaching arithmetic.

Three deterministic operations, each based on a published classification or
formula:

* salinity class from electrical conductivity (ECe, dS/m),
* pH class from a 1:2.5 (or 1:1/water) measurement,
* leaching requirement from irrigation-water and target-soil salinity.

What this tool deliberately does **not** do is tell you which crop to grow or
what dose to apply: crop-specific tolerance tables and fertilizer rates belong in
the knowledge base, where they carry a real source. Every result says so.
"""

from __future__ import annotations

from typing import Any

from agroai.tools.base import (
    NUMBER,
    CHOICE,
    Tool,
    ToolParameter,
    ToolResult,
    ToolSpec,
)

#: Standard ECe salinity classes (dS/m) used by FAO/USDA soil descriptions.
SALINITY_CLASSES: tuple[tuple[float, str, str], ...] = (
    (2.0, "non-saline", "sho'rlanmagan"),
    (4.0, "slightly saline", "kuchsiz sho'rlangan"),
    (8.0, "moderately saline", "o'rtacha sho'rlangan"),
    (16.0, "strongly saline", "kuchli sho'rlangan"),
    (float("inf"), "very strongly saline", "juda kuchli sho'rlangan"),
)

#: pH classes (1:1 or 1:2.5 water suspension).
PH_CLASSES: tuple[tuple[float, str, str], ...] = (
    (4.5, "extremely acidic", "juda kislotali"),
    (5.5, "strongly acidic", "kuchli kislotali"),
    (6.5, "moderately acidic", "o'rtacha kislotali"),
    (7.3, "neutral", "neytral"),
    (8.5, "moderately alkaline", "o'rtacha ishqoriy"),
    (9.5, "strongly alkaline", "kuchli ishqoriy"),
    (float("inf"), "very strongly alkaline", "juda kuchli ishqoriy"),
)

SALINITY_REFERENCE = "FAO/USDA soil salinity classes (ECe, dS/m)"
PH_REFERENCE = "Standard soil pH classes (1:1 / 1:2.5 water suspension)"
LEACHING_REFERENCE = "FAO leaching requirement: LR = ECw / (5 * ECe - ECw)"

VERIFY_NOTE = (
    "Klassifikatsiya standart chegaralarga asoslangan; yakuniy agrotexnik "
    "xulosa uchun mahalliy tuproq tahlili va agronom xulosasini tekshiring. "
    "/ Classification uses standard thresholds; confirm with a local soil test."
)


def classify_salinity(ec_ds_m: float) -> tuple[str, str]:
    """Return ``(english_class, uzbek_class)`` for ECe in dS/m."""
    for threshold, english, uzbek in SALINITY_CLASSES:
        if ec_ds_m < threshold:
            return english, uzbek
    return SALINITY_CLASSES[-1][1], SALINITY_CLASSES[-1][2]


def classify_ph(ph: float) -> tuple[str, str]:
    """Return ``(english_class, uzbek_class)`` for a soil pH value."""
    for threshold, english, uzbek in PH_CLASSES:
        if ph < threshold:
            return english, uzbek
    return PH_CLASSES[-1][1], PH_CLASSES[-1][2]


def leaching_requirement(ec_water: float, ec_target: float) -> float:
    """FAO leaching requirement as a fraction of applied water.

    ``LR = ECw / (5 * ECe - ECw)``. Raises :class:`ValueError` when the inputs
    make the formula meaningless (denominator at or below zero), because
    returning a number there would be misleading.
    """
    denominator = 5 * ec_target - ec_water
    if denominator <= 0:
        raise ValueError(
            "leaching requirement is undefined here: 5 * ECe must exceed ECw "
            "(check your irrigation-water and target-soil salinity values)"
        )
    return ec_water / denominator


class SoilTool(Tool):
    """Classify soil salinity/pH and compute the leaching requirement."""

    spec = ToolSpec(
        name="soil_tool",
        description=(
            "Soil analysis helper: action='salinity' (ECe in dS/m), "
            "action='ph', or action='leaching_requirement' "
            "(irrigation-water ECw and target soil ECe)."
        ),
        category="soil",
        parameters=(
            ToolParameter(
                "action",
                CHOICE,
                "One of: salinity, ph, leaching_requirement.",
                choices=("salinity", "ph", "leaching_requirement"),
            ),
            ToolParameter(
                "ec_ds_m",
                NUMBER,
                "Electrical conductivity (ECe) in dS/m - for 'salinity'.",
                required=False,
            ),
            ToolParameter("ph", NUMBER, "Soil pH - for 'ph'.", required=False),
            ToolParameter(
                "ec_water",
                NUMBER,
                "Irrigation water EC (ECw, dS/m) - for 'leaching_requirement'.",
                required=False,
            ),
            ToolParameter(
                "ec_target",
                NUMBER,
                "Target soil EC (ECe, dS/m) - for 'leaching_requirement'.",
                required=False,
            ),
        ),
        reference=SALINITY_REFERENCE,
    )

    def run(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "")).strip().lower()
        if action == "salinity":
            return self._salinity(kwargs)
        if action == "ph":
            return self._ph(kwargs)
        if action == "leaching_requirement":
            return self._leaching(kwargs)
        return self.fail(f"unknown action {action!r}; use salinity, ph or leaching_requirement")

    # ------------------------------------------------------------------ #
    def _salinity(self, kwargs: dict[str, Any]) -> ToolResult:
        ec = self.optional_number(kwargs, "ec_ds_m")
        if ec is None:
            return self.fail("ec_ds_m is required for action='salinity'")
        if ec < 0:
            return self.fail("ec_ds_m must not be negative")
        english, uzbek = classify_salinity(ec)
        return ToolResult(
            tool=self.name,
            ok=True,
            output=f"ECe {ec:g} dS/m -> {english} ({uzbek})",
            data={"ec_ds_m": ec, "class": english, "class_uz": uzbek},
            reference=SALINITY_REFERENCE,
            notes=[
                VERIFY_NOTE,
                "Ekinlarning sho'rlikka chidamliligi ekinga bog'liq - bilimlar "
                "bazasidan manba bilan tekshiring. / Crop salinity tolerance is "
                "crop-specific: take those thresholds from the knowledge base.",
            ],
        )

    def _ph(self, kwargs: dict[str, Any]) -> ToolResult:
        ph = self.optional_number(kwargs, "ph")
        if ph is None:
            return self.fail("ph is required for action='ph'")
        if not 0 <= ph <= 14:
            return self.fail("ph must be between 0 and 14")
        english, uzbek = classify_ph(ph)
        notes = [VERIFY_NOTE]
        if english.endswith("alkaline"):
            notes.append(
                "Ishqoriy tuproqda o'g'it tanlash va mikroelementlarning "
                "mavjudligi o'zgaradi. / Alkaline soils change nutrient "
                "availability; check crop-specific guidance in the knowledge base."
            )
        return ToolResult(
            tool=self.name,
            ok=True,
            output=f"pH {ph:g} -> {english} ({uzbek})",
            data={"ph": ph, "class": english, "class_uz": uzbek},
            reference=PH_REFERENCE,
            notes=notes,
        )

    def _leaching(self, kwargs: dict[str, Any]) -> ToolResult:
        ec_water = self.optional_number(kwargs, "ec_water")
        ec_target = self.optional_number(kwargs, "ec_target")
        if ec_water is None or ec_target is None:
            return self.fail(
                "ec_water and ec_target are required for action='leaching_requirement'"
            )
        try:
            fraction = leaching_requirement(ec_water, ec_target)
        except ValueError as exc:
            return self.fail(str(exc))
        percent = round(fraction * 100, 1)
        return ToolResult(
            tool=self.name,
            ok=True,
            output=(
                f"Leaching requirement LR = {fraction:.4f} "
                f"({percent}% qo'shimcha suv / extra water)"
            ),
            data={
                "ec_water": ec_water,
                "ec_target": ec_target,
                "leaching_requirement": fraction,
                "extra_water_percent": percent,
            },
            reference=LEACHING_REFERENCE,
            notes=[
                VERIFY_NOTE,
                "Hisob suv va tuproq sho'rligi o'lchovlariga tayanadi; drenaj "
                "holati va ekin turi hisobga olinmagan. / The figure depends on "
                "your measurements; drainage conditions and crop are not modelled.",
            ],
        )


__all__ = [
    "LEACHING_REFERENCE",
    "PH_CLASSES",
    "PH_REFERENCE",
    "SALINITY_CLASSES",
    "SALINITY_REFERENCE",
    "VERIFY_NOTE",
    "SoilTool",
    "classify_ph",
    "classify_salinity",
    "leaching_requirement",
]
