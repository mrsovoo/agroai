"""Weather tool: honest about the network, useful about the numbers.

Two distinct capabilities:

* ``spraying_conditions`` - checks a set of *user-supplied* readings against
  common spray-window limits. Pure arithmetic on your numbers, no network.
* ``current`` - live weather. AgroAI Core has no network provider, so this
  reports exactly that instead of inventing a forecast.

The product label always outranks general guidance: every result says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agroai.tools.base import (
    CHOICE,
    NUMBER,
    Tool,
    ToolParameter,
    ToolResult,
    ToolSpec,
)

REFERENCE_SPRAY = (
    "Umumiy purkash oynasi chegaralari (shamol, harorat, namlik, yomg'ir) - "
    "kengaytirish xizmatlarining odatiy tavsiyalari"
)

LABEL_NOTE = (
    "Mahsulot yorlig'i (label) har doim ustuvor: yorliqdagi chegaralar va "
    "dozaga rioya qiling. / The product label always outranks general guidance."
)

NETWORK_DETAIL = (
    "Live weather is unavailable: AgroAI Core has no weather provider "
    "configured and deliberately makes no network calls. Supply the readings "
    "yourself (temperature, humidity, wind, rain forecast) to use "
    "action='spraying_conditions'."
)


@dataclass(frozen=True)
class SprayWindow:
    """Outcome of one spraying-condition check."""

    temperature: str
    humidity: str
    wind: str
    rain: str
    verdict: str

    def lines(self) -> list[str]:
        return [
            f"harorat/temperature: {self.temperature}",
            f"namlik/humidity: {self.humidity}",
            f"shamol/wind: {self.wind}",
            f"yomg'ir/rain: {self.rain}",
            f"Xulosa/verdict: {self.verdict}",
        ]


def spraying_window(
    temperature_c: float,
    humidity_pct: float,
    wind_kmh: float,
    rain_expected: bool,
) -> SprayWindow:
    """Evaluate a spray window from supplied readings (no network, no guessing)."""
    if 15 <= temperature_c <= 30:
        temperature = "qulay / suitable"
    elif 10 <= temperature_c <= 32:
        temperature = "chegarada / marginal"
    else:
        temperature = "purkash uchun yaroqsiz (juda sovuq yoki juda issiq) / unsuitable"

    if 40 <= humidity_pct <= 90:
        humidity = "qulay / suitable"
    elif humidity_pct < 40:
        humidity = "juda quruq (bug'lanish/tomchi oqimi xavfi) / too dry"
    else:
        humidity = "juda nam (sekin qurish, kasallik xavfi) / too humid"

    if wind_kmh <= 8:
        wind = "qulay / suitable"
    elif wind_kmh <= 15:
        wind = "chegarada / marginal"
    else:
        wind = "juda kuchli shamol / too windy (drift risk)"

    rain = (
        "yomg'ir kutilmoqda / rain expected"
        if rain_expected
        else "yomg'ir kutilmaydi / no rain expected"
    )

    blocking = (
        humidity.startswith("juda quruq")
        or humidity.startswith("juda nam")
        or wind.startswith("juda kuchli")
        or temperature.startswith("purkash uchun yaroqsiz")
        or rain_expected
    )
    marginal = "chegarada" in temperature or "chegarada" in wind
    if blocking:
        verdict = "purkash tavsiya etilmaydi / do not spray now"
    elif marginal:
        verdict = "ehtiyotkorlik bilan / spray with caution"
    else:
        verdict = "sharoit qulay / conditions look suitable"

    return SprayWindow(
        temperature=temperature,
        humidity=humidity,
        wind=wind,
        rain=rain,
        verdict=verdict,
    )


class WeatherTool(Tool):
    """Spray-window checking from supplied readings, plus honest live weather."""

    spec = ToolSpec(
        name="weather_tool",
        description=(
            "Weather helper: action='spraying_conditions' (temperature_c, "
            "humidity_pct, wind_kmh, rain_expected) evaluates whether spraying is "
            "advisable; action='current' needs live weather and is unavailable."
        ),
        category="weather",
        parameters=(
            ToolParameter(
                "action",
                CHOICE,
                "One of: spraying_conditions, current.",
                choices=("spraying_conditions", "current"),
            ),
            ToolParameter("temperature_c", NUMBER, "Air temperature in Celsius.", required=False),
            ToolParameter("humidity_pct", NUMBER, "Relative humidity in percent.", required=False),
            ToolParameter("wind_kmh", NUMBER, "Wind speed in km/h.", required=False),
            ToolParameter(
                "rain_expected",
                CHOICE,
                "yes or no - is rain forecast for the next hours?",
                required=False,
                choices=("yes", "no"),
            ),
        ),
        reference=REFERENCE_SPRAY,
        requires_network=False,
    )

    def run(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "")).strip().lower()
        if action == "spraying_conditions":
            return self._spraying_conditions(kwargs)
        if action == "current":
            return self._current()
        return self.fail(f"unknown action {action!r}; use spraying_conditions or current")

    # ------------------------------------------------------------------ #
    def _spraying_conditions(self, kwargs: dict[str, Any]) -> ToolResult:
        temperature = self.optional_number(kwargs, "temperature_c")
        humidity = self.optional_number(kwargs, "humidity_pct")
        wind = self.optional_number(kwargs, "wind_kmh")
        rain_raw = str(kwargs.get("rain_expected", "")).strip().lower()

        missing = [
            name
            for name, value in (
                ("temperature_c", temperature),
                ("humidity_pct", humidity),
                ("wind_kmh", wind),
            )
            if value is None
        ]
        if missing:
            return self.fail(
                "spraying_conditions needs: " + ", ".join(missing)
                + " (readings you measured - this tool does not fetch weather)"
            )
        if rain_raw not in {"yes", "no"}:
            return self.fail("rain_expected must be 'yes' or 'no'")
        if not 0 <= humidity <= 100:
            return self.fail("humidity_pct must be between 0 and 100")

        window = spraying_window(temperature, humidity, wind, rain_raw == "yes")
        return ToolResult(
            tool=self.name,
            ok=True,
            output="\n".join(window.lines()),
            data={
                "temperature_c": temperature,
                "humidity_pct": humidity,
                "wind_kmh": wind,
                "rain_expected": rain_raw == "yes",
                "verdict": window.verdict,
            },
            reference=REFERENCE_SPRAY,
            notes=[
                LABEL_NOTE,
                "Bu tekshiruv faqat siz bergan o'lchovlarga asoslanadi. / The check "
                "uses only the readings you supplied.",
            ],
        )

    def _current(self) -> ToolResult:
        return self.fail(
            NETWORK_DETAIL,
            data={"action": "current", "requires_network": True},
            reference="",
        )


__all__ = [
    "LABEL_NOTE",
    "NETWORK_DETAIL",
    "REFERENCE_SPRAY",
    "SprayWindow",
    "WeatherTool",
    "spraying_window",
]
