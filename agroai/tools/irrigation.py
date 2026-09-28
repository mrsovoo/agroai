"""Irrigation tool: water requirement, volume and drip runtime arithmetic.

Deterministic arithmetic on values the user supplies:

* ``crop_water_requirement`` - ``ETc = ET0 x Kc``, converted to m3 for an area;
* ``irrigation_volume`` - depth (mm) over an area (ha) to cubic metres;
* ``drip_runtime`` - how long a drip system must run to deliver that depth.

The tool never invents an ET0 value (that is weather data) and never invents a
crop coefficient Kc (that is crop-specific and must come from the knowledge base
with a source). Both are required inputs, and the result says where they came
from - or that the user supplied them.
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

REFERENCE_ETC = "ETc = ET0 x Kc (FAO-56 crop evapotranspiration method)"
REFERENCE_VOLUME = "1 mm of water over 1 ha = 10 m3 (definition)"
REFERENCE_DRIP = (
    "Drip application rate: 1 L/m2 = 1 mm, so mm/h = emitter flow / "
    "(emitter spacing x row spacing)"
)

INPUT_NOTE = (
    "ET0 (ob-havo ma'lumoti) va Kc (ekinga xos) tashqaridan berilishi shart - "
    "bu vosita ularni o'ylab topmaydi. / ET0 and Kc must be supplied; this tool "
    "does not invent weather data or crop coefficients."
)


class IrrigationTool(Tool):
    """Irrigation calculator: crop water requirement, volume, drip runtime."""

    spec = ToolSpec(
        name="irrigation_tool",
        description=(
            "Irrigation calculations: action='crop_water_requirement' "
            "(et0_mm, kc, area_ha), action='irrigation_volume' (depth_mm, area_ha), "
            "action='drip_runtime' (depth_mm, emitter_flow_lph, "
            "emitter_spacing_m, row_spacing_m)."
        ),
        category="irrigation",
        parameters=(
            ToolParameter(
                "action",
                CHOICE,
                "One of: crop_water_requirement, irrigation_volume, drip_runtime.",
                choices=("crop_water_requirement", "irrigation_volume", "drip_runtime"),
            ),
            ToolParameter("et0_mm", NUMBER, "Reference ET0 in mm/day.", required=False),
            ToolParameter("kc", NUMBER, "Crop coefficient Kc (unitless).", required=False),
            ToolParameter("area_ha", NUMBER, "Area in hectares.", required=False),
            ToolParameter("depth_mm", NUMBER, "Water depth in mm.", required=False),
            ToolParameter(
                "emitter_flow_lph", NUMBER, "Drip emitter flow in L/hour.", required=False
            ),
            ToolParameter(
                "emitter_spacing_m", NUMBER, "Distance between emitters (m).", required=False
            ),
            ToolParameter(
                "row_spacing_m", NUMBER, "Distance between drip rows (m).", required=False
            ),
        ),
        reference=REFERENCE_ETC,
    )

    def run(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "")).strip().lower()
        if action == "crop_water_requirement":
            return self._water_requirement(kwargs)
        if action == "irrigation_volume":
            return self._volume(kwargs)
        if action == "drip_runtime":
            return self._drip_runtime(kwargs)
        return self.fail(
            f"unknown action {action!r}; use crop_water_requirement, "
            "irrigation_volume or drip_runtime"
        )

    # ------------------------------------------------------------------ #
    def _water_requirement(self, kwargs: dict[str, Any]) -> ToolResult:
        et0 = self.optional_number(kwargs, "et0_mm")
        kc = self.optional_number(kwargs, "kc")
        area = self.optional_number(kwargs, "area_ha")
        if et0 is None or kc is None:
            return self.fail("et0_mm and kc are required for action='crop_water_requirement'")
        if et0 < 0:
            return self.fail("et0_mm must not be negative")
        if kc <= 0 or kc > 3:
            return self.fail("kc must be between 0 and 3 (a typical range)")

        etc_mm = et0 * kc
        data: dict[str, Any] = {"et0_mm": et0, "kc": kc, "etc_mm": round(etc_mm, 2)}
        output = f"ETc = {et0:g} x {kc:g} = {etc_mm:.2f} mm/kun (mm/day)"
        if area is not None:
            if area < 0:
                return self.fail("area_ha must not be negative")
            cubic_metres = etc_mm * area * 10
            data.update({"area_ha": area, "volume_m3_per_day": round(cubic_metres, 2)})
            output += f"; {area:g} ga uchun {cubic_metres:.1f} m3/kun"

        return ToolResult(
            tool=self.name,
            ok=True,
            output=output,
            data=data,
            reference=REFERENCE_ETC,
            notes=[
                INPUT_NOTE,
                "Bu sof hisob-kitob: yomg'ir, tuproq namligi va sug'orish "
                "samaradorligi qo'shilmagan. / Pure arithmetic: rainfall, soil "
                "moisture and application efficiency are not included.",
            ],
        )

    def _volume(self, kwargs: dict[str, Any]) -> ToolResult:
        depth = self.optional_number(kwargs, "depth_mm")
        area = self.optional_number(kwargs, "area_ha")
        if depth is None or area is None:
            return self.fail("depth_mm and area_ha are required for action='irrigation_volume'")
        if depth < 0 or area < 0:
            return self.fail("depth_mm and area_ha must not be negative")

        cubic_metres = depth * area * 10
        litres = cubic_metres * 1000
        return ToolResult(
            tool=self.name,
            ok=True,
            output=(
                f"{depth:g} mm suv / {area:g} ga = {cubic_metres:.1f} m3 "
                f"({litres:.0f} litr)"
            ),
            data={
                "depth_mm": depth,
                "area_ha": area,
                "volume_m3": round(cubic_metres, 2),
                "volume_litres": round(litres, 1),
            },
            reference=REFERENCE_VOLUME,
            notes=["Sug'orish samaradorligi (FIE) hisobga olinmagan. / Application "
                   "efficiency (FIE) is not included."],
        )

    def _drip_runtime(self, kwargs: dict[str, Any]) -> ToolResult:
        depth = self.optional_number(kwargs, "depth_mm")
        flow = self.optional_number(kwargs, "emitter_flow_lph")
        emitter_spacing = self.optional_number(kwargs, "emitter_spacing_m")
        row_spacing = self.optional_number(kwargs, "row_spacing_m")
        if depth is None or flow is None or emitter_spacing is None or row_spacing is None:
            return self.fail(
                "depth_mm, emitter_flow_lph, emitter_spacing_m and row_spacing_m "
                "are required for action='drip_runtime'"
            )
        if flow <= 0 or emitter_spacing <= 0 or row_spacing <= 0:
            return self.fail("emitter flow and spacings must be greater than zero")
        if depth < 0:
            return self.fail("depth_mm must not be negative")

        mm_per_hour = flow / (emitter_spacing * row_spacing)
        hours = depth / mm_per_hour if mm_per_hour else 0.0
        return ToolResult(
            tool=self.name,
            ok=True,
            output=(
                f"Application rate {mm_per_hour:.2f} mm/soat; {depth:g} mm berish uchun "
                f"{hours:.2f} soat kerak"
            ),
            data={
                "depth_mm": depth,
                "emitter_flow_lph": flow,
                "emitter_spacing_m": emitter_spacing,
                "row_spacing_m": row_spacing,
                "application_rate_mm_per_hour": round(mm_per_hour, 3),
                "runtime_hours": round(hours, 2),
            },
            reference=REFERENCE_DRIP,
            notes=[
                "Emitting uniformity, bosim yo'qolishi va filtrlar hisobga "
                "olinmagan. / Emission uniformity, pressure loss and filtration "
                "are not modelled.",
            ],
        )


__all__ = [
    "INPUT_NOTE",
    "REFERENCE_DRIP",
    "REFERENCE_ETC",
    "REFERENCE_VOLUME",
    "IrrigationTool",
]
