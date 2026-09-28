"""Tests for the tools: registry, validation and every implemented tool."""

from __future__ import annotations

import pytest

from agroai.tools import (
    CalculatorTool,
    IrrigationTool,
    SoilTool,
    ToolRegistry,
    UnitConverterTool,
    VeterinaryTool,
    WeatherTool,
    get_default_registry,
)
from agroai.tools.base import ToolError
from agroai.tools.calculator import CalculatorError, evaluate_expression
from agroai.tools.soil import classify_ph, classify_salinity, leaching_requirement
from agroai.tools.veterinary import ALWAYS_OBSERVE, structure_case
from agroai.tools.weather import spraying_window


# --------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------- #
def test_default_registry_is_populated_and_described() -> None:
    registry = get_default_registry()
    assert set(registry.names()) == {
        "calculator",
        "unit_converter",
        "soil_tool",
        "irrigation_tool",
        "weather_tool",
        "veterinary_tool",
    }
    described = registry.describe()
    assert "calculator" in described
    assert "soil_tool" in described
    spec = registry.get("soil_tool").spec
    assert spec.category == "soil"
    assert "action" in spec.parameter_names()


def test_registry_rejects_duplicates_and_reports_unknown_tools() -> None:
    registry = ToolRegistry([CalculatorTool()])
    with pytest.raises(ToolError):
        registry.register(CalculatorTool())

    result = registry.run("does_not_exist", x=1)
    assert result.ok is False
    assert "Unknown tool" in result.error


def test_tool_call_validates_required_parameters() -> None:
    result = get_default_registry().run("unit_converter", value=1)
    assert result.ok is False
    assert "from_unit" in result.error
    assert "to_unit" in result.error


def test_tool_spec_to_dict_is_json_safe() -> None:
    spec = get_default_registry().get("irrigation_tool").spec
    data = spec.to_dict()
    assert data["name"] == "irrigation_tool"
    assert isinstance(data["parameters"], list)


# --------------------------------------------------------------------- #
# Calculator
# --------------------------------------------------------------------- #
def test_calculator_evaluates_expressions() -> None:
    result = CalculatorTool().call(expression="12 * 3.5")
    assert result.ok is True
    assert result.data["result"] == 42.0

    assert evaluate_expression("sqrt(400)") == 20.0
    assert evaluate_expression("(2 + 3) * 4 - min(1, 2)") == 19.0


@pytest.mark.parametrize(
    "expression",
    ["1 / 0", "import os", "__import__('os')", "2 ** 999", "", "unknown(3)", "1 + 'a'"],
)
def test_calculator_refuses_unsafe_or_invalid_input(expression: str) -> None:
    result = CalculatorTool().call(expression=expression)
    assert result.ok is False
    assert result.error


def test_evaluate_expression_reports_long_input() -> None:
    with pytest.raises(CalculatorError):
        evaluate_expression("1 + " * 200 + "1")


# --------------------------------------------------------------------- #
# Unit converter
# --------------------------------------------------------------------- #
def test_unit_converter_area_and_mass() -> None:
    tool = UnitConverterTool()
    assert tool.call(value=1, from_unit="ga", to_unit="m2").data["result"] == 10_000.0
    assert tool.call(value=2, from_unit="t", to_unit="kg").data["result"] == 2000.0
    # 10 000 ppm is 1 % - a ratio conversion, not a length conversion.
    ratio = tool.call(value=10_000, from_unit="ppm", to_unit="%")
    assert ratio.data["result"] == 1.0
    assert ratio.data["dimension"] == "concentration"


def test_unit_converter_rejects_cross_dimension_and_unknown_units() -> None:
    tool = UnitConverterTool()
    cross = tool.call(value=1, from_unit="ga", to_unit="kg")
    assert cross.ok is False
    assert "cannot convert" in cross.error

    unknown = tool.call(value=1, from_unit="ga", to_unit="parsec")
    assert unknown.ok is False
    assert "unknown unit" in unknown.error


# --------------------------------------------------------------------- #
# Soil
# --------------------------------------------------------------------- #
def test_soil_salinity_and_ph_classification() -> None:
    assert classify_salinity(1.0)[0] == "non-saline"
    assert classify_salinity(6.5)[0] == "moderately saline"
    assert classify_salinity(30.0)[0] == "very strongly saline"
    assert classify_ph(4.0)[0] == "extremely acidic"
    assert classify_ph(7.0)[0] == "neutral"
    assert classify_ph(9.0)[0] == "strongly alkaline"


def test_soil_tool_actions() -> None:
    tool = SoilTool()
    salinity = tool.call(action="salinity", ec_ds_m=6.5)
    assert salinity.ok is True
    assert salinity.data["class"] == "moderately saline"
    assert salinity.reference
    assert salinity.notes

    ph = tool.call(action="ph", ph=8.0)
    assert ph.ok is True
    assert ph.data["class"] == "moderately alkaline"

    leaching = tool.call(action="leaching_requirement", ec_water=1.0, ec_target=4.0)
    assert leaching.ok is True
    assert round(leaching.data["leaching_requirement"], 4) == 0.0526


def test_soil_tool_validation_and_edge_cases() -> None:
    tool = SoilTool()
    assert tool.call(action="salinity", ec_ds_m=-1).ok is False
    assert tool.call(action="ph", ph=20).ok is False
    assert tool.call(action="nope").ok is False
    assert tool.call(action="salinity").ok is False

    impossible = tool.call(action="leaching_requirement", ec_water=10.0, ec_target=1.0)
    assert impossible.ok is False
    assert "undefined" in impossible.error

    with pytest.raises(ValueError):
        leaching_requirement(10.0, 1.0)


# --------------------------------------------------------------------- #
# Irrigation
# --------------------------------------------------------------------- #
def test_irrigation_crop_water_requirement() -> None:
    result = IrrigationTool().call(
        action="crop_water_requirement", et0_mm=5, kc=1.2, area_ha=1
    )
    assert result.ok is True
    assert result.data["etc_mm"] == 6.0
    assert result.data["volume_m3_per_day"] == 60.0
    # ET0 and Kc are inputs, never invented by the tool.
    assert "ET0" in " ".join(result.notes)


def test_irrigation_volume_and_drip_runtime() -> None:
    tool = IrrigationTool()
    volume = tool.call(action="irrigation_volume", depth_mm=10, area_ha=1)
    assert volume.data["volume_m3"] == 100.0

    drip = tool.call(
        action="drip_runtime",
        depth_mm=6,
        emitter_flow_lph=2,
        emitter_spacing_m=0.3,
        row_spacing_m=1.0,
    )
    assert drip.ok is True
    assert drip.data["application_rate_mm_per_hour"] == pytest.approx(6.667, abs=0.001)
    assert drip.data["runtime_hours"] == pytest.approx(0.9, abs=0.01)


def test_irrigation_validation() -> None:
    tool = IrrigationTool()
    assert tool.call(action="crop_water_requirement", et0_mm=5).ok is False
    assert tool.call(action="crop_water_requirement", et0_mm=5, kc=9).ok is False
    assert tool.call(action="drip_runtime", depth_mm=6, emitter_flow_lph=0).ok is False
    assert tool.call(action="unknown").ok is False


# --------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------- #
def test_spraying_window_verdicts() -> None:
    good = spraying_window(22.0, 60.0, 5.0, False)
    assert "suitable" in good.verdict

    windy = spraying_window(22.0, 60.0, 25.0, False)
    assert "do not spray" in windy.verdict

    rain = spraying_window(22.0, 60.0, 5.0, True)
    assert "do not spray" in rain.verdict

    marginal = spraying_window(11.0, 60.0, 10.0, False)
    assert "caution" in marginal.verdict


def test_weather_tool_spraying_conditions_and_live_weather() -> None:
    tool = WeatherTool()
    result = tool.call(
        action="spraying_conditions",
        temperature_c=22,
        humidity_pct=60,
        wind_kmh=5,
        rain_expected="no",
    )
    assert result.ok is True
    assert result.data["verdict"]
    assert any("label" in note for note in result.notes)

    live = tool.call(action="current")
    assert live.ok is False
    assert "network" in live.error

    missing = tool.call(action="spraying_conditions", temperature_c=22)
    assert missing.ok is False
    assert tool.call(
        action="spraying_conditions",
        temperature_c=22,
        humidity_pct=60,
        wind_kmh=5,
        rain_expected="maybe",
    ).ok is False


# --------------------------------------------------------------------- #
# Veterinary
# --------------------------------------------------------------------- #
def test_structure_case_maps_symptoms_to_systems() -> None:
    case = structure_case("cattle", "yotal va ishtaha yo'q, harorat ko'tarilgan")
    assert "respiratory" in case.systems
    assert "digestive" in case.systems
    assert case.urgent is False
    assert case.observations[0] == ALWAYS_OBSERVE[0]


def test_structure_case_flags_urgency_and_handles_unknown_signs() -> None:
    urgent = structure_case("cow", "qon aralash ich ketish")
    assert urgent.urgent is True

    vague = structure_case("sheep", "o'zini yaxshi his qilmayapti")
    assert vague.systems == ()
    assert any("aniqlanmadi" in observation for observation in vague.observations)


def test_veterinary_tool_never_diagnoses_or_doses() -> None:
    tool = VeterinaryTool()
    checklist = tool.call(
        action="observation_checklist", species="cattle", symptoms="yotal, ishtaha yo'q"
    )
    assert checklist.ok is True
    assert checklist.data["systems"]
    assert checklist.requires_verified_sources is True
    assert any("does not diagnose" in note for note in checklist.notes)

    dosage = tool.call(action="dosage_lookup", question="qancha antibiotik beriladi?")
    assert dosage.ok is False
    assert "ishonchli manbadan" in dosage.error
    assert dosage.requires_verified_sources is True

    assert tool.call(action="observation_checklist", species="cattle").ok is False
    assert tool.call(action="nope").ok is False
