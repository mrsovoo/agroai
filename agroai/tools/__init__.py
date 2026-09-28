"""AgroAI tools: deterministic, offline helpers the agent can call.

Knowledge tools (``search_knowledge``, ``get_document``, ...) are already
available through the retriever and are not duplicated here; this package holds
the *computational* tools:

* :mod:`agroai.tools.calculator` - safe arithmetic and unit conversion;
* :mod:`agroai.tools.soil` - salinity/pH classification, leaching requirement;
* :mod:`agroai.tools.irrigation` - ETc, volumes, drip runtime;
* :mod:`agroai.tools.weather` - spray-window checks (live weather is absent);
* :mod:`agroai.tools.veterinary` - case structure, never a diagnosis or a dose.
"""

from __future__ import annotations

from agroai.tools.base import (
    CHOICE,
    INTEGER,
    NUMBER,
    STRING,
    TEXT,
    Tool,
    ToolError,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    UnknownToolError,
    get_default_registry,
)
from agroai.tools.calculator import (
    CalculatorError,
    CalculatorTool,
    UnitConverterTool,
    evaluate_expression,
)
from agroai.tools.irrigation import IrrigationTool
from agroai.tools.soil import SoilTool, classify_ph, classify_salinity, leaching_requirement
from agroai.tools.veterinary import VeterinaryTool, structure_case
from agroai.tools.weather import WeatherTool, spraying_window

__all__ = [
    "CHOICE",
    "INTEGER",
    "NUMBER",
    "STRING",
    "TEXT",
    "CalculatorError",
    "CalculatorTool",
    "IrrigationTool",
    "SoilTool",
    "Tool",
    "ToolError",
    "ToolParameter",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "UnitConverterTool",
    "UnknownToolError",
    "VeterinaryTool",
    "WeatherTool",
    "classify_ph",
    "classify_salinity",
    "evaluate_expression",
    "get_default_registry",
    "leaching_requirement",
    "spraying_window",
    "structure_case",
]
