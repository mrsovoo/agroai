"""Data tools: safe arithmetic and unit conversion.

No crop advice, no dosages - only arithmetic and conversions the user can verify
by hand. The evaluator never uses ``eval``: it walks a parsed AST and allows a
small, explicit set of operators; anything else is rejected with an error.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from agroai.tools.base import (
    NUMBER,
    TEXT,
    Tool,
    ToolParameter,
    ToolResult,
    ToolSpec,
)

_BINARY_OPERATORS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
}

#: Guards against accidental resource abuse (e.g. ``9 ** 9 ** 9``).
_MAX_POWER = 64


class CalculatorError(ValueError):
    """Raised for expressions the calculator refuses to evaluate."""


def evaluate_expression(expression: str) -> float:
    """Evaluate a small arithmetic expression safely.

    Supports ``+ - * / // % **``, parentheses, unary signs and a few functions
    (``sqrt``, ``round``, ``min``, ``max``, ``abs``, ``ceil``, ``floor``).
    """
    expression = (expression or "").strip()
    if not expression:
        raise CalculatorError("empty expression")
    if len(expression) > 200:
        raise CalculatorError("expression too long (max 200 characters)")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"cannot parse expression: {exc.msg}") from exc
    return float(_eval_node(tree.body))


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculatorError(f"only numbers are allowed, got {node.value!r}")
        return float(node.value)

    if isinstance(node, ast.BinOp):
        handler = _BINARY_OPERATORS.get(type(node.op))
        if handler is None:
            raise CalculatorError(f"operator not allowed: {type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_POWER:
                raise CalculatorError("exponent is too large")
        try:
            return float(handler(left, right))
        except ZeroDivisionError as exc:
            raise CalculatorError("division by zero") from exc

    if isinstance(node, ast.UnaryOp):
        handler = _UNARY_OPERATORS.get(type(node.op))
        if handler is None:
            raise CalculatorError(f"operator not allowed: {type(node.op).__name__}")
        return float(handler(_eval_node(node.operand)))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise CalculatorError("only sqrt/round/min/max/abs/ceil/floor may be called")
        if node.keywords:
            raise CalculatorError("keyword arguments are not allowed")
        arguments = [_eval_node(argument) for argument in node.args]
        try:
            return float(_FUNCTIONS[node.func.id](*arguments))
        except (TypeError, ValueError) as exc:
            raise CalculatorError(f"{node.func.id}() failed: {exc}") from exc

    raise CalculatorError(f"unsupported expression element: {type(node).__name__}")


class CalculatorTool(Tool):
    """Evaluate an arithmetic expression."""

    spec = ToolSpec(
        name="calculator",
        description=(
            "Evaluate an arithmetic expression (+ - * / // % **, parentheses, "
            "sqrt/round/min/max/abs/ceil/floor)."
        ),
        category="data",
        parameters=(
            ToolParameter(
                "expression",
                TEXT,
                "Arithmetic expression, e.g. '12 * 3.5' or 'sqrt(400)'.",
            ),
        ),
    )

    def run(self, **kwargs: Any) -> ToolResult:
        expression = str(kwargs.get("expression", "")).strip()
        try:
            value = evaluate_expression(expression)
        except CalculatorError as exc:
            return self.fail(str(exc))
        pretty = int(value) if value.is_integer() else value
        return ToolResult(
            tool=self.name,
            ok=True,
            output=f"{expression} = {pretty}",
            data={"expression": expression, "result": value},
        )


#: Conversion table: every unit expressed in the base unit of its dimension.
_UNIT_TABLE: dict[str, tuple[str, float]] = {
    # area (base: m2)
    "m2": ("area", 1.0),
    "ga": ("area", 10_000.0),
    "ha": ("area", 10_000.0),
    "gektar": ("area", 10_000.0),
    "sotka": ("area", 100.0),
    "km2": ("area", 1_000_000.0),
    # mass (base: kg)
    "kg": ("mass", 1.0),
    "g": ("mass", 0.001),
    "t": ("mass", 1000.0),
    "tonna": ("mass", 1000.0),
    "s": ("mass", 100.0),  # centner
    "sentner": ("mass", 100.0),
    # volume (base: m3)
    "m3": ("volume", 1.0),
    "l": ("volume", 0.001),
    "litr": ("volume", 0.001),
    "ml": ("volume", 0.000001),
    # length (base: m)
    "m": ("length", 1.0),
    "cm": ("length", 0.01),
    "mm": ("length", 0.001),
    "km": ("length", 1000.0),
    "sm": ("length", 0.01),
}

#: Concentration-ish units handled separately (they are ratios, not lengths).
_RATIO_UNITS = {
    "ppm": 1.0,
    "%": 10_000.0,  # 1 % = 10 000 ppm
    "mg/kg": 1.0,
    "mg/l": 1.0,
    "g/l": 1000.0,
}


class UnitConverterTool(Tool):
    """Convert between agricultural units of area, mass, volume and length."""

    spec = ToolSpec(
        name="unit_converter",
        description=(
            "Convert between units: area (m2, ga/ha, sotka, km2), mass "
            "(kg, g, t, sentner), volume (m3, l, ml), length (m, cm, mm, km) "
            "and concentrations (ppm, %, mg/kg, mg/l, g/l)."
        ),
        category="data",
        parameters=(
            ToolParameter("value", NUMBER, "Numeric value to convert."),
            ToolParameter("from_unit", TEXT, "Source unit, e.g. 'ga'."),
            ToolParameter("to_unit", TEXT, "Target unit, e.g. 'm2'."),
        ),
    )

    def run(self, **kwargs: Any) -> ToolResult:
        value = self.numbers(kwargs, "value")["value"]
        source = str(kwargs.get("from_unit", "")).strip().lower()
        target = str(kwargs.get("to_unit", "")).strip().lower()

        if source in _RATIO_UNITS and target in _RATIO_UNITS:
            result = value * _RATIO_UNITS[source] / _RATIO_UNITS[target]
            dimension = "concentration"
        else:
            if source not in _UNIT_TABLE or target not in _UNIT_TABLE:
                known = sorted(set(_UNIT_TABLE) | set(_RATIO_UNITS))
                return self.fail(
                    f"unknown unit ({source!r} or {target!r}). Known units: {', '.join(known)}"
                )
            source_dimension, source_factor = _UNIT_TABLE[source]
            target_dimension, target_factor = _UNIT_TABLE[target]
            if source_dimension != target_dimension:
                return self.fail(
                    f"cannot convert {source_dimension} ({source}) to "
                    f"{target_dimension} ({target})"
                )
            dimension = source_dimension
            result = value * source_factor / target_factor

        rounded = round(result, 6)
        pretty = int(rounded) if float(rounded).is_integer() else rounded
        return ToolResult(
            tool=self.name,
            ok=True,
            output=f"{value:g} {source} = {pretty} {target}",
            data={
                "value": value,
                "from_unit": source,
                "to_unit": target,
                "result": result,
                "dimension": dimension,
            },
            notes=["Conversion is arithmetic only; verify the unit you measured."],
        )


__all__ = [
    "CalculatorError",
    "CalculatorTool",
    "UnitConverterTool",
    "evaluate_expression",
]
