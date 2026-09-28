"""Tools: the contract, parameter validation and the registry.

A tool is a small, deterministic, offline function the agent can call instead of
guessing: unit conversion, salinity classification, irrigation arithmetic. Tools
that would need the network or an unverified source are registered too, but they
answer *honestly* - "not available" - rather than inventing a value.

Every result carries ``notes`` and, where a published method was used,
``reference``. A tool never states a citation it cannot support: references are
method names (e.g. "FAO leaching requirement"), never fabricated URLs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

NUMBER = "number"
INTEGER = "integer"
STRING = "string"
TEXT = "text"
CHOICE = "choice"


class ToolError(RuntimeError):
    """Raised when a tool is called with parameters it cannot use."""


class UnknownToolError(ToolError):
    """Raised when the registry is asked for a tool it does not have."""


@dataclass(frozen=True)
class ToolParameter:
    """One named input of a tool."""

    name: str
    type: str = NUMBER
    description: str = ""
    required: bool = True
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolSpec:
    """Self-description of a tool: what it is for and what it needs."""

    name: str
    description: str
    category: str = "general"
    parameters: tuple[ToolParameter, ...] = ()
    #: True when the tool can only answer with live data from the internet.
    requires_network: bool = False
    #: True when the tool's output is only trustworthy with a verified source.
    requires_verified_sources: bool = False
    #: Published method the tool implements, if any (never a URL that may rot).
    reference: str = ""
    #: True when the tool is implemented; False for registered-but-future tools.
    implemented: bool = True

    def parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": [
                {
                    "name": parameter.name,
                    "type": parameter.type,
                    "required": parameter.required,
                    "description": parameter.description,
                }
                for parameter in self.parameters
            ],
            "requires_network": self.requires_network,
            "requires_verified_sources": self.requires_verified_sources,
            "reference": self.reference,
            "implemented": self.implemented,
        }


@dataclass
class ToolResult:
    """The outcome of a tool call, including how to present it honestly."""

    tool: str
    ok: bool
    output: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    reference: str = ""
    error: str = ""
    requires_verified_sources: bool = False

    @property
    def status(self) -> str:
        return "ok" if self.ok else "unavailable"

    def as_text(self) -> str:
        """Render the result for a prompt or the terminal."""
        lines = [f"[{self.tool}] {self.output}".strip()]
        if self.reference:
            lines.append(f"  Usul/manba: {self.reference}")
        for note in self.notes:
            lines.append(f"  Eslatma: {note}")
        if not self.ok and self.error:
            lines.append(f"  Xato/Error: {self.error}")
        return "\n".join(line for line in lines if line)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "output": self.output,
            "data": dict(self.data),
            "notes": list(self.notes),
            "reference": self.reference,
            "error": self.error,
        }


class Tool(ABC):
    """Base class for every AgroAI tool."""

    spec: ToolSpec

    @property
    def name(self) -> str:
        return self.spec.name

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool. Must not perform network I/O."""

    # ------------------------------------------------------------------ #
    # Parameter handling (shared, so every tool validates the same way)
    # ------------------------------------------------------------------ #
    def check_parameters(self, kwargs: dict[str, Any]) -> list[str]:
        """Return a description of each parameter problem (empty = valid)."""
        problems: list[str] = []
        for parameter in self.spec.parameters:
            raw = kwargs.get(parameter.name)
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                if parameter.required:
                    problems.append(f"{parameter.name} is required ({parameter.type})")
                continue
            if parameter.type in {NUMBER, INTEGER}:
                try:
                    number = float(raw)
                except (TypeError, ValueError):
                    problems.append(f"{parameter.name} must be a number, got {raw!r}")
                    continue
                if parameter.type == INTEGER and number != int(number):
                    problems.append(f"{parameter.name} must be a whole number, got {raw!r}")
            elif parameter.type == CHOICE and parameter.choices:
                if str(raw).strip().lower() not in parameter.choices:
                    problems.append(
                        f"{parameter.name} must be one of {', '.join(parameter.choices)}"
                    )
        return problems

    def call(self, **kwargs: Any) -> ToolResult:
        """Validate parameters, then run - never raising for bad user input."""
        problems = self.check_parameters(kwargs)
        if problems:
            return ToolResult(
                tool=self.name,
                ok=False,
                error="; ".join(problems),
                notes=[f"Required parameters: {', '.join(self.spec.parameter_names())}"],
                requires_verified_sources=self.spec.requires_verified_sources,
            )
        return self.run(**kwargs)

    # ------------------------------------------------------------------ #
    # Input helpers for tool implementations
    # ------------------------------------------------------------------ #
    def numbers(self, kwargs: dict[str, Any], *names: str) -> dict[str, float]:
        """Coerce the named parameters to floats, raising :class:`ToolError`."""
        values: dict[str, float] = {}
        for name in names:
            try:
                values[name] = float(kwargs[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ToolError(f"{self.name}: {name} must be a number") from exc
        return values

    def optional_number(self, kwargs: dict[str, Any], name: str) -> float | None:
        raw = kwargs.get(name)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return None
        try:
            return float(raw)
        except (TypeError, ValueError) as exc:
            raise ToolError(f"{self.name}: {name} must be a number") from exc

    def fail(self, message: str, **extra: Any) -> ToolResult:
        return ToolResult(
            tool=self.name,
            ok=False,
            error=message,
            requires_verified_sources=self.spec.requires_verified_sources,
            **extra,
        )


class ToolRegistry:
    """The set of tools the agent may call, looked up by name."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolError(f"A tool named {tool.name!r} is already registered.")
        self._tools[tool.name] = tool

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise UnknownToolError(
                f"Unknown tool {name!r}. Available: {', '.join(self.names()) or '(none)'}"
            ) from exc

    def specs(self, *, category: str | None = None) -> list[ToolSpec]:
        specs = [tool.spec for tool in self._tools.values()]
        if category:
            specs = [spec for spec in specs if spec.category == category]
        return sorted(specs, key=lambda spec: (spec.category, spec.name))

    def run(self, name: str, **kwargs: Any) -> ToolResult:
        """Run a tool by name. Unknown names produce a failed result, not a crash."""
        try:
            tool = self.get(name)
        except UnknownToolError as exc:
            return ToolResult(tool=name, ok=False, error=str(exc))
        return tool.call(**kwargs)

    def describe(self) -> str:
        """Human readable listing used by the CLI and the planner prompt."""
        lines: list[str] = []
        for spec in self.specs():
            marker = "" if spec.implemented else "  (not implemented)"
            lines.append(f"- {spec.name} [{spec.category}]: {spec.description}{marker}")
        return "\n".join(lines)


def get_default_registry() -> ToolRegistry:
    """The standard AgroAI tool set (imports are local to avoid cycles)."""
    from agroai.tools.calculator import CalculatorTool, UnitConverterTool
    from agroai.tools.irrigation import IrrigationTool
    from agroai.tools.soil import SoilTool
    from agroai.tools.veterinary import VeterinaryTool
    from agroai.tools.weather import WeatherTool

    return ToolRegistry(
        [
            CalculatorTool(),
            UnitConverterTool(),
            SoilTool(),
            IrrigationTool(),
            WeatherTool(),
            VeterinaryTool(),
        ]
    )


#: Type alias used by callers that inject a registry factory in tests.
RegistryFactory = Callable[[], ToolRegistry]

__all__ = [
    "CHOICE",
    "INTEGER",
    "NUMBER",
    "STRING",
    "TEXT",
    "RegistryFactory",
    "Tool",
    "ToolError",
    "ToolParameter",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "UnknownToolError",
    "get_default_registry",
]
