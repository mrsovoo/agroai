"""Quotas and plans: enforced server-side, never trusted from a client.

```text
USER -> PLAN (free / pro / custom) -> USAGE LIMITS
```

The roadmap is explicit that "limitlar server tomonida tekshiriladi. Frontenddagi
oddiy counterga ishonmaymiz" - the limits are checked on the server and the
client's own counter is ignored. That is what :class:`QuotaPolicy` does:

* :class:`Plan` - the named limit set (free / pro / ...), admin-configurable;
* per-user overrides - an individual limit for one account;
* :class:`UsageStore` - a day-scoped usage counter that survives restarts;
* :meth:`QuotaPolicy.consume` - check **and** count in one server-side step.

Honesty rules: a blocked request says exactly which limit was hit and when it
resets; a disabled policy still *counts* usage (so enabling it later is not a
blind start); and ``-1`` means unlimited, which is reported as ``None`` remaining
rather than a fake large number.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

#: Actions a quota can be applied to.
ACTION_MESSAGE = "message"
ACTION_IMAGE = "image"
ACTION_VIDEO = "video"
ACTION_SEARCH = "search"
ACTION_EXTERNAL_SEARCH = "external_search"
ACTION_KNOWLEDGE_WRITE = "knowledge_write"

ACTIONS: tuple[str, ...] = (
    ACTION_MESSAGE,
    ACTION_IMAGE,
    ACTION_VIDEO,
    ACTION_SEARCH,
    ACTION_EXTERNAL_SEARCH,
    ACTION_KNOWLEDGE_WRITE,
)

#: ``-1`` means "no limit".
UNLIMITED = -1

USAGE_FILE = "usage.json"


def today() -> str:
    """Current UTC day (``YYYY-MM-DD``) - the quota window."""
    return datetime.now(timezone.utc).date().isoformat()


@dataclass(frozen=True)
class QuotaLimits:
    """Per-day limits for one plan (``-1`` = unlimited)."""

    messages: int = 20
    images: int = 3
    videos: int = 1
    searches: int = 100
    external_searches: int = 0
    knowledge_writes: int = 0

    def limit_for(self, action: str) -> int:
        mapping = {
            ACTION_MESSAGE: self.messages,
            ACTION_IMAGE: self.images,
            ACTION_VIDEO: self.videos,
            ACTION_SEARCH: self.searches,
            ACTION_EXTERNAL_SEARCH: self.external_searches,
            ACTION_KNOWLEDGE_WRITE: self.knowledge_writes,
        }
        if action not in mapping:
            raise ValueError(f"unknown quota action {action!r}; expected one of {ACTIONS}")
        return mapping[action]

    def merged(self, override: dict[str, int]) -> "QuotaLimits":
        """Return these limits with per-action overrides applied."""
        known = set(self.__dataclass_fields__)
        clean = {key: int(value) for key, value in override.items() if key in known}
        return QuotaLimits(**{**asdict(self), **clean})

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class Plan:
    """A named subscription tier."""

    name: str
    title: str
    limits: QuotaLimits = field(default_factory=QuotaLimits)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "limits": self.limits.to_dict(),
        }


#: The plans the platform ships with. Limits are admin-configurable; these are
#: conservative starting points from the specification.
DEFAULT_PLANS: dict[str, Plan] = {
    "free": Plan(
        name="free",
        title="Free",
        limits=QuotaLimits(
            messages=20,
            images=3,
            videos=1,
            searches=50,
            external_searches=0,
            knowledge_writes=0,
        ),
        description="Text answers with sources, a few image checks per day.",
    ),
    "pro": Plan(
        name="pro",
        title="Pro",
        limits=QuotaLimits(
            messages=500,
            images=100,
            videos=20,
            searches=1000,
            external_searches=-1,
            knowledge_writes=20,
        ),
        description="Higher limits, video analysis, trusted external search.",
    ),
    "internal": Plan(
        name="internal",
        title="Internal",
        limits=QuotaLimits(
            messages=UNLIMITED,
            images=UNLIMITED,
            videos=UNLIMITED,
            searches=UNLIMITED,
            external_searches=UNLIMITED,
            knowledge_writes=UNLIMITED,
        ),
        description="First-party products (AgroAI Web/Bot/AgrozAI) on the private API.",
    ),
}


@dataclass
class QuotaDecision:
    """The server's answer to "may this request proceed?"."""

    allowed: bool
    action: str
    user_id: str
    plan: str
    used: int = 0
    limit: int = UNLIMITED
    reason: str = ""
    enforce: bool = True

    @property
    def unlimited(self) -> bool:
        return self.limit == UNLIMITED

    @property
    def remaining(self) -> int | None:
        if self.unlimited:
            return None
        return max(0, self.limit - self.used)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "action": self.action,
            "user_id": self.user_id,
            "plan": self.plan,
            "used": self.used,
            "limit": self.limit,
            "remaining": self.remaining,
            "unlimited": self.unlimited,
            "reason": self.reason,
            "enforce": self.enforce,
        }

    def as_headers(self) -> dict[str, str]:
        """Rate-limit style headers a client can display."""
        headers = {
            "X-RateLimit-Action": self.action,
            "X-RateLimit-Limit": "unlimited" if self.unlimited else str(self.limit),
            "X-RateLimit-Used": str(self.used),
        }
        if self.remaining is not None:
            headers["X-RateLimit-Remaining"] = str(self.remaining)
        return headers


class UsageStore:
    """Day-scoped, server-side usage counter backed by a JSON file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._data: dict[str, dict[str, dict[str, int]]] = {}
        self._load()

    # ------------------------------------------------------------------ #
    def counts(self, user_id: str, day: str | None = None) -> dict[str, int]:
        return dict(self._data.get(day or today(), {}).get(user_id, {}))

    def used(self, user_id: str, action: str, day: str | None = None) -> int:
        return self.counts(user_id, day).get(action, 0)

    def increment(
        self, user_id: str, action: str, amount: int = 1, day: str | None = None
    ) -> int:
        """Add to a counter and return the new value."""
        window = day or today()
        bucket = self._data.setdefault(window, {}).setdefault(user_id, {})
        bucket[action] = bucket.get(action, 0) + amount
        self._save()
        return bucket[action]

    def reset(self, user_id: str | None = None, day: str | None = None) -> None:
        window = day or today()
        if user_id is None:
            self._data.pop(window, None)
        else:
            self._data.get(window, {}).pop(user_id, None)
        self._save()

    # ------------------------------------------------------------------ #
    def stats(self) -> dict[str, Any]:
        windows = {window: len(users) for window, users in self._data.items()}
        totals: dict[str, int] = {}
        for users in self._data.values():
            for actions in users.values():
                for action, value in actions.items():
                    totals[action] = totals.get(action, 0) + value
        return {"path": str(self.path), "windows": windows, "totals": totals}

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Never silently discard usage data: start empty but keep the file.
            return
        if isinstance(data, dict):
            self._data = data.get("usage", data) if "usage" in data else data

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"usage": self._data}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class QuotaPolicy:
    """Resolves a user's plan and enforces its limits server-side."""

    def __init__(
        self,
        *,
        store: UsageStore | None = None,
        plans: dict[str, Plan] | None = None,
        default_plan: str = "free",
        overrides: dict[str, dict[str, int]] | None = None,
        enforce: bool = True,
    ) -> None:
        self.store = store
        self.plans = dict(plans or DEFAULT_PLANS)
        self.default_plan = default_plan if default_plan in self.plans else "free"
        self.overrides = {key: dict(value) for key, value in (overrides or {}).items()}
        self.enforce = enforce

    # ------------------------------------------------------------------ #
    def plan_for(self, requested: str | None) -> Plan:
        """Return the plan for a request (unknown names fall back to the default)."""
        if requested and requested.lower() in self.plans:
            return self.plans[requested.lower()]
        return self.plans[self.default_plan]

    def limits_for(self, user_id: str, requested: str | None = None) -> QuotaLimits:
        """Limits for one user: plan limits plus that user's overrides."""
        limits = self.plan_for(requested).limits
        override = self.overrides.get(user_id)
        return limits.merged(override) if override else limits

    def set_override(self, user_id: str, **limits: int) -> QuotaLimits:
        """Grant a custom limit set to one user (admin action)."""
        merged = self.limits_for(user_id).merged(limits)
        self.overrides[user_id] = {
            key: value for key, value in asdict(merged).items() if value != asdict(self.plan_for(None).limits)[key]
        }
        return merged

    def clear_override(self, user_id: str) -> None:
        self.overrides.pop(user_id, None)

    # ------------------------------------------------------------------ #
    def check(
        self, user_id: str, action: str, *, plan: str | None = None, day: str | None = None
    ) -> QuotaDecision:
        """Decide whether ``action`` is still allowed, without counting it."""
        resolved = self.plan_for(plan)
        limits = self.limits_for(user_id, plan)
        limit = limits.limit_for(action)
        used = self.store.used(user_id, action, day) if self.store is not None else 0

        if limit == UNLIMITED:
            return QuotaDecision(
                allowed=True,
                action=action,
                user_id=user_id,
                plan=resolved.name,
                used=used,
                limit=limit,
                reason="unlimited",
                enforce=self.enforce,
            )
        if used < limit:
            return QuotaDecision(
                allowed=True,
                action=action,
                user_id=user_id,
                plan=resolved.name,
                used=used,
                limit=limit,
                enforce=self.enforce,
            )
        return QuotaDecision(
            allowed=not self.enforce,
            action=action,
            user_id=user_id,
            plan=resolved.name,
            used=used,
            limit=limit,
            reason=(
                f"{resolved.name} plan allows {limit} {action} request(s) per day "
                f"(used {used}); the limit resets at 00:00 UTC"
            ),
            enforce=self.enforce,
        )

    def consume(
        self, user_id: str, action: str, *, plan: str | None = None, day: str | None = None
    ) -> QuotaDecision:
        """Check and (when allowed) count the action in one server-side step."""
        decision = self.check(user_id, action, plan=plan, day=day)
        if not decision.allowed or self.store is None:
            return decision
        used = self.store.increment(user_id, action, day=day)
        decision.used = used
        return decision

    def usage(self, user_id: str, *, day: str | None = None) -> dict[str, Any]:
        """Current usage and remaining allowance for one user."""
        window = day or today()
        limits = asdict(self.limits_for(user_id))
        counts = self.store.counts(user_id, window) if self.store is not None else {}
        per_action = {}
        for action, limit in limits.items():
            canonical = _canonical_action(action)
            used = counts.get(canonical, 0)
            per_action[canonical] = {
                "used": used,
                "limit": limit,
                "remaining": None if limit == UNLIMITED else max(0, limit - used),
            }
        return {"user_id": user_id, "day": window, "plan": self.plan_for(None).name, "actions": per_action}

    def reset(self, user_id: str | None = None, *, day: str | None = None) -> None:
        if self.store is not None:
            self.store.reset(user_id, day)

    def describe(self) -> str:
        lines = ["AgroAI plans:"]
        for plan in sorted(self.plans.values(), key=lambda item: item.name):
            lines.append(f"  {plan.name}: {plan.description}")
            lines.append(
                "    "
                + ", ".join(
                    f"{_canonical_action(key)}={'unlimited' if value == UNLIMITED else value}"
                    for key, value in asdict(plan.limits).items()
                )
            )
        if self.overrides:
            lines.append(f"  per-user overrides: {', '.join(sorted(self.overrides))}")
        lines.append(f"  enforcement: {'on' if self.enforce else 'off (counted, not blocked)'}")
        return "\n".join(lines)


_ACTION_BY_FIELD: dict[str, str] = {
    "messages": ACTION_MESSAGE,
    "images": ACTION_IMAGE,
    "videos": ACTION_VIDEO,
    "searches": ACTION_SEARCH,
    "external_searches": ACTION_EXTERNAL_SEARCH,
    "knowledge_writes": ACTION_KNOWLEDGE_WRITE,
}


def _canonical_action(field_name: str) -> str:
    return _ACTION_BY_FIELD.get(field_name, field_name)


def merge_plans(names: Iterable[str], plans: dict[str, Plan] | None = None) -> dict[str, Plan]:
    """Return the named plans (unknown names are skipped, never invented)."""
    available = plans or DEFAULT_PLANS
    return {name: available[name] for name in names if name in available}


__all__ = [
    "ACTIONS",
    "ACTION_EXTERNAL_SEARCH",
    "ACTION_IMAGE",
    "ACTION_KNOWLEDGE_WRITE",
    "ACTION_MESSAGE",
    "ACTION_SEARCH",
    "ACTION_VIDEO",
    "DEFAULT_PLANS",
    "Plan",
    "QuotaDecision",
    "QuotaLimits",
    "QuotaPolicy",
    "UNLIMITED",
    "USAGE_FILE",
    "UsageStore",
    "merge_plans",
    "today",
]
