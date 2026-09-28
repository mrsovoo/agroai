"""Skills: domains that scope knowledge, tools and prompt guidance.

* :mod:`agroai.skills.base` - :class:`Skill`, :class:`SkillRegistry`;
* :mod:`agroai.skills.catalog` - the fifteen default domain skills;
* :mod:`agroai.skills.router` - deterministic routing, and the rule that
  retrieval is only scoped when one skill clearly wins.

Skills are declarative data (categories, tools, keywords, one guidance line), so
the planner can use them, the CLI can list them and doctor can report them -
without any skill ever asserting a fact.
"""

from __future__ import annotations

from agroai.skills.base import (
    Skill,
    SkillMatch,
    SkillRegistry,
    get_skill_registry,
)
from agroai.skills.catalog import DEFAULT_SKILLS
from agroai.skills.router import (
    CATEGORY_SKILL_HINTS,
    SCOPE_MARGIN,
    RoutingResult,
    SkillRouter,
    get_skill_router,
    skills_context,
)

__all__ = [
    "CATEGORY_SKILL_HINTS",
    "DEFAULT_SKILLS",
    "SCOPE_MARGIN",
    "RoutingResult",
    "Skill",
    "SkillMatch",
    "SkillRegistry",
    "SkillRouter",
    "get_skill_registry",
    "get_skill_router",
    "skills_context",
]
