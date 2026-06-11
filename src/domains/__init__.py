"""Domain registry for the V3 schema-driven architecture.

Import this module to auto-register all built-in domain profiles.  Third-party
domains can be registered via ``register(profile)``.
"""

from __future__ import annotations

from domains.base import DomainProfile

_registry: dict[str, DomainProfile] = {}


def register(profile: DomainProfile) -> None:
    """Register a domain profile so it can be looked up by name."""
    _registry[profile.domain_name] = profile


def get_domain(name: str) -> DomainProfile:
    """Return the DomainProfile registered under *name*."""
    if name not in _registry:
        available = list(_registry)
        raise KeyError(f"Unknown domain: '{name}'. Available: {available}")
    return _registry[name]


def list_domains() -> list[str]:
    """Return the names of all registered domains."""
    return list(_registry)


# ── Auto-register built-in domains ──────────────────────────────────
from domains.ai_ml import AI_ML_DOMAIN      # noqa: E402
from domains.biology import BIOLOGY_DOMAIN  # noqa: E402
from domains.materials_science import MATERIALS_SCIENCE_DOMAIN  # noqa: E402
register(AI_ML_DOMAIN)
register(BIOLOGY_DOMAIN)
register(MATERIALS_SCIENCE_DOMAIN)
