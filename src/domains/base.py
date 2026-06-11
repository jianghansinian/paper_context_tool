"""Schema definition classes for the V3 schema-driven architecture.

These types form the single source of truth from which both LLM prompts and
markdown output are derived.  See ``docs/design_schema_driven.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────
# Leaf definitions
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ColumnDef:
    """A column in a table-valued field."""
    name: str
    label_en: str
    label_zh: str


@dataclass
class FieldDef:
    """Definition of a single analysis field extracted by the LLM.

    Each field maps to one key in the JSON the LLM returns.  The *kind*
    controls both the JSON type the LLM is asked for and how the markdown
    renderer displays the value.
    """
    name: str
    kind: str           # "text" | "list[str]" | "component_table" | "formula_table"
                        # | "result_table" | "key_value_table" | "structured_list"
    label_en: str
    label_zh: str
    prompt: str         # Instruction for the LLM (a single sentence or question).
    required: bool = True
    columns: list[ColumnDef] = field(default_factory=list)

    # ── helpers ──

    @property
    def is_table(self) -> bool:
        return self.kind.endswith("_table")

    @property
    def is_list(self) -> bool:
        return self.kind in ("list[str]", "structured_list")


@dataclass
class SectionDef:
    """A section (or sub-section) in the output markdown report.

    Sections are rendered in definition order.  A section whose *every*
    field is empty (and that has no non-empty sub-sections) is silently
    skipped unless ``always_show`` is set.
    """
    name: str
    level: int                              # 1 = ##, 2 = ###, 3 = ####
    title_en: str
    title_zh: str
    fields: list[str] = field(default_factory=list)
    condition: Optional[str] = None         # Python expression evaluated at render time
    subsections: list["SectionDef"] = field(default_factory=list)
    always_show: bool = False               # Render even if all fields are empty


# ──────────────────────────────────────────────────────────────────────
# Profile types
# ──────────────────────────────────────────────────────────────────────


@dataclass
class PaperTypeProfile:
    """The complete schema for one paper type within a domain.

    *fields* drive LLM prompt generation; *sections* drive markdown rendering.
    """
    type_name: str
    description: str            # Used in the paper-type detection prompt.
    fields: list[FieldDef]
    sections: list[SectionDef]

    # ── helpers ──

    def get_field(self, name: str) -> Optional[FieldDef]:
        for f in self.fields:
            if f.name == name:
                return f
        return None

    @property
    def required_fields(self) -> list[FieldDef]:
        return [f for f in self.fields if f.required]

    @property
    def optional_fields(self) -> list[FieldDef]:
        return [f for f in self.fields if not f.required]


@dataclass
class DomainProfile:
    """A complete configuration for one academic domain.

    A domain contains one or more *paper types*, each with its own analysis
    schema.  ``default_paper_type`` is used when type detection confidence is
    too low.
    """
    domain_name: str
    domain_description: str     # Used in the domain-detection prompt.
    paper_types: list[PaperTypeProfile]
    default_paper_type: str

    # ── helpers ──

    def get_paper_type(self, name: str) -> Optional[PaperTypeProfile]:
        for pt in self.paper_types:
            if pt.type_name == name:
                return pt
        return None

    @property
    def default_profile(self) -> PaperTypeProfile:
        pt = self.get_paper_type(self.default_paper_type)
        if pt is None and self.paper_types:
            return self.paper_types[0]
        if pt is None:
            raise ValueError(f"Domain '{self.domain_name}' has no paper types.")
        return pt
