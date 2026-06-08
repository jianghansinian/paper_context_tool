"""Core data models for V3 pipeline (seed-paper-centric structured understanding)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class CitationType(Enum):
    SUPPORTING = "supporting"
    CONTRASTING = "contrasting"
    FOUNDATIONAL = "foundational"
    RELATED = "related_work"
    NOT_CLASSIFIED = "not_classified"


@dataclass
class Component:
    name: str
    purpose: str
    details: Optional[str] = None
    referenced_figure: Optional[str] = None


@dataclass
class Formula:
    name: str
    latex: Optional[str] = None
    explanation: str = ""
    significance: str = ""


@dataclass
class Result:
    dataset: str
    metric: str
    value: str
    comparison: Optional[str] = None


@dataclass
class StructuredUnderstanding:
    problem: str = ""
    motivation: str = ""
    key_insight: str = ""
    architecture_overview: str = ""
    components: list[Component] = field(default_factory=list)
    formulas: list[Formula] = field(default_factory=list)
    architecture_figure: Optional[str] = None
    design_rationale: Optional[str] = None
    related_work_context: Optional[str] = None
    intuitive_analogy: Optional[str] = None
    training_data: str = ""
    data_engineering: Optional[str] = None
    training_stages: list[dict] = field(default_factory=list)
    loss_functions: list[str] = field(default_factory=list)
    optimizer: str = ""
    training_procedure: str = ""
    inference_procedure: str = ""
    post_processing: Optional[str] = None
    deployment_architecture: Optional[str] = None
    deployment_value: Optional[str] = None
    field_evolution: Optional[str] = None
    core_question: Optional[str] = None
    evaluation_setup: Optional[str] = None
    industry_comparison: list[dict] = field(default_factory=list)
    main_results: list[Result] = field(default_factory=list)
    ablation_results: list[str] = field(default_factory=list)
    qualitative_results: Optional[str] = None
    contributions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    synthesis: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StructuredUnderstanding":
        components = [Component(**c) for c in d.get("components", [])]
        formulas = [Formula(**f) for f in d.get("formulas", [])]
        main_results = [Result(**r) for r in d.get("main_results", [])]
        return cls(
            problem=d.get("problem", ""),
            motivation=d.get("motivation", ""),
            key_insight=d.get("key_insight", ""),
            architecture_overview=d.get("architecture_overview", ""),
            components=components,
            formulas=formulas,
            architecture_figure=d.get("architecture_figure"),
            design_rationale=d.get("design_rationale"),
            related_work_context=d.get("related_work_context"),
            intuitive_analogy=d.get("intuitive_analogy"),
            training_data=d.get("training_data", ""),
            data_engineering=d.get("data_engineering"),
            training_stages=d.get("training_stages", []),
            loss_functions=d.get("loss_functions", []),
            optimizer=d.get("optimizer", ""),
            training_procedure=d.get("training_procedure", ""),
            inference_procedure=d.get("inference_procedure", ""),
            post_processing=d.get("post_processing"),
            deployment_architecture=d.get("deployment_architecture"),
            deployment_value=d.get("deployment_value"),
            field_evolution=d.get("field_evolution"),
            core_question=d.get("core_question"),
            evaluation_setup=d.get("evaluation_setup"),
            industry_comparison=d.get("industry_comparison", []),
            main_results=main_results,
            ablation_results=d.get("ablation_results", []),
            qualitative_results=d.get("qualitative_results"),
            contributions=d.get("contributions", []),
            limitations=d.get("limitations", []),
            synthesis=d.get("synthesis"),
        )


@dataclass
class Paper:
    id: str
    arxiv_id: Optional[str] = None
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int = 0
    abstract: str = ""
    full_text: Optional[str] = None
    citation_count: int = 0
    url: str = ""
    source: str = ""  # "arxiv", "semantic_scholar", "pdf_file", "openalex"
    reference_ids: list[str] = field(default_factory=list)
    structured: Optional[StructuredUnderstanding] = None
    user_description: str = ""

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "abstract": self.abstract,
            "full_text": self.full_text,
            "citation_count": self.citation_count,
            "url": self.url,
            "source": self.source,
            "reference_ids": self.reference_ids,
            "user_description": self.user_description,
        }
        if self.structured is not None:
            d["structured"] = self.structured.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Paper":
        struct = d.get("structured")
        return cls(
            id=d.get("id", ""),
            arxiv_id=d.get("arxiv_id"),
            title=d.get("title", ""),
            authors=d.get("authors", []),
            year=d.get("year", 0),
            abstract=d.get("abstract", ""),
            full_text=d.get("full_text"),
            citation_count=d.get("citation_count", 0),
            url=d.get("url", ""),
            source=d.get("source", ""),
            reference_ids=d.get("reference_ids", []),
            structured=StructuredUnderstanding.from_dict(struct) if struct else None,
            user_description=d.get("user_description", ""),
        )


@dataclass
class Reference:
    paper_id: str
    paper_title: str = ""
    context: str = ""
    citation_type: CitationType = CitationType.NOT_CLASSIFIED
    is_key_reference: bool = False

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "context": self.context,
            "citation_type": self.citation_type.value,
            "is_key_reference": self.is_key_reference,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Reference":
        ct = d.get("citation_type", "not_classified")
        if isinstance(ct, str):
            try:
                ct = CitationType(ct)
            except ValueError:
                ct = CitationType.NOT_CLASSIFIED
        return cls(
            paper_id=d.get("paper_id", ""),
            paper_title=d.get("paper_title", ""),
            context=d.get("context", ""),
            citation_type=ct,
            is_key_reference=d.get("is_key_reference", False),
        )
