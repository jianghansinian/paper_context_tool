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


# ═══════════════════════════════════════════════════════════════════
# V4 data models — Research Narrative Engine
# ═══════════════════════════════════════════════════════════════════


@dataclass
class Claim:
    """A paper's core assertion — the atomic unit of V4 evolution modeling.

    Claim is NOT a method description. It is a falsifiable judgment about
    what works, what is better, and why.

    Good: "Sparse queries can match dense BEV accuracy at 40% lower FLOPs"
    Bad:  "We propose a sparse query mechanism for BEV detection"
    """
    paper_id: str
    paper_title: str
    year: int
    statement: str           # The claim itself (falsifiable assertion)
    evidence: str            # Supporting evidence (results, ablations, benchmarks)
    problem_addressed: str   # What problem does this claim address?
    claim_type: str          # "improves" | "extends" | "replaces" | "introduces"
    claim_level: str = "methodological"  # "paradigm" | "methodological" | "engineering"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Claim":
        return cls(
            paper_id=d.get("paper_id", ""),
            paper_title=d.get("paper_title", ""),
            year=d.get("year", 0),
            statement=d.get("statement", ""),
            evidence=d.get("evidence", ""),
            problem_addressed=d.get("problem_addressed", ""),
            claim_type=d.get("claim_type", "introduces"),
            claim_level=d.get("claim_level", "methodological"),
        )


@dataclass
class ClaimRelation:
    """A directed relationship between two Claims — the edge in the evolution graph.

    Built by a two-step classifier: (1) same_lineage check, (2) relation type.
    Only YES-lineage pairs proceed to relation classification; NO → parallel.
    """
    source_paper: str          # Paper title
    target_paper: str          # Paper title
    source_claim: str          # Claim statement text
    target_claim: str          # Claim statement text
    relation: str              # attack|replace|improve|extend|support|parallel
    explanation: str           # One sentence explaining why
    source_year: int
    target_year: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ClaimRelation":
        return cls(
            source_paper=d.get("source_paper", ""),
            target_paper=d.get("target_paper", ""),
            source_claim=d.get("source_claim", ""),
            target_claim=d.get("target_claim", ""),
            relation=d.get("relation", "unknown"),
            explanation=d.get("explanation", ""),
            source_year=d.get("source_year", 0),
            target_year=d.get("target_year", 0),
        )


@dataclass
class Tension:
    """A research tension — a contradiction that drives the field forward.

    Tension is the first citizen of V5 narrative. It connects what happened inside
    the field (Claims, Relations) to what the reader wants to know (why did X replace Y?).
    """
    tension: str               # Short label (e.g. "Dense vs Sparse Representation")
    description: str           # Detailed description of the contradiction
    introduced_by: list[str]   # Paper titles that first exposed this tension
    resolved_by: list[str]     # Paper titles that advanced or favored a direction
    status: str                # direction_clear | direction_forming | open
    dimension: str             # representation|geometry|system|evaluation
    domain_scope: str = ""     # e.g. "in detection/tracking", "in occupancy"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Tension":
        return cls(
            tension=d.get("tension", ""),
            description=d.get("description", ""),
            introduced_by=d.get("introduced_by", []),
            resolved_by=d.get("resolved_by", []),
            status=d.get("status", "open"),
            dimension=d.get("dimension", "system"),
            domain_scope=d.get("domain_scope", ""),
        )


@dataclass
class Direction:
    """A research direction — where the evidence points on a question.

    Direction is the CONCLUSION of a ResearchQuestion's debate. It captures
    what answer the evidence favors, how strong that evidence is, and which
    papers support or oppose it.

    Distinct from ParadigmShift: Direction is per-RQ ("the community favors
    implicit geometry"), ParadigmShift is cross-RQ ("Dense→Sparse changed
    the field's representation philosophy").
    """
    statement: str               # e.g. "Implicit geometry learning is sufficient"
    support_papers: list[str]    # Papers whose evidence supports this direction
    opposing_papers: list[str]   # Papers that challenge or complicate this direction
    confidence: str = "medium"   # "high" | "medium" | "low"
    evidence_summary: str = ""   # 1-2 sentences summarizing key evidence

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Direction":
        return cls(
            statement=d.get("statement", ""),
            support_papers=d.get("support_papers", []),
            opposing_papers=d.get("opposing_papers", []),
            confidence=d.get("confidence", "medium"),
            evidence_summary=d.get("evidence_summary", ""),
        )


@dataclass
class ParadigmShift:
    """A paradigm shift — when the field's collective belief fundamentally changed.

    Strictly distinguished from technique evolution: a paradigm shift overturns
    a core assumption; technique evolution improves within the same assumption.
    """
    shift_name: str            # e.g. "Dense BEV → Sparse Representation"
    description: str           # 2-3 sentences: what changed and why it mattered
    old_paradigm: str          # What the field believed before
    new_paradigm: str          # What the field believed after
    catalyst_papers: list[str] # Papers that triggered or crystallized this shift
    magnitude: str             # paradigm_shift | optimization | convergence
    level: str                 # research_question | method | evaluation
    dimension: str             # representation | geometry | system | evaluation
    year_range: str            # e.g. "2022-2024"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ParadigmShift":
        return cls(
            shift_name=d.get("shift_name", ""),
            description=d.get("description", ""),
            old_paradigm=d.get("old_paradigm", ""),
            new_paradigm=d.get("new_paradigm", ""),
            catalyst_papers=d.get("catalyst_papers", []),
            magnitude=d.get("magnitude", "incremental"),
            level=d.get("level", "method"),
            dimension=d.get("dimension", "system"),
            year_range=d.get("year_range", ""),
        )


@dataclass
class Phase:
    """A time period with a core contradiction — V8's narrative chapter unit.

    Phase is not a new entity — it emerges from clustering Tensions by time+theme.
    Each Phase is a chapter in the narrative, linked by causal chain:
    Phase N's unresolved_problem → Phase N+1's core_contradiction.
    """
    name: str                        # e.g. "Dense BEV Era"
    time_range: str                  # e.g. "2020-2022"
    core_contradiction: str          # 1-sentence core contradiction
    key_papers: list[str]            # 3-6 key paper titles
    core_debate: str                 # What the field was debating in this phase
    unresolved_problem: str          # → becomes next phase's motivation
    tensions: list[Tension] = field(default_factory=list)  # Tensions clustered into this phase

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tensions"] = [t.to_dict() for t in self.tensions]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Phase":
        return cls(
            name=d.get("name", ""),
            time_range=d.get("time_range", ""),
            core_contradiction=d.get("core_contradiction", ""),
            key_papers=d.get("key_papers", []),
            core_debate=d.get("core_debate", ""),
            unresolved_problem=d.get("unresolved_problem", ""),
            tensions=[Tension.from_dict(t) for t in d.get("tensions", [])],
        )


@dataclass
class ResearchQuestion:
    """A research question — V8: Phase content, not chapter title.

    RQs capture the questions the field debated. They are more stable than Tensions
    but don't make good chapter titles (break time causal chain). In V8, RQs are
    the "plot" within a Phase — each Phase may involve 1-2 RQs debated by papers.
    """
    question: str              # Full question text, e.g. "Is explicit depth supervision necessary?"
    short_name: str            # Short label, e.g. "Depth Necessity"
    description: str           # 1-2 sentences of context
    level: str                 # "field" | "paradigm" | "engineering"
    status: str                # direction_clear | direction_forming | open
    positions: list[dict]      # [{"paper": "...", "position": "...", "evidence": "..."}]
    introduced_by: list[str]   # Paper titles that raised this question
    tensions: list[Tension] = field(default_factory=list)  # Subordinate tensions within this RQ
    direction: Optional[Direction] = None  # Where the evidence points

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.direction:
            d["direction"] = self.direction.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchQuestion":
        direction = d.get("direction")
        return cls(
            question=d.get("question", ""),
            short_name=d.get("short_name", ""),
            description=d.get("description", ""),
            level=d.get("level", "paradigm"),
            status=d.get("status", "open"),
            positions=d.get("positions", []),
            introduced_by=d.get("introduced_by", []),
            tensions=[Tension.from_dict(t) for t in d.get("tensions", [])],
            direction=Direction.from_dict(direction) if direction else None,
        )


@dataclass
class NarrativeSection:
    """One chapter of the V8 narrative — organized by Phase (time period).

    V8 change: each section centers on one Phase (time-based tension cluster),
    not on ResearchQuestion. Phases are linked by causal chain:
    Phase N's unresolved_problem → Phase N+1's core_contradiction.

    The professor-teaching model: "Last week we ended with a problem: [unresolved].
    This week: [Phase name]. The core debate was..."
    """
    title: str                 # Phase name, e.g. "Phase 1: Dense BEV Era (2020-2022)"
    phase: Phase               # The Phase this section covers
    claims: list[Claim] = field(default_factory=list)
    claim_relations: list[ClaimRelation] = field(default_factory=list)
    paradigm_shifts: list[ParadigmShift] = field(default_factory=list)
    direction: Optional[Direction] = None  # Where evidence points within this phase
    narrative: str = ""        # Professor-style lecture narrative

    def to_dict(self) -> dict:
        d = {
            "title": self.title,
            "phase": self.phase.to_dict(),
            "claims": [c.to_dict() for c in self.claims],
            "claim_relations": [r.to_dict() for r in self.claim_relations],
            "paradigm_shifts": [p.to_dict() for p in self.paradigm_shifts],
            "narrative": self.narrative,
        }
        if self.direction:
            d["direction"] = self.direction.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "NarrativeSection":
        direction = d.get("direction")
        return cls(
            title=d.get("title", ""),
            phase=Phase.from_dict(d.get("phase", {})),
            claims=[Claim.from_dict(c) for c in d.get("claims", [])],
            claim_relations=[ClaimRelation.from_dict(r) for r in d.get("claim_relations", [])],
            paradigm_shifts=[ParadigmShift.from_dict(p) for p in d.get("paradigm_shifts", [])],
            direction=Direction.from_dict(direction) if direction else None,
            narrative=d.get("narrative", ""),
        )


@dataclass
class ResearchNarrative:
    """V8 output: a field's complete technical evolution story.

    Organized by Phase (NarrativeSection), linked by causal chain.
    RQs are Phase content, not chapter titles.
    """
    field_name: str
    seed_paper_id: Optional[str] = None
    overview: str = ""
    sections: list[NarrativeSection] = field(default_factory=list)
    phases: list[Phase] = field(default_factory=list)
    paradigm_shifts: list[ParadigmShift] = field(default_factory=list)
    research_questions: list[ResearchQuestion] = field(default_factory=list)
    tensions: list[Tension] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    claim_relations: list[ClaimRelation] = field(default_factory=list)
    synthesis: str = ""

    def to_dict(self) -> dict:
        return {
            "field_name": self.field_name,
            "seed_paper_id": self.seed_paper_id,
            "overview": self.overview,
            "sections": [s.to_dict() for s in self.sections],
            "phases": [p.to_dict() for p in self.phases],
            "paradigm_shifts": [p.to_dict() for p in self.paradigm_shifts],
            "research_questions": [q.to_dict() for q in self.research_questions],
            "tensions": [t.to_dict() for t in self.tensions],
            "claims": [c.to_dict() for c in self.claims],
            "claim_relations": [r.to_dict() for r in self.claim_relations],
            "synthesis": self.synthesis,
        }
