"""Tests for V3 paper data models."""
import pytest
from paper import (
    Paper, StructuredUnderstanding, Reference, CitationType,
    Component, Formula, Result,
)


class TestPaper:
    def test_create_minimal_paper(self):
        p = Paper(id="test:1")
        assert p.id == "test:1"
        assert p.title == ""
        assert p.authors == []
        assert p.structured is None

    def test_create_full_paper(self):
        p = Paper(
            id="test:2",
            arxiv_id="2001.00001",
            title="Test Paper",
            authors=["Alice", "Bob"],
            year=2022,
            abstract="An abstract.",
            citation_count=50,
            url="https://arxiv.org/abs/2001.00001",
            source="arxiv",
            user_description="Focus on method",
        )
        assert p.title == "Test Paper"
        assert len(p.authors) == 2
        assert p.year == 2022
        assert p.user_description == "Focus on method"

    def test_to_dict_and_back(self):
        paper = Paper(
            id="test:3",
            title="Roundtrip",
            year=2023,
            abstract="Test roundtrip.",
            source="pdf_file",
        )
        d = paper.to_dict()
        restored = Paper.from_dict(d)
        assert restored.title == paper.title
        assert restored.id == paper.id
        assert restored.year == paper.year
        assert restored.structured is None

    def test_to_dict_with_structured(self):
        paper = Paper(
            id="test:4",
            title="With Structure",
            year=2023,
            source="arxiv",
            structured=StructuredUnderstanding(
                problem="A problem",
                key_insight="An insight",
            ),
        )
        d = paper.to_dict()
        assert d["structured"]["problem"] == "A problem"
        restored = Paper.from_dict(d)
        assert restored.structured is not None
        assert restored.structured.problem == "A problem"


class TestStructuredUnderstanding:
    def test_empty_understanding(self):
        su = StructuredUnderstanding()
        assert su.problem == ""
        assert su.components == []
        assert su.formulas == []
        assert su.contributions == []

    def test_full_understanding(self):
        su = StructuredUnderstanding(
            problem="P",
            motivation="M",
            key_insight="K",
            architecture_overview="A",
            components=[Component(name="C1", purpose="P1")],
            formulas=[Formula(name="F1", explanation="E1", significance="S1")],
            main_results=[Result(dataset="D", metric="M", value="V")],
            loss_functions=["L1"],
            optimizer="Adam",
            contributions=["C1"],
            limitations=["L1"],
        )
        assert len(su.components) == 1
        assert su.components[0].name == "C1"
        assert len(su.formulas) == 1
        assert len(su.main_results) == 1
        assert su.contributions == ["C1"]

    def test_to_dict_and_back(self):
        su = StructuredUnderstanding(
            problem="Test problem",
            components=[
                Component(name="Encoder", purpose="Encode", details="CNN"),
                Component(name="Decoder", purpose="Decode", referenced_figure="Fig 1"),
            ],
            formulas=[Formula(name="Loss", latex="L", explanation="Loss fn", significance="Key")],
            main_results=[Result(dataset="D", metric="mAP", value="50", comparison="+2")],
        )
        d = su.to_dict()
        restored = StructuredUnderstanding.from_dict(d)
        assert restored.problem == su.problem
        assert len(restored.components) == 2
        assert restored.components[0].name == "Encoder"
        assert restored.components[1].referenced_figure == "Fig 1"
        assert len(restored.formulas) == 1
        assert restored.formulas[0].latex == "L"
        assert len(restored.main_results) == 1
        assert restored.main_results[0].dataset == "D"

    def test_from_dict_missing_fields(self):
        su = StructuredUnderstanding.from_dict({})
        assert su.problem == ""
        assert su.components == []
        assert su.formulas == []

    def test_from_dict_malformed_lists(self):
        su = StructuredUnderstanding.from_dict({
            "components": [{"name": "C", "purpose": "P"}],
            "formulas": [{"name": "F"}],
            "main_results": [{"dataset": "D", "metric": "M", "value": "V"}],
        })
        assert len(su.components) == 1
        assert su.components[0].details is None


class TestReference:
    def test_create_reference(self):
        ref = Reference(
            paper_id="test:ref1",
            paper_title="Ref Paper",
            context="cited in method section",
            citation_type=CitationType.SUPPORTING,
            is_key_reference=True,
        )
        assert ref.paper_id == "test:ref1"
        assert ref.citation_type == CitationType.SUPPORTING
        assert ref.is_key_reference is True

    def test_reference_roundtrip(self):
        ref = Reference(
            paper_id="test:ref2",
            paper_title="Roundtrip Ref",
            context="context text",
            citation_type=CitationType.FOUNDATIONAL,
            is_key_reference=False,
        )
        d = ref.to_dict()
        restored = Reference.from_dict(d)
        assert restored.paper_id == ref.paper_id
        assert restored.citation_type == CitationType.FOUNDATIONAL

    def test_reference_defaults(self):
        ref = Reference(paper_id="test:default")
        assert ref.citation_type == CitationType.NOT_CLASSIFIED
        assert ref.is_key_reference is False

    def test_reference_from_dict_unknown_type(self):
        ref = Reference.from_dict({
            "paper_id": "test:unknown",
            "citation_type": "bogus_type",
        })
        assert ref.citation_type == CitationType.NOT_CLASSIFIED


class TestCitationType:
    def test_values(self):
        assert CitationType.SUPPORTING.value == "supporting"
        assert CitationType.CONTRASTING.value == "contrasting"
        assert CitationType.FOUNDATIONAL.value == "foundational"
        assert CitationType.RELATED.value == "related_work"


class TestComponent:
    def test_component_with_all_fields(self):
        c = Component(
            name="Backbone",
            purpose="Feature extraction",
            details="ResNet-50, output stride 16",
            referenced_figure="Figure 2(a)",
        )
        assert c.name == "Backbone"
        assert c.details == "ResNet-50, output stride 16"
        assert c.referenced_figure == "Figure 2(a)"


class TestFormula:
    def test_formula_with_latex(self):
        f = Formula(
            name="Cross-Attention",
            latex="QK^T/\\sqrt{d}",
            explanation="Scaled dot-product attention",
            significance="Enables efficient multi-camera fusion",
        )
        assert f.latex is not None
        assert "QK" in f.latex


class TestResult:
    def test_result_basic(self):
        r = Result(dataset="nuScenes", metric="NDS", value="51.7")
        assert r.dataset == "nuScenes"
        assert r.comparison is None
