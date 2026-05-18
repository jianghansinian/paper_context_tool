from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Small paper set (existing — used by key_paper, embedding tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_papers():
    return [
        {
            "title": "Paper A: Method for Object Detection",
            "abstract": "A novel approach using convolutional networks for real-time object detection.",
            "year": 2020,
            "citation_count": 500,
            "link": "https://arxiv.org/abs/2001.00001",
        },
        {
            "title": "Paper B: Improved Object Detection with Attention",
            "abstract": "Extending detection methods with transformer attention mechanisms.",
            "year": 2021,
            "citation_count": 300,
            "link": "https://arxiv.org/abs/2101.00001",
        },
        {
            "title": "Paper C: Semantic Segmentation Review",
            "abstract": "A comprehensive survey of semantic segmentation techniques.",
            "year": 2019,
            "citation_count": 1000,
            "link": "https://arxiv.org/abs/1901.00001",
        },
        {
            "title": "Paper D: Efficient Segmentation with Mobile Networks",
            "abstract": "Lightweight architectures for real-time semantic segmentation.",
            "year": 2022,
            "citation_count": 150,
            "link": "https://arxiv.org/abs/2201.00001",
        },
    ]


@pytest.fixture
def sample_embeddings():
    np.random.seed(42)
    return np.random.randn(20, 1024).astype(float)


# ---------------------------------------------------------------------------
# Larger paper set for relevance filter testing
# ---------------------------------------------------------------------------

@pytest.fixture
def mixed_relevance_papers():
    """Papers with some clearly relevant and some clearly irrelevant entries."""
    return [
        {
            "title": "BEVFormer: Bird's Eye View Object Detection with Transformers",
            "abstract": "We propose BEVFormer, a transformer-based architecture for bird's eye view perception in autonomous driving.",
            "year": 2022,
            "citation_count": 800,
            "link": "https://arxiv.org/abs/2203.17270",
        },
        {
            "title": "Lift, Splat, Shoot: Encoding Images for BEV Perception",
            "abstract": "A method for lifting camera images into a bird's eye view representation for autonomous driving.",
            "year": 2020,
            "citation_count": 900,
            "link": "https://arxiv.org/abs/2006.12345",
        },
        {
            "title": "Faces of Inequality: Gender, Class and Welfare States",
            "abstract": "An analysis of inequality patterns across different welfare state regimes.",
            "year": 2000,
            "citation_count": 500,
            "link": "https://doi.org/10.1093/sp/7.2.127",
        },
        {
            "title": "BEVDepth: Acquisition of Reliable Depth for Multi-View 3D Object Detection",
            "abstract": "Learning depth estimation for bird's eye view detection in autonomous driving scenarios.",
            "year": 2023,
            "citation_count": 300,
            "link": "https://doi.org/10.1609/aaai.v37i2.25233",
        },
        {
            "title": "The Organization of Behavior: A Neuropsychological Theory",
            "abstract": "A classic textbook on neuropsychology and behavior organization.",
            "year": 1950,
            "citation_count": 50000,
            "link": "https://doi.org/10.1001/jama.1950.02910470083028",
        },
        {
            "title": "BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View",
            "abstract": "A unified bird's eye view representation for multi-sensor fusion in autonomous driving.",
            "year": 2023,
            "citation_count": 400,
            "link": "https://doi.org/10.1109/icra48891.2023.10160968",
        },
        {
            "title": "Three faces of biofilms: a microbial lifestyle",
            "abstract": "Review of biofilm formation and its role in microbial communities.",
            "year": 2021,
            "citation_count": 200,
            "link": "https://doi.org/10.1038/s41522-021-00251-2",
        },
    ]


# ---------------------------------------------------------------------------
# Mock LLM client
# ---------------------------------------------------------------------------

class MockChatCompletionMessage:
    def __init__(self, content: str):
        self.content = content


class MockChatCompletionChoice:
    def __init__(self, content: str):
        self.message = MockChatCompletionMessage(content)


class MockChatCompletion:
    def __init__(self, content: str):
        self.choices = [MockChatCompletionChoice(content)]


def _mock_create(content: str):
    def create(*args, **kwargs):
        return MockChatCompletion(content)
    return create


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    return client


@pytest.fixture
def relevance_filter_response():
    """Canned LLM response for relevance filtering — marks BEV papers relevant, others irrelevant."""
    return """[
  {"index": 0, "judgment": "relevant", "reason": "BEV perception for autonomous driving"},
  {"index": 1, "judgment": "relevant", "reason": "BEV perception method"},
  {"index": 2, "judgment": "irrelevant", "reason": "Sociology paper, not related to autonomous driving"},
  {"index": 3, "judgment": "relevant", "reason": "BEV depth estimation for autonomous driving"},
  {"index": 4, "judgment": "irrelevant", "reason": "Neuropsychology textbook"},
  {"index": 5, "judgment": "relevant", "reason": "Multi-sensor fusion BEV for autonomous driving"},
  {"index": 6, "judgment": "irrelevant", "reason": "Microbiology paper"}
]"""


@pytest.fixture
def branch_analysis_response():
    return """{
  "branch_name": "Camera-based BEV Perception",
  "narrative": "This branch focuses on using camera inputs to construct bird's eye view representations.",
  "key_papers": [
    {"title": "Lift, Splat, Shoot", "year": 2020, "link": "https://arxiv.org/abs/2006.12345",
     "significance": "First to propose", "importance_rank": 1}
  ],
  "paradigm_shifts": [],
  "technical_forks": []
}"""


@pytest.fixture
def evolution_analysis_response():
    return """{
  "overview": "The field evolved from early camera-only approaches to multi-modal fusion.",
  "cross_branch_relationships": [
    {"branches": ["Camera-based", "Multi-modal"], "relationship": "precursor_to",
     "description": "Camera-only approaches laid the foundation for multi-modal fusion."}
  ],
  "temporal_ordering": ["Camera-based BEV", "Multi-modal Fusion BEV"]
}"""


@pytest.fixture
def validation_response():
    return """{
  "quality_score": 8,
  "issues": [
    {"severity": "warning", "description": "Small cluster size", "location": "Branch: LiDAR-based BEV"}
  ],
  "missing_topics": [],
  "suggested_improvements": ["Consider adding temporal fusion approaches"]
}"""


# ---------------------------------------------------------------------------
# V3 fixtures: Paper models, structured analysis responses, mock SS API
# ---------------------------------------------------------------------------

@pytest.fixture
def seed_paper_v3():
    """A representative seed paper for V3 pipeline tests."""
    from paper import Paper
    return Paper(
        id="ss:seed123",
        arxiv_id="2203.17270",
        title="BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers",
        authors=["Zhiqi Li", "Wenhai Wang", "Hao Li", "Enze Xie", "Chonghao Sima", "Tong Lu", "Yu Qiao", "Jifeng Dai"],
        year=2022,
        abstract="We propose BEVFormer, a new framework for learning unified BEV representations with spatiotemporal transformers. BEVFormer supports autonomous driving perception tasks including 3D object detection and map segmentation.",
        citation_count=800,
        url="https://arxiv.org/abs/2203.17270",
        source="arxiv",
        full_text="[Page 1]\nBEVFormer: Learning Bird's-Eye-View Representation\n\nAbstract\nWe propose BEVFormer...\n\n1. Introduction\nBird's eye view perception is critical for autonomous driving...\n\n2. Related Work\n2.1 Vision-based BEV Perception\nLift-Splat-Shoot [1] projects image features to BEV...\n\n3. Method\n3.1 Overall Architecture\nFigure 1 shows the overall architecture. BEVFormer consists of three key components: BEV Queries, Spatial Cross-Attention, and Temporal Self-Attention.\n\n3.2 BEV Queries\nBEV queries are learnable parameters Q ∈ R^{H×W×C} that represent the BEV grid. Each query corresponds to a spatial location in the BEV plane.\n\n3.3 Spatial Cross-Attention\nSCA(Q, F) = DeformAttn(Q, p, F) where p projects query positions to image features.\n\n3.4 Temporal Self-Attention\nTSA aligns historical BEV features B_{t-1} with current queries Q_t using ego-motion compensation.\n\n4. Experiments\n4.1 3D Object Detection\nOn nuScenes test set, BEVFormer achieves 51.7% NDS and 41.6% mAP.\n\n4.2 Ablation Study\nTable 2 shows that removing temporal self-attention drops NDS by 2.1%.\n\n5. Conclusion\nWe introduced BEVFormer, a spatiotemporal transformer for BEV perception.\n\nReferences\n[1] J. Philion and S. Fidler. Lift, Splat, Shoot. ECCV 2020.",
        user_description="I want to understand the temporal attention mechanism",
    )


@pytest.fixture
def structured_analysis_response():
    """Canned LLM response for structured paper analysis."""
    return """{
  "problem": "3D object detection from multi-camera images requires unified BEV representation",
  "motivation": "Existing BEV methods either rely on dense depth estimation or lack temporal modeling",
  "key_insight": "Use learnable BEV queries with spatiotemporal transformers to aggregate multi-camera and temporal information",
  "architecture_overview": "Encoder-decoder structure where BEV queries attend to multi-camera image features via spatial cross-attention and incorporate temporal information via temporal self-attention",
  "architecture_figure": "Figure 1 shows the encoder-decoder pipeline: multi-camera images → backbone CNN → FPN → BEV queries with spatial cross-attention + temporal self-attention → task heads",
  "components": [
    {"name": "BEV Queries", "purpose": "Learnable grid parameters representing BEV locations", "details": "Q in R^{200×200×256}, pre-defined 3D reference points", "referenced_figure": "Figure 2(a)"},
    {"name": "Spatial Cross-Attention", "purpose": "Aggregate multi-camera image features into BEV queries", "details": "Deformable attention with 4 sampling points per query, 8 attention heads", "referenced_figure": "Figure 2(b)"},
    {"name": "Temporal Self-Attention", "purpose": "Align and fuse historical BEV features with current frame", "details": "Ego-motion compensation via pose transformation, then deformable self-attention", "referenced_figure": "Figure 2(c)"}
  ],
  "formulas": [
    {"name": "Spatial Cross-Attention", "latex": "SCA(Q_p, F_t) = \\\\sum_{k=1}^{K} A_{pk} \\\\cdot W F_t(p + \\\\Delta p_k)", "explanation": "Each BEV query Q_p attends to K sampling points in the multi-scale image features F_t", "significance": "Core mechanism for projecting 2D image features into 3D BEV space"},
    {"name": "Temporal Self-Attention", "latex": "TSA(Q_t, B_{t-1}) = DeformAttn(Q_t, B_{t-1}')", "explanation": "Aligns historical BEV B_{t-1} with current queries Q_t via ego-motion transform", "significance": "Enables temporal feature fusion, critical for velocity estimation and occlusion handling"}
  ],
  "training_data": "nuScenes dataset — 700 training scenes, 150 validation scenes",
  "loss_functions": ["Focal Loss for classification", "L1 Loss for bounding box regression"],
  "optimizer": "AdamW, initial LR 2e-4, cosine schedule, batch size 16",
  "training_procedure": "Backbone pretrained on FCOS3D, then end-to-end. Data augmentation: random flip, rotation ±22.5°, scaling 0.5-1.5. BEVFormer-Base ~20 epochs on 8× V100.",
  "inference_procedure": "Multi-camera images → backbone → FPN → BEVFormer encoder (6 layers) → task-specific heads (bbox, cls) → NMS",
  "post_processing": "NMS with IoU threshold 0.6",
  "main_results": [
    {"dataset": "nuScenes val", "metric": "NDS", "value": "51.7", "comparison": "+3.5 vs DETR3D (48.2)"},
    {"dataset": "nuScenes val", "metric": "mAP", "value": "41.6", "comparison": "+5.1 vs PETR (36.5)"}
  ],
  "ablation_results": [
    "Removing Temporal Self-Attention drops NDS by 2.1%",
    "Removing Spatial Cross-Attention (dense depth) drops NDS by 3.4%",
    "4 sampling points in deformable attention is optimal"
  ],
  "qualitative_results": "BEVFormer correctly detects heavily occluded vehicles at long range (>50m) where DETR3D fails",
  "contributions": [
    "First spatiotemporal transformer for unified BEV representation",
    "BEV Queries with spatial cross-attention for efficient multi-camera fusion",
    "Temporal self-attention with ego-motion alignment"
  ],
  "limitations": [
    "Requires ego-motion data for temporal alignment",
    "High computational cost (320ms inference on V100)",
    "Only evaluated on nuScenes — generalization to other datasets not shown"
  ]
}"""


@pytest.fixture
def route_analysis_response():
    """Canned LLM response for technical route grouping."""
    return """{
  "overview": "The field of BEV perception has three main technical approaches: LSS-based depth projection, Transformer-based query projection, and MLP-based implicit projection.",
  "branches": [
    {
      "name": "Transformer-based BEV Projection",
      "description": "Uses learnable queries with cross-attention to project image features to BEV space. No explicit depth estimation.",
      "paper_indices": [0, 2],
      "is_mainstream": true,
      "common_technical_tags": ["transformer-encoder", "cross-attention", "learnable-queries", "no-explicit-depth"]
    },
    {
      "name": "LSS-based Depth Projection",
      "description": "Estimates per-pixel depth distribution and splats image features into BEV voxels.",
      "paper_indices": [1],
      "is_mainstream": true,
      "common_technical_tags": ["depth-estimation", "voxel-splatting", "explicit-depth", "CNN-backbone"]
    }
  ]
}"""


@pytest.fixture
def comparison_response():
    """Canned LLM response for comparative analysis."""
    return """{
  "comparison_matrix": [
    {
      "dimension": "Depth Estimation",
      "seed_paper": "Implicit via cross-attention — no explicit depth",
      "mainstream_approach": "Explicit per-pixel depth prediction (LSS-based)",
      "advantage": "Avoids depth estimation errors; more robust to calibration noise"
    },
    {
      "dimension": "Temporal Modeling",
      "seed_paper": "Temporal self-attention with ego-motion alignment",
      "mainstream_approach": "No temporal modeling or simple concatenation",
      "advantage": "Better velocity estimation and occlusion handling"
    }
  ],
  "narrative": "BEVFormer departs from the dominant LSS-based paradigm by using attention instead of depth projection. Its key innovation is the spatiotemporal transformer, which was missing in prior work.",
  "unique_positioning": "BEVFormer is the first method to unify spatial and temporal attention in a single BEV representation."
}"""


@pytest.fixture
def mock_ss_references_response():
    """Canned Semantic Scholar references API response."""
    return {
        "data": [
            {
                "contexts": ["Lift-Splat-Shoot projects image features to BEV using depth estimation"],
                "intents": ["methodology", "background"],
                "isInfluential": True,
                "citedPaper": {
                    "paperId": "ss:lss001",
                    "title": "Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D",
                    "authors": [{"name": "Jonah Philion"}, {"name": "Sanja Fidler"}],
                    "year": 2020,
                    "citationCount": 900,
                    "externalIds": {"ArXiv": "2008.05700"},
                    "abstract": "We propose a method for encoding images into a bird's-eye-view representation.",
                    "url": "https://arxiv.org/abs/2008.05700"
                }
            },
            {
                "contexts": ["DETR3D uses 3D object queries to detect objects in multi-view images"],
                "intents": ["methodology"],
                "isInfluential": True,
                "citedPaper": {
                    "paperId": "ss:detr3d001",
                    "title": "DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries",
                    "authors": [{"name": "Yue Wang"}, {"name": "Vitor Guizilini"}],
                    "year": 2021,
                    "citationCount": 700,
                    "externalIds": {"ArXiv": "2110.06922"},
                    "abstract": "DETR3D is a 3D object detection method.",
                    "url": "https://arxiv.org/abs/2110.06922"
                }
            },
            {
                "contexts": [],
                "intents": [],
                "isInfluential": False,
                "citedPaper": {
                    "paperId": "ss:nonauto001",
                    "title": "Faces of Inequality: Gender, Class and Welfare States",
                    "authors": [{"name": "Some Author"}],
                    "year": 2000,
                    "citationCount": 500,
                    "externalIds": {},
                    "abstract": "Not related.",
                    "url": ""
                }
            }
        ]
    }


@pytest.fixture
def mock_ss_citations_response():
    """Canned Semantic Scholar citations API response."""
    return {
        "data": [
            {
                "citingPaper": {
                    "paperId": "ss:bevfusion001",
                    "title": "BEVFusion: Multi-Task Multi-Sensor Fusion with Unified BEV",
                    "authors": [{"name": "Zhijian Liu"}, {"name": "Haotian Tang"}],
                    "year": 2023,
                    "citationCount": 400,
                    "externalIds": {"ArXiv": "2205.13790"},
                    "abstract": "BEVFusion unifies multi-modal sensor data...",
                    "url": "https://arxiv.org/abs/2205.13790"
                }
            },
            {
                "citingPaper": {
                    "paperId": "ss:bevpool001",
                    "title": "BEVPoolv2: A Cutting-edge Implementation of BEVDet",
                    "authors": [{"name": "Junjie Huang"}, {"name": "Guan Huang"}],
                    "year": 2022,
                    "citationCount": 100,
                    "externalIds": {},
                    "abstract": "BEVPoolv2 optimizes BEV pooling...",
                    "url": ""
                }
            }
        ]
    }


@pytest.fixture
def sample_paper_v3():
    """A simple paper fixture for unit tests."""
    from paper import Paper
    return Paper(
        id="test:paper1",
        arxiv_id="2001.00001",
        title="Test Paper: A Novel Method for BEV Perception",
        authors=["Test Author"],
        year=2021,
        abstract="A novel method for BEV perception.",
        citation_count=100,
        url="https://arxiv.org/abs/2001.00001",
        source="arxiv",
        full_text="Introduction\nThis paper proposes a novel method.\n\nMethod\nOur architecture consists of encoder and decoder.\n\nExperiments\nResults show improvement.",
    )


@pytest.fixture
def sample_structured_understanding():
    """A pre-built StructuredUnderstanding for testing."""
    from paper import StructuredUnderstanding, Component, Formula, Result
    return StructuredUnderstanding(
        problem="Test problem",
        motivation="Test motivation",
        key_insight="Test insight",
        architecture_overview="Test architecture",
        components=[
            Component(name="Encoder", purpose="Extract features", details="CNN backbone"),
            Component(name="Decoder", purpose="Generate output", details="Transformer decoder"),
        ],
        formulas=[
            Formula(name="Loss", latex="L = L1 + L2", explanation="Combined loss", significance="Main training objective"),
        ],
        architecture_figure="Figure 1 shows the pipeline.",
        training_data="Test dataset",
        loss_functions=["L1 Loss", "L2 Loss"],
        optimizer="Adam, lr=1e-4",
        training_procedure="Train end-to-end for 100 epochs.",
        inference_procedure="Forward pass through encoder and decoder.",
        post_processing="NMS with threshold 0.5",
        main_results=[
            Result(dataset="TestSet", metric="mAP", value="42.0", comparison="+2.0 vs baseline"),
        ],
        ablation_results=["Removing decoder drops 5.0 mAP"],
        qualitative_results="Visual results show improvement.",
        contributions=["Novel encoder-decoder architecture"],
        limitations=["Only tested on one dataset"],
    )
