import numpy as np
import pytest


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
