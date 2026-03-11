You are a senior Python engineer.

Build a minimal MVP tool called "Paper Context Tool".

Goal:
Automatically organize research papers into:
Field → Branch → Key Papers → Timeline

The tool should:

1. Load research papers from a JSON file
2. Generate embeddings using an OpenAI-compatible Embedding API
3. Match papers to branches using cosine similarity
4. Identify key papers using citation count
5. Generate a timeline sorted by year
6. Export the result as a Markdown file

Tech stack:
- Python
- OpenAI-compatible embeddings (configurable provider)
- numpy
- scikit-learn
- requests
- simple JSON storage

Project structure:

paper_context_tool/
    main.py
    config.py
    embedding.py
    classifier.py
    timeline.py
    markdown_export.py
    data/
        papers.json
        branches.json
    output/
        field_map.md

Data format:

papers.json
[
  {
    "title": "...",
    "abstract": "...",
    "year": 2022,
    "citation_count": 320,
    "link": "https://arxiv.org/abs/..."
  }
]

branches.json
[
  {
    "field": "BEV Perception",
    "branch": "Camera-based BEV",
    "seed_papers": [
        "Lift, Splat, Shoot",
        "BEVDet"
    ]
  }
]

Functional requirements:

Step 1:
Generate embeddings for paper title + abstract.

Step 2:
Compute similarity between paper embedding and branch seed embeddings.

Step 3:
Assign the paper to the most similar branch.

Step 4:
Inside each branch:
- sort papers by citation_count
- choose top 5 as key papers

Step 5:
Generate timeline sorted by year.

Step 6:
Export Markdown file.

Markdown format:

# Field: {field}

## Branch: {branch}

Key Papers
- PaperName (Year) [link]

Timeline
Year → Paper

Write clean modular Python code.

Include a main() function to run the full pipeline.

Add clear comments in the code.