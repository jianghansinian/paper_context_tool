def build_timeline(papers):
    """Sort papers chronologically for timeline view."""
    papers_sorted = sorted(papers, key=lambda paper: int(paper.get("year", 0)))
    return [
        {
            "title": paper.get("title", ""),
            "year": int(paper.get("year", 0)),
            "link": paper.get("link", ""),
        }
        for paper in papers_sorted
    ]
