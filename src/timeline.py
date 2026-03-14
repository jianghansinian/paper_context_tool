def build_timeline(papers):
    """Sort papers chronologically for timeline view."""
    papers_sorted = sorted(
        papers,
        key=lambda paper: paper["year"],
    )
    return papers_sorted
