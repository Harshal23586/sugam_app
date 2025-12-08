def calculate_rag_score(document_text: str) -> float:
    """
    Dummy scoring function.
    Replace with real scoring formula.
    """
    if not document_text:
        return 0.0
    return round(len(document_text.split()) / 100, 2)
