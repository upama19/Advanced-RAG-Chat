def decide_tool(user_input):
    text = user_input.lower()

    # Explicit document-based intent
    doc_keywords = [
        "paper",
        "pdf",
        "this research",
        "this paper",
        "experiment",
        "results",
        "evaluation",
        "methodology",
        "final",
        "conclusion",
    ]

    if any(k in text for k in doc_keywords):
        return "retrieve"

    if "summarize" in text:
        return "retrieve"

    if "explain" in text or "what is" in text or "how does" in text:
        return "retrieve"

    return "direct"
