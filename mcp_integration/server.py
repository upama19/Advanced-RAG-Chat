from pathlib import Path
import pickle
import faiss

from mcp.server.fastmcp import FastMCP

from embeddings.build_index import build_index_for_pdf
from generation.rag_answer import answer_question

mcp = FastMCP("pdf-rag-chat")

ACTIVE_DOCUMENT = {
    "name": None,
    "path": None,
}


@mcp.tool()
def index_pdf(pdf_path: str) -> str:
    """
    Index a local PDF file and make it the active document.
    """
    path = Path(pdf_path)

    if not path.exists():
        return f"File not found: {pdf_path}"

    if path.suffix.lower() != ".pdf":
        return "Only PDF files are supported."

    index, chunks = build_index_for_pdf(path)

    faiss.write_index(index, "embeddings/faiss.index")
    with open("embeddings/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    ACTIVE_DOCUMENT["name"] = path.name
    ACTIVE_DOCUMENT["path"] = str(path.resolve())

    return f"Indexed {path.name} successfully."


@mcp.tool()
def ask_document(question: str) -> str:
    """
    Ask a question about the currently active PDF document.
    """
    if ACTIVE_DOCUMENT["name"] is None:
        return "No active document. Run index_pdf first."

    answer, _ = answer_question(question, active_document=ACTIVE_DOCUMENT["name"])
    return answer


@mcp.resource("document://active")
def get_active_document() -> str:
    """
    Return information about the current active document.
    """
    if ACTIVE_DOCUMENT["name"] is None:
        return "No active document."

    return (
        f"Active document: {ACTIVE_DOCUMENT['name']}\n"
        f"Path: {ACTIVE_DOCUMENT['path']}"
    )


if __name__ == "__main__":
    mcp.run()
