from pathlib import Path
from ingestion.load_docs import load_pdf
from ingestion.chunker import semantic_chunk_text


def process_pdf(pdf_path: Path):
    pages = load_pdf(pdf_path)
    all_chunks = []

    source = pdf_path.name

    for page in pages:
        page_chunks = semantic_chunk_text(
            text=page["text"], source=source, page_number=page["page_number"]
        )
        all_chunks.extend(page_chunks)

    return all_chunks


if __name__ == "__main__":
    pdf_path = Path("data/Unet.pdf")
    chunks = process_pdf(pdf_path)

    print(f"Total chunks created: {len(chunks)}")
    print("\n--- SAMPLE CHUNK ---\n")
    print(chunks[0])
