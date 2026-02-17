from pypdf import PdfReader
from pathlib import Path


def load_pdf(pdf_path: Path):
    reader = PdfReader(pdf_path)
    pages = []

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({"page_number": page_idx + 1, "text": text})

    return pages


if __name__ == "__main__":
    pdf_path = Path("data/Unet.pdf")
    pages = load_pdf(pdf_path)

    print(f"Total pages loaded: {len(pages)}")
    print("\n --- SAMPLE TEXT ---\n")
    print(pages[0]["text"][:500])
