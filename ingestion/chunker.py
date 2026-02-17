import re


def split_into_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunk_text(text, source, page_number, max_chars=800, overlap_sentences=2):
    sentences = split_into_sentences(text)

    chunks = []
    current_chunk = []
    current_length = 0
    chunk_index = 0

    for sentence in sentences:
        if current_length + len(sentence) > max_chars:
            chunks_text = "".join(current_chunk)

            chunks.append({
                "id" : f"{source}_page_{page_number}_chunk_{chunk_index}",
                "text" : chunks_text,
                "metadata" : {
                    "source": source,
                    "page_number": page_number,
                    "chunk_index": chunk_index
                }
            })
            chunk_index += 1
            current_chunk = current_chunk[-overlap_sentences:]
            current_length = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += len(sentence)

    if current_chunk:
        chunks_text = "".join(current_chunk)
        chunks.append({
            "id" : f"{source}_page_{page_number}_chunk_{chunk_index}",
            "text" : chunks_text,
            "metadata" : {
                "source": source,
                "page_number": page_number,
                "chunk_index": chunk_index
            }
        })

    return chunks


if __name__ == "__main__":
    sample_text = (
        "U-Net is a convolutional neural network. "
        "It is widely used for biomedical image segmentation. "
        "The architecture consists of an encoder and decoder. "
        "Skip connections help preserve spatial information."
    )

    chunks = semantic_chunk_text(sample_text, max_chars=100)

    for i, c in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(c)
