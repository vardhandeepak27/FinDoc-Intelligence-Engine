
import re


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50):
    if not text:
        return []

    if chunk_size <= overlap:
        raise ValueError("Chunk size must be greater than overlap")

    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence exceeds the target size,
        # save the current chunk first.
        if current_chunk and len(current_chunk) + len(sentence) + 1 > chunk_size:
            chunks.append(current_chunk.strip())

            # Preserve overlap from the previous chunk
            overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk = (
                current_chunk + " " + sentence
                if current_chunk
                else sentence
            )

    # Add the final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())


    return chunks


