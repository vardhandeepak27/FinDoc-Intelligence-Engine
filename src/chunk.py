import re


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50):
    if not text:
        return []

    if chunk_size <= overlap:
        raise ValueError("Chunk size must be greater than overlap")

    # Split text into sentences while preserving their content.
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_sentences = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        extra_length = len(sentence) + (1 if current_sentences else 0)

        if current_sentences and current_length + extra_length > chunk_size:
            chunks.append(" ".join(current_sentences))

            # Keep complete sentences for overlap.
            overlap_sentences = []
            overlap_length = 0

            for previous in reversed(current_sentences):
                added_length = len(previous) + (
                    1 if overlap_sentences else 0
                )

                if overlap_length + added_length > overlap:
                    break

                overlap_sentences.insert(0, previous)
                overlap_length += added_length

            current_sentences = overlap_sentences
            current_length = overlap_length

        if current_sentences:
            current_length += 1

        current_sentences.append(sentence)
        current_length += len(sentence)

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks
