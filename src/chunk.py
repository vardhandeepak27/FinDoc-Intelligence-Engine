from ingest import extract_text

def chunk_text(text: str, chunk_size: int = 50, overlap: int = 20):

    if not text:
        return []
    
    if chunk_size <= overlap:
        raise ValueError("Chunk size must be greater than overlap")
    
    chunks = []

    start = 0
    
    while(start<len(text)):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start = end - overlap

    return chunks


