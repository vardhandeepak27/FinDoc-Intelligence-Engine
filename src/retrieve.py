import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import extract_text
from chunk import chunk_text

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Build index
text = extract_text("data/sample.pdf")
chunks = chunk_text(text, chunk_size=300, overlap=50)

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i}:")
    print(repr(chunk))

embeddings = model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("Total chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)
print("Index vectors:", index.ntotal)


def retrieve(query, top_k=5):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    print(f"\nQuery: {query}\n")

    for rank in range(len(indices[0])):
        idx = int(indices[0][rank])

        if idx == -1:
            continue

        distance = float(distances[0][rank])

        result = {
            "chunk_id": idx,
            "text": chunks[idx],
            "distance": distance
        }

        results.append(result)

        print(f"Result {rank + 1} | Distance: {distance:.4f}")
        print(chunks[idx])
        print("-" * 50)

    return results



results = retrieve("What is the work from home policy?")

print("\nReturned results:")
print(results)