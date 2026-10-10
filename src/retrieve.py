import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import extract_text
from chunk import chunk_text

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Build index
text = extract_text("data/sample.pdf")
chunks = chunk_text(text, chunk_size=100, overlap=20)

embeddings = model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("Total chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)
print("Index vectors:", index.ntotal)


def retrieve(query, top_k=3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    print(f"\nQuery: {query}\n")

    # DEBUG: print raw arrays
    print("Indices:", indices)
    print("Distances:", distances)

    for rank in range(top_k):
        idx = int(indices[0][rank])
        distance = float(distances[0][rank])

        print(f"Result {rank + 1} | Distance: {distance:.4f}")
        print(chunks[idx])
        print("-" * 50)


retrieve("How many paid leaves do employees receive?")
retrieve("Can paid leaves be carried forward?")
retrieve("What is the work from home policy?")