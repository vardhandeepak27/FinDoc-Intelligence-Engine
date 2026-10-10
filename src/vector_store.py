import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import extract_text
from chunk import chunk_text

# 1. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Extract text from PDF
text = extract_text("data/sample.pdf")

# 3. Create chunks
chunks = chunk_text(text, chunk_size=100, overlap=20)

# 4. Generate embeddings
embeddings = model.encode(chunks)

# 5. Convert to float32 (FAISS requirement)
embeddings = np.array(embeddings).astype("float32")

# 6. Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# 7. Add vectors to the index
index.add(embeddings)

print(f"Total chunks: {len(chunks)}")
print(f"Vector dimension: {dimension}")
print(f"Vectors stored in FAISS: {index.ntotal}")