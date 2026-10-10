from sentence_transformers import SentenceTransformer
from ingest import extract_text
from chunk import chunk_text

model = SentenceTransformer("all-MiniLM-L6-v2")
text = extract_text("data/sample.pdf")
chunks = chunk_text(text)

embeddings = model.encode(chunks)

print("Embedding Type:", type(embeddings))
print("Length of Embedding:", len(embeddings))
print("Length of vecotr: ", len(embeddings[0]))
print("Vector: ", embeddings[0])
