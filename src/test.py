from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence1 = "Employees receive 24 annual paid leaves."
sentence2 = "Workers are entitled to twenty-four days of annual leave.."
sentence3 = "Apple released a new iPhone."

embeddings = model.encode([sentence1, sentence2, sentence3])

print(util.cos_sim(embeddings[0], embeddings[1]))
print(util.cos_sim(embeddings[0], embeddings[2]))