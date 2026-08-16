from fastembed import TextEmbedding
import numpy as np

model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")  # free, local, 384-dim

texts = [
    "The customer requested a refund for a defective product.",
    "User wants their money back because the item was broken.",
    "The weather in Kolkata today is humid.",
]

embeddings = list(model.embed(texts))
print(len(embeddings[0]))  # 384

# print(embeddings)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Refund vs money-back:", cosine_similarity(embeddings[0], embeddings[1]))  # high
print("Refund vs weather:", cosine_similarity(embeddings[0], embeddings[2]))     # low