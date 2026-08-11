import numpy as np
from sentence_transformers import SentenceTransformer

def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

# all-MiniLM-L6-v2 has 384 features
model = SentenceTransformer("all-MiniLM-L6-v2")

s1 = "The road is blocked due to traffic"
s2 = "There is heavy traffic in the road"

e1 = model.encode(s1)
e2 = model.encode(s2)

print(e1.shape)
print(cosine_similarity(e1, e2))