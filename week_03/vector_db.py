from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Connect to Qdrant Cloud
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

COLLECTION_NAME = "items"

# Create embedding model
model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

texts = [
    "The customer requested a refund for a defective product.",
    "User wants their money back because the item was broken.",
    "The weather in Kolkata today is humid.",
]

# Generate embeddings
embeddings = list(model.embed(texts))

# Create Qdrant collection
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )

    print(f"Created collection: {COLLECTION_NAME}")
else:
    print(f"Collection already exists: {COLLECTION_NAME}")

# Store embeddings in Qdrant
points = []

for i, (text, embedding) in enumerate(zip(texts, embeddings)):
    points.append(
        PointStruct(
            id=i + 1,
            vector=embedding.tolist(),
            payload={
                "text": text,
            },
        )
    )

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
)

print(f"Inserted {len(points)} documents into Qdrant.")

query = "I need my money returned because the product was broken."

query_embedding = list(model.embed([query]))[0]

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding.tolist(),
    limit=3,
).points

for result in results:
    print("Score:", result.score)
    print("Text:", result.payload["text"])
    print()

client.delete_collection(collection_name="items")

print("Collection deleted successfully.")