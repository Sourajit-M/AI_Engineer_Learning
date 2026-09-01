import uuid
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
qdrant_client = QdrantClient(
    api_key=QDRANT_API_KEY,
    url=QDRANT_URL
)
COLLECTION_NAME = "qa_docs"


def init_bm25(chunks: list[str]) -> BM25Okapi:
    """Tokenize chunks and initialize BM25 retriever."""
    tokenized = [c.lower().split() for c in chunks]
    return BM25Okapi(tokenized)


def bm25_search(query: str, bm25: BM25Okapi, top_k: int = 10) -> list[int]:
    """Search using keyword-based BM25."""
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:top_k]  # returns chunk indices, ranked


def semantic_search(
    query: str,
    embed_model: SentenceTransformer,
    qdrant_client: QdrantClient,
    collection_name: str,
    top_k: int = 10,
) -> list[int]:
    """Search using dense vector embeddings via Qdrant."""
    query_emb = embed_model.encode(query).tolist()

    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_emb,
        limit=top_k,
    )

    # Extract original chunk index stored in Qdrant payload
    return [hit.payload["chunk_index"] for hit in results.points]


def hybrid_search(
    query: str,
    chunks: list[str],
    bm25: BM25Okapi,
    embed_model: SentenceTransformer,
    qdrant_client: QdrantClient,
    collection_name: str,
    top_k: int = 5,
    k_rrf: int = 60,
) -> list[str]:
    """Fuse keyword and semantic results using Reciprocal Rank Fusion (RRF)."""
    bm25_ranked = bm25_search(query, bm25, top_k=10)
    semantic_ranked = semantic_search(
        query, embed_model, qdrant_client, collection_name, top_k=10
    )

    rrf_scores = {}

    # Add points based on BM25 ranking
    for rank, index in enumerate(bm25_ranked):
        points = 1 / (k_rrf + rank + 1)
        rrf_scores[index] = rrf_scores.get(index, 0) + points

    # Add points based on semantic ranking
    for rank, index in enumerate(semantic_ranked):
        points = 1 / (k_rrf + rank + 1)
        rrf_scores[index] = rrf_scores.get(index, 0) + points

    # Sort chunks by their combined score
    sorted_chunks = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # Get the indices of the best chunks
    top_indices = []

    for index, score in sorted_chunks[:top_k]:
        top_indices.append(index)

    # Return the actual chunks
    top_chunks = []

    for index in top_indices:
        top_chunks.append(chunks[index])

    return top_chunks


# Example Usage Setup
def ingest_chunks(chunks: list[str]):
    """Ingest chunks into Qdrant with chunk index metadata."""
    embeddings = embed_model.encode(chunks).tolist()

    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384, distance=models.Distance.COSINE
            ),
        )

    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"chunk_{i}")),
            vector=emb,
            payload={"text": chunk, "chunk_index": i},
        )
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)


# Running the functional pipeline
chunks = [
    "The refund policy allows returns within 30 days of purchase.",
    "Items returned after 30 days will receive store credit only.",
    "The weather in Kolkata today is warm and clear.",
]

ingest_chunks(chunks)
bm25_instance = init_bm25(chunks)

query = "How long do I have to return an item?"
results = hybrid_search(
    query=query,
    chunks=chunks,
    bm25=bm25_instance,
    embed_model=embed_model,
    qdrant_client=qdrant_client,
    collection_name=COLLECTION_NAME,
    top_k=2,
)

print(results)

#Cross Encoder

reranker = CrossEncoder("BAAI/bge-reranker-base")  # free, local

def rerank(query: str, candidates: list[str], top_k: int = 5) -> list[str]:
    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k]]

print("#"*50)
print("******RERANK*********")
final_chunks = rerank(query, results, top_k=1)
print(final_chunks)