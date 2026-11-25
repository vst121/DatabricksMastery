"""
DatabricksMastery | GenAI
03 - Embeddings

Experiments:
1. Generate text embeddings
2. Inspect vector representations
3. Calculate cosine similarity
4. Implement semantic search

Key concepts:
Dense vectors, vector dimensions, semantic similarity,
cosine similarity, query-document retrieval.
"""

# ============================================================
# 1. IMPORTS AND CONFIGURATION
# ============================================================

import math
from openai import OpenAI


workspace_url = (
    "https://" + spark.conf.get("spark.databricks.workspaceUrl")
)

MODEL_ENDPOINT = "system.ai.gte-large-en"

token = (
    dbutils.notebook.entry_point
    .getDbutils()
    .notebook()
    .getContext()
    .apiToken()
    .get()
)

client = OpenAI(
    api_key=token,
    base_url=f"{workspace_url}/ai-gateway/mlflow/v1"
)

print("Embedding model:", MODEL_ENDPOINT)


# ============================================================
# 2. SAMPLE DOCUMENTS
# ============================================================

documents = [
    {
        "id": 1,
        "text": "Machine learning models identify patterns from training data."
    },
    {
        "id": 2,
        "text": "Large language models generate human-like text using transformer architectures."
    },
    {
        "id": 3,
        "text": "Vector databases store embeddings and support similarity search."
    },
    {
        "id": 4,
        "text": "RAG combines information retrieval with language model generation."
    },
    {
        "id": 5,
        "text": "Microservices communicate through APIs and asynchronous messaging."
    }
]


# ============================================================
# 3. GENERATE EMBEDDINGS
# ============================================================

def embed_texts(texts):
    """Generate embeddings for a batch of text inputs."""

    response = client.embeddings.create(
        model=MODEL_ENDPOINT,
        input=texts
    )

    return [item.embedding for item in response.data]


texts = [document["text"] for document in documents]

document_vectors = embed_texts(texts)

for document, vector in zip(documents, document_vectors):
    document["embedding"] = vector

print("Documents embedded:", len(document_vectors))
print("Vector dimensions:", len(document_vectors[0]))


# ============================================================
# 4. INSPECT VECTOR REPRESENTATIONS
# ============================================================

sample_vector = document_vectors[0]

print("\nOriginal text:")
print(documents[0]["text"])

print("\nFirst 10 vector values:")
print(sample_vector[:10])

print("\nVector dimensions:", len(sample_vector))


# ============================================================
# 5. COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector_a, vector_b):
    """Calculate cosine similarity between two vectors."""

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


similarity = cosine_similarity(
    document_vectors[0],
    document_vectors[1]
)

print("\nSimilarity between documents 1 and 2:")
print(round(similarity, 4))


# ============================================================
# 6. SEMANTIC SEARCH
# ============================================================

query = "How can an AI application retrieve relevant information from documents?"

query_vector = embed_texts([query])[0]

results = []

for document in documents:

    score = cosine_similarity(
        query_vector,
        document["embedding"]
    )

    results.append({
        "id": document["id"],
        "text": document["text"],
        "similarity": score
    })


results.sort(
    key=lambda item: item["similarity"],
    reverse=True
)

print("\nQuery:", query)
print("\nTop matching documents:")

for rank, result in enumerate(results[:3], start=1):

    print(f"\nRank: {rank}")
    print("Document ID:", result["id"])
    print("Similarity:", round(result["similarity"], 4))
    print("Text:", result["text"])


# ============================================================
# 7. ENGINEERING SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EMBEDDINGS - EXPERIMENT SUMMARY")
print("=" * 60)

print("""
1. Embeddings represent text as dense numerical vectors.
2. Similar meanings can have nearby vector representations.
3. Cosine similarity measures vector direction similarity.
4. Semantic search retrieves documents by meaning, not just keywords.
5. Embeddings are a core building block of RAG.

Production considerations:
- Chunk documents before embedding.
- Use the same embedding model for documents and queries.
- Store vectors with document IDs and metadata.
- Select similarity metrics appropriate to the model.
- Evaluate retrieval quality independently from LLM generation.
""")
