
"""
06 - RAG Evaluation

Evaluate retrieval quality using labeled queries and ranked results.

Metrics:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Hit Rate@K
"""

from statistics import mean


# 1. Evaluation dataset
# Relevant document IDs are the ground-truth labels.
evaluation_data = [
    {
        "query": "How does RAG work?",
        "relevant_docs": {"DOC-001", "DOC-002"},
        "retrieved_docs": ["DOC-003", "DOC-001", "DOC-004", "DOC-002"],
    },
    {
        "query": "What is an embedding?",
        "relevant_docs": {"DOC-005", "DOC-006"},
        "retrieved_docs": ["DOC-005", "DOC-007", "DOC-006", "DOC-008"],
    },
    {
        "query": "What is event-driven architecture?",
        "relevant_docs": {"DOC-009"},
        "retrieved_docs": ["DOC-010", "DOC-011", "DOC-009", "DOC-012"],
    },
]

K = 3


# 2. Evaluation metrics
def precision_at_k(retrieved, relevant, k):
    results = retrieved[:k]
    if not results:
        return 0.0

    return sum(doc in relevant for doc in results) / k


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0

    results = retrieved[:k]
    return sum(doc in relevant for doc in results) / len(relevant)


def reciprocal_rank(retrieved, relevant):
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank

    return 0.0


def hit_rate_at_k(retrieved, relevant, k):
    return float(any(doc in relevant for doc in retrieved[:k]))


# 3. Evaluate each query
query_metrics = []

for item in evaluation_data:
    retrieved = item["retrieved_docs"]
    relevant = item["relevant_docs"]

    metrics = {
        "query": item["query"],
        "precision": precision_at_k(retrieved, relevant, K),
        "recall": recall_at_k(retrieved, relevant, K),
        "mrr": reciprocal_rank(retrieved, relevant),
        "hit_rate": hit_rate_at_k(retrieved, relevant, K),
    }

    query_metrics.append(metrics)

    print(f"\nQuery: {metrics['query']}")
    print(f"Retrieved: {retrieved[:K]}")
    print(f"Relevant:  {sorted(relevant)}")
    print(f"Precision@{K}: {metrics['precision']:.3f}")
    print(f"Recall@{K}:    {metrics['recall']:.3f}")
    print(f"Reciprocal Rank: {metrics['mrr']:.3f}")
    print(f"Hit Rate@{K}:  {metrics['hit_rate']:.3f}")


# 4. Aggregate evaluation
print("\n" + "=" * 45)
print("AGGREGATE RETRIEVAL EVALUATION")
print("=" * 45)

for metric in ("precision", "recall", "mrr", "hit_rate"):
    score = mean(item[metric] for item in query_metrics)
    print(f"{metric.upper():<15}: {score:.3f}")


# 5. Engineering observations
print("\nENGINEERING OBSERVATIONS")
print("- Precision measures retrieval relevance.")
print("- Recall measures retrieval coverage.")
print("- MRR rewards relevant results appearing earlier.")
print("- Hit Rate measures whether retrieval found any relevant result.")
print("- Evaluate retrieval separately from answer generation.")
