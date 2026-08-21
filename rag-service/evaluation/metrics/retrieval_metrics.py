"""
Retrieval metrics for RAG Evaluation.
Includes Hit@K, Mean Reciprocal Rank (MRR), Precision@K, and Recall@K.
"""

from typing import Any, Optional


def is_relevant(chunk: Any, expected_document: Optional[str]) -> bool:
    """
    Check if a retrieved chunk matches the expected document name or ID.
    """
    if not expected_document:
        return False
    doc_name = getattr(chunk, "document_name", None) or (chunk.get("document_name") if isinstance(chunk, dict) else None)
    doc_id = getattr(chunk, "document_id", None) or (chunk.get("document_id") if isinstance(chunk, dict) else None)
    expected = expected_document.lower().strip()
    
    if doc_name and doc_name.lower().strip() == expected:
        return True
    if doc_id and str(doc_id).lower().strip() == expected:
        return True
    return False


def calculate_hit_at_k(retrieved_chunks: list[Any], expected_document: Optional[str], k: int) -> float:
    """
    Hit@K: 1.0 if a relevant chunk appears in the top K retrieved results, else 0.0.
    """
    if not expected_document:
        return 0.0
    for chunk in retrieved_chunks[:k]:
        if is_relevant(chunk, expected_document):
            return 1.0
    return 0.0


def calculate_mrr(retrieved_chunks: list[Any], expected_document: Optional[str]) -> float:
    """
    Reciprocal Rank of the first relevant retrieved result.
    Mean Reciprocal Rank (MRR) is the average across all queries.
    """
    if not expected_document:
        return 0.0
    for idx, chunk in enumerate(retrieved_chunks):
        if is_relevant(chunk, expected_document):
            return 1.0 / (idx + 1)
    return 0.0


def calculate_precision_at_k(retrieved_chunks: list[Any], expected_document: Optional[str], k: int) -> float:
    """
    Precision@K: (relevant retrieved results in top K) / K.
    """
    if not expected_document or k <= 0:
        return 0.0
    relevant_count = sum(1 for chunk in retrieved_chunks[:k] if is_relevant(chunk, expected_document))
    return relevant_count / k


def calculate_recall_at_k(retrieved_chunks: list[Any], expected_document: Optional[str], k: int, total_relevant: int = 1) -> float:
    """
    Recall@K: (relevant retrieved results in top K) / total expected relevant results.
    Defaults to total_relevant=1 as the dataset schema maps to one target document.
    """
    if not expected_document or total_relevant <= 0:
        return 0.0
    relevant_count = sum(1 for chunk in retrieved_chunks[:k] if is_relevant(chunk, expected_document))
    # Cap recall at 1.0
    return min(1.0, relevant_count / total_relevant)
