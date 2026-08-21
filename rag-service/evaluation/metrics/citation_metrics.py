"""
Citation metrics for RAG Evaluation.
Checks source accuracy (no fabricated citations) and metadata fidelity.
"""

from typing import Any


def evaluate_citations(
    citations: list[Any],
    retrieved_chunks: list[Any],
) -> dict[str, float]:
    """
    Evaluate the citation quality of RAG query output.

    Parameters
    ----------
    citations : list[RAGSource]
        List of RAGSource citations returned in the response.
    retrieved_chunks : list[RetrievedChunk]
        List of chunks returned from the Retrieval layer.

    Returns
    -------
    dict
        {
            "citation_source_accuracy": float,
            "citation_metadata_accuracy": float
        }
    """
    if not citations:
        # If no citations returned, accuracy is perfect if there were no retrieved chunks,
        # or if they weren't used. We default to 1.0.
        return {
            "citation_source_accuracy": 1.0,
            "citation_metadata_accuracy": 1.0,
        }

    # Map retrieved chunks by chunk_id
    chunk_map = {}
    for chunk in retrieved_chunks:
        chunk_id = getattr(chunk, "chunk_id", None) or chunk.get("chunk_id")
        if chunk_id:
            chunk_map[chunk_id] = chunk

    valid_sources = 0
    correct_metadata_sources = 0

    for cit in citations:
        cit_chunk_id = getattr(cit, "chunk_id", None) or cit.get("chunk_id")
        if not cit_chunk_id or cit_chunk_id not in chunk_map:
            # Fabricated citation or maps to nothing retrieved
            continue

        valid_sources += 1
        original_chunk = chunk_map[cit_chunk_id]

        # Check metadata match
        metadata_match = True
        fields_to_check = [
            "document_id",
            "document_name",
            "page_number",
            "slide_number",
            "slide_title",
            "heading",
            "subject",
            "topic",
        ]

        for field in fields_to_check:
            original_val = getattr(original_chunk, field, None)
            if original_val is None and isinstance(original_chunk, dict):
                original_val = original_chunk.get(field)
            
            citation_val = getattr(cit, field, None)
            if citation_val is None and isinstance(cit, dict):
                citation_val = cit.get(field)

            if original_val != citation_val:
                metadata_match = False
                break

        # Check similarity score preservation (with small float tolerance)
        orig_score = getattr(original_chunk, "similarity_score", None)
        if orig_score is None and isinstance(original_chunk, dict):
            orig_score = original_chunk.get("similarity_score")
        
        cit_score = getattr(cit, "similarity_score", None)
        if cit_score is None and isinstance(cit, dict):
            cit_score = cit.get("similarity_score")

        if orig_score is not None and cit_score is not None:
            if abs(orig_score - cit_score) > 1e-4:
                metadata_match = False

        if metadata_match:
            correct_metadata_sources += 1

    source_accuracy = valid_sources / len(citations) if citations else 1.0
    metadata_accuracy = correct_metadata_sources / len(citations) if citations else 1.0

    return {
        "citation_source_accuracy": source_accuracy,
        "citation_metadata_accuracy": metadata_accuracy,
    }
