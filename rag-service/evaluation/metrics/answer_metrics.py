"""
Answer quality metrics for RAG Evaluation.
Includes keyword coverage, and grounding/faithfulness checks.
"""

from typing import Optional


def calculate_keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    """
    Keyword Coverage: Fraction of expected keywords that appear in the answer (case-insensitive).
    """
    if not expected_keywords:
        return 1.0  # vacuously full coverage if no keywords expected
    if not answer:
        return 0.0

    answer_lower = answer.lower()
    found_count = 0
    for keyword in expected_keywords:
        if keyword.lower() in answer_lower:
            found_count += 1

    return found_count / len(expected_keywords)


def evaluate_groundedness(grounded: bool, expected_grounded: Optional[bool]) -> float:
    """
    Returns 1.0 if the grounded flag matches the expected_grounded value, else 0.0.
    """
    if expected_grounded is None:
        return 1.0
    return 1.0 if grounded == expected_grounded else 0.0
