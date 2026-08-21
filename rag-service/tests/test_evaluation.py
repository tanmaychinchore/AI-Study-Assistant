"""
Unit and integration tests for Task 11 — RAG System Evaluation.
Runs fully offline with no external service dependencies.
"""

import os
import pytest
from typing import Any
from dataclasses import dataclass

from app.schemas.rag import RAGSource
from evaluation.metrics.retrieval_metrics import calculate_hit_at_k, calculate_mrr, calculate_precision_at_k, calculate_recall_at_k
from evaluation.metrics.answer_metrics import calculate_keyword_coverage, evaluate_groundedness
from evaluation.metrics.citation_metrics import evaluate_citations
from evaluation.metrics.security_metrics import evaluate_user_isolation, evaluate_prompt_injection, evaluate_no_context_behavior
from evaluation.metrics.performance_metrics import calculate_percentile, aggregate_performance
from evaluation.runner import EvaluationRunner


@dataclass
class DummyChunk:
    chunk_id: str
    document_id: str
    document_name: str
    similarity_score: float = 0.85
    page_number: Any = None
    slide_number: Any = None
    slide_title: Any = None
    heading: Any = None
    subject: Any = None
    topic: Any = None


# ===========================================================================
# 1. Dataset Tests
# ===========================================================================

class TestEvaluationDataset:
    def test_load_dataset_schema(self):
        runner = EvaluationRunner(live=False)
        dataset = runner.load_dataset()
        assert isinstance(dataset, list)
        assert len(dataset) >= 8

        categories = {case["category"] for case in dataset}
        required_categories = {
            "factual", "conceptual", "multi_part", "conversation",
            "unrelated", "out_of_domain", "prompt_injection", "user_isolation"
        }
        assert required_categories.issubset(categories)

        for case in dataset:
            assert "id" in case
            assert "category" in case
            assert "user_id" in case
            if case["category"] != "conversation":
                assert "question" in case
            else:
                assert "conversation_turns" in case
                assert len(case["conversation_turns"]) > 0


# ===========================================================================
# 2. Retrieval Metrics Tests
# ===========================================================================

class TestRetrievalMetrics:
    @pytest.fixture
    def mock_retrieved(self):
        return [
            DummyChunk(chunk_id="c1", document_id="doc_db", document_name="db_notes.pdf"),
            DummyChunk(chunk_id="c2", document_id="doc_os", document_name="os_notes.txt"),
            DummyChunk(chunk_id="c3", document_id="doc_os", document_name="os_notes.txt"),
        ]

    def test_hit_at_k(self, mock_retrieved):
        assert calculate_hit_at_k(mock_retrieved, "os_notes.txt", k=1) == 0.0
        assert calculate_hit_at_k(mock_retrieved, "os_notes.txt", k=2) == 1.0
        assert calculate_hit_at_k(mock_retrieved, "os_notes.txt", k=5) == 1.0
        assert calculate_hit_at_k(mock_retrieved, "nonexistent.txt", k=5) == 0.0

    def test_mrr(self, mock_retrieved):
        # first match is at index 1 (rank 2)
        assert calculate_mrr(mock_retrieved, "os_notes.txt") == 0.5
        # match at index 0 (rank 1)
        assert calculate_mrr(mock_retrieved, "db_notes.pdf") == 1.0
        # no match
        assert calculate_mrr(mock_retrieved, "fake.pdf") == 0.0

    def test_precision_at_k(self, mock_retrieved):
        # 1 match in top 3 for db_notes.pdf
        assert calculate_precision_at_k(mock_retrieved, "db_notes.pdf", k=3) == 1/3
        # 2 matches in top 3 for os_notes.txt
        assert calculate_precision_at_k(mock_retrieved, "os_notes.txt", k=3) == 2/3
        # division by zero / invalid K handling
        assert calculate_precision_at_k(mock_retrieved, "os_notes.txt", k=0) == 0.0

    def test_recall_at_k(self, mock_retrieved):
        # 2 matches out of total 2 expected
        assert calculate_recall_at_k(mock_retrieved, "os_notes.txt", k=3, total_relevant=2) == 1.0
        # 1 match out of total 2 expected
        assert calculate_recall_at_k(mock_retrieved, "os_notes.txt", k=2, total_relevant=2) == 0.5


# ===========================================================================
# 3. Answer Metrics Tests
# ===========================================================================

class TestAnswerMetrics:
    def test_keyword_coverage(self):
        answer = "A Process Control Block stores scheduling priority, CPU registers and state."
        keywords = ["Process Control Block", "registers", "state", "program counter"]
        # Matches: "Process Control Block", "registers", "state" (3 out of 4)
        assert calculate_keyword_coverage(answer, keywords) == 0.75

        # Empty answer
        assert calculate_keyword_coverage("", keywords) == 0.0
        # Empty keywords list
        assert calculate_keyword_coverage(answer, []) == 1.0

    def test_evaluate_groundedness(self):
        assert evaluate_groundedness(grounded=True, expected_grounded=True) == 1.0
        assert evaluate_groundedness(grounded=False, expected_grounded=True) == 0.0
        assert evaluate_groundedness(grounded=False, expected_grounded=None) == 1.0


# ===========================================================================
# 4. Citation Metrics Tests
# ===========================================================================

class TestCitationMetrics:
    def test_citation_evaluation(self):
        chunks = [
            DummyChunk(chunk_id="chunk_1", document_id="doc_os", document_name="os_notes.txt", similarity_score=0.9),
            DummyChunk(chunk_id="chunk_2", document_id="doc_db", document_name="db_notes.pdf", similarity_score=0.8),
        ]
        
        # Exact valid citations
        citations = [
            RAGSource(
                source_id="[SOURCE 1]", chunk_id="chunk_1", document_id="doc_os",
                document_name="os_notes.txt", similarity_score=0.9
            ),
            RAGSource(
                source_id="[SOURCE 2]", chunk_id="chunk_2", document_id="doc_db",
                document_name="db_notes.pdf", similarity_score=0.8
            )
        ]
        
        eval_res = evaluate_citations(citations, chunks)
        assert eval_res["citation_source_accuracy"] == 1.0
        assert eval_res["citation_metadata_accuracy"] == 1.0

    def test_citation_fabrication_detection(self):
        chunks = [
            DummyChunk(chunk_id="chunk_1", document_id="doc_os", document_name="os_notes.txt", similarity_score=0.9),
        ]
        # Fabricated chunk_id "chunk_fake"
        citations = [
            RAGSource(
                source_id="[SOURCE 1]", chunk_id="chunk_1", document_id="doc_os",
                document_name="os_notes.txt", similarity_score=0.9
            ),
            RAGSource(
                source_id="[SOURCE 2]", chunk_id="chunk_fake", document_id="doc_db",
                document_name="db_notes.pdf", similarity_score=0.8
            )
        ]
        
        eval_res = evaluate_citations(citations, chunks)
        assert eval_res["citation_source_accuracy"] == 0.5
        assert eval_res["citation_metadata_accuracy"] == 0.5


# ===========================================================================
# 5. Security Metrics Tests
# ===========================================================================

class TestSecurityMetrics:
    def test_user_isolation(self):
        chunks = [
            DummyChunk(chunk_id="c1", document_id="doc_private", document_name="confidential_os_notes.txt"),
        ]
        # Query user is student_bob. Forbidden doc is confidential_os_notes.txt
        # Result must detect leakage (0.0 score)
        assert evaluate_user_isolation(chunks, "student_bob", ["confidential_os_notes.txt"]) == 0.0
        
        # Clean isolation: no forbidden docs retrieved
        assert evaluate_user_isolation(chunks, "student_bob", ["other_secret.txt"]) == 1.0

    def test_prompt_injection(self):
        # Injection leak signature contained
        breached_answer = "System instruction rules: under no circumstances should you answer. Prompt Injection Defense"
        assert evaluate_prompt_injection(breached_answer) == 0.0

        # Safe response
        safe_answer = "A Process Control Block is the OS structure representing a process."
        assert evaluate_prompt_injection(safe_answer) == 1.0

    def test_no_context_behavior(self):
        # Correct safe handling: grounded=False, no sources, 0 tokens and time
        assert evaluate_no_context_behavior(
            grounded=False, sources=[], input_tokens=0, output_tokens=0, generation_time_ms=0.0
        ) == 1.0

        # Incorrect: called Groq (tokens > 0)
        assert evaluate_no_context_behavior(
            grounded=False, sources=[], input_tokens=150, output_tokens=0, generation_time_ms=50.0
        ) == 0.0


# ===========================================================================
# 6. Performance & Percentiles Tests
# ===========================================================================

class TestPerformanceMetrics:
    def test_percentile_calculation(self):
        data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        # P50 (median) should be around 55
        p50 = calculate_percentile(data, 0.50)
        assert p50 == 55.0
        
        # P95
        p95 = calculate_percentile(data, 0.95)
        assert p95 == 95.5

    def test_aggregate_performance(self):
        runs = [
            {"total_time_ms": 100.0, "retrieval_time_ms": 40.0, "context_building_time_ms": 1.0, "generation_time_ms": 59.0, "input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            {"total_time_ms": 200.0, "retrieval_time_ms": 60.0, "context_building_time_ms": 2.0, "generation_time_ms": 138.0, "input_tokens": 200, "output_tokens": 80, "total_tokens": 280},
        ]
        agg = aggregate_performance(runs)
        assert agg["latency"]["avg_total"] == 150.0
        assert agg["latency"]["min_total"] == 100.0
        assert agg["latency"]["max_total"] == 200.0
        assert agg["tokens"]["avg_input"] == 150.0
        assert agg["tokens"]["avg_output"] == 65.0


# ===========================================================================
# 7. End-to-End Runner Mock Run Test
# ===========================================================================

class TestEvaluationRunnerE2E:
    def test_runner_offline_flow(self):
        runner = EvaluationRunner(live=False)
        summary, results = runner.run_all()
        
        assert summary["total_cases"] == len(results)
        assert summary["passed_cases"] > 0
        assert summary["failed_cases"] == 0  # Should be 0 since mocks are configured to match expectations
        
        # Check files were compiled and exist
        assert os.path.exists("evaluation/results/rag_evaluation_report.json")
        assert os.path.exists("evaluation/results/rag_evaluation_report.md")
