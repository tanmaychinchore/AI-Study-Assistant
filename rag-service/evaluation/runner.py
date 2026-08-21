"""
Evaluation runner for RAG System.
Supports offline (mocked) and live evaluation.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.conversation import MessageRole
from app.schemas.llm import ChatMessage, GenerationResult
from app.schemas.rag import RAGRequest, RAGResult, RAGSource, RAGRetrievalStatistics, RAGGenerationStatistics
from app.schemas.retrieval import RetrievedChunk
from app.services.conversation_service import ConversationService
from app.services.rag_service import RAGService
from evaluation.metrics.retrieval_metrics import calculate_hit_at_k, calculate_mrr, calculate_precision_at_k, calculate_recall_at_k
from evaluation.metrics.answer_metrics import calculate_keyword_coverage, evaluate_groundedness
from evaluation.metrics.citation_metrics import evaluate_citations
from evaluation.metrics.security_metrics import evaluate_user_isolation, evaluate_prompt_injection, evaluate_no_context_behavior
from evaluation.metrics.performance_metrics import aggregate_performance
from evaluation.report import compile_report, print_cli_dashboard

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Offline Mock Services
# ---------------------------------------------------------------------------

class MockEmbeddingService:
    """Mocks BGE-M3 local embedding loading for offline testing."""
    is_loaded = True
    
    def embed_query(self, query: str) -> list[float]:
        return [0.1] * 1024


class MockAstraDBService:
    """Mocks Astra DB client connection for offline testing."""
    is_ready = True
    
    def __init__(self):
        self.mock_chunks: list[dict[str, Any]] = []

    def configure_search_results(self, chunks: list[dict[str, Any]]) -> None:
        self.mock_chunks = chunks

    def vector_search(
        self,
        query_vector: list[float],
        user_id: str,
        top_k: int,
        document_id: Optional[str] = None,
        subject: Optional[str] = None,
        topic: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> tuple[list[dict[str, Any]], float, int]:
        # Enforce user isolation: only return chunks belonging to query user_id
        results = []
        for chunk in self.mock_chunks:
            chunk_user = chunk.get("user_id", user_id)
            if chunk_user == user_id:
                # Apply optional threshold filter
                score = chunk.get("similarity_score", 0.0)
                if similarity_threshold is not None and score < similarity_threshold:
                    continue
                # Apply optional doc_id filter
                if document_id and chunk.get("document_id") != document_id:
                    continue
                results.append(chunk)
        return results[:top_k], 45.2, len(results)


class MockGroqService:
    """Mocks Groq API endpoint for offline testing."""
    model = "llama-3.3-70b-versatile"

    def __init__(self):
        self.call_count = 0
        self.mock_response_text = ""
        self.input_tokens = 0
        self.output_tokens = 0

    def configure_response(self, text: str, input_tokens: int = 150, output_tokens: int = 80) -> None:
        self.mock_response_text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.call_count = 0

    def generate(
        self,
        messages: list[Any],
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> GenerationResult:
        self.call_count += 1
        time.sleep(0.05)  # Simulate small network delay
        return GenerationResult(
            content=self.mock_response_text,
            model=model or self.model,
            finish_reason="stop",
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
            latency_ms=50.0,
            request_id="mock_request_123"
        )


# ---------------------------------------------------------------------------
# Runner Engine
# ---------------------------------------------------------------------------

class EvaluationRunner:
    """
    RAG system evaluation runner supporting both offline/mocked and live runs.
    """
    def __init__(self, live: bool = False):
        self.live = live
        self.dataset_path = "evaluation/datasets/rag_evaluation.json"

        if not live:
            logger.info("Initializing Evaluation Runner in OFFLINE (Mocked) Mode.")
            self.emb_svc = MockEmbeddingService()
            self.db_svc = MockAstraDBService()
            self.groq_svc = MockGroqService()
            
            # Setup real service wrappers but inject mock sub-services
            from app.services.retrieval_service import RetrievalService
            self.retrieval_svc = RetrievalService(
                embedding_service=self.emb_svc,
                astra_service=self.db_svc
            )
            self.rag_svc = RAGService(
                retrieval_service=self.retrieval_svc,
                groq_service=self.groq_svc
            )
            
            # Use mongomock Client inside ConversationService
            import mongomock
            self.mock_mongo = mongomock.MongoClient()
            self.conv_svc = ConversationService(
                database_name="test_eval_assistant",
                conversations_collection="eval_conversations",
                messages_collection="eval_messages",
                client=self.mock_mongo
            )
        else:
            logger.info("Initializing Evaluation Runner in LIVE Mode.")
            # In live mode, import real services
            from app.services.embedding_service import EmbeddingService
            from app.services.astra_db_service import AstraDBService
            from app.services.groq_service import GroqService
            from app.services.retrieval_service import RetrievalService
            
            self.emb_svc = EmbeddingService()
            self.emb_svc.load_model()
            
            self.db_svc = AstraDBService()
            self.db_svc.connect()
            
            self.groq_svc = GroqService()
            
            self.retrieval_svc = RetrievalService(
                embedding_service=self.emb_svc,
                astra_service=self.db_svc
            )
            self.rag_svc = RAGService(
                retrieval_service=self.retrieval_svc,
                groq_service=self.groq_svc
            )
            self.conv_svc = ConversationService()
            self.conv_svc.connect()

    def load_dataset(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Evaluation dataset not found at {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _setup_offline_mocks(self, case: dict[str, Any]) -> None:
        """Configure mock services to simulate retrieval and text generation for a case."""
        if self.live:
            return

        category = case.get("category")
        expected_doc = case.get("expected_document")
        keywords = case.get("expected_keywords", [])
        ref_answer = case.get("reference_answer", "")
        expected_grounded = case.get("expected_grounded", True)
        
        # 1. Setup Retrieval Mock Results
        chunks = []
        if expected_grounded:
            # Create a relevant matching chunk
            text_1 = f"This reference text contains Process Control Block and mentions keywords: {' '.join(keywords)}."
            chunks.append({
                "chunk_id": f"chunk_relevant_{case['id']}",
                "document_id": f"doc_{expected_doc}" if expected_doc else "doc_123",
                "document_name": expected_doc or "os_notes.txt",
                "text": text_1,
                "similarity_score": 0.88,
                "user_id": case.get("user_id", "student_alice"),
                "char_count": len(text_1),
                "file_type": "txt",
                "chunk_index": 0
            })
            # Add an unrelated chunk
            text_2 = "This is completely different material with DBMS queries."
            chunks.append({
                "chunk_id": f"chunk_unrelated_{case['id']}",
                "document_id": "doc_unrelated",
                "document_name": "other_notes.txt",
                "text": text_2,
                "similarity_score": 0.45,
                "user_id": case.get("user_id", "student_alice"),
                "char_count": len(text_2),
                "file_type": "txt",
                "chunk_index": 1
            })
        else:
            # Unrelated / out-of-domain: return either nothing or scores below threshold
            if category in ("unrelated", "out_of_domain", "user_isolation"):
                pass  # Empty list
            else:
                text_3 = "Irrelevant noise text."
                chunks.append({
                    "chunk_id": "chunk_low_score",
                    "document_id": "doc_low",
                    "document_name": "other.txt",
                    "text": text_3,
                    "similarity_score": 0.12,
                    "user_id": case.get("user_id", "student_alice"),
                    "char_count": len(text_3),
                    "file_type": "txt",
                    "chunk_index": 0
                })

        # Inject into mock database service
        self.db_svc.configure_search_results(chunks)

        # 2. Setup Groq LLM Mock Response
        # Include reference answer & keywords so metrics evaluate to 100%
        mocked_answer = ref_answer
        if keywords:
            mocked_answer += f" Key concepts: {', '.join(keywords)}."
            
        if category == "prompt_injection":
            # For prompt injection, answer must be clean and not reveal instructions
            mocked_answer = "A Process Control Block is a data structure storing process state and counter."
            
        self.groq_svc.configure_response(mocked_answer)

    def run_case(self, case: dict[str, Any]) -> dict[str, Any]:
        """
        Run a single evaluation test case.
        """
        case_id = case["id"]
        category = case["category"]
        user_id = case.get("user_id", "student_alice")
        
        self._setup_offline_mocks(case)

        # Handle conversational multi-turn cases separately
        if category == "conversation" or "conversation_turns" in case:
            return self.run_conversation_case(case)

        question = case["question"]
        expected_doc = case.get("expected_document")
        expected_keywords = case.get("expected_keywords", [])
        expected_grounded = case.get("expected_grounded", True)
        
        # Build standard request
        req = RAGRequest(
            query=question,
            user_id=user_id,
            top_k=5,
            similarity_threshold=0.50 if not expected_grounded else None
        )

        overall_start = time.perf_counter()
        
        # Run standard RAG pipeline
        result = self.rag_svc.query(req)
        
        total_time_ms = round((time.perf_counter() - overall_start) * 1000, 2)

        # Collect results
        retrieved_chunks = result.sources
        answer = result.answer
        grounded = result.grounded
        
        input_tokens = result.generation_statistics.input_tokens
        output_tokens = result.generation_statistics.output_tokens
        generation_time = result.generation_statistics.generation_time_ms

        # Compute Metrics
        hit_1 = calculate_hit_at_k(retrieved_chunks, expected_doc, 1)
        hit_3 = calculate_hit_at_k(retrieved_chunks, expected_doc, 3)
        hit_5 = calculate_hit_at_k(retrieved_chunks, expected_doc, 5)
        mrr = calculate_mrr(retrieved_chunks, expected_doc)
        precision_5 = calculate_precision_at_k(retrieved_chunks, expected_doc, 5)
        recall_5 = calculate_recall_at_k(retrieved_chunks, expected_doc, 5)

        keyword_cov = calculate_keyword_coverage(answer, expected_keywords)
        grounded_match = evaluate_groundedness(grounded, expected_grounded)

        citations_eval = evaluate_citations(result.sources, retrieved_chunks)
        
        # Security evaluations
        user_leakage = 0.0
        if category == "user_isolation":
            # For user isolation test: User B (student_bob) asked about User A's confidential file.
            # Leakage is 1.0 if User A's document is retrieved, else 0.0.
            user_leakage = 1.0 - evaluate_user_isolation(retrieved_chunks, user_id, [expected_doc])
            
        prompt_injection_pass = 1.0
        if category == "prompt_injection":
            prompt_injection_pass = evaluate_prompt_injection(answer)

        no_context_pass = 1.0
        if not expected_grounded:
            no_context_pass = evaluate_no_context_behavior(
                grounded, result.sources, input_tokens, output_tokens, generation_time
            )

        success = True
        if expected_grounded and (hit_5 < 1.0 or keyword_cov < settings.EVAL_MIN_KEYWORD_COVERAGE):
            success = False
        if category == "user_isolation" and user_leakage > 0.0:
            success = False
        if category == "prompt_injection" and prompt_injection_pass < 1.0:
            success = False
        if not expected_grounded and no_context_pass < 1.0:
            success = False

        return {
            "case_id": case_id,
            "category": category,
            "question": question,
            "expected_document": expected_doc,
            "success": success,
            "chunks_retrieved": len(retrieved_chunks),
            "grounded": grounded,
            "keyword_coverage": keyword_cov,
            "hit_at_1": hit_1,
            "hit_at_3": hit_3,
            "hit_at_5": hit_5,
            "mrr": mrr,
            "precision_at_5": precision_5,
            "recall_at_5": recall_5,
            "grounded_match": grounded_match,
            "citation_source_accuracy": citations_eval["citation_source_accuracy"],
            "citation_metadata_accuracy": citations_eval["citation_metadata_accuracy"],
            "user_leakage": user_leakage,
            "prompt_injection_pass": prompt_injection_pass,
            "no_context_pass": no_context_pass,
            "total_time_ms": total_time_ms,
            "retrieval_time_ms": result.retrieval_statistics.retrieval_time_ms,
            "context_building_time_ms": result.context_building_time_ms,
            "generation_time_ms": generation_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": result.generation_statistics.total_tokens,
        }

    def run_conversation_case(self, case: dict[str, Any]) -> dict[str, Any]:
        """
        Run multi-turn conversation evaluation turns.
        """
        case_id = case["id"]
        category = case["category"]
        user_id = case.get("user_id", "student_alice")
        turns = case.get("conversation_turns", [])

        # 1. Initialize conversation
        conv = self.conv_svc.create_conversation(user_id=user_id, title=f"Eval Conv {case_id}")
        conv_id = conv.conversation_id

        last_turn_result = {}
        
        # Run each turn
        for turn_idx, turn in enumerate(turns):
            question = turn["question"]
            expected_doc = turn.get("expected_document")
            expected_keywords = turn.get("expected_keywords", [])
            
            # Setup offline mocks specifically for this turn
            self._setup_offline_mocks({
                "id": f"{case_id}_t{turn_idx}",
                "category": category,
                "user_id": user_id,
                "expected_document": expected_doc,
                "expected_keywords": expected_keywords,
                "reference_answer": f"Answer for turn {turn_idx}: {question}",
                "expected_grounded": True
            })

            # Retrieve recent conversation turns
            history_objs = self.conv_svc.get_recent_history(
                conversation_id=conv_id, user_id=user_id, limit=settings.CHAT_MAX_HISTORY_MESSAGES
            )
            chat_history = [
                ChatMessage(role="user" if m.role == MessageRole.USER else "assistant", content=m.content)
                for m in history_objs
            ]

            # Run RAG query
            req = RAGRequest(query=question, user_id=user_id, top_k=5)
            
            overall_start = time.perf_counter()
            result = self.rag_svc.query(req, conversation_history=chat_history)
            total_time_ms = round((time.perf_counter() - overall_start) * 1000, 2)

            # Persist to database
            self.conv_svc.append_message(conv_id, user_id, MessageRole.USER, question)
            self.conv_svc.append_message(conv_id, user_id, MessageRole.ASSISTANT, result.answer)

            # Record final turn metrics
            if turn_idx == len(turns) - 1:
                retrieved_chunks = result.sources
                hit_1 = calculate_hit_at_k(retrieved_chunks, expected_doc, 1)
                hit_3 = calculate_hit_at_k(retrieved_chunks, expected_doc, 3)
                hit_5 = calculate_hit_at_k(retrieved_chunks, expected_doc, 5)
                mrr = calculate_mrr(retrieved_chunks, expected_doc)
                precision_5 = calculate_precision_at_k(retrieved_chunks, expected_doc, 5)
                recall_5 = calculate_recall_at_k(retrieved_chunks, expected_doc, 5)

                keyword_cov = calculate_keyword_coverage(result.answer, expected_keywords)
                citations_eval = evaluate_citations(result.sources, retrieved_chunks)

                success = hit_5 >= 1.0 and keyword_cov >= settings.EVAL_MIN_KEYWORD_COVERAGE

                last_turn_result = {
                    "case_id": case_id,
                    "category": category,
                    "question": question,
                    "expected_document": expected_doc,
                    "success": success,
                    "chunks_retrieved": len(retrieved_chunks),
                    "grounded": result.grounded,
                    "keyword_coverage": keyword_cov,
                    "hit_at_1": hit_1,
                    "hit_at_3": hit_3,
                    "hit_at_5": hit_5,
                    "mrr": mrr,
                    "precision_at_5": precision_5,
                    "recall_at_5": recall_5,
                    "grounded_match": 1.0 if result.grounded else 0.0,
                    "citation_source_accuracy": citations_eval["citation_source_accuracy"],
                    "citation_metadata_accuracy": citations_eval["citation_metadata_accuracy"],
                    "user_leakage": 0.0,
                    "prompt_injection_pass": 1.0,
                    "no_context_pass": 1.0,
                    "total_time_ms": total_time_ms,
                    "retrieval_time_ms": result.retrieval_statistics.retrieval_time_ms,
                    "context_building_time_ms": result.context_building_time_ms,
                    "generation_time_ms": result.generation_statistics.generation_time_ms,
                    "input_tokens": result.generation_statistics.input_tokens,
                    "output_tokens": result.generation_statistics.output_tokens,
                    "total_tokens": result.generation_statistics.total_tokens,
                }

        # Cleanup conversation database entries after verification
        self.conv_svc.delete_conversation(conv_id, user_id)
        return last_turn_result

    def run_all(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """
        Run the complete evaluation suite across the dataset and compute summaries.
        """
        dataset = self.load_dataset()
        case_results = []

        for case in dataset:
            logger.info("Executing Evaluation Case: id=%s  category=%s", case["id"], case["category"])
            res = self.run_case(case)
            case_results.append(res)

        total_cases = len(case_results)
        passed_cases = sum(1 for r in case_results if r["success"])
        failed_cases = total_cases - passed_cases

        # Aggregate performance metrics
        perf_data = aggregate_performance(case_results)

        # Filter for cases that actually expect grounded retrieval (factual, conceptual, multi_part, conversation, prompt_injection)
        grounded_cases = [r for r in case_results if r["expected_document"] and r["category"] not in ("user_isolation", "unrelated", "out_of_domain")]
        ret_denominator = len(grounded_cases) if grounded_cases else 1

        # Average Retrieval Metrics (computed only over grounded cases expecting matches)
        avg_hit_1 = sum(r["hit_at_1"] for r in grounded_cases) / ret_denominator
        avg_hit_3 = sum(r["hit_at_3"] for r in grounded_cases) / ret_denominator
        avg_hit_5 = sum(r["hit_at_5"] for r in grounded_cases) / ret_denominator
        avg_mrr = sum(r["mrr"] for r in grounded_cases) / ret_denominator
        avg_prec_5 = sum(r["precision_at_5"] for r in grounded_cases) / ret_denominator
        avg_recall_5 = sum(r["recall_at_5"] for r in grounded_cases) / ret_denominator

        # Average Answer Quality Metrics (computed only over grounded cases expecting matches)
        avg_keyword = sum(r["keyword_coverage"] for r in grounded_cases) / ret_denominator
        avg_grounded = sum(r["grounded_match"] for r in case_results) / total_cases
        answer_success = sum(1 for r in grounded_cases if r["keyword_coverage"] >= settings.EVAL_MIN_KEYWORD_COVERAGE) / ret_denominator

        # Average Citation Metrics (computed over cases that returned citations)
        cases_with_citations = [r for r in case_results if r["chunks_retrieved"] > 0]
        cit_denominator = len(cases_with_citations) if cases_with_citations else 1
        avg_source_acc = sum(r["citation_source_accuracy"] for r in cases_with_citations) / cit_denominator
        avg_metadata_acc = sum(r["citation_metadata_accuracy"] for r in cases_with_citations) / cit_denominator

        # Security Breach aggregations
        user_isolation_cases = [r for r in case_results if r["category"] == "user_isolation"]
        leakage_rate = sum(r["user_leakage"] for r in user_isolation_cases) / len(user_isolation_cases) if user_isolation_cases else 0.0

        injection_cases = [r for r in case_results if r["category"] == "prompt_injection"]
        injection_breach_rate = sum(1.0 - r["prompt_injection_pass"] for r in injection_cases) / len(injection_cases) if injection_cases else 0.0

        no_context_cases = [r for r in case_results if r["no_context_pass"] is not None and r["category"] in ("unrelated", "out_of_domain")]
        no_context_safe_rate = sum(r["no_context_pass"] for r in no_context_cases) / len(no_context_cases) if no_context_cases else 1.0

        # Pass / Fail criteria checks
        hit_at_5_pass = avg_hit_5 >= settings.EVAL_MIN_HIT_AT_5
        keyword_pass = avg_keyword >= settings.EVAL_MIN_KEYWORD_COVERAGE
        citation_pass = avg_source_acc == 1.0
        isolation_pass = leakage_rate == 0.0
        injection_pass = injection_breach_rate == 0.0
        nocontext_pass = no_context_safe_rate == 1.0

        summary = {
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "failed_cases": failed_cases,
            "thresholds": {
                "hit_at_5": settings.EVAL_MIN_HIT_AT_5,
                "keyword_coverage": settings.EVAL_MIN_KEYWORD_COVERAGE,
            },
            "pass_fail": {
                "hit_at_5": hit_at_5_pass,
                "keyword_coverage": keyword_pass,
                "citation_source_accuracy": citation_pass,
                "user_isolation": isolation_pass,
                "prompt_injection": injection_pass,
                "no_context": nocontext_pass,
            },
            "retrieval": {
                "hit_at_1": round(avg_hit_1, 2),
                "hit_at_3": round(avg_hit_3, 2),
                "hit_at_5": round(avg_hit_5, 2),
                "mrr": round(avg_mrr, 3),
                "precision_at_5": round(avg_prec_5, 2),
                "recall_at_5": round(avg_recall_5, 2),
            },
            "answer": {
                "keyword_coverage": round(avg_keyword, 2),
                "grounded_response_rate": round(avg_grounded, 2),
                "answer_success_rate": round(answer_success, 2),
            },
            "citations": {
                "citation_source_accuracy": round(avg_source_acc, 2),
                "citation_metadata_accuracy": round(avg_metadata_acc, 2),
            },
            "security": {
                "user_isolation_leakage_rate": round(leakage_rate, 2),
                "prompt_injection_breach_rate": round(injection_breach_rate, 2),
                "no_context_safe_handling_rate": round(no_context_safe_rate, 2),
            },
            "performance": perf_data
        }

        # Compile and save Markdown and JSON reports
        compile_report(summary, case_results)
        
        return summary, case_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Study Assistant — RAG Evaluation Runner")
    parser.add_argument("--live", action="store_true", help="Run in LIVE mode using external cloud APIs.")
    args = parser.parse_args()

    runner = EvaluationRunner(live=args.live)
    summary, case_results = runner.run_all()
    
    # Print aligned terminal dashboard
    print_cli_dashboard(summary, len(case_results))
