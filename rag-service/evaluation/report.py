"""
Report compilation, JSON/Markdown saving, and CLI printing for RAG Evaluation.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any


def compile_report(
    summary: dict[str, Any],
    case_results: list[dict[str, Any]],
    output_dir: str = "evaluation/results"
) -> tuple[str, str]:
    """
    Compile the evaluation results into JSON and Markdown reports and save them.
    Returns (json_path, markdown_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    
    full_report = {
        "timestamp": timestamp,
        "summary": summary,
        "results": case_results
    }
    
    # 1. Save JSON Report
    json_path = os.path.join(output_dir, "rag_evaluation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    # 2. Compile Markdown Report
    markdown_path = os.path.join(output_dir, "rag_evaluation_report.md")
    
    md_lines = [
        "# AI Study Assistant — RAG Evaluation Report",
        f"\n**Timestamp (UTC)**: {timestamp}",
        "\n## 1. Summary Metrics",
        "\n| Metric Category | Metric | Value | Target / Threshold | Pass / Fail |",
        "|---|---|---|---|---|",
    ]
    
    # Add Retrieval Metrics
    ret = summary.get("retrieval", {})
    md_lines.append(f"| Retrieval | Hit@1 | {ret.get('hit_at_1', 0.0):.2f} | - | - |")
    md_lines.append(f"| Retrieval | Hit@3 | {ret.get('hit_at_3', 0.0):.2f} | - | - |")
    md_lines.append(f"| Retrieval | Hit@5 | {ret.get('hit_at_5', 0.0):.2f} | {summary.get('thresholds', {}).get('hit_at_5', 0.70):.2f} | {'PASS' if summary.get('pass_fail', {}).get('hit_at_5') else 'FAIL'} |")
    md_lines.append(f"| Retrieval | MRR | {ret.get('mrr', 0.0):.3f} | - | - |")
    md_lines.append(f"| Retrieval | Precision@5 | {ret.get('precision_at_5', 0.0):.2f} | - | - |")
    md_lines.append(f"| Retrieval | Recall@5 | {ret.get('recall_at_5', 0.0):.2f} | - | - |")
    
    # Add Answer Quality Metrics
    ans = summary.get("answer", {})
    md_lines.append(f"| Answer Quality | Keyword Coverage | {ans.get('keyword_coverage', 0.0):.2f} | {summary.get('thresholds', {}).get('keyword_coverage', 0.70):.2f} | {'PASS' if summary.get('pass_fail', {}).get('keyword_coverage') else 'FAIL'} |")
    md_lines.append(f"| Answer Quality | Grounded Response Rate | {ans.get('grounded_response_rate', 0.0):.2f} | - | - |")
    md_lines.append(f"| Answer Quality | Answer Success Rate | {ans.get('answer_success_rate', 0.0):.2f} | - | - |")

    # Add Citation Metrics
    cit = summary.get("citations", {})
    md_lines.append(f"| Citations | Source Accuracy | {cit.get('citation_source_accuracy', 0.0):.2f} | 1.00 | {'PASS' if summary.get('pass_fail', {}).get('citation_source_accuracy') else 'FAIL'} |")
    md_lines.append(f"| Citations | Metadata Accuracy | {cit.get('citation_metadata_accuracy', 0.0):.2f} | - | - |")

    # Add Security Metrics
    sec = summary.get("security", {})
    md_lines.append(f"| Security | User Isolation Leakage Rate | {sec.get('user_isolation_leakage_rate', 0.0):.2f} | 0.00 | {'PASS' if summary.get('pass_fail', {}).get('user_isolation') else 'FAIL'} |")
    md_lines.append(f"| Security | Prompt Injection Breach Rate | {sec.get('prompt_injection_breach_rate', 0.0):.2f} | 0.00 | {'PASS' if summary.get('pass_fail', {}).get('prompt_injection') else 'FAIL'} |")
    md_lines.append(f"| Security | No-Context Safe Handling | {sec.get('no_context_safe_handling_rate', 0.0):.2f} | 1.00 | {'PASS' if summary.get('pass_fail', {}).get('no_context') else 'FAIL'} |")

    # Latency Metrics
    perf = summary.get("performance", {}).get("latency", {})
    md_lines.extend([
        "\n## 2. Performance & Latency (ms)",
        "\n| Phase | Min | Max | Average | P50 (Median) | P95 |",
        "|---|---|---|---|---|---|",
        f"| **Total End-to-End** | {perf.get('min_total', 0.0):.1f} | {perf.get('max_total', 0.0):.1f} | {perf.get('avg_total', 0.0):.1f} | {perf.get('p50_total', 0.0):.1f} | {perf.get('p95_total', 0.0):.1f} |",
        f"| Retrieval | - | - | {perf.get('avg_retrieval', 0.0):.1f} | - | - |",
        f"| Context Building | - | - | {perf.get('avg_context', 0.0):.1f} | - | - |",
        f"| LLM Generation | - | - | {perf.get('avg_generation', 0.0):.1f} | - | - |",
    ])

    # Token Metrics
    tokens = summary.get("performance", {}).get("tokens", {})
    md_lines.extend([
        "\n## 3. Token Consumption (Average)",
        f"\n- **Average Input Tokens**: {tokens.get('avg_input', 0.0):.1f}",
        f"- **Average Output Tokens**: {tokens.get('avg_output', 0.0):.1f}",
        f"- **Average Total Tokens**: {tokens.get('avg_total', 0.0):.1f}",
    ])

    # Case Breakdown Table
    md_lines.extend([
        "\n## 4. Test Case Breakdown",
        "\n| ID | Category | Question | Expected Doc | Chunks | Grounded | Coverage | Result |",
        "|---|---|---|---|---|---|---|---|",
    ])
    for result in case_results:
        cid = result.get("case_id")
        category = result.get("category")
        q = result.get("question", "")[:40] + ("..." if len(result.get("question", "")) > 40 else "")
        expected = result.get("expected_document") or "-"
        chunks_retrieved = result.get("chunks_retrieved", 0)
        grounded = "Yes" if result.get("grounded") else "No"
        cov = f"{result.get('keyword_coverage', 0.0):.2f}"
        res = "PASS" if result.get("success") else "FAIL"
        md_lines.append(f"| {cid} | {category} | {q} | {expected} | {chunks_retrieved} | {grounded} | {cov} | **{res}** |")

    # Evaluation Limitations section
    md_lines.extend([
        "\n## 5. Evaluation Limitations",
        "\n> [!WARNING]",
        "> 1. **Small Dataset**: The evaluation uses a small targeted evaluation dataset designed for functional checking. It does not replace large-scale benchmarking.",
        "> 2. **Keyword Coverage limitations**: Keyword coverage checks case-insensitive presence of key phrases. It does not measure semantic exactness or correct grammatical synthesis.",
        "> 3. **Rule-Based Groundedness**: Deterministic verification handles out-of-domain and zero-context assertions reliably but cannot guarantee detection of all creative LLM hallucinations.",
        "> 4. **Environment Dependency**: Performance latencies and API response times are highly variable depending on API queue size, network transit, local CPU usage, and Groq server loads.",
        "> 5. **Mock vs Live Evaluation**: Offline/Mocked evaluation validates the code pipelines, statistics aggregation, and edge cases, but live evaluation results vary based on model versions and document index contents."
    ])

    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return json_path, markdown_path


def print_cli_dashboard(summary: dict[str, Any], total_cases: int) -> None:
    """
    Print an aligned CLI dashboard summary.
    """
    ret = summary.get("retrieval", {})
    ans = summary.get("answer", {})
    cit = summary.get("citations", {})
    sec = summary.get("security", {})
    perf = summary.get("performance", {}).get("latency", {})
    tokens = summary.get("performance", {}).get("tokens", {})
    pf = summary.get("pass_fail", {})

    print("=====================================")
    print("AI STUDY ASSISTANT — RAG EVALUATION")
    print("=====================================")
    print(f"Cases: {total_cases}")
    print()
    print("Retrieval")
    print("---------")
    print(f"Hit@1:  {ret.get('hit_at_1', 0.0):.2f}")
    print(f"Hit@3:  {ret.get('hit_at_3', 0.0):.2f}")
    print(f"Hit@5:  {ret.get('hit_at_5', 0.0):.2f}  [{'PASS' if pf.get('hit_at_5') else 'FAIL'}]")
    print(f"MRR:    {ret.get('mrr', 0.0):.3f}")
    print()
    print("Answer Quality")
    print("--------------")
    print(f"Keyword Coverage: {ans.get('keyword_coverage', 0.0):.2f}  [{'PASS' if pf.get('keyword_coverage') else 'FAIL'}]")
    print(f"Groundedness:     {ans.get('grounded_response_rate', 0.0):.2f}")
    print()
    print("Citations")
    print("---------")
    print(f"Source Accuracy:   {cit.get('citation_source_accuracy', 0.0):.2f}  [{'PASS' if pf.get('citation_source_accuracy') else 'FAIL'}]")
    print(f"Metadata Accuracy: {cit.get('citation_metadata_accuracy', 0.0):.2f}")
    print()
    print("Security")
    print("--------")
    print(f"User Isolation:   {'PASS' if pf.get('user_isolation') else 'FAIL'}")
    print(f"Prompt Injection: {'PASS' if pf.get('prompt_injection') else 'FAIL'}")
    print(f"No-Context Safe:  {'PASS' if pf.get('no_context') else 'FAIL'}")
    print()
    print("Performance")
    print("-----------")
    print(f"Average: {perf.get('avg_total', 0.0):.1f} ms")
    print(f"P50:     {perf.get('p50_total', 0.0):.1f} ms")
    print(f"P95:     {perf.get('p95_total', 0.0):.1f} ms")
    print()
    print("Tokens")
    print("------")
    print(f"Average Input:  {int(tokens.get('avg_input', 0.0))}")
    print(f"Average Output: {int(tokens.get('avg_output', 0.0))}")
    print(f"Average Total:  {int(tokens.get('avg_total', 0.0))}")
    print()
    print("=====================================")
    print("Evaluation Complete")
    print("=====================================")
