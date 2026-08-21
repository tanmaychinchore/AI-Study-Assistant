"""
Performance and token metrics for RAG Evaluation.
Aggregates latencies (including P50, P95 percentiles) and token consumption metrics.
"""

import math
from typing import Any


def calculate_percentile(sorted_data: list[float], percentile: float) -> float:
    """
    Calculate the percentile of a sorted list of numbers.
    """
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * percentile
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return d0 + d1


def aggregate_performance(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate latencies and token counts across multiple evaluation runs.
    """
    if not runs:
        return {
            "latency": {
                "avg_total": 0.0, "min_total": 0.0, "max_total": 0.0, "p50_total": 0.0, "p95_total": 0.0,
                "avg_retrieval": 0.0, "avg_context": 0.0, "avg_generation": 0.0
            },
            "tokens": {
                "avg_input": 0.0, "avg_output": 0.0, "avg_total": 0.0
            }
        }

    total_times = []
    retrieval_times = []
    context_times = []
    generation_times = []

    input_tokens = []
    output_tokens = []
    total_tokens = []

    for run in runs:
        total_times.append(run.get("total_time_ms", 0.0))
        retrieval_times.append(run.get("retrieval_time_ms", 0.0))
        context_times.append(run.get("context_building_time_ms", 0.0))
        generation_times.append(run.get("generation_time_ms", 0.0))

        input_tokens.append(run.get("input_tokens", 0))
        output_tokens.append(run.get("output_tokens", 0))
        total_tokens.append(run.get("total_tokens", 0))

    total_times.sort()
    
    avg_total = sum(total_times) / len(total_times)
    min_total = total_times[0]
    max_total = total_times[-1]
    p50_total = calculate_percentile(total_times, 0.50)
    p95_total = calculate_percentile(total_times, 0.95)

    avg_retrieval = sum(retrieval_times) / len(retrieval_times)
    avg_context = sum(context_times) / len(context_times)
    avg_generation = sum(generation_times) / len(generation_times)

    avg_input = sum(input_tokens) / len(input_tokens)
    avg_output = sum(output_tokens) / len(output_tokens)
    avg_total_tokens = sum(total_tokens) / len(total_tokens)

    return {
        "latency": {
            "avg_total": round(avg_total, 2),
            "min_total": round(min_total, 2),
            "max_total": round(max_total, 2),
            "p50_total": round(p50_total, 2),
            "p95_total": round(p95_total, 2),
            "avg_retrieval": round(avg_retrieval, 2),
            "avg_context": round(avg_context, 2),
            "avg_generation": round(avg_generation, 2),
        },
        "tokens": {
            "avg_input": round(avg_input, 2),
            "avg_output": round(avg_output, 2),
            "avg_total": round(avg_total_tokens, 2),
        }
    }
