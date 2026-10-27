"""Compare reranker models on the labelled Constitution retrieval set.

Run with the indexed corpus and Qdrant available:

    python -m eval.benchmark_rerankers

Override the candidates with a comma-separated environment variable:

    $env:RERANKER_BENCHMARK_MODELS="cross-encoder/ms-marco-MiniLM-L-6-v2,BAAI/bge-reranker-base"
"""
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import TOP_K_RETRIEVAL, TOP_K_RERANK
from eval.evaluate import load_testset, recall_at_k
from retrieval.hybrid import hybrid_search
from retrieval.rerank import rerank


def main() -> None:
    models = [
        model.strip()
        for model in os.environ.get(
            "RERANKER_BENCHMARK_MODELS",
            "cross-encoder/ms-marco-MiniLM-L-6-v2,BAAI/bge-reranker-base",
        ).split(",")
        if model.strip()
    ]
    testset = load_testset()
    candidate_sets = [
        (case, hybrid_search(case["question"], top_k=TOP_K_RETRIEVAL))
        for case in testset
    ]

    print(f"Evaluating {len(testset)} questions at Recall@{TOP_K_RERANK}\n")
    for model_name in models:
        hits = 0
        durations = []
        for case, candidates in candidate_sets:
            started = time.perf_counter()
            ranked = rerank(
                case["question"],
                candidates,
                top_k=TOP_K_RERANK,
                model_name=model_name,
            )
            durations.append((time.perf_counter() - started) * 1000)
            hits += recall_at_k(ranked, case)

        recall = hits / len(testset)
        latency = sum(durations) / len(durations)
        print(f"{model_name}")
        print(f"  Recall@{TOP_K_RERANK}: {recall:.2f}")
        print(f"  Average rerank latency: {latency:.1f} ms\n")


if __name__ == "__main__":
    main()
