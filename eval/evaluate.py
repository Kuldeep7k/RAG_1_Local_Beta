"""Evaluation harness for Constitution-specific retrieval and answer checks."""
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.hybrid import hybrid_search
from retrieval.rerank import rerank
from generation.prompts import build_prompt
from generation.llm import generate
from config import TOP_K_RETRIEVAL, TOP_K_RERANK

TESTSET_PATH = Path(__file__).parent / "testset.json"


def recall_at_k(retrieved_chunks: list, case: dict) -> bool:
    expected_article = case.get("expected_article_ref")
    if expected_article is not None:
        return any(
            c.get("article_ref") == expected_article for c in retrieved_chunks
        )
    expected_document = case["source_document"]
    expected_page = case["source_page"]
    return any(
        c["source"] == expected_document and c["page"] == expected_page
        for c in retrieved_chunks
    )


def word_overlap_score(answer: str, expected: str) -> float:
    expected_words = set(expected.lower().split())
    if not expected_words:
        return 0.0
    answer_words = set(answer.lower().split())
    return len(expected_words & answer_words) / len(expected_words)


def load_testset() -> list:
    testset = json.loads(TESTSET_PATH.read_text())
    if not testset:
        raise ValueError(
            "eval/testset.json is empty. Add reviewed Constitution questions "
            "before running evaluation."
        )
    return testset


def run_evaluation(include_answers: bool = False):
    testset = load_testset()
    results = []

    for case in testset:
        candidates = hybrid_search(case["question"], top_k=TOP_K_RETRIEVAL)
        started = time.perf_counter()
        top_chunks = rerank(case["question"], candidates, top_k=TOP_K_RERANK)
        rerank_ms = (time.perf_counter() - started) * 1000

        hit = recall_at_k(top_chunks, case)
        result = {
            "question": case["question"],
            "retrieval_hit": hit,
            "rerank_ms": round(rerank_ms, 1),
        }

        if include_answers:
            messages = build_prompt(case["question"], top_chunks)
            answer = generate(messages)
            result["answer"] = answer
            result["word_overlap"] = round(
                word_overlap_score(answer, case["expected_answer"]), 2
            )
        results.append(result)

    recall = sum(r["retrieval_hit"] for r in results) / len(results)
    avg_rerank_ms = sum(r["rerank_ms"] for r in results) / len(results)

    print(f"\nModel: configured reranker")
    print(f"Recall@{TOP_K_RERANK}: {recall:.2f}")
    print(f"Average rerank latency: {avg_rerank_ms:.1f} ms")
    if include_answers:
        avg_overlap = sum(r["word_overlap"] for r in results) / len(results)
        print(f"Average word overlap (faithfulness proxy): {avg_overlap:.2f}")
    print()

    for r in results:
        status = "HIT " if r["retrieval_hit"] else "MISS"
        print(f"[{status}] {r['question']} ({r['rerank_ms']:.1f} ms)")
        if include_answers:
            print(f"  -> {r['answer'][:150]}...")
        print()

    return results


if __name__ == "__main__":
    run_evaluation(include_answers="--answers" in sys.argv)
