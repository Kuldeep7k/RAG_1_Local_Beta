"""Evaluation harness for labelled retrieval and answer checks.

Populate testset.json with real, reviewed questions before using a metric to
set a reranker threshold or accept a pipeline change.
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.hybrid import hybrid_search
from retrieval.rerank import rerank
from generation.prompts import build_prompt
from generation.llm import generate
from config import TOP_K_RETRIEVAL, TOP_K_RERANK

TESTSET_PATH = Path(__file__).parent / "testset.json"


def recall_at_k(retrieved_chunks: list, expected_document: str, expected_page: int) -> bool:
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


def run_evaluation():
    testset = json.loads(TESTSET_PATH.read_text())
    if not testset:
        raise ValueError(
            "eval/testset.json is empty. Add reviewed questions, expected "
            "answers, source_document, and source_page before running evaluation."
        )
    results = []

    for case in testset:
        candidates = hybrid_search(case["question"], top_k=TOP_K_RETRIEVAL)
        top_chunks = rerank(case["question"], candidates, top_k=TOP_K_RERANK)

        hit = recall_at_k(top_chunks, case["source_document"], case["source_page"])

        messages = build_prompt(case["question"], top_chunks)
        answer = generate(messages)
        overlap = word_overlap_score(answer, case["expected_answer"])

        results.append({
            "question": case["question"],
            "retrieval_hit": hit,
            "answer": answer,
            "word_overlap": round(overlap, 2),
        })

    recall = sum(r["retrieval_hit"] for r in results) / len(results)
    avg_overlap = sum(r["word_overlap"] for r in results) / len(results)

    print(f"\nRecall@{TOP_K_RERANK}: {recall:.2f}")
    print(f"Average word overlap (faithfulness proxy): {avg_overlap:.2f}\n")

    for r in results:
        status = "HIT " if r["retrieval_hit"] else "MISS"
        print(f"[{status}] {r['question']}\n  -> {r['answer'][:150]}...\n")

    return results


if __name__ == "__main__":
    run_evaluation()
