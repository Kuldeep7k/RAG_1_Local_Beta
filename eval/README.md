# Evaluation dataset

`testset.json` contains Constitution of India questions keyed to the expected
article reference. Add reviewed questions as the corpus grows. Each record
must contain:

```json
{
  "question": "What does Article 14 state?",
  "expected_answer": "It guarantees equality before the law and equal protection of the laws.",
  "expected_article_ref": "14"
}
```

Run `python -m eval.evaluate` with the indexed Constitution corpus before
changing retrieval settings. To compare the current reranker with a candidate:

```powershell
python -m eval.benchmark_rerankers
```

The standard evaluation checks retrieval only. Add `--answers` when Ollama is
running and you also want the answer-generation word-overlap check:

```powershell
python -m eval.evaluate --answers
```

Set `RERANKER_BENCHMARK_MODELS` to a comma-separated list to compare other
models. The default remains
`cross-encoder/ms-marco-MiniLM-L-6-v2`; configure the application model with
`RERANKER_MODEL`.

The test set checks article-level retrieval instead of hard-coding PDF page
numbers, because PDF page numbering can vary between source editions. Keep
questions covering exact references, ordinary semantic questions, and
questions whose answer depends on nearby constitutional text. Rebuild the
Qdrant collection only after an embedding model shows a measured improvement.
