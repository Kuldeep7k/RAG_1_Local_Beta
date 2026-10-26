# Evaluation dataset

Populate `testset.json` with reviewed questions from your own documents.
Each record must contain:

```json
{
  "question": "What does Article 14 state?",
  "expected_answer": "It guarantees equality before the law and equal protection of the laws.",
  "source_document": "india_constitution_2026.pdf",
  "source_page": 1
}
```

Use the actual citation page from the uploaded PDF; do not guess it. Keep
questions covering exact references, ordinary semantic questions, negative
questions, and permission boundaries. Run `python -m eval.evaluate` before
changing retrieval settings, and calibrate a reranker threshold only from the
measured results.
