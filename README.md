# Local RAG (Windows, no Docker, 4GB VRAM)

> **Current scope:** This project is currently configured and demonstrated
> using the Constitution of India PDF as its primary knowledge source. The
> ingestion pipeline supports additional PDF, DOCX, TXT, and Markdown
> documents, so it can be extended to other knowledge bases.

## One-time setup

```
ollama pull llama3.2:3b
```

Download the Qdrant Windows binary from https://github.com/qdrant/qdrant/releases
and place `qdrant.exe` somewhere convenient.

```
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Validate the repository without starting Ollama or Qdrant
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m compileall -q -x "venv|storage|snapshots|tmp" .
```

## Running the system (five processes, each in its own terminal)

Pick one secret string once and reuse it everywhere below, e.g. `myqdrantkey123`.

**Terminal 1 - vector DB**
```powershell
$env:QDRANT__SERVICE__HOST="127.0.0.1"
$env:QDRANT__SERVICE__API_KEY="myqdrantkey123"
.\qdrant.exe
```
Binds Qdrant to localhost only and requires the key for every request.

**Terminal 2 - API**
```powershell
.\venv\Scripts\Activate.ps1
$env:QDRANT_API_KEY="myqdrantkey123"
$env:RAG_AUTH_USERS='{"local_user":{"role":"employee","access_level":"employee","department":null}}'
$env:RAG_AUTH_TOKENS='{"replace-with-a-random-token":"local_user"}'
uvicorn main:app --reload --reload-exclude "venv/*"
```
`QDRANT_API_KEY` must contain the same value as `QDRANT__SERVICE__API_KEY`.

**Terminal 3 - ingestion worker**
```powershell
.\venv\Scripts\Activate.ps1
$env:QDRANT_API_KEY="myqdrantkey123"
python worker.py
```

**Terminal 4 - chat UI**
```powershell
.\venv\Scripts\Activate.ps1
$env:RAG_API_TOKEN="replace-with-the-same-random-token"
streamlit run ui.py
```
Opens http://localhost:8501 with the chat window, upload sidebar, and retrieval debug panel.

**Terminal 5 - Ollama**
```powershell
ollama serve
```
Keep Ollama running while using chat. If Ollama is already running as a
desktop service, do not start a second `ollama serve` process.

Embeddings run locally in the worker with
`sentence-transformers/all-MiniLM-L6-v2`; no Ollama connection is needed for
ingestion. Ollama is used only by the chat API for LLM answer generation.

**Every `$env:` line must be re-run in a fresh terminal.**

## Usage

Uploads are limited to 25 MiB and the API accepts PDF, DOCX, TXT, and MD files.
Chat and upload requests require a bearer token configured in
`RAG_AUTH_TOKENS`; the API derives identity from that token and ignores
caller-supplied identity fields. Configure matching users in `RAG_AUTH_USERS`.

Chunks default to 220 estimated tokens with 35-token word-aligned overlap.
These limits leave headroom below the 256-token context limit of the default
MiniLM embedding model and prevent embedding-time truncation.

The worker claims each job atomically and processes it exactly once. Any
provider, parsing, indexing, or unexpected failure marks the job permanently
failed and stores the error; it is never retried automatically. Abandoned
processing jobs are also marked failed after `JOB_STALE_SECONDS`, and every
terminal historical job is removed when the worker starts. UTF-8 text files
that cannot be decoded are rejected.

**Ingest directly:**
```powershell
python -m ingestion.ingest data\raw\some_file.pdf --access-level employee
```

**Check or cancel your jobs:**
```powershell
$headers = @{ Authorization = ("Bearer " + $env:RAG_API_TOKEN) }
Invoke-RestMethod "http://localhost:8000/jobs" -Headers $headers
Invoke-RestMethod "http://localhost:8000/jobs/1" -Headers $headers
Invoke-RestMethod -Method Post "http://localhost:8000/jobs/1/cancel" -Headers $headers
```

`/health` is a liveness check. `/ready` verifies SQLite and Qdrant. Job
responses omit server paths and redact internal errors. Cancellation is
cooperative between embedding batches and removes partial index state.

**Search raw chunks for debugging:**
```powershell
python debug_search.py "some phrase"
python debug_search.py --page 32
```

## Evaluation

Fill in `eval/testset.json` with real questions and answers, then:
```
python -m eval.evaluate
```

## Notes

- Protected API routes require an Authorization header containing the configured bearer token.
- `RAG_AUTH_TOKENS` maps tokens to user IDs; `RAG_AUTH_USERS` maps users to access scopes.
- All state lives in `data/rag.db` and Qdrant's `storage/` directory.
- If retrieval quality is poor, check `ollama list` and verify Qdrant is running before ingestion.
- Always bind Qdrant to `127.0.0.1` and set `QDRANT__SERVICE__API_KEY`.
- `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `EMBEDDING_VERSION`, and
  `QDRANT_COLLECTION` are configurable. The default embedding dimension is 384.
- `OLLAMA_BASE_URL` controls the Ollama provider connection.

