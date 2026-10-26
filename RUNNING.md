# Running Local RAG on Windows

> **Current scope:** This setup currently uses the Constitution of India PDF
> as the primary knowledge source. You can upload additional PDF, DOCX, TXT,
> and Markdown files through the application.

## First-time setup

Install Ollama, then download the required models:

```powershell
ollama pull llama3.2:3b
```

Download `qdrant.exe` from https://github.com/qdrant/qdrant/releases and place it in the project root.

Create and install the Python environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Start the application

Use five PowerShell terminals. Choose one Qdrant secret and reuse it in the API and worker terminals.

Example secret: `myqdrantkey123`

### Terminal 1: Qdrant

```powershell
$env:QDRANT__SERVICE__HOST="127.0.0.1"
$env:QDRANT__SERVICE__API_KEY="myqdrantkey123"
.\qdrant.exe
```

Keep this terminal running. Qdrant listens locally at `http://127.0.0.1:6333`.

### Terminal 2: FastAPI

```powershell
.\venv\Scripts\Activate.ps1
$env:QDRANT_API_KEY="myqdrantkey123"
$env:RAG_AUTH_USERS='{"local_user":{"role":"employee","access_level":"employee","department":null}}'
$env:RAG_AUTH_TOKENS='{"pAQIDD3zc6ijNavXr1BoVTvM7eEdqVf2":"local_user"}'
uvicorn main:app --reload --reload-exclude "venv/"
```

`QDRANT_API_KEY` must match `QDRANT__SERVICE__API_KEY`.

### Terminal 3: Ingestion worker

```powershell
.\venv\Scripts\Activate.ps1
$env:QDRANT_API_KEY="myqdrantkey123"
python worker.py
```

Keep this terminal running. It processes uploaded documents in the background.
Each job is processed once. Provider, parsing, or indexing failures are stored
as permanent `failed` jobs and are not retried automatically. Jobs abandoned
by a stopped worker are also marked failed after `JOB_STALE_SECONDS`. Terminal
job history is removed when the worker starts.

### Terminal 4: Streamlit UI

```powershell
.\venv\Scripts\Activate.ps1
$env:RAG_API_TOKEN="pAQIDD3zc6ijNavXr1BoVTvM7eEdqVf2"
streamlit run ui.py
```

Open http://localhost:8501.

### Terminal 5: Ollama

```powershell
ollama serve
```

Keep this terminal running while using chat. Ollama is required only for chat
answer generation. Embeddings run directly in
the worker with `sentence-transformers/all-MiniLM-L6-v2`, so ingestion does not
depend on the Ollama service. Verify the LLM model with:

```powershell
ollama list
```


## Use the application

1. Open the Streamlit URL.
2. Upload a PDF, DOCX, TXT, or Markdown file.
3. Wait for ingestion to finish.
4. Ask questions in the chat box.
5. Open retrieval details or sources to inspect grounding.

Uploads are limited to 25 MiB by default. The worker parses, chunks, embeds, and indexes each document in the background.

## Check job status

Use the same bearer token configured in the UI:

```powershell
$headers = @{ Authorization = ("Bearer " + $env:RAG_API_TOKEN) }
Invoke-RestMethod "http://localhost:8000/jobs" -Headers $headers
Invoke-RestMethod "http://localhost:8000/jobs/1" -Headers $headers
```

## Stop ingestion

The UI provides a **Stop ingestion** action while a document is being indexed. Cancellation is cooperative and takes effect between embedding batches.

You can also cancel a job through the API:

```powershell
$headers = @{ Authorization = ("Bearer " + $env:RAG_API_TOKEN) }
Invoke-RestMethod -Method Post "http://localhost:8000/jobs/1/cancel" -Headers $headers
```

## Stop the application

Stop each terminal with `Ctrl+C` in this order:

1. Streamlit UI
2. Ingestion worker
3. FastAPI
4. Qdrant
5. Ollama

## Cleanup caches

Preview cleanup first:

```powershell
.\cleanup.ps1 -WhatIf
```

Remove runtime logs, Python caches, and test caches:

```powershell
.\cleanup.ps1
```

To include temporary upload files:

```powershell
.\cleanup.ps1 -IncludeTempUploads
```

The cleanup script preserves `data/`, uploaded documents, Qdrant `storage/`, `snapshots/`, `venv/`, and `qdrant.exe`.

## Validation

Run the test suite and compile checks without starting external services:

```powershell
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m compileall -q -x "venv|storage|snapshots|tmp" .
```

