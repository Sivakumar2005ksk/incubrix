# IncuBrix Prompt-Based Video Editing Engine

An editing engine that turns natural language instructions (*"cut out the intro"*, *"remove pricing"*, *"add caption 'Welcome!'"*) into exact, reversible edits on a video, and renders the result non-destructively with frame-accurate audio/video synchronization.

---

## Architecture Overview

```
                      ┌────────────────────────────┐
                      │    Session Runner / CLI    │
                      └─────────────┬──────────────┘
                                    │ HTTP / CLI
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FASTAPI SERVICE LAYER                           │
│  • Idempotency Gate (Idempotency-Key)   • Structured JSON Logs         │
│  • Async Job Manager (202 Accepted)     • Health/Ready Probes          │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼                                ▼
       ┌────────────────────────┐       ┌────────────────────────┐
       │     INTENT COMPILER    │       │     TRANSCRIPT ENGINE  │
       │  • 3-Way Decision Gate │       │  • Word Tokens (w_001) │
       │  • Time Policy Resolver│       │  • Silences (s_001)    │
       └────────────┬───────────┘       └────────────┬───────────┘
                    │                                │
                    └────────────────┬───────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  CONTENT-ANCHORED STATE MACHINE                        │
│  • Pure Projection Function f(Source, Ops) -> Playhead                 │
│  • Non-Destructive Event Sourced History                               │
│  • Selective Restore & Branching Undo/Redo                             │
│  • OpenTimelineIO (OTIO) Exporter                                      │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     FFMPEG RENDERING COMPOSITOR                        │
│  • Audio Ducking (sidechaincompress)  • Frame-accurate PTS Concats     │
│  • Drawtext / Subtitle Burn-In        • Mute / Bleep Audio Synthesis   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Setup in Under 10 Commands

```bash
# 1. Clone repository
git clone <repo-url>
cd incubrix

# 2. Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment configuration
cp .env.example .env

# 5. Run the full test suite (Zero API keys required!)
python run_tests.py

# 6. Run the 8-step worked example session
python -m src.cli.session_runner -i fixtures/worked_example.yaml -o results/worked_example

# 7. Start the FastAPI microservice
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Or via Docker Compose:
```bash
docker compose up -d
```

---

## How to Run Tests and the Batch Runner

### 1. Run Offline Tests (Unit, Integration, Determinism)
```bash
python run_tests.py
```

### 2. Run the Scripted Session Runner
```bash
python -m src.cli.session_runner --input fixtures/worked_example.yaml --output results/
```
Outputs generated in `./results/`:
- `results.json`: Execution log for each step (status, summary, hash, duration).
- `final_timeline.json`: Complete state machine projection.
- `final_timeline.otio`: OpenTimelineIO export for Premiere Pro/Resolve.
- `manifest.json`: Execution manifest with versions, seeds, timings, and cost estimate.

### 3. Run the Batch Runner on a Folder of Sessions
```bash
python -m src.cli.session_runner --input ./fixtures --output ./batch_results/
```

---

## Standardized Error Codes Catalog (Integration Standard #5)

All error payloads follow:
```json
{
  "error": {
    "code": "UPPER_SNAKE_CASE",
    "message": "Human-readable explanation",
    "retryable": false,
    "details": {}
  }
}
```

| Error Code | HTTP Status | Meaning |
| :--- | :--- | :--- |
| `SESSION_NOT_FOUND` | 404 | The requested `session_id` does not exist in memory or storage. |
| `JOB_NOT_FOUND` | 404 | The requested background render `job_id` does not exist. |
| `OUT_OF_BOUNDS_TIMESTAMP`| 400 | Specified cut/speed timestamp exceeds video duration. |
| `IDEMPOTENCY_CONFLICT` | 409 | Conflicting request with the same Idempotency-Key. |
| `INVALID_INSTRUCTION` | 422 | Command is syntactically invalid or cannot be parsed into an operation. |
| `INTERNAL_SERVER_ERROR` | 500 | Unhandled internal exception during execution. |

---

## Environment Settings Catalog (Integration Standard #6)

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `PORT` | `8000` | Port for the HTTP API server |
| `HOST` | `0.0.0.0` | Bind host address |
| `STORAGE_TYPE` | `local` | `'local'` or `'s3'` for MinIO/AWS |
| `LOCAL_STORAGE_DIR` | `./storage` | Directory where rendered outputs are stored |
| `S3_ENDPOINT_URL` | `http://localhost:9000` | MinIO / S3 endpoint URL |
| `S3_ACCESS_KEY` | `minioadmin` | S3 Access Key |
| `S3_SECRET_KEY` | `minioadmin` | S3 Secret Key |
| `S3_BUCKET_NAME` | `incubrix-outputs` | S3 bucket name |
| `WEBHOOK_SECRET_KEY` | `incubrix-super-secret-hmac-key`| Secret used to sign HMAC-SHA256 callback webhooks |
| `STT_PROVIDER` | `mock` | `'mock'` (deterministic offline) or `'whisper'` |
| `NLP_PROVIDER` | `deterministic` | `'deterministic'` (offline) or `'llm'` |
