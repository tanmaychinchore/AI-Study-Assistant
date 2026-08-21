# AI Study Assistant — RAG Service

Production-quality RAG (Retrieval-Augmented Generation) microservice for the AI Study Assistant platform.
Enables multi-format document ingestion, cleaning, chunking, semantic retrieval, citations, conversation persistence, evaluation, and security hardening.

---

## 🚀 System Architecture

```
                         CLIENT
                           │
                           ▼
                        FastAPI
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   Documents          Retrieval          Conversations
        │                  │                  │
        ▼                  ▼                  ▼
   Extraction          BGE-M3            MongoDB
        │                  │
        ▼                  ▼
   Cleaning           Astra DB
        │                  │
        ▼                  ▼
   Chunking          Retrieved Chunks
        │                  │
        └──────────┬───────┘
                   ▼
                RAGService
                   │
                   ▼
               Groq LLM
                   │
                   ▼
             Grounded Answer (with Citations)
```

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **API Framework** | FastAPI | Async Python web framework |
| **Database (Vector)** | Astra DB Serverless | Hosted Cassandra database for 1024d embeddings |
| **Database (NoSQL)** | MongoDB | Document database for conversation logs & message history |
| **Embedding Model** | BAAI/bge-m3 | 1024-dimension multilingual embedding model (local CPU/GPU execution) |
| **LLM Orchestration**| Groq API | High-speed LLM generation using Llama-3.3-70b-versatile |
| **Document Parsing** | PyMuPDF, python-pptx, python-docx | High-fidelity extractors for PDF, PPTX, DOCX, and TXT |

---

## 🔑 Environment Variables Reference

Copy `.env.example` to `.env` and fill in the required keys:

| Variable | Required? | Default | Description / Example |
| :--- | :---: | :---: | :--- |
| `ENVIRONMENT` | Yes | `development` | Environment mode (`development` or `production`) |
| `LOG_LEVEL` | Yes | `INFO` | Logger verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `CORS_ALLOWED_ORIGINS`| Yes | `http://localhost:8000,http://127.0.0.1:8000` | Comma-separated list of allowed origins |
| `GROQ_API_KEY` | Yes | | API credentials for Groq Cloud service |
| `ASTRA_DB_API_ENDPOINT`| Yes | | Astra DB instance endpoint |
| `ASTRA_DB_APPLICATION_TOKEN`| Yes | | Astra DB Application token (starts with `AstraCS:`) |
| `ASTRA_DB_KEYSPACE` | Yes | `default_keyspace` | Target Astra DB keyspace |
| `MONGODB_URI` | Yes | `mongodb://localhost:27017` | MongoDB connection URI |
| `MONGODB_DATABASE` | Yes | `ai_study_assistant` | Conversation logs database name |

---

## 📦 Getting Started

### Prerequisites
- **Python 3.11** installed
- Local **MongoDB** running (e.g., `mongodb://localhost:27017`)
- Astra DB account & Groq API key

### Installation

1. Clone the repository and navigate to the `rag-service` directory:
   ```bash
   cd rag-service
   ```

2. Create and activate a Python 3.11 virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   # source venv/bin/activate   # macOS / Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment:
   ```bash
   copy .env.example .env
   # Edit .env and supply keys
   ```

---

## 🖥️ Running the Server

### Development Mode (with reloading)
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Production Mode (multi-worker)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 🧪 Testing and Verification

Run the test suite offline (using mocks/mongomock/local tests):
```bash
# Run all unit, integration, and production safety tests
venv\Scripts\python.exe -m pytest tests/ -v
```

---

## 🚀 RAG System Quality Evaluation (Task 11)

Measure retrieval quality, groundedness accuracy, citation fidelity, security compliance, and latency performance using the evaluation runner:

### Run Offline Evaluation (Default)
Uses mocked database schemas and LLM responses to verify metric calculations:
```bash
python -m evaluation.runner
```

### Run Live Evaluation
Executes queries against live Astra DB and Groq services:
```bash
python -m evaluation.runner --live
```

---

## 🔒 Production Hardening & Safety Checklist (Task 12)

- [x] **CORS Locked**: Wildcard CORS blocked when credentials enabled; configurable allowed origins.
- [x] **Strict Size Check**: Upload size checked before processing; maximum file size limited to **50 MB**.
- [x] **Magic Byte Verification**: Verifies file headers (PDF, ZIP/OOXML, text) to block disguised malicious uploads.
- [x] **Path Traversal Protection**: Upload filenames sanitized to prevent directory traversal.
- [x] **Redacted Tracebacks**: Standardized HTTP 500 exceptions hide system internals, database credentials, and tracebacks from client responses.
- [x] **No Logs Leakage**: Environment variable validation redacts API keys and database tokens from console logs.
- [x] **Development Endpoint Guard**: Embedding and vector test endpoints return HTTP 403 Forbidden in `production` mode.
- [x] **Health and Readiness Probes**: Separate `/health/liveness` and `/health/readiness` endpoints report database connection status.
- [x] **Guaranteed Temp Cleanup**: Uploaded temporary files are cleaned up under `finally` statements.
- [x] **Request Traceability**: Every call receives an `X-Correlation-ID` header, propagated through all logging filters.
