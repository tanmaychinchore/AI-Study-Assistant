# AI Study Assistant — RAG Service

Production-quality RAG (Retrieval-Augmented Generation) microservice for the AI Study Assistant platform.

## Tech Stack

| Component        | Technology           |
| ---------------- | -------------------- |
| API Framework    | FastAPI              |
| Orchestration    | LangChain            |
| Embedding Model  | BAAI/bge-m3 (1024d)  |
| Vector Database  | Astra DB Serverless  |
| LLM Provider     | Groq                 |
| Document Parsing | PyMuPDF, python-pptx, python-docx |

## Quick Start

```bash
# 1. Create virtual environment (Python 3.11)
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env
# Edit .env with your API keys

# 4. Run the service
python run.py
```

The service starts at **http://127.0.0.1:8000**.

- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/api/v1/health

## Project Structure

```
rag-service/
├── app/
│   ├── main.py              # FastAPI application
│   ├── api/
│   │   ├── router.py        # Central API router
│   │   └── routes/
│   │       └── health.py    # Health check endpoint
│   ├── core/
│   │   ├── config.py        # Pydantic settings
│   │   └── logging.py       # Structured logging
│   ├── schemas/
│   │   └── response.py      # Response models
│   ├── services/            # Business logic (upcoming)
│   ├── loaders/             # Document format loaders (upcoming)
│   └── utils/               # Utilities (upcoming)
├── tests/                   # Test suite (upcoming)
├── .env.example             # Environment template
├── requirements.txt         # Python dependencies
├── run.py                   # Uvicorn entry point
└── README.md
```

## Development Milestones

- [x] Task 1: Project & FastAPI Foundation
- [ ] Task 2: Multi-format Document Extraction
- [ ] Task 3: Document Cleaning & Chunking
- [ ] Task 4: BGE-M3 Embedding Service
- [ ] Task 5: Astra DB Vector Store
- [ ] Task 6: Complete Indexing Pipeline
- [ ] Task 7: Semantic Retrieval Engine
- [ ] Task 8: Groq LLM Integration
- [ ] Task 9: Grounded RAG Query Pipeline
- [ ] Task 10: Citations & Conversation Context
- [ ] Task 11: Evaluation & Retrieval Improvements
- [ ] Task 12: Production Cleanup & Documentation
