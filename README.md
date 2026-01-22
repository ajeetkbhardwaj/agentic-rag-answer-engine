%s
# AI Answering System (Agentic RAG)

This repository implements a production-style Retrieval-Augmented Generation (RAG) system with agentic orchestration. It supports user-uploaded documents, live web search, evidence fusion, and citation-backed answers. The implementation is modular and LLM-provider agnostic.

Quick contents
- `ingestion/` - document loaders, chunking, LlamaIndex integration
- `agents/` - router, doc_rag, web search, fusion, answer generator
- `llm/` - LLM abstraction and provider implementations
- `orchestration/` - simple orchestrator wiring agents
- `api/` - FastAPI backend
- `ui.py` - Gradio UI for uploads & queries
- `config/` - YAML + .env.example

Requirements

Install dependencies (recommended in a venv):

```bash
python -m venv .venv
.venv\\Scripts\\activate    # Windows
pip install -r requirements.txt
```

Configuration

Copy `.env.example` to `.env` and populate API keys for your chosen LLM provider and search tools.

Running the backend (FastAPI)

```bash
# Run FastAPI server
python -m api.app
# or
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Running the UI (Gradio)

```bash
python ui.py
```

Usage

1. Start the backend or run the UI directly.
2. Upload documents (PDF/DOCX/TXT/CSV) via the UI or POST `/upload` to the API.
3. Ask a question via the UI or POST `/query` with `{"query": "..."}`.
4. Answers include inline citations and a source list.

Notes & Next Steps

- The LLM integrations are placeholders that expect API keys and may require updates based on provider SDK versions.
- For production, add authentication, rate limiting, robust scraping, and proper vector DB hosting (Qdrant/Pinecone).
- Consider adding tests and CI for reproducibility.

References
- LlamaIndex, LangChain, LangGraph, CrewAI
- See `config/config.yaml` for default system settings
