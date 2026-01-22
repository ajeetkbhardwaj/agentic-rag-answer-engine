"""
FastAPI backend for the AI Answering System.

Endpoints:
- GET  /health
- POST /upload  (multipart file upload)
- POST /query   (JSON: {"query": "...", "top_k": 5})

This wires into the orchestrator and DocRAG ingestion.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import shutil
import logging

from config import config
from orchestration.graph import Orchestrator
from agents.doc_rag_agent import get_doc_rag_agent

logger = logging.getLogger(__name__)
app = FastAPI(title=config.app.get('name', 'AI Answering System') if hasattr(config, 'app') else 'AI Answering System')

# CORS (allow local testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload dir exists
UPLOAD_DIR = Path(getattr(config, 'UPLOAD_DIR', './storage/uploads'))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

orch = Orchestrator()
doc_agent = get_doc_rag_agent()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get('/health')
async def health():
    return JSONResponse({'status': 'ok'})


@app.post('/upload')
async def upload_file(file: UploadFile = File(...)):
    # Save file
    filename = Path(file.filename).name
    save_path = UPLOAD_DIR / filename
    try:
        with save_path.open('wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed saving upload {filename}: {e}")
        raise HTTPException(status_code=500, detail='Failed to save file')
    finally:
        file.file.close()

    # Ingest into DocRAG
    try:
        doc_id = doc_agent.ingest_file(str(save_path))
    except Exception as e:
        logger.error(f"Ingestion failed for {filename}: {e}")
        raise HTTPException(status_code=500, detail='Ingestion failed')

    return JSONResponse({'status': 'uploaded', 'filename': filename, 'document_id': doc_id})


@app.post('/query')
async def query(req: QueryRequest):
    # Detect whether we have uploaded docs
    has_uploaded_docs = any(UPLOAD_DIR.iterdir())
    metadata = {'has_uploaded_docs': has_uploaded_docs, 'top_k': req.top_k}

    try:
        result = orch.run_query(req.query, metadata=metadata)
    except Exception as e:
        logger.error(f"Query orchestration failed: {e}")
        raise HTTPException(status_code=500, detail='Query failed')

    return JSONResponse(result)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(getattr(config, 'PORT', 8000)))
