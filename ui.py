"""
Gradio web UI for the AI Answering System.

Features:
- Upload a document (PDF/DOCX/TXT/CSV) and ingest into the DocRAG index
- Ask a query and view grounded answer + cited sources

This UI imports the local orchestrator for direct, in-process calls.
"""
import gradio as gr
from pathlib import Path
import shutil
import logging

from config import config
from orchestration.graph import Orchestrator
from agents.doc_rag_agent import get_doc_rag_agent

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(getattr(config, 'UPLOAD_DIR', './storage/uploads'))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

orch = Orchestrator()
doc_agent = get_doc_rag_agent()


def _save_and_ingest(file_obj):
    if file_obj is None:
        return "No file provided"
    try:
        # Gradio provides a TemporaryFile-like object with .name path
        src_path = Path(file_obj.name)
        dest = UPLOAD_DIR / src_path.name
        shutil.copy(src_path, dest)
        doc_id = doc_agent.ingest_file(str(dest))
        return f"Uploaded and ingested as {doc_id}"
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return f"Upload failed: {e}"


def _ask(query: str):
    if not query:
        return "", ""
    has_uploaded_docs = any(UPLOAD_DIR.iterdir())
    res = orch.run_query(query, metadata={'has_uploaded_docs': has_uploaded_docs})
    answer = res.get('answer', '')
    sources = res.get('sources_text', '')
    return answer, sources


with gr.Blocks(title="AI Answering System") as demo:
    gr.Markdown("""
    # AI Answering System
    Upload documents and ask questions. Answers include inline citations and a sources list.
    """)

    with gr.Row():
        file_input = gr.File(label="Upload Document", file_count="single", type="file")
        upload_btn = gr.Button("Upload & Ingest")
        upload_status = gr.Textbox(label="Upload Status", interactive=False)

    upload_btn.click(fn=_save_and_ingest, inputs=file_input, outputs=upload_status)

    gr.Markdown("---")

    query_input = gr.Textbox(label="Ask a question", placeholder="What are the latest supply chain risk mitigation strategies?", lines=2)
    ask_btn = gr.Button("Ask")
    answer_output = gr.Textbox(label="Answer", lines=8)
    sources_output = gr.Textbox(label="Sources", lines=6)

    ask_btn.click(fn=_ask, inputs=query_input, outputs=[answer_output, sources_output])


if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
