"""
Main entrypoint for the AI Answering System.

Modes:
- `server` -> runs FastAPI via uvicorn
- `ui` -> runs Gradio UI
- `demo` -> runs a small local demo query
"""
import argparse
import subprocess
import sys
from pathlib import Path

from config import config


def run_server():
    import uvicorn
    uvicorn.run('api.app:app', host='0.0.0.0', port=int(getattr(config, 'PORT', 8000)), reload=True)


def run_ui():
    import ui


def demo_query(q: str):
    from orchestration.graph import Orchestrator
    orch = Orchestrator()
    res = orch.run_query(q, metadata={'has_uploaded_docs': False})
    print('Decision:', res['decision'])
    print('\nAnswer:\n', res['answer'])
    print('\nSources:\n', res['sources_text'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['server', 'ui', 'demo'], help='Mode to run')
    parser.add_argument('--query', '-q', type=str, help='Query for demo mode')
    args = parser.parse_args()

    if args.mode == 'server':
        run_server()
    elif args.mode == 'ui':
        run_ui()
    elif args.mode == 'demo':
        if not args.query:
            print('Provide --query for demo')
            sys.exit(1)
        demo_query(args.query)
