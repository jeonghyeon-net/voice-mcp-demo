#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Ollama Cloud 설정
export OLLAMA_API_KEY="0dc326afa3434f9b83121dcfb172d975.uAkl56KLUXjEqF-51XpTpVPA"
export OLLAMA_MODEL="gpt-oss:120b"

python echo.py
