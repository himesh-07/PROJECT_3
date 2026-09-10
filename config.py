"""
config.py
─────────
Central configuration for ChefMind AI.

MODE selection (set in .env or environment):
  USE_LOCAL=true   →  Ollama LLM + HuggingFace embeddings (100% free, no IBM account)
  USE_LOCAL=false  →  IBM watsonx Granite + IBM slate embeddings (default)
  LLM_BACKEND=gemini → Google Gemini API (free tier available)
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

# Load .env file from the project root (if present)
load_dotenv()


# ── Mode switch ───────────────────────────────────────────────────────────────

# Set USE_LOCAL=true in .env to run fully offline with Ollama (no IBM account needed)
USE_LOCAL: bool = os.getenv("USE_LOCAL", "false").lower() == "true"


# ── watsonx.ai credentials (only required when USE_LOCAL=false) ───────────────

WATSONX_API_KEY: str = os.getenv("WATSONX_API_KEY", "")
WATSONX_URL: str = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT_ID: str = os.getenv("WATSONX_PROJECT_ID", "")

# IBM Granite model for generation
WATSONX_MODEL_ID: str = os.getenv("WATSONX_MODEL_ID", "ibm/granite-13b-chat-v2")


# ── Local model (Ollama) ──────────────────────────────────────────────────────

# Model to use when USE_LOCAL=true. Must be pulled via: ollama pull <name>
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")


# ── Gemini ────────────────────────────────────────────────────────────────────

# Get free API key from: https://aistudio.google.com/app/apikey
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# Gemini model to use (gemini-1.5-flash is free tier)
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ── LLM backend selector ──────────────────────────────────────────────────────
# Options: "gemini" | "ollama" | "watsonx"
# USE_LOCAL=true forces "ollama" unless LLM_BACKEND is explicitly set
_llm_env = os.getenv("LLM_BACKEND", "").lower()
if _llm_env:
    LLM_BACKEND: str = _llm_env
elif USE_LOCAL:
    LLM_BACKEND = "ollama"
else:
    LLM_BACKEND = "watsonx"


# ── Embedding model ───────────────────────────────────────────────────────────

# IBM slate model (used when USE_LOCAL=false)
WATSONX_EMBEDDING_MODEL_ID: str = "ibm/slate-125m-english-rtrvr"

# Free HuggingFace model (used when USE_LOCAL=true) — downloads ~90 MB once
LOCAL_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# Convenience alias used by retriever.py and ingest.py
EMBEDDING_MODEL_ID: str = LOCAL_EMBEDDING_MODEL if USE_LOCAL else WATSONX_EMBEDDING_MODEL_ID


# ── Vector store (Chroma, local persistent) ───────────────────────────────────

CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME: str = "chefmind_recipes"


# ── Chunking parameters ───────────────────────────────────────────────────────

CHUNK_SIZE: int = 512        # characters per chunk
CHUNK_OVERLAP: int = 64      # overlap between adjacent chunks


# ── Retrieval parameters ──────────────────────────────────────────────────────

TOP_K_RESULTS: int = 4       # number of recipe chunks to retrieve


# ── Generation parameters (IBM watsonx only) ──────────────────────────────────

GEN_PARAMS: dict = {
    "max_new_tokens": 1200,
    "temperature": 0.3,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}


# ── Paths ─────────────────────────────────────────────────────────────────────

RECIPES_DIR: str = os.getenv("RECIPES_DIR", "./data/recipes")
